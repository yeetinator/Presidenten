import asyncio
import torch
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from game import Presidenten, PlayerType
from playerTypes.random_bot import PresidentenRandomBot
from playerTypes.baseline_bot import PresidentenBaselineBot
from playerTypes.dmc_bot import PresidentenDMCBot, PresidentenValueNet

app = FastAPI(title="Presidenten Game Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _parse_helper(suits):
    char_to_value = {"T": 10, "J": 11, "Q": 12, "K": 13, "A": 14, "2": 15}
    return [
        (
            char_to_value[r]
            if (r := suit if len(suit) == 1 else suit[:-1]) in char_to_value
            else int(r)
        )
        for suit in suits
    ]


def parse_suits_to_move(suits):
    if not suits:
        return (0, 0, 0)

    vals = _parse_helper(suits)
    count = len(vals)
    twos_used = vals.count(15)
    non_twos = [v for v in vals if v != 15]
    card_val = non_twos[0] if non_twos else 15

    return (card_val, count, twos_used)


def parse_suits_to_selection(suits):
    if not suits:
        return []
    return _parse_helper(suits)


def parse_moves_to_suits(moves: list):
    suits_moves = []
    value_to_char = {10: "T", 11: "J", 12: "Q", 13: "K", 14: "A", 15: "2"}

    for card_val, count, twos_used in moves:
        if card_val == 0:
            continue

        suit = value_to_char.get(card_val, str(card_val))
        suits_moves.append([suit] * (count - twos_used) + ["2"] * twos_used)
    return suits_moves


def get_finish_suits(suits, state):
    if not suits:
        return []
    return [card for card in state["suited_hand"] if card.startswith(suits[0])]


def make_json_serializable(state: dict):
    clean_state = state.copy()
    if "passed" in clean_state and isinstance(clean_state["passed"], set):
        clean_state["passed"] = list(clean_state["passed"])

    if "active_players" in clean_state and isinstance(
        clean_state["active_players"], set
    ):
        clean_state["active_players"] = list(clean_state["active_players"])
    return clean_state


def enrich_state(state: dict, assign_p: dict[int, PlayerType]):
    clean_state = make_json_serializable(state)
    clean_state["player_types"] = {
        p_id: get_player_type_label(player_type)
        for p_id, player_type in assign_p.items()
    }
    clean_state["can_pass"] = (0, 0, 0) in state["legal_moves"]
    clean_state["legal_moves_suits"] = parse_moves_to_suits(state["legal_moves"])

    return clean_state


def get_exchange_count(env: Presidenten, role: str) -> int:
    for high_role, low_role, count in env.role_pairs:
        if role == high_role or role == low_role:
            return count
    return 0


def get_player_type_label(player_type: PlayerType) -> str:
    labels = {
        PlayerType.HUMAN: "Human",
        PlayerType.RANDOM: "Random",
        PlayerType.BASELINE: "Baseline",
        PlayerType.DMC: "DMC",
    }
    return labels.get(player_type, "Unknown")


async def run_exchange_phase(
    env: Presidenten,
    assigned_players: dict[
        int,
        PresidentenRandomBot | PresidentenBaselineBot | PresidentenDMCBot,
    ],
    websocket: WebSocket,
    human_id: int,
    assign_p: dict[int, PlayerType],
):
    cards_to_pass: dict[int | str, list[int]] = {}
    human_role = env.roles[human_id]
    required_cards = get_exchange_count(env, human_role)
    can_choose = human_role in {"President", "Vice-President", "Secretary"}

    for p_id, role in env.roles.items():
        if role == "Citizen" or p_id == human_id:
            continue

        bot = assigned_players[p_id]
        state = env._get_state(p_id)
        chosen_cards = await asyncio.to_thread(bot.choose_cards_to_pass, state)
        cards_to_pass[p_id] = chosen_cards
    await websocket.send_json(
        {
            "type": "EXCHANGE_PROMPT",
            "state": enrich_state(env._get_state(human_id), assign_p),
            "required_cards": required_cards,
            "can_choose": can_choose,
        }
    )

    while True:
        raw_data = await websocket.receive_json()
        if raw_data.get("type") != "EXCHANGE_CARDS":
            continue

        suits = raw_data.get("suits", []) if can_choose else []
        cards = parse_suits_to_selection(suits)

        if not isinstance(cards, list):
            continue

        cards_to_pass[human_id] = cards
        break

    print(f"Cards to pass: {cards_to_pass}")
    env.exchange_log = {}
    for pair in env.role_pairs:
        high_role, low_role, count = pair
        env.exchange_cards(pair, cards_to_pass)

        await websocket.send_json(
            {
                "type": "GAME_LOG",
                "message": f"{high_role} and {low_role} have exchanged {count} card(s).",
            }
        )
        await websocket.send_json(
            {
                "type": "STATE_UPDATE",
                "state": enrich_state(env._get_state(human_id), assign_p),
            }
        )
        await asyncio.sleep(1.2)


async def run(
    env: Presidenten,
    assigned_players: dict[
        int,
        PresidentenRandomBot | PresidentenBaselineBot | PresidentenDMCBot,
    ],
    assign_p: dict,
    websocket: WebSocket,
    human_id,
    human_waiting=False,
):
    while not env.game_over:
        curr_id = env.curr_turn
        if curr_id is None:
            break

        state = env._get_state(curr_id)
        if state["is_finish_prompt"] or env.pending_finish:
            if curr_id == human_id or (
                env.pending_finish and env.pending_finish["queue"][0][2] == human_id
            ):
                JUMP_IN_WINDOW = 1.5
                await websocket.send_json(
                    {
                        "type": "JUMP_IN_PROMPT",
                        "state": enrich_state(env._get_state(human_id), assign_p),
                        "message": "JUMP IN!",
                    }
                )

                try:
                    raw_data = await asyncio.wait_for(
                        websocket.receive_json(), timeout=JUMP_IN_WINDOW
                    )
                    if raw_data.get("type") == "PLAY_MOVE" and raw_data.get(
                        "jump", True
                    ):
                        suits_array = raw_data.get("suits", [])
                        parsed_array = get_finish_suits(suits_array, state)
                        chosen_move = parse_suits_to_move(parsed_array)
                        env.step(human_id, chosen_move, parsed_array)

                        await websocket.send_json(
                            {
                                "type": "GAME_LOG",
                                "message": "Succesful jump in!",
                            }
                        )
                        await websocket.send_json(
                            {
                                "type": "STATE_UPDATE",
                                "state": enrich_state(
                                    env._get_state(human_id), assign_p
                                ),
                            }
                        )
                        await asyncio.sleep(1.5)

                        if env.was_pile_reset:
                            env.clear_pile()
                            await websocket.send_json(
                                {
                                    "type": "STATE_UPDATE",
                                    "state": enrich_state(
                                        env._get_state(human_id), assign_p
                                    ),
                                }
                            )
                        if env.game_over:
                            break
                        return
                except asyncio.TimeoutError:
                    await websocket.send_json(
                        {
                            "type": "GAME_LOG",
                            "message": "Too slow!",
                        }
                    )
                    if env.pending_finish:
                        if (
                            curr_id
                            == env._get_next_active_player(
                                env.pending_finish["resume_turn"]
                            )
                            and env.pending_finish["resume_played"]
                        ):
                            env.pending_finish["resume_turn"] = curr_id

                        curr_id = env.pending_finish["resume_turn"]
                        env.pending_finish["resume_played"] = True
            else:
                await asyncio.sleep(0.5)

                bot = assigned_players[curr_id]
                chosen_move = await asyncio.to_thread(bot.get_move, state, env)
                env.step(curr_id, chosen_move)
                if chosen_move != (0, 0, 0):
                    await websocket.send_json(
                        {
                            "type": "GAME_LOG",
                            "message": f"Player {curr_id} ({env.roles[curr_id]}): jumped in with {Presidenten.visualize_move(chosen_move)}!",
                        }
                    )
                    await websocket.send_json(
                        {
                            "type": "STATE_UPDATE",
                            "state": enrich_state(env._get_state(human_id), assign_p),
                        }
                    )
                    await asyncio.sleep(1.5)

                    if env.was_pile_reset:
                        env.clear_pile()
                        await websocket.send_json(
                            {
                                "type": "STATE_UPDATE",
                                "state": enrich_state(
                                    env._get_state(human_id), assign_p
                                ),
                            }
                        )
                elif human_waiting:
                    return True
                elif not env.pending_finish:
                    await asyncio.sleep(1)
                if env.was_pile_reset:
                    env.clear_pile()
                    await websocket.send_json(
                        {
                            "type": "STATE_UPDATE",
                            "state": enrich_state(env._get_state(human_id), assign_p),
                        }
                    )
                    await asyncio.sleep(1.5)
                continue

        if assign_p[curr_id] == PlayerType.HUMAN:
            pending_bool = False if env.pending_finish else True
            await websocket.send_json(
                {
                    "type": "STATE_UPDATE",
                    "state": enrich_state(
                        env._get_state(human_id, only_finish=pending_bool), assign_p
                    ),
                    "clearJump": pending_bool,
                }
            )
            return

        bot = assigned_players[curr_id]
        chosen_move = await asyncio.to_thread(
            bot.get_move, env._get_state(curr_id), env
        )

        if (
            chosen_move != (0, 0, 0)
            and env.pending_finish
            and env.pending_finish["queue"][0][2] == human_id
        ):
            await websocket.send_json(
                {
                    "type": "GAME_LOG",
                    "message": f"Player {curr_id} ({env.roles[curr_id]}) played before you could jump in!",
                }
            )
            env.step(human_id, (0, 0, 0))

            if env.pending_finish:
                if env.pending_finish["queue"][0][2] == human_id:
                    env.step(human_id, (0, 0, 0))
                else:
                    continue
        env.step(curr_id, chosen_move)

        await websocket.send_json(
            {
                "type": "GAME_LOG",
                "message": f"Player {curr_id} ({env.roles[curr_id]}): {Presidenten.visualize_move(chosen_move)}",
            }
        )
        await websocket.send_json(
            {
                "type": "STATE_UPDATE",
                "state": enrich_state(env._get_state(human_id), assign_p),
            }
        )

        if env.was_pile_reset and not env.game_over:
            await asyncio.sleep(1.5)

            env.clear_pile()
            await websocket.send_json(
                {
                    "type": "STATE_UPDATE",
                    "state": enrich_state(env._get_state(human_id), assign_p),
                }
            )
            await asyncio.sleep(1.5)

            if env.curr_turn == human_id:
                return
        elif not env.pending_finish and not env.game_over:
            if env.curr_turn == human_id:
                return
            await asyncio.sleep(1.5)

    last_active_player = list(env.playing)[0] if env.playing else None
    if last_active_player is None:
        for p_id in range(env.players):
            if env.hands[p_id]:
                last_active_player = p_id
                break

    if last_active_player is not None:
        await asyncio.sleep(1.5)
        if last_active_player != human_id:
            await websocket.send_json(
                {"type": "REVEAL_BOT", "seat": last_active_player}
            )
            await asyncio.sleep(2.5)

    env.assign_roles()
    await websocket.send_json(
        {
            "type": "ROUND_OVER",
            "scores": env.scores,
            "roles": env.roles,
            "out_order": env.out_order,
        }
    )


@app.websocket("/ws/game")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    env = None
    assigned_players = {}
    assign_p = {}
    human_id = 0

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "START_GAME":
                num_players = data.get("num_players", 4)
                player_config = data.get(
                    "player_types",
                )
                assign_p = {p_id: PlayerType(t) for p_id, t in enumerate(player_config)}
                env = Presidenten(num_players)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                for p_id, p_type in assign_p.items():
                    if p_type == PlayerType.HUMAN:
                        human_id = p_id
                    elif p_type == PlayerType.RANDOM:
                        assigned_players[p_id] = PresidentenRandomBot(p_id)
                    elif p_type == PlayerType.BASELINE:
                        assigned_players[p_id] = PresidentenBaselineBot(p_id)
                    elif p_type == PlayerType.DMC:
                        dmc_model = PresidentenValueNet().to(device)
                        base_dir = os.path.dirname(os.path.abspath(__file__))
                        model_path = os.path.join(
                            base_dir, "playerTypes", "best_model_27500.pt"
                        )

                        try:
                            load_path = torch.load(model_path, map_location=device)
                            dmc_model.load_state_dict(load_path["model_state_dict"])
                        except Exception as e:
                            print(
                                f"Warning: Could not load weights from {model_path}. Reason: {e}. Using untrained model instead."
                            )

                        dmc_model.eval()
                        assigned_players[p_id] = PresidentenDMCBot(
                            p_id, dmc_model, device
                        )

                env.full_reset(next_round=False)
                await websocket.send_json(
                    {
                        "type": "LOG_ALERT",
                        "message": "Game lobby created. Dealing hands...",
                    }
                )
                await websocket.send_json(
                    {
                        "type": "STATE_UPDATE",
                        "state": enrich_state(env._get_state(human_id), assign_p),
                    }
                )
                await asyncio.sleep(1.5)
                await run(
                    env,
                    assigned_players,
                    assign_p,
                    websocket,
                    human_id,
                )
            elif msg_type == "PLAY_MOVE":
                if env is None or env.game_over:
                    continue

                is_turn = env.curr_turn == human_id or (
                    env.pending_finish and env.pending_finish["queue"][0][2] == human_id
                )
                if not is_turn:
                    continue

                if (
                    "jump" not in data
                    and env.pending_finish
                    and env.pending_finish["queue"][0][2] == human_id
                ):
                    env.step(human_id, (0, 0, 0))

                    if env.pending_finish:
                        if env.pending_finish["queue"][0][2] == human_id:
                            env.step(human_id, (0, 0, 0))
                        else:
                            bot_passed = await run(
                                env,
                                assigned_players,
                                assign_p,
                                websocket,
                                human_id,
                                True,
                            )
                            if not bot_passed:
                                continue

                suits_array = data.get("suits", [])
                if data.get("jump"):
                    suits_array = get_finish_suits(
                        suits_array, env._get_state(human_id)
                    )

                chosen_move = parse_suits_to_move(suits_array)
                env.step(human_id, chosen_move, suits_array)

                await websocket.send_json(
                    {
                        "type": "GAME_LOG",
                        "message": f"({env.roles[human_id]}) You played: {Presidenten.visualize_move(chosen_move)}",
                    }
                )
                await websocket.send_json(
                    {
                        "type": "STATE_UPDATE",
                        "state": enrich_state(env._get_state(human_id), assign_p),
                    }
                )

                if env.was_pile_reset and not env.pending_finish and not env.game_over:
                    await asyncio.sleep(1.5)

                    env.clear_pile()
                    await websocket.send_json(
                        {
                            "type": "STATE_UPDATE",
                            "state": enrich_state(env._get_state(human_id), assign_p),
                        }
                    )
                else:
                    await asyncio.sleep(1.5)
                await run(
                    env,
                    assigned_players,
                    assign_p,
                    websocket,
                    human_id,
                )
            elif msg_type == "NEXT_ROUND":
                if env and env.game_over:
                    env.full_reset(next_round=True)
                    await websocket.send_json(
                        {
                            "type": "LOG_ALERT",
                            "message": f"Starting Round {env.round}...",
                        }
                    )
                    await run_exchange_phase(
                        env,
                        assigned_players,
                        websocket,
                        human_id,
                        assign_p,
                    )
                    await websocket.send_json(
                        {
                            "type": "STATE_UPDATE",
                            "state": enrich_state(env._get_state(human_id), assign_p),
                        }
                    )
                    await asyncio.sleep(1.5)
                    await run(
                        env,
                        assigned_players,
                        assign_p,
                        websocket,
                        human_id,
                    )
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Unexpected error: {e}")
