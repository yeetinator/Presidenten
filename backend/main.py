import asyncio
from dataclasses import dataclass, field
from pathlib import Path

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from game import President, PlayerType
from playerTypes.random_bot import PresidentRandomBot
from playerTypes.baseline_bot import PresidentBaselineBot
from playerTypes.dmc_bot import PresidentDMCBot, PresidentValueNet

app = FastAPI(title="President Game Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BotPlayer = PresidentRandomBot | PresidentBaselineBot | PresidentDMCBot

MESSAGE_DELAY = 1.5
EXCHANGE_DELAY = 1.2
BOT_THINK_DELAY = 0.5
JUMP_IN_WINDOW = 1.5
REVEAL_DELAY = 2
PASS_DELAY = 1.0

PLAYER_TYPE_LABELS = {
    PlayerType.HUMAN: "Human",
    PlayerType.RANDOM: "Random",
    PlayerType.BASELINE: "Baseline",
    PlayerType.DMC: "DMC",
}

DISCONNECT_MESSAGE_TYPE = "__DISCONNECT__"


@dataclass
class GameTimings:
    message_delay: float = MESSAGE_DELAY
    exchange_delay: float = EXCHANGE_DELAY
    bot_think_delay: float = BOT_THINK_DELAY
    jump_in_window: float = JUMP_IN_WINDOW
    reveal_delay: float = REVEAL_DELAY
    pass_delay: float = PASS_DELAY

    def set_fast_forward(self):
        self.message_delay = 0.75
        self.bot_think_delay = 0.25
        self.pass_delay = 0.5

    def reset(self):
        self.message_delay = MESSAGE_DELAY
        self.exchange_delay = EXCHANGE_DELAY
        self.bot_think_delay = BOT_THINK_DELAY
        self.jump_in_window = JUMP_IN_WINDOW
        self.reveal_delay = REVEAL_DELAY
        self.pass_delay = PASS_DELAY


@dataclass
class GameSession:
    env: President | None = None
    assigned_players: dict[int, BotPlayer] = field(default_factory=dict)
    assign_p: dict[int, PlayerType] = field(default_factory=dict)
    human_id: int = 0
    timings: GameTimings = field(default_factory=GameTimings)
    ready_event: asyncio.Event = field(default_factory=asyncio.Event)
    next_round_event: asyncio.Event = field(default_factory=asyncio.Event)
    dealt_cards_event: asyncio.Event = field(default_factory=asyncio.Event)
    game_over_event: asyncio.Event = field(default_factory=asyncio.Event)


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


def get_exchange_count(env: President, role: str) -> int:
    for high_role, low_role, count in env.role_pairs:
        if role == high_role or role == low_role:
            return count
    return 0


def get_player_type_label(player_type: PlayerType) -> str:
    return PLAYER_TYPE_LABELS.get(player_type, "Unknown")


async def send_game_log(websocket: WebSocket, message: str):
    await websocket.send_json({"type": "GAME_LOG", "message": message})


async def send_state_update(
    websocket: WebSocket,
    env: President,
    human_id: int,
    assign_p: dict[int, PlayerType],
    *,
    clear_jump: bool | None = None,
):
    payload = {
        "type": "STATE_UPDATE",
        "state": enrich_state(env._get_state(human_id), assign_p),
    }
    if clear_jump is not None:
        payload["clearJump"] = clear_jump
    await websocket.send_json(payload)


async def websocket_router(
    websocket: WebSocket,
    play_queue: asyncio.Queue,
    control_queue: asyncio.Queue,
    disconnect_event: asyncio.Event,
):
    try:
        while True:
            message = await websocket.receive_json()
            message_type = message.get("type")

            if message_type in {"PLAY_MOVE", "EXCHANGE_CARDS"}:
                await play_queue.put(message)
            else:
                await control_queue.put(message)
    except WebSocketDisconnect:
        pass
    finally:
        disconnect_event.set()
        sentinel = {"type": DISCONNECT_MESSAGE_TYPE}
        await play_queue.put(sentinel)
        await control_queue.put(sentinel)


async def next_queued_message(
    input_queue: asyncio.Queue,
    pending_messages: list[dict],
    disconnect_event: asyncio.Event,
    *,
    allowed_types: set[str] | None = None,
    timeout: float | None = None,
):
    if disconnect_event.is_set():
        raise WebSocketDisconnect

    if allowed_types is not None:
        for idx, pending_message in enumerate(pending_messages):
            if pending_message.get("type") in allowed_types:
                return pending_messages.pop(idx)

    while True:
        if disconnect_event.is_set():
            raise WebSocketDisconnect

        if timeout is None:
            raw_data = await input_queue.get()
        else:
            raw_data = await asyncio.wait_for(input_queue.get(), timeout=timeout)

        if raw_data.get("type") == DISCONNECT_MESSAGE_TYPE:
            raise WebSocketDisconnect

        if allowed_types is None or raw_data.get("type") in allowed_types:
            return raw_data

        pending_messages.append(raw_data)


def apply_fast_forward(timings: GameTimings):
    timings.set_fast_forward()


async def control_listener(
    control_queue: asyncio.Queue,
    session: GameSession,
    disconnect_event: asyncio.Event,
):
    while not disconnect_event.is_set():
        try:
            control_message = await next_queued_message(
                control_queue,
                [],
                disconnect_event,
                allowed_types={
                    "START_GAME",
                    "NEXT_ROUND",
                    "FAST_FORWARD",
                    "DEALT_CARDS",
                },
            )
        except WebSocketDisconnect:
            return

        message_type = control_message.get("type")

        if message_type == "FAST_FORWARD":
            apply_fast_forward(session.timings)
            continue

        if message_type == "START_GAME":
            num_players = control_message.get("num_players", 4)
            player_config = control_message.get("player_types") or []
            session.assign_p = {
                p_id: PlayerType(player_type)
                for p_id, player_type in enumerate(player_config)
            }
            session.env = President(num_players)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            session.assigned_players, session.human_id = build_assigned_players(
                session.assign_p, device
            )
            session.env.full_reset(next_round=False)
            session.ready_event.set()
            continue

        if message_type == "NEXT_ROUND":
            session.next_round_event.set()
            continue

        if message_type == "DEALT_CARDS":
            session.dealt_cards_event.set()


async def wait_for_event(event: asyncio.Event, disconnect_event: asyncio.Event):
    while not event.is_set():
        if disconnect_event.is_set():
            raise WebSocketDisconnect
        await asyncio.sleep(0.05)


def build_assigned_players(
    assign_p: dict[int, PlayerType], device: torch.device
) -> tuple[dict[int, BotPlayer], int]:
    assigned_players: dict[int, BotPlayer] = {}
    human_id = 0

    model_path = Path(__file__).resolve().parent / "playerTypes" / "model_gen_29750.pt"

    for p_id, p_type in assign_p.items():
        if p_type == PlayerType.HUMAN:
            human_id = p_id
        elif p_type == PlayerType.RANDOM:
            assigned_players[p_id] = PresidentRandomBot(p_id)
        elif p_type == PlayerType.BASELINE:
            assigned_players[p_id] = PresidentBaselineBot(p_id)
        elif p_type == PlayerType.DMC:
            dmc_model = PresidentValueNet().to(device)

            try:
                load_path = torch.load(model_path, map_location=device)
                dmc_model.load_state_dict(load_path["model_state_dict"])
            except Exception as e:
                print(
                    f"Warning: Could not load weights from {model_path}. Reason: {e}. Using untrained model instead."
                )

            dmc_model.eval()
            assigned_players[p_id] = PresidentDMCBot(p_id, dmc_model, device)

    return assigned_players, human_id


async def wait_for_exchange_cards(
    input_queue: asyncio.Queue,
    pending_messages: list[dict],
    disconnect_event: asyncio.Event,
    can_choose: bool,
):
    while True:
        raw_data = await next_queued_message(
            input_queue,
            pending_messages,
            disconnect_event,
            allowed_types={"EXCHANGE_CARDS"},
        )

        suits = raw_data.get("suits", []) if can_choose else []
        return parse_suits_to_selection(suits)


async def run_exchange_phase(
    env: President,
    assigned_players: dict[int, BotPlayer],
    websocket: WebSocket,
    human_id: int,
    assign_p: dict[int, PlayerType],
    input_queue: asyncio.Queue,
    pending_messages: list[dict],
    disconnect_event: asyncio.Event,
    timings: GameTimings,
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

    cards_to_pass[human_id] = await wait_for_exchange_cards(
        input_queue,
        pending_messages,
        disconnect_event,
        can_choose,
    )

    print(f"Cards to pass: {cards_to_pass}")
    env.exchange_log = {}
    for pair in env.role_pairs:
        high_role, low_role, count = pair
        env.exchange_cards(pair, cards_to_pass)

        await send_game_log(
            websocket,
            f"{high_role} and {low_role} have exchanged {count} card(s).",
        )
        await send_state_update(websocket, env, human_id, assign_p)
        await asyncio.sleep(timings.exchange_delay)


async def run(
    env: President,
    assigned_players: dict[int, BotPlayer],
    assign_p: dict,
    websocket: WebSocket,
    human_id,
    input_queue: asyncio.Queue,
    pending_messages: list[dict],
    disconnect_event: asyncio.Event,
    timings: GameTimings,
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
                await websocket.send_json(
                    {
                        "type": "JUMP_IN_PROMPT",
                        "state": enrich_state(env._get_state(human_id), assign_p),
                        "message": "JUMP IN!",
                    }
                )

                try:
                    raw_data = await next_queued_message(
                        input_queue,
                        pending_messages,
                        disconnect_event,
                        allowed_types={"PLAY_MOVE"},
                        timeout=timings.jump_in_window,
                    )
                    if raw_data.get("type") == "PLAY_MOVE" and raw_data.get(
                        "jump", True
                    ):
                        suits_array = raw_data.get("suits", [])
                        parsed_array = get_finish_suits(suits_array, state)
                        chosen_move = parse_suits_to_move(parsed_array)
                        env.step(human_id, chosen_move, parsed_array)

                        await send_game_log(websocket, "Succesful jump in!")
                        await send_state_update(websocket, env, human_id, assign_p)
                        await asyncio.sleep(MESSAGE_DELAY)

                        if env.was_pile_reset:
                            env.clear_pile()
                            if env.curr_turn != human_id:
                                await send_state_update(
                                    websocket, env, human_id, assign_p
                                )
                        if env.game_over:
                            break
                        continue
                except asyncio.TimeoutError:
                    await send_game_log(websocket, "Too slow!")
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
                await asyncio.sleep(timings.bot_think_delay)

                bot = assigned_players[curr_id]
                chosen_move = await asyncio.to_thread(bot.get_move, state, env)
                env.step(curr_id, chosen_move)
                if chosen_move != (0, 0, 0):
                    await send_game_log(
                        websocket,
                        f"Player {curr_id} ({env.roles[curr_id]}): jumped in with {President.visualize_move(chosen_move)}!",
                    )
                    await send_state_update(websocket, env, human_id, assign_p)
                    await asyncio.sleep(timings.message_delay)

                    if env.was_pile_reset:
                        env.clear_pile()
                        await send_state_update(websocket, env, human_id, assign_p)
                elif not env.pending_finish:
                    await asyncio.sleep(timings.pass_delay)
                if env.was_pile_reset:
                    env.clear_pile()
                    await send_state_update(websocket, env, human_id, assign_p)
                    await asyncio.sleep(timings.message_delay)
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
            raw_data = await next_queued_message(
                input_queue,
                pending_messages,
                disconnect_event,
                allowed_types={"PLAY_MOVE"},
            )

            suits_array = raw_data.get("suits", [])
            if raw_data.get("jump"):
                suits_array = get_finish_suits(suits_array, env._get_state(human_id))
            elif env.pending_finish and env.pending_finish["queue"][0][2] == human_id:
                env.step(human_id, (0, 0, 0))
                if env.pending_finish:
                    if env.pending_finish["queue"][0][2] == human_id:
                        env.step(human_id, (0, 0, 0))
                    else:
                        bot_move = await asyncio.to_thread(
                            assigned_players[
                                env.pending_finish["queue"][0][2]
                            ].get_move,
                            env._get_state(env.pending_finish["queue"][0][2]),
                            env,
                        )
                        if bot_move != (0, 0, 0):
                            env.curr_turn = env.pending_finish["queue"][0][2]
                            continue
                        else:
                            env.step(env.pending_finish["queue"][0][2], (0, 0, 0))

            chosen_move = parse_suits_to_move(suits_array)
            env.step(human_id, chosen_move, suits_array)

            await send_game_log(
                websocket,
                f"({env.roles[human_id]}) You played: {President.visualize_move(chosen_move)}",
            )
            await send_state_update(websocket, env, human_id, assign_p)

            if env.was_pile_reset and not env.pending_finish and not env.game_over:
                await asyncio.sleep(timings.message_delay)

                env.clear_pile()
                await send_state_update(websocket, env, human_id, assign_p)
            else:
                await asyncio.sleep(timings.message_delay)
            continue

        bot = assigned_players[curr_id]
        chosen_move = await asyncio.to_thread(
            bot.get_move, env._get_state(curr_id), env
        )

        if (
            chosen_move != (0, 0, 0)
            and env.pending_finish
            and env.pending_finish["queue"][0][2] == human_id
        ):
            await send_game_log(
                websocket,
                f"Player {curr_id} ({env.roles[curr_id]}) played before you could jump in!",
            )
            env.step(human_id, (0, 0, 0))

            if env.pending_finish:
                if env.pending_finish["queue"][0][2] == human_id:
                    env.step(human_id, (0, 0, 0))
                else:
                    continue
        env.step(curr_id, chosen_move)

        await send_game_log(
            websocket,
            f"Player {curr_id} ({env.roles[curr_id]}): {President.visualize_move(chosen_move)}",
        )
        if env.curr_turn != human_id or env.was_pile_reset:
            await send_state_update(websocket, env, human_id, assign_p)

        if env.was_pile_reset and not env.game_over:
            await asyncio.sleep(timings.message_delay)

            env.clear_pile()
            if env.curr_turn == human_id:
                continue

            await send_state_update(websocket, env, human_id, assign_p)
            await asyncio.sleep(timings.message_delay)
        elif not env.pending_finish and not env.game_over and env.curr_turn != human_id:
            await asyncio.sleep(timings.message_delay)

    last_active_player = list(env.playing)[0] if env.playing else None
    if last_active_player is None:
        for p_id in range(env.players):
            if env.hands[p_id]:
                last_active_player = p_id
                break

    if last_active_player is not None:
        await asyncio.sleep(timings.message_delay)
        if last_active_player != human_id:
            await websocket.send_json(
                {"type": "REVEAL_BOT", "seat": last_active_player}
            )
            await asyncio.sleep(timings.reveal_delay)

    env.assign_roles()
    await websocket.send_json(
        {
            "type": "ROUND_OVER",
            "scores": env.scores,
            "roles": env.roles,
            "out_order": env.out_order,
        }
    )


async def run_connection(
    websocket: WebSocket,
    play_queue: asyncio.Queue,
    control_queue: asyncio.Queue,
    disconnect_event: asyncio.Event,
):
    session = GameSession()
    pending_messages: list[dict] = []
    control_task = asyncio.create_task(
        control_listener(control_queue, session, disconnect_event)
    )

    try:
        await wait_for_event(session.ready_event, disconnect_event)

        if session.env is None:
            return

        await websocket.send_json(
            {
                "type": "LOG_ALERT",
                "message": "Game lobby created. Dealing hands...",
            }
        )
        await websocket.send_json(
            {
                "type": "DEAL_CARDS",
                "state": enrich_state(
                    session.env._get_state(session.human_id), session.assign_p
                ),
            }
        )
        await wait_for_event(session.dealt_cards_event, disconnect_event)

        session.timings.reset()
        if session.env.curr_turn != session.human_id:
            await asyncio.sleep(session.timings.message_delay)
        await run(
            session.env,
            session.assigned_players,
            session.assign_p,
            websocket,
            session.human_id,
            play_queue,
            pending_messages,
            disconnect_event,
            session.timings,
        )

        while not disconnect_event.is_set():
            await wait_for_event(session.next_round_event, disconnect_event)
            session.next_round_event.clear()

            if session.env is None or not session.env.game_over:
                continue

            session.env.full_reset(next_round=True)
            await websocket.send_json(
                {
                    "type": "LOG_ALERT",
                    "message": f"Starting Round {session.env.round}...",
                }
            )
            await websocket.send_json(
                {
                    "type": "DEAL_CARDS",
                    "state": enrich_state(
                        session.env._get_state(session.human_id), session.assign_p
                    ),
                }
            )
            await wait_for_event(session.dealt_cards_event, disconnect_event)

            session.timings.reset()
            await run_exchange_phase(
                session.env,
                session.assigned_players,
                websocket,
                session.human_id,
                session.assign_p,
                play_queue,
                pending_messages,
                disconnect_event,
                session.timings,
            )
            await send_state_update(
                websocket, session.env, session.human_id, session.assign_p
            )
            await asyncio.sleep(session.timings.message_delay)
            await run(
                session.env,
                session.assigned_players,
                session.assign_p,
                websocket,
                session.human_id,
                play_queue,
                pending_messages,
                disconnect_event,
                session.timings,
            )
    except WebSocketDisconnect:
        return
    finally:
        control_task.cancel()
        await asyncio.gather(control_task, return_exceptions=True)


@app.websocket("/ws/game")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    play_queue: asyncio.Queue = asyncio.Queue()
    control_queue: asyncio.Queue = asyncio.Queue()
    disconnect_event = asyncio.Event()

    router_task = asyncio.create_task(
        websocket_router(websocket, play_queue, control_queue, disconnect_event)
    )
    game_task = asyncio.create_task(
        run_connection(websocket, play_queue, control_queue, disconnect_event)
    )

    try:
        await asyncio.gather(router_task, game_task)
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise
    finally:
        router_task.cancel()
        game_task.cancel()
        await asyncio.gather(router_task, game_task, return_exceptions=True)
