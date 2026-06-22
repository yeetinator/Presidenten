import asyncio
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from game import Presidenten, PlayerType
from playerTypes.random_bot import PresidentenRandomBot
from playerTypes.baseline_bot import PresidentenBaselineBot
from playerTypes.ismcts_bot import PresidentenISMCTSBot
from playerTypes.dmc_bot import PresidentenDMCBot, PresidentenValueNet
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Initializing system lifespan context...")
    print("Spawning ISMCTS persistent process pool with 10 workers.")

    shared_executor = ProcessPoolExecutor(max_workers=10)
    app.state.shared_executor = shared_executor

    yield

    print("Intercepted server shutdown signal sequence.")
    print("Safely terminating ISMCTS background worker processes...")
    shared_executor.shutdown(wait=False)


app = FastAPI(title="Presidenten Game Server", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def make_json_serializable(state: dict):
    clean_state = state.copy()
    if "passed" in clean_state and isinstance(clean_state["passed"], set):
        clean_state["passed"] = list(clean_state["passed"])
    if "active_players" in clean_state and isinstance(
        clean_state["active_players"], set
    ):
        clean_state["active_players"] = list(clean_state["active_players"])
    return clean_state


async def run(
    env: Presidenten,
    assigned_players: dict[
        int,
        PresidentenRandomBot
        | PresidentenBaselineBot
        | PresidentenISMCTSBot
        | PresidentenDMCBot,
    ],
    assign_p: dict,
    websocket: WebSocket,
    human_id,
    shared_executor,
):
    while not env.game_over:
        curr_id = env.curr_turn
        if curr_id is None:
            break

        state = env._get_state(curr_id)
        if state["is_finish_prompt"]:
            if curr_id == human_id:
                JUMP_IN_WINDOW = 1.5
                await websocket.send_json(
                    {
                        "type": "JUMP_IN_PROMPT",
                        "state": make_json_serializable(state),
                        "timeout_seconds": JUMP_IN_WINDOW,
                        "message": "JUMP IN!",
                    }
                )
                try:
                    raw_data = await asyncio.wait_for(
                        websocket.receive_json(), timeout=JUMP_IN_WINDOW
                    )
                    if raw_data.get("type") == "PLAY_MOVE":
                        move_array = raw_data.get("move")
                        env.step(human_id, tuple(move_array))
                        await websocket.send_json(
                            {
                                "type": "GAME_LOG",
                                "message": "Succesful jump in!",
                            }
                        )
                except asyncio.TimeoutError:
                    await websocket.send_json(
                        {
                            "type": "GAME_LOG",
                            "message": "Too slow!",
                        }
                    )
                    env.step(human_id, (0, 0, 0))
                continue
            else:
                await asyncio.sleep(0.5)
                bot = assigned_players[curr_id]

                if isinstance(bot, PresidentenISMCTSBot):
                    chosen_move = await asyncio.to_thread(
                        bot.get_move, state, env, shared_executor, "s"
                    )
                else:
                    chosen_move = await asyncio.to_thread(bot.get_move, state, env)

                env.step(curr_id, chosen_move)
                if chosen_move != (0, 0, 0):
                    await websocket.send_json(
                        {
                            "type": "GAME_LOG",
                            "message": f"Bot {curr_id} jumped in!",
                        }
                    )
                continue

        if assign_p[curr_id] == PlayerType.HUMAN:
            await websocket.send_json(
                {"type": "STATE_UPDATE", "state": make_json_serializable(state)}
            )
            return

        bot = assigned_players[curr_id]
        if isinstance(bot, PresidentenISMCTSBot):
            chosen_move = await asyncio.to_thread(
                bot.get_move, state, env, shared_executor, "s"
            )
        else:
            chosen_move = await asyncio.to_thread(bot.get_move, state, env)

        env.step(curr_id, chosen_move)
        await websocket.send_json(
            {
                "type": "GAME_LOG",
                "message": f"Player {curr_id} ({env.roles[curr_id]}): {Presidenten.visualize_move(chosen_move)}",
            }
        )

        human_perspective_state = env._get_state(human_id)
        await websocket.send_json(
            {
                "type": "STATE_UPDATE",
                "state": make_json_serializable(human_perspective_state),
            }
        )
        await asyncio.sleep(1)

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
    shared_executor = websocket.app.state.shared_executor

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
                    elif p_type == PlayerType.ISMCTS:
                        assigned_players[p_id] = PresidentenISMCTSBot(p_id, 1000)
                    elif p_type == PlayerType.DMC:
                        dmc_model = PresidentenValueNet().to(device)
                        model_path = "backend/playerTypes/best_model.pt"

                        try:
                            load_path = torch.load(model_path, map_location=device)
                            dmc_model.load_state_dict(load_path["model_state_dict"])
                        except Exception:
                            print(f"Warning: Could not load weights from {model_path}.")
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
                await run(
                    env,
                    assigned_players,
                    assign_p,
                    websocket,
                    human_id,
                    shared_executor,
                )
            elif msg_type == "PLAY_MOVE":
                if env is None or env.game_over:
                    continue

                move_array = data.get("move")
                chosen_move = tuple(move_array)

                env.step(human_id, chosen_move)
                await websocket.send_json(
                    {
                        "type": "GAME_LOG",
                        "message": f"({env.roles[human_id]}) You played: {Presidenten.visualize_move(chosen_move)}",
                    }
                )
                await run(
                    env,
                    assigned_players,
                    assign_p,
                    websocket,
                    human_id,
                    shared_executor,
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
                    await run(
                        env,
                        assigned_players,
                        assign_p,
                        websocket,
                        human_id,
                        shared_executor,
                    )
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Unexpected error: {e}")
