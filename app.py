from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from game import Presidenten
from playerTypes.baseline_bot import PresidentenBaselineBot
from playerTypes.ismcts_bot import PresidentenISMCTSBot
from playerTypes.random_bot import PresidentenRandomBot

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"

BOT_LABELS = {
    "random": "Random",
    "baseline": "Baseline",
    "ismcts": "ISMCTS",
}

DEFAULT_CONFIG = {
    "human_player": 0,
    "player_kinds": ["human", "baseline", "baseline", "baseline"],
    "ismcts_iterations": 140,
    "max_rounds": 10,
}

app = FastAPI(title="Presidenten GUI")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"ok": True})


def build_bot(kind: str, player_id: int, iterations: int):
    if kind == "random":
        return PresidentenRandomBot(player_id=player_id)
    if kind == "baseline":
        return PresidentenBaselineBot(player_id=player_id)
    if kind == "ismcts":
        return PresidentenISMCTSBot(player_id=player_id, iterations=iterations)
    raise ValueError(f"Unsupported bot kind: {kind}")


def normalize_config(raw_config: dict[str, Any] | None) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if raw_config:
        config.update(raw_config)

    human_player = int(config.get("human_player", 0))
    if human_player < 0 or human_player >= 4:
        human_player = 0

    player_kinds = list(config.get("player_kinds", DEFAULT_CONFIG["player_kinds"]))
    if len(player_kinds) != 4:
        player_kinds = list(DEFAULT_CONFIG["player_kinds"])

    for index, kind in enumerate(player_kinds):
        if index == human_player:
            player_kinds[index] = "human"
        elif kind not in BOT_LABELS:
            player_kinds[index] = "baseline"

    iterations = int(
        config.get("ismcts_iterations", DEFAULT_CONFIG["ismcts_iterations"])
    )
    iterations = max(20, min(iterations, 1000))

    max_rounds = int(config.get("max_rounds", DEFAULT_CONFIG["max_rounds"]))
    max_rounds = max(1, min(max_rounds, 100))

    return {
        "human_player": human_player,
        "player_kinds": player_kinds,
        "ismcts_iterations": iterations,
        "max_rounds": max_rounds,
    }


class WebGameSession:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = normalize_config(config)
        self.env = Presidenten(players=4, verbose=False)
        self.bots: dict[int, Any] = {}
        self.finalized = False
        self.reset_match()

    @property
    def human_player(self) -> int:
        return int(self.config["human_player"])

    @property
    def max_rounds(self) -> int:
        return int(self.config["max_rounds"])

    def reset_match(self) -> None:
        self.bots = {
            player_id: build_bot(
                self.config["player_kinds"][player_id],
                player_id,
                self.config["ismcts_iterations"],
            )
            for player_id in range(4)
            if player_id != self.human_player
        }
        self.env = Presidenten(players=4, verbose=False)
        self.env.full_reset()
        self.finalized = False

    def start_next_round(self) -> None:
        self.env.full_reset(next_round=True)
        self.finalized = False

    def player_label(self, player_id: int) -> str:
        if player_id == self.human_player:
            return f"You (Seat {player_id})"
        kind = self.config["player_kinds"][player_id]
        return f"Seat {player_id} ({BOT_LABELS[kind]})"

    def finalize_round(self) -> None:
        if self.finalized or not self.env.game_over:
            return
        self.env.assign_roles()
        self.finalized = True

    def current_state(self, message: str | None = None) -> dict[str, Any]:
        self.finalize_round()

        human_hand = self.env.hands[self.human_player].copy()
        legal_moves = self.env.get_legal_moves(self.human_player)
        current_turn = self.env.curr_turn
        phase = "game_over" if self.env.game_over else "play"

        return {
            "message": message,
            "phase": phase,
            "finished": self.env.game_over,
            "game_over": self.env.game_over,
            "round_number": self.env.round,
            "max_rounds": self.max_rounds,
            "round": self.env.round,
            "human_player": self.human_player,
            "current_turn": current_turn,
            "current_turn_label": self.player_label(current_turn),
            "human_is_current": current_turn == self.human_player
            and not self.env.game_over,
            "your_turn": current_turn == self.human_player and not self.env.game_over,
            "your_role": self.env.roles[self.human_player],
            "hand": Presidenten.visualize_hand(human_hand),
            "hand_values": human_hand,
            "legal_moves": [list(move) for move in legal_moves],
            "legal_move_labels": [
                Presidenten.visualize_move(move) for move in legal_moves
            ],
            "last_move": list(self.env.last_move),
            "last_move_label": Presidenten.visualize_move(self.env.last_move),
            "passed": sorted(self.env.passed),
            "active_players": sorted(self.env.playing),
            "player_states": [
                {
                    "player_id": player_id,
                    "label": self.player_label(player_id),
                    "kind": self.config["player_kinds"][player_id],
                    "role": self.env.roles[player_id],
                    "hand_count": len(self.env.hands[player_id]),
                    "score": self.env.scores[player_id][0],
                    "round_wins": self.env.scores[player_id][1],
                    "is_human": player_id == self.human_player,
                    "is_active": player_id in self.env.playing,
                }
                for player_id in range(self.env.players)
            ],
            "scores": [
                {
                    "player_id": player_id,
                    "score": self.env.scores[player_id][0],
                    "round_wins": self.env.scores[player_id][1],
                }
                for player_id in range(self.env.players)
            ],
            "history": [
                {
                    "player_id": player_id,
                    "label": self.player_label(player_id),
                    "move": list(move),
                    "move_label": Presidenten.visualize_move(move),
                }
                for player_id, move in self.env.history[-40:]
            ],
            "final_order": list(self.env.out_order),
            "ended_2": list(self.env.ended_2),
            "exchange_requirement": None,
        }


async def play_bot_turn(session: WebGameSession) -> dict[str, Any]:
    player_id = session.env.curr_turn
    bot = session.bots[player_id]
    state = session.env._get_state(player_id)

    move = await asyncio.to_thread(lambda: bot.get_move(state, real_env=session.env))
    if move not in state["legal_moves"]:
        move = state["legal_moves"][0]

    session.env.step(player_id, move)
    return {
        "player_id": player_id,
        "move": list(move),
        "move_label": Presidenten.visualize_move(move),
    }


async def advance_until_human_or_end(
    session: WebGameSession, websocket: WebSocket
) -> None:
    while not session.env.game_over and session.env.curr_turn != session.human_player:
        action = await play_bot_turn(session)
        await websocket.send_json(
            {
                "type": "log",
                "message": f"{session.player_label(action['player_id'])} played {action['move_label']}.",
                "action": action,
                "state": session.current_state(),
            }
        )

    if session.env.game_over:
        await websocket.send_json(
            {
                "type": "game_over",
                "message": "Round complete.",
                "state": session.current_state("Round complete."),
            }
        )

        if session.env.round < session.max_rounds:
            session.start_next_round()
            await websocket.send_json(
                {
                    "type": "state",
                    "message": f"Round {session.env.round} started.",
                    "state": session.current_state(
                        f"Round {session.env.round} started."
                    ),
                }
            )
            await advance_until_human_or_end(session, websocket)
    else:
        await websocket.send_json(
            {
                "type": "state",
                "state": session.current_state("Your turn."),
            }
        )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    session = WebGameSession()
    try:
        await websocket.send_json(
            {
                "type": "state",
                "state": session.current_state("Connected. Start a new game."),
            }
        )

        while True:
            payload = await websocket.receive_json()
            message_type = payload.get("type")

            if message_type == "start":
                session = WebGameSession(payload.get("config"))
                await websocket.send_json(
                    {
                        "type": "state",
                        "state": session.current_state("New game started."),
                    }
                )
                await advance_until_human_or_end(session, websocket)
                continue

            if message_type == "restart":
                session.reset_match()
                await websocket.send_json(
                    {
                        "type": "state",
                        "state": session.current_state("Game restarted."),
                    }
                )
                await advance_until_human_or_end(session, websocket)
                continue

            if message_type == "move":
                if session.env.game_over:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": "The round is already over. Restart to play again.",
                        }
                    )
                    continue

                if session.env.curr_turn != session.human_player:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": "It is not your turn yet.",
                            "state": session.current_state(),
                        }
                    )
                    continue

                move = payload.get("move")
                if not isinstance(move, list) or len(move) != 3:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": "Move must be a three-item array.",
                        }
                    )
                    continue

                chosen_move = tuple(int(value) for value in move)
                legal_moves = session.env.get_legal_moves(session.human_player)
                if chosen_move not in legal_moves:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": "That move is not legal in the current state.",
                            "state": session.current_state(),
                        }
                    )
                    continue

                session.env.step(session.human_player, chosen_move)
                if session.env.game_over:
                    await websocket.send_json(
                        {
                            "type": "game_over",
                            "message": f"You played {Presidenten.visualize_move(chosen_move)}.",
                            "state": session.current_state(
                                f"You played {Presidenten.visualize_move(chosen_move)}."
                            ),
                        }
                    )
                    continue

                await websocket.send_json(
                    {
                        "type": "state",
                        "message": f"You played {Presidenten.visualize_move(chosen_move)}.",
                        "state": session.current_state(
                            f"You played {Presidenten.visualize_move(chosen_move)}."
                        ),
                    }
                )
                await advance_until_human_or_end(session, websocket)
                continue

            await websocket.send_json(
                {
                    "type": "error",
                    "message": "Unknown command.",
                }
            )
    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8010, reload=True)
