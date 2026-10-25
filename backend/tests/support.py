"""Helpers shared by the tests: a fake browser socket and a simple scripted player."""

import asyncio
import contextlib
import functools
import json

from fastapi import WebSocketDisconnect

import main
from game import PlayerType, President
from game.engine import MAPPING_SV
from playerTypes.player import Player


def async_test(test):
    """Run an async test function in its own event loop (no pytest plugin needed)."""

    @functools.wraps(test)
    def wrapper(*args, **kwargs):
        return asyncio.run(test(*args, **kwargs))

    return wrapper


class FakeWebSocket:
    """Stands in for the browser: what the server sends is queued for the test to read."""

    def __init__(self):
        self.from_server: asyncio.Queue = asyncio.Queue()
        self.to_server: asyncio.Queue = asyncio.Queue()
        self.closed = False

    async def accept(self):
        pass

    async def send_json(self, data):
        if self.closed:  # what uvicorn raises once the browser has gone
            raise RuntimeError(
                "Unexpected ASGI message 'websocket.send', after sending 'websocket.close'"
            )
        await self.from_server.put(
            json.loads(json.dumps(data))
        )  # a JSON round trip, like the wire

    async def receive_json(self):
        message = await self.to_server.get()
        if message is None:
            raise WebSocketDisconnect
        return message

    async def send(self, message: dict):
        await self.to_server.put(message)

    async def recv(self, *types: str, timeout: float = 5.0) -> dict:
        """The next server message of one of `types` (of any type if none are given)."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            message = await asyncio.wait_for(
                self.from_server.get(), max(0.0, deadline - loop.time())
            )
            if not types or message["type"] in types:
                return message

    async def nothing_for(self, seconds: float, *types: str) -> bool:
        """Whether no message of `types` arrives within `seconds`."""
        try:
            await self.recv(*types, timeout=seconds)
        except asyncio.TimeoutError:
            return True
        return False

    async def disconnect(self):
        self.closed = True
        await self.to_server.put(None)


@contextlib.asynccontextmanager
async def connected():
    """A FakeWebSocket with the real websocket_endpoint serving it."""
    ws = FakeWebSocket()
    endpoint = asyncio.create_task(main.websocket_endpoint(ws))
    try:
        yield ws
    finally:
        if not ws.closed:
            await ws.disconnect()
        await asyncio.wait_for(endpoint, 5)


async def start_game(ws: FakeWebSocket, player_types: list[int]) -> dict:
    await ws.send(
        {
            "type": "START_GAME",
            "num_players": len(player_types),
            "player_types": player_types,
        }
    )
    return await ws.recv("DEAL_CARDS", "START_REJECTED")


def card_value(card: str) -> int:
    rank = card[:-1]
    return MAPPING_SV[rank] if rank in MAPPING_SV else int(rank)


def suits_for(state: dict, values: list[str]) -> list[str]:
    """Suit cards from the hand matching rank `values`, as a player would select them."""
    hand, cards = list(state["suited_hand"]), []
    for value in values:
        card = next(c for c in hand if c[:-1] == value)
        hand.remove(card)
        cards.append(card)
    return cards


async def answer(ws: FakeWebSocket, message: dict) -> bool:
    """Reply to a prompt the simplest legal way; False if `message` is no prompt for us."""
    kind, prompt_id = message["type"], message.get("prompt_id")
    if kind == "DEAL_CARDS":
        await ws.send({"type": "DEALT_CARDS"})
    elif kind == "EXCHANGE_PROMPT":
        count, hand = message["required_cards"], message["state"]["suited_hand"]
        cards = hand[:count] if message["can_choose"] else []
        await ws.send(
            {"type": "EXCHANGE_CARDS", "suits": cards, "prompt_id": prompt_id}
        )
    elif kind == "JUMP_IN_PROMPT":
        finish = message["state"]["legal_moves_suits"][0]
        await ws.send(
            {"type": "PLAY_MOVE", "suits": finish, "jump": True, "prompt_id": prompt_id}
        )
    elif kind == "STATE_UPDATE" and prompt_id is not None:
        state = message["state"]
        move = (
            suits_for(state, state["legal_moves_suits"][0])
            if state["legal_moves_suits"]
            else []
        )
        await ws.send({"type": "PLAY_MOVE", "suits": move, "prompt_id": prompt_id})
    else:
        return False
    return True


async def play_until_round_over(ws: FakeWebSocket, timeout: float = 20.0) -> dict:
    """Answer every prompt until the round ends; returns the ROUND_OVER message."""
    while True:
        message = await ws.recv(timeout=timeout)
        if message["type"] == "ROUND_OVER":
            return message
        await answer(ws, message)


def table_with_hands(
    hands: dict[int, list[str]],
    bots: dict[int, Player],
    *,
    turn: int,
    pile: tuple[list[str], int] | None = None,
    passed: set[int] | None = None,
) -> tuple[main.Table, FakeWebSocket, asyncio.Queue]:
    """A round in progress with the given suit-card hands (seat 0 is the human), served
    over a FakeWebSocket. `pile` is the cards on it and who played them. Returns the
    table for main.play_round, the socket, and the queue to put the human's replies in.
    """
    env = President(len(hands), track_suits=True)
    env.full_reset()
    env.suited_hands = {p: sorted(cards, key=card_value) for p, cards in hands.items()}
    env.hands = {p: sorted(card_value(c) for c in cards) for p, cards in hands.items()}
    env.first_turn = False
    env.curr_turn = turn
    env.passed = set(passed or ())
    if pile:
        cards, env.pile_leader = pile
        env.suit_last_move, env.pile = list(cards), [card_value(c) for c in cards]
        env.last_move = main.parse_suits_to_move(cards)

    session = main.GameSession(
        env=env,
        assigned_players=bots,
        assign_p={0: PlayerType.HUMAN} | {p: PlayerType.BASELINE for p in bots},
    )
    session.timings.reset()
    ws, replies = FakeWebSocket(), asyncio.Queue()
    return main.Table(session, env, ws, replies, [], asyncio.Event()), ws, replies
