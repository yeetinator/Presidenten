"""Whole games through the real server, with a player that clicks like someone at the
browser: it follows App.svelte's rules for when cards, Pass and Jump In are enabled and
clicks after a random delay, so some replies land just as timeouts expire or piles clear.
"""

import asyncio
import random

import pytest

import main
from game import President
from support import async_test, card_value, connected, suits_for


@pytest.fixture
def problems(monkeypatch):
    """Flags any illegal move the engine applies for the human, and any hand that stops
    matching its suit cards."""
    found = []
    real_step = President.step

    def checked_step(env, p_id, move, suits=None):
        if p_id == 0 and move not in env.get_legal_moves(0, None, only_finish=False):
            found.append(f"illegal human move {move} on {env.last_move}")
        result = real_step(env, p_id, move, suits)
        for p in range(env.players):
            if sorted(card_value(c) for c in env.suited_hands[p]) != sorted(
                env.hands[p]
            ):
                found.append(
                    f"player {p}: hand {env.hands[p]} vs suit cards {env.suited_hands[p]}"
                )
        return result

    monkeypatch.setattr(President, "step", checked_step)
    return found


def finish_moves(state):
    """The jump-in moves the UI offers (gameStore.getAutoFinishMove)."""
    groups = []
    for card in state["suit_last_move"]:
        if groups and groups[-1][0] == card[:-1]:
            groups[-1].append(card[:-1])
        else:
            groups.append([card[:-1]])
    moves = [[group[0]] * (4 - len(group)) for group in groups if len(group) < 4]
    return [move for move in moves if move in state["legal_moves_suits"]]


class BrowserPlayer:
    def __init__(self, ws, rng):
        self.ws, self.rng = ws, rng
        self.shown = (
            0  # bumps with every state on screen; a click is lost when it changes
        )
        self.jump_visible = False

    def click_later(self, message, shown):
        async def click():
            await asyncio.sleep(self.rng.uniform(0.05, 1.2) * main.MESSAGE_DELAY)
            if self.shown == shown:
                await self.ws.send(message)

        asyncio.create_task(click())

    def on_state(self, message):
        kind, state = message["type"], message["state"]
        self.shown += 1
        if kind == "JUMP_IN_PROMPT":
            self.jump_visible = True
        elif kind == "STATE_UPDATE":
            # a turn with clearJump False offers the jump-in next to normal moves
            self.jump_visible = message.get("clearJump") is False

        my_turn = state["curr_turn"] == 0
        turn_pending = (
            my_turn and not state["can_pass"] and not state["legal_moves_suits"]
        )
        jump_visible = self.jump_visible and state["is_finish_prompt"]
        pass_visible = (
            my_turn
            and state["can_pass"]
            and state["pile_leader"] != 0
            and not turn_pending
        )
        reply = {"type": "PLAY_MOVE", "prompt_id": message.get("prompt_id")}

        jumps = finish_moves(state) if jump_visible else []
        wants_to_pass = pass_visible and self.rng.random() < 0.3
        if jumps and self.rng.random() < 0.7:
            self.click_later({**reply, "suits": jumps[0], "jump": True}, self.shown)
        elif (
            my_turn
            and not turn_pending
            and state["legal_moves_suits"]
            and not wants_to_pass
        ):
            move = self.rng.choice(state["legal_moves_suits"])
            if (
                move in jumps
            ):  # clicking a card that completes the quad jumps straight in
                self.click_later({**reply, "suits": move, "jump": True}, self.shown)
            else:
                self.click_later({**reply, "suits": suits_for(state, move)}, self.shown)
        elif pass_visible:
            self.click_later({**reply, "suits": []}, self.shown)

    async def play(self, rounds):
        finished = 0
        while finished < rounds:
            message = await self.ws.recv(timeout=20)
            kind = message["type"]
            if kind == "DEAL_CARDS":
                self.shown += 1
                await self.ws.send({"type": "DEALT_CARDS"})
            elif kind == "EXCHANGE_PROMPT":
                self.shown += 1
                count, hand = message["required_cards"], message["state"]["suited_hand"]
                cards = self.rng.sample(hand, count) if message["can_choose"] else []
                self.click_later(
                    {
                        "type": "EXCHANGE_CARDS",
                        "suits": cards,
                        "prompt_id": message["prompt_id"],
                    },
                    self.shown,
                )
            elif kind in ("STATE_UPDATE", "JUMP_IN_PROMPT"):
                self.on_state(message)
            elif kind == "ROUND_OVER":
                finished += 1
                if finished < rounds:
                    await self.ws.send({"type": "NEXT_ROUND"})


@pytest.mark.parametrize(
    "seed, player_types",
    [
        (1, [0, 2, 2, 2]),
        (2, [0, 1, 2, 1]),
        (3, [0, 2, 1, 2, 2]),
        (4, [0, 1, 2, 1, 2, 1]),
        (5, [0, 2, 2, 1, 2, 1, 2]),
    ],
)
@async_test
async def test_games_played_like_in_the_browser_stay_legal(
    seed, player_types, problems
):
    random.seed(seed)
    async with connected() as ws:
        await ws.send(
            {
                "type": "START_GAME",
                "num_players": len(player_types),
                "player_types": player_types,
            }
        )
        await BrowserPlayer(ws, random.Random(seed)).play(rounds=3)

    assert problems == []
