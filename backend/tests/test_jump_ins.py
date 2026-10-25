"""The house rules for a human jump-in: the window opens again every time the next
player passes, a pass by the human doesn't give it up, and when the last pass would
clear the pile the human still gets one more window first. Any play ends the chance."""

import asyncio

import pytest

import main
from playerTypes.baseline_bot import PresidentBaselineBot
from support import async_test, table_with_hands

FOURS = ["4", "4"]


@pytest.fixture(autouse=True)
def short_jump_window(monkeypatch):
    monkeypatch.setattr(
        main, "JUMP_IN_WINDOW", 0.1
    )  # the scripted human answers at once


class Plays(PresidentBaselineBot):
    """A baseline bot that plays `move` whenever it may."""

    def __init__(self, player_id, move):
        super().__init__(player_id)
        self.move = move

    def get_move(self, state, *args, **kwargs):
        if self.move in state["legal_moves"]:
            return self.move
        return super().get_move(state, *args, **kwargs)


HUMAN = ["4S", "4D", "TH", "KH"]  # holds the other two 4s
CANT_BEAT_A_PAIR = {2: ["6H", "8D", "QS", "KC"], 3: ["7C", "9H", "JD", "AC"]}


async def play_out(table, ws, replies, *, jump_on_window=None, on_my_turn="pass"):
    """Let the human's jump-in windows lapse (or jump in the given one) and answer the
    human's turn. Returns what happened until the pile clears, as short event names."""
    game = asyncio.create_task(main.play_round(table))
    events, windows, seen_passed = [], 0, set(table.env.passed)
    try:
        while True:
            message = await ws.recv(timeout=3)
            kind, state = message["type"], message.get("state", {})
            if kind == "JUMP_IN_PROMPT":
                windows += 1
                if windows == jump_on_window:
                    events.append("jumps")
                    jump = {"suits": FOURS, "jump": True}
                    await replies.put(
                        {"type": "PLAY_MOVE", **jump, "prompt_id": message["prompt_id"]}
                    )
                else:
                    events.append("window lapses")
            elif kind == "STATE_UPDATE":
                if not state["suit_last_move"]:
                    events.append("pile clears")
                    return events
                if len(state["suit_last_move"]) == 4:
                    events.append("quad on the pile")
                if set(state["passed"]) - seen_passed:
                    events.append("pass")
                    seen_passed |= set(state["passed"])
                if "prompt_id" in message:
                    events.append(f"my turn: {on_my_turn}")
                    move = (
                        {"suits": []}
                        if on_my_turn == "pass"
                        else {"suits": FOURS, "jump": True}
                    )
                    await replies.put(
                        {"type": "PLAY_MOVE", **move, "prompt_id": message["prompt_id"]}
                    )
            elif kind == "ROUND_OVER":
                return events
    finally:
        game.cancel()


def bot_1_plays_fours(**table_args):
    hands = {0: HUMAN, 1: ["4H", "4C", "8S", "JS"], **CANT_BEAT_A_PAIR}
    bots = {
        1: Plays(1, (4, 2, 0)),
        2: PresidentBaselineBot(2),
        3: PresidentBaselineBot(3),
    }
    # bot 3 led a single 3, so bot 1's pair of 4s can go on it
    return table_with_hands(hands, bots, turn=1, pile=(["3C"], 3), **table_args)


@async_test
async def test_each_pass_reopens_the_window():
    events = await play_out(*bot_1_plays_fours(), jump_on_window=2)
    assert events[:3] == ["window lapses", "pass", "jumps"]
    assert "quad on the pile" in events


@async_test
async def test_passing_myself_keeps_the_jump_in_and_the_last_pass_still_gives_a_window():
    events = await play_out(*bot_1_plays_fours())
    # bots 2 and 3 pass, the human passes last: one more window before the pile clears
    assert events[-4:] == ["my turn: pass", "pass", "window lapses", "pile clears"]


@async_test
async def test_after_passing_earlier_the_last_pass_still_gives_a_window():
    events = await play_out(*bot_1_plays_fours(passed={0}))
    assert "my turn: pass" not in events
    assert events[-3:] == ["pass", "window lapses", "pile clears"]


@async_test
async def test_jumping_in_the_window_after_the_last_pass_works():
    events = await play_out(*bot_1_plays_fours(passed={0}), jump_on_window=3)
    # bots 2 and 3 pass; the third window is the one after the last pass
    assert events == [
        "window lapses",
        "pass",
        "window lapses",
        "pass",
        "jumps",
        "quad on the pile",
        "pile clears",
    ]


@async_test
async def test_a_play_ends_the_jump_in_chance():
    hands = {
        0: HUMAN,
        1: ["4H", "4C", "8S", "JS"],
        2: ["9S", "9D", "QS", "KC"],
        3: ["7C", "9H", "JD", "AC"],
    }
    bots = {1: Plays(1, (4, 2, 0)), 2: Plays(2, (9, 2, 0)), 3: PresidentBaselineBot(3)}
    events = await play_out(*table_with_hands(hands, bots, turn=1, pile=(["3C"], 3)))
    assert events.count("window lapses") == 1, "no window may open after bot 2 played"


@async_test
async def test_my_pass_that_is_not_the_last_keeps_the_jump_in_for_the_next_passes():
    hands = {
        0: HUMAN,
        1: ["6H", "8D", "QS", "KC"],
        2: ["7C", "9H", "JD", "AC"],
        3: ["4H", "4C", "8S", "JS"],
    }
    bots = {
        1: PresidentBaselineBot(1),
        2: PresidentBaselineBot(2),
        3: Plays(3, (4, 2, 0)),
    }
    events = await play_out(*table_with_hands(hands, bots, turn=3, pile=(["3C"], 2)))
    # bot 3 plays the 4s; the human is next and passes; bots 1 and 2 then pass
    assert events == [
        "window lapses",
        "my turn: pass",
        "pass",  # the human's own pass: the jump-in stays open
        "window lapses",
        "pass",
        "window lapses",
        "pass",  # the last pass: one more window before the pile clears
        "window lapses",
        "pile clears",
    ]
