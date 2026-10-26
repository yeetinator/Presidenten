"""The house rules for a human jump-in.

A human who already passed on the pile keeps the chance while the others pass: the
window opens again after every pass, and after the last pass it opens once more before
the pile clears. A human who hasn't passed gets a window before each other player's
turn, and on their own turn may jump in, or play other cards or pass, both of which
give the jump-in up. Any play by another player ends the chance."""

import asyncio

import pytest

import main
from playerTypes.baseline_bot import PresidentBaselineBot
from support import async_test, table_with_hands

FOURS = ["4", "4"]
KINGS = ["K", "K", "K"]


@pytest.fixture(autouse=True)
def short_jump_window(monkeypatch):
    # the scripted human answers at once
    monkeypatch.setattr(main, "JUMP_IN_WINDOW", 0.1)


class Plays(PresidentBaselineBot):
    """A baseline bot that plays `move` whenever it may."""

    def __init__(self, player_id, move):
        super().__init__(player_id)
        self.move = move

    def get_move(self, state, *args, **kwargs):
        if self.move in state["legal_moves"]:
            return self.move
        return super().get_move(state, *args, **kwargs)


async def play_out(
    table, ws, replies, *, jump_on_window=None, jump_with=FOURS, on_my_turn="pass"
):
    """Let the human's jump-in windows lapse (or jump in the given one) and answer the
    human's turn with "pass", "jump" or a list of cards to play. Returns what happened
    until the pile clears, as short event names."""
    game = asyncio.create_task(main.play_round(table))
    events, windows, seen_passed = [], 0, set(table.env.passed)
    jump = {"suits": jump_with, "jump": True}
    try:
        while True:
            message = await ws.recv(timeout=3)
            kind, state = message["type"], message.get("state", {})
            reply = {"type": "PLAY_MOVE", "prompt_id": message.get("prompt_id")}
            if kind == "JUMP_IN_PROMPT":
                windows += 1
                if windows == jump_on_window:
                    events.append("jumps")
                    await replies.put({**reply, **jump})
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
                    if on_my_turn == "pass":
                        events.append("my turn: pass")
                        await replies.put({**reply, "suits": []})
                    elif on_my_turn == "jump":
                        events.append("my turn: jump")
                        await replies.put({**reply, **jump})
                    else:
                        events.append(f"my turn: play {' '.join(on_my_turn)}")
                        await replies.put({**reply, "suits": on_my_turn})
            elif kind == "ROUND_OVER":
                return events
    finally:
        game.cancel()


HOLDS_FOURS = ["4S", "4D", "TH", "KH"]  # the human holds the other two 4s
CANT_BEAT_A_PAIR = {2: ["6H", "8D", "QS", "KC"], 3: ["7C", "9H", "JD", "AC"]}


def bot_1_plays_fours(**table_args):
    hands = {0: HOLDS_FOURS, 1: ["4H", "4C", "8S", "JS"], **CANT_BEAT_A_PAIR}
    bots = {
        1: Plays(1, (4, 2, 0)),
        2: PresidentBaselineBot(2),
        3: PresidentBaselineBot(3),
    }
    # bot 3 led a single 3, so bot 1's pair of 4s can go on it
    return table_with_hands(hands, bots, turn=1, pile=(["3C"], 3), **table_args)


def bot_2_plays_a_king():
    """Bot 2 plays a K, bot 3 can't beat it, and then it is the human's turn holding
    the other three Ks and an A."""
    hands = {
        0: ["KH", "KD", "KS", "AH"],
        1: ["4C", "6H", "8D", "TS"],
        2: ["KC", "5D", "7S", "9H"],
        3: ["5C", "7H", "9D", "JD"],
    }
    bots = {
        1: PresidentBaselineBot(1),
        2: Plays(2, (13, 1, 0)),
        3: PresidentBaselineBot(3),
    }
    return table_with_hands(hands, bots, turn=2, pile=(["3C"], 1))


# --- before the human's turn, and when the human hasn't passed --------------------


@async_test
async def test_each_pass_before_my_turn_reopens_the_window():
    events = await play_out(*bot_1_plays_fours(), jump_on_window=2)
    assert events[:3] == ["window lapses", "pass", "jumps"]
    assert "quad on the pile" in events


@async_test
async def test_my_turn_offers_the_jump_in_next_to_other_moves_right_away():
    table, ws, replies = bot_2_plays_a_king()
    game = asyncio.create_task(main.play_round(table))
    try:
        window = await ws.recv("JUMP_IN_PROMPT")  # before bot 3's turn
        assert window["state"]["legal_moves_suits"] == [KINGS]

        turn = await ws.recv("JUMP_IN_PROMPT", "STATE_UPDATE")
        assert turn["type"] == "STATE_UPDATE", "no jump-only window on the human's turn"
        assert "prompt_id" in turn and turn["clearJump"] is False
        assert turn["state"]["is_finish_prompt"]
        assert KINGS in turn["state"]["legal_moves_suits"]
        assert ["A"] in turn["state"]["legal_moves_suits"]
    finally:
        game.cancel()


@async_test
async def test_playing_other_cards_on_my_turn_gives_up_the_jump_in():
    table, ws, replies = bot_2_plays_a_king()
    events = await play_out(table, ws, replies, on_my_turn=["AH"])

    assert events[:3] == ["window lapses", "pass", "my turn: play AH"]
    assert "window lapses" not in events[3:] and "quad on the pile" not in events
    assert table.env.hands[0] == [13, 13, 13], "the A was played, the Ks kept"


@async_test
async def test_jumping_in_on_my_turn_works():
    table, ws, replies = bot_2_plays_a_king()
    events = await play_out(table, ws, replies, on_my_turn="jump", jump_with=KINGS)

    assert events[:4] == ["window lapses", "pass", "my turn: jump", "quad on the pile"]
    assert table.env.hands[0] == [14]


@async_test
async def test_passing_on_my_turn_gives_up_the_jump_in():
    events = await play_out(*bot_1_plays_fours())
    # bots 2 and 3 pass, then the human passes: no window before the pile clears
    assert events == [
        "window lapses",
        "pass",
        "window lapses",
        "pass",
        "my turn: pass",
        "pass",
        "pile clears",
    ]


@async_test
async def test_passing_on_my_turn_gives_up_the_jump_in_for_the_rest_of_the_pile():
    hands = {
        0: HOLDS_FOURS,
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
    # bot 3 plays the 4s, the human is next and passes, then bots 1 and 2 pass
    assert events == ["my turn: pass", "pass", "pass", "pass", "pile clears"]


# --- when the human already passed on this pile ------------------------------------


@async_test
async def test_after_passing_the_window_reopens_and_the_last_pass_gives_one_more():
    events = await play_out(*bot_1_plays_fours(passed={0}))
    # bots 2 and 3 pass; bot 3's pass is the last one
    assert events == [
        "window lapses",
        "pass",
        "window lapses",
        "pass",
        "window lapses",
        "pile clears",
    ]


@async_test
async def test_jumping_in_the_window_after_the_last_pass_works():
    events = await play_out(*bot_1_plays_fours(passed={0}), jump_on_window=3)
    assert events == [
        "window lapses",
        "pass",
        "window lapses",
        "pass",
        "jumps",
        "quad on the pile",
        "pile clears",
    ]


@pytest.mark.parametrize("human_passed", [False, True])
@async_test
async def test_a_play_on_top_ends_the_jump_in_chance(human_passed):
    hands = {
        0: HOLDS_FOURS,
        1: ["4H", "4C", "8S", "JS"],
        2: ["9S", "9D", "QS", "KC"],
        3: ["7C", "9H", "JD", "AC"],
    }
    bots = {1: Plays(1, (4, 2, 0)), 2: Plays(2, (9, 2, 0)), 3: PresidentBaselineBot(3)}
    passed = {0} if human_passed else None
    table, ws, replies = table_with_hands(
        hands, bots, turn=1, pile=(["3C"], 3), passed=passed
    )
    events = await play_out(table, ws, replies)
    assert events.count("window lapses") == 1, "no window may open after bot 2 played"
    assert "quad on the pile" not in events
