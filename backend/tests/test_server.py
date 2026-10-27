import asyncio
import time

import pytest
import torch

import main
from game import PlayerType, President
from playerTypes.baseline_bot import PresidentBaselineBot
from support import (
    FakeWebSocket,
    answer,
    async_test,
    connected,
    play_until_round_over,
    start_game,
    suits_for,
    table_with_hands,
)

BOTS_ONLY = [0, 2, 2, 2]  # the human plus three baseline bots


@pytest.mark.parametrize(
    "suits, move",
    [
        ([], (0, 0, 0)),
        (["7H"], (7, 1, 0)),
        (["7H", "7S"], (7, 2, 0)),
        (["7H", "2C"], (7, 2, 1)),  # a 2 next to other cards is a wildcard
        (["2D"], (15, 1, 0)),  # 2s on their own are their own rank, as in the engine
        (["2D", "2S"], (15, 2, 0)),
    ],
)
def test_selected_cards_parse_to_engine_moves(suits, move):
    assert main.parse_suits_to_move(suits) == move


def test_display_states_offer_no_moves():
    env = President(4, track_suits=True)
    state = env.full_reset()

    shown = main.enrich_state(state, {0: PlayerType.HUMAN})
    prompted = main.enrich_state(state, {0: PlayerType.HUMAN}, accepts_moves=True)

    assert shown["legal_moves_suits"] == [] and shown["can_pass"] is False
    assert prompted["legal_moves_suits"]


@pytest.mark.parametrize(
    "start, reason",
    [
        ({"num_players": 3, "player_types": [0, 2, 2]}, "A game needs 4 to 7 players."),
        (
            {"num_players": 8, "player_types": [0] + [2] * 7},
            "A game needs 4 to 7 players.",
        ),
        (
            {"num_players": "4", "player_types": BOTS_ONLY},
            "A game needs 4 to 7 players.",
        ),
        ({"player_types": BOTS_ONLY}, "A game needs 4 to 7 players."),
        (
            {"num_players": 4, "player_types": [0, 2, 2]},
            "Every seat needs a player type.",
        ),
        ({"num_players": 4, "player_types": [0, 2, 2, 9]}, "Unknown player type."),
        (
            {"num_players": 4, "player_types": [0, 2, 2, 1]},
            "That player type isn't available in the web game.",
        ),
        (
            {"num_players": 4, "player_types": [2, 0, 2, 2]},
            "Seat 0 must be the only human player.",
        ),
        (
            {"num_players": 4, "player_types": [0, 0, 2, 2]},
            "Seat 0 must be the only human player.",
        ),
    ],
)
@async_test
async def test_invalid_start_game_is_rejected_with_a_reason(start, reason):
    async with connected() as ws:
        await ws.send({"type": "START_GAME", **start})
        assert await ws.recv() == {"type": "START_REJECTED", "message": reason}

        # the connection stays usable for a valid start
        assert (await start_game(ws, BOTS_ONLY))["type"] == "DEAL_CARDS"


@pytest.mark.parametrize("players", [4, 5, 6, 7])
@async_test
async def test_valid_start_game_deals_every_seat(players):
    async with connected() as ws:
        deal = await start_game(ws, [0] + [2] * (players - 1))
        assert deal["type"] == "DEAL_CARDS"
        assert len(deal["state"]["player_roles"]) == players


@async_test
async def test_a_second_start_game_is_rejected():
    async with connected() as ws:
        await start_game(ws, BOTS_ONLY)
        await ws.send(
            {"type": "START_GAME", "num_players": 4, "player_types": BOTS_ONLY}
        )
        assert (await ws.recv("START_REJECTED"))[
            "message"
        ] == "A game is already running."


@async_test
async def test_every_round_waits_for_the_deal_animation():
    async with connected() as ws:
        await start_game(ws, BOTS_ONLY)
        assert await ws.nothing_for(0.3, "STATE_UPDATE", "JUMP_IN_PROMPT")
        await ws.send({"type": "DEALT_CARDS"})
        await play_until_round_over(ws)

        await ws.send({"type": "NEXT_ROUND"})
        await ws.recv("DEAL_CARDS")
        assert await ws.nothing_for(0.3, "EXCHANGE_PROMPT", "STATE_UPDATE")
        await ws.send({"type": "DEALT_CARDS"})
        assert (await ws.recv("EXCHANGE_PROMPT"))["type"] == "EXCHANGE_PROMPT"


async def next_turn_prompt(ws: FakeWebSocket) -> dict:
    """Play on until the server asks the human to make a normal move."""
    while True:
        message = await ws.recv()
        if (
            message["type"] == "STATE_UPDATE"
            and "prompt_id" in message
            and message["state"]["legal_moves_suits"]
        ):
            return message
        if message["type"] != "JUMP_IN_PROMPT":  # let jump-ins lapse
            await answer(ws, message)


@async_test
async def test_replies_to_older_prompts_are_ignored():
    async with connected() as ws:
        await start_game(ws, BOTS_ONLY)
        await ws.send({"type": "DEALT_CARDS"})
        prompt = await next_turn_prompt(ws)
        state = prompt["state"]
        move = suits_for(state, state["legal_moves_suits"][0])

        await ws.send(
            {"type": "PLAY_MOVE", "suits": move, "prompt_id": prompt["prompt_id"] - 1}
        )
        await ws.send({"type": "PLAY_MOVE", "suits": move})
        assert await ws.nothing_for(
            0.3, "STATE_UPDATE"
        ), "stale or untagged replies must be ignored"

        await ws.send(
            {"type": "PLAY_MOVE", "suits": move, "prompt_id": prompt["prompt_id"]}
        )
        after = await ws.recv("STATE_UPDATE")
        assert len(after["state"]["suited_hand"]) == len(state["suited_hand"]) - len(
            move
        )
        assert "prompt_id" not in after and not after["state"]["legal_moves_suits"]


@async_test
async def test_exchange_replies_to_older_prompts_are_ignored():
    async with connected() as ws:
        await start_game(ws, BOTS_ONLY)
        await ws.send({"type": "DEALT_CARDS"})
        await play_until_round_over(ws)
        await ws.send({"type": "NEXT_ROUND"})
        await ws.recv("DEAL_CARDS")
        await ws.send({"type": "DEALT_CARDS"})
        prompt = await ws.recv("EXCHANGE_PROMPT")
        count, hand = prompt["required_cards"], prompt["state"]["suited_hand"]
        cards = hand[:count] if prompt["can_choose"] else []

        await ws.send(
            {
                "type": "EXCHANGE_CARDS",
                "suits": cards,
                "prompt_id": prompt["prompt_id"] - 1,
            }
        )
        assert await ws.nothing_for(0.3, "STATE_UPDATE")
        await ws.send(
            {"type": "EXCHANGE_CARDS", "suits": cards, "prompt_id": prompt["prompt_id"]}
        )
        assert (await ws.recv("STATE_UPDATE"))["type"] == "STATE_UPDATE"


@async_test
async def test_stale_replies_do_not_stretch_the_jump_in_window():
    queue, disconnect = asyncio.Queue(), asyncio.Event()

    async def stale_replies():
        for _ in range(20):
            await asyncio.sleep(0.01)
            await queue.put({"type": "PLAY_MOVE", "prompt_id": 1, "suits": []})

    spam = asyncio.create_task(stale_replies())
    started = time.monotonic()
    with pytest.raises(asyncio.TimeoutError):
        await main.wait_for_reply(queue, [], disconnect, "PLAY_MOVE", 2, timeout=0.1)
    assert time.monotonic() - started < 0.15
    spam.cancel()


class LeadsPairOfFives(PresidentBaselineBot):
    def get_move(self, state, *args, **kwargs):
        if (5, 2, 0) in state["legal_moves"]:
            return (5, 2, 0)
        return super().get_move(state, *args, **kwargs)


@async_test
async def test_human_going_out_by_jumping_in_still_finishes_the_round():
    hands = {
        0: ["5S", "5D"],  # the human holds exactly the other two 5s
        1: ["5H", "5C", "9S", "KD"],
        2: ["3C", "6H", "8D", "QS", "AH"],
        3: ["4C", "7H", "TD", "JS", "AS"],
    }
    bots = {
        1: LeadsPairOfFives(1),
        2: PresidentBaselineBot(2),
        3: PresidentBaselineBot(3),
    }
    table, ws, replies = table_with_hands(hands, bots, turn=1)
    game = asyncio.create_task(main.play_round(table))

    try:
        jumped = False
        while True:
            message = await ws.recv(timeout=3)
            if message["type"] == "ROUND_OVER":
                break
            if message["type"] == "JUMP_IN_PROMPT":
                jumped = True
                await replies.put(
                    {
                        "type": "PLAY_MOVE",
                        "suits": ["5", "5"],
                        "jump": True,
                        "prompt_id": message["prompt_id"],
                    }
                )
        assert jumped and message["out_order"][0] == 0
    finally:
        game.cancel()


@async_test
async def test_disconnecting_mid_round_ends_the_session_quietly():
    async with connected() as ws:
        await start_game(ws, BOTS_ONLY)
        await ws.send({"type": "DEALT_CARDS"})
        bot_turns = 0
        while bot_turns < 3:
            message = await ws.recv()
            await answer(ws, message)
            if message["type"] == "STATE_UPDATE" and message["state"][
                "curr_turn"
            ] not in (0, None):
                bot_turns += 1
        await ws.disconnect()  # the endpoint must finish without raising (checked on exit)


def test_dmc_model_is_loaded_once_and_shared():
    device = torch.device("cpu")
    main.load_dmc_model.cache_clear()
    model = main.load_dmc_model(device)

    assert main.load_dmc_model(device) is model
    seats = {0: PlayerType.HUMAN, 1: PlayerType.DMC, 2: PlayerType.DMC}
    players, _ = main.build_assigned_players(seats, device)
    assert players[1].model is players[2].model is model
