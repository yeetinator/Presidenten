import random

import pytest

from game import President
from playerTypes.baseline_bot import PresidentBaselineBot
from playerTypes.random_bot import PresidentRandomBot
from support import card_value


def table_with(hands: dict[int, list[str]], turn: int) -> President:
    """A 4-player game mid-round with the given suit-card hands."""
    env = President(4, track_suits=True)
    env.full_reset()
    env.suited_hands = {p: sorted(cards, key=card_value) for p, cards in hands.items()}
    env.hands = {p: sorted(card_value(c) for c in cards) for p, cards in hands.items()}
    env.first_turn = False
    env.curr_turn = turn
    return env


def test_jumping_in_with_last_cards_passes_the_lead_on():
    env = table_with(
        {0: ["5H", "5C", "9S"], 1: ["5S", "5D"], 2: ["3C", "6H"], 3: ["4C", "7H"]},
        turn=0,
    )
    env.step(0, (5, 2, 0), ["5H", "5C"])
    assert env.curr_turn == 1 and env._get_state(1)["is_finish_prompt"]

    env.step(1, (5, 2, 0), ["5S", "5D"])

    assert env.hands[1] == [] and env.out_order == [1]
    assert env.curr_turn == 2, "an empty-handed jumper must not keep the lead"
    assert env.was_pile_reset


def test_jumper_with_cards_left_leads_the_new_pile():
    env = table_with(
        {
            0: ["5H", "5C", "9S"],
            1: ["5S", "5D", "KH"],
            2: ["3C", "6H"],
            3: ["4C", "7H"],
        },
        turn=0,
    )
    env.step(0, (5, 2, 0), ["5H", "5C"])
    env.step(1, (5, 2, 0), ["5S", "5D"])

    assert env.curr_turn == 1 and env.was_pile_reset


def test_a_player_who_passed_can_still_jump_in():
    env = table_with(
        {
            0: ["5H", "5C", "9S"],
            1: ["5S", "5D", "KH"],
            2: ["3C", "6H"],
            3: ["4C", "7H"],
        },
        turn=0,
    )
    env.passed = {1}  # passing on this pile earlier doesn't forfeit a jump-in

    env.step(0, (5, 2, 0), ["5H", "5C"])

    assert env.curr_turn == 1
    assert env._get_state(1)["is_finish_prompt"]
    assert (5, 2, 0) in env.get_legal_moves(1)


def test_jumping_in_with_the_second_finish_option_plays_those_cards():
    env = table_with(
        {
            0: ["7H", "7C", "2S", "9S"],
            1: ["7S", "7D", "2H", "2C", "2D", "KH"],
            2: ["3C", "6H"],
            3: ["4C", "5H"],
        },
        turn=0,
    )
    env.step(0, (7, 3, 1), ["7H", "7C", "2S"])
    assert env.get_legal_moves(1) == [(0, 0, 0), (7, 2, 0), (15, 3, 0)]

    env.step(1, (15, 3, 0), ["2H", "2C", "2D"])

    assert env.hands[1] == [7, 7, 13]
    assert sorted(env.suited_hands[1]) == ["7D", "7S", "KH"]
    assert env.pile[-3:] == [15, 15, 15]


def all_twos_played(env):
    env._history_flat_counts[15] = 4
    return env


def test_the_highest_unplayed_card_clears_the_pile():
    env = all_twos_played(
        table_with(
            {0: ["AH", "4C"], 1: ["5H", "6C"], 2: ["7D", "8S"], 3: ["9C", "TH"]},
            turn=0,
        )
    )
    env.step(0, (14, 1, 0), ["AH"])

    assert env.was_pile_reset and env.curr_turn == 0


def test_an_unplayed_2_keeps_the_pile_open_even_in_the_players_own_hand():
    env = table_with(
        {0: ["AH", "2S"], 1: ["5H", "6C"], 2: ["7D", "8S"], 3: ["9C", "TH"]},
        turn=0,
    )
    env._history_flat_counts[15] = 3
    env.step(0, (14, 1, 0), ["AH"])

    # Clearing would tell the others that the last 2 is in player 0's hand
    assert not env.was_pile_reset and env.curr_turn == 1


def test_a_pile_clears_when_nobody_left_to_answer_holds_enough_cards():
    env = table_with(
        {0: ["5H", "5C", "4D"], 1: ["9S", "9H"], 2: ["KH"], 3: ["AC"]},
        turn=0,
    )
    env.step(0, (5, 2, 0), ["5H", "5C"])
    assert not env.was_pile_reset and env.curr_turn == 1

    env.step(1, (0, 0, 0))

    assert env.was_pile_reset and env.curr_turn == 0


def test_a_jump_in_comes_before_an_early_clear():
    env = all_twos_played(
        table_with(
            {0: ["AH", "4C"], 1: ["AS", "AD", "AC", "6C"], 2: ["7D", "8S"], 3: ["9C", "TH"]},
            turn=0,
        )
    )
    env.step(0, (14, 1, 0), ["AH"])
    assert env.curr_turn == 1 and env._get_state(1)["is_finish_prompt"]

    env.step(1, (0, 0, 0))

    assert env.was_pile_reset and env.curr_turn == 0


@pytest.mark.parametrize("players", [4, 5, 6, 7])
def test_hands_and_suit_cards_stay_in_sync_over_whole_games(players):
    random.seed(players)
    env = President(players, track_suits=True)
    bots = {
        p: (PresidentRandomBot(p, seed=players) if p % 2 else PresidentBaselineBot(p))
        for p in range(players)
    }

    for round_idx in range(5):
        state = env.full_reset(next_round=round_idx > 0)
        if round_idx:
            cards = {
                p: bots[p].choose_cards_to_pass(env._get_state(p))
                for p in range(players)
            }
            for pair in env.role_pairs:
                env.exchange_cards(pair, cards)
            state = env._get_state(env.curr_turn)

        for _ in range(1000):
            if env.game_over:
                break
            player = env.curr_turn
            move = bots[player].get_move(state)
            assert move in state["legal_moves"] or (
                move == (0, 0, 0) and not state["legal_moves"]
            )
            state, _ = env.step(player, move)
            if env.was_pile_reset:
                env.clear_pile()

            for p in range(players):
                assert sorted(card_value(c) for c in env.suited_hands[p]) == sorted(
                    env.hands[p]
                )
        assert env.game_over, "a round must end"

        env.assign_roles()
        assert sorted(env.out_order) == list(range(players))
