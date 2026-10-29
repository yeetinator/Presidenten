import pytest
import torch
import torch.nn as nn

from game import President
from playerTypes.dmc_bot import NUM_ACTIONS, PASS_ACTION, PresidentDMCBot, action_indices

CPU = torch.device("cpu")


class FixedQ(nn.Module):
    def __init__(self, q):
        super().__init__()
        self.q = torch.tensor(q, dtype=torch.float32)

    def forward(self, x):
        return self.q.expand(x.shape[0], -1)


def q_values(pass_q, plays):
    q = [-1.0] * NUM_ACTIONS
    q[PASS_ACTION] = pass_q
    for move, value in plays.items():
        q[int(action_indices([move])[0])] = value
    return q


def endgame_state(active):
    """Seat 0 answers a single 4 from seat 1, with a 5 and a 9 to beat it."""
    env = President(4, lightweight=True)
    env.full_reset()
    env.first_turn = False
    env.hands = {0: [5, 9], 1: [7, 12], 2: [10] if active == 3 else [], 3: []}
    env.playing = set(range(active))
    env.out_order = [p for p in range(4) if p not in env.playing]
    env.last_move = (4, 1, 0)
    env.pile_leader = 1
    env.curr_turn = 0
    return env._get_state(0)


LOST = q_values(-0.97, {(5, 1, 0): -1.02, (9, 1, 0): -0.99})


def test_plays_on_in_a_lost_two_player_endgame():
    bot = PresidentDMCBot(0, FixedQ(LOST), CPU)
    assert bot.get_move(endgame_state(active=2)) == (9, 1, 0)


@pytest.mark.parametrize(
    "q, active, kwargs",
    [
        (LOST, 3, {}),  # more than 2 players left
        (q_values(-0.5, {(5, 1, 0): -0.6, (9, 1, 0): -0.7}), 2, {}),  # not lost
        (LOST, 2, {"record": True}),  # training games keep the learned choice
    ],
)
def test_keeps_passing_otherwise(q, active, kwargs):
    bot = PresidentDMCBot(0, FixedQ(q), CPU, **kwargs)
    assert bot.get_move(endgame_state(active)) == (0, 0, 0)
