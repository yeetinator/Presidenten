import random

import numpy as np
import pytest
import torch

from game import President
from playerTypes.dmc_bot import (
    COUNT_BUCKETS,
    NUM_RANKS,
    PresidentDMCBot,
    PresidentValueNet,
    _rank_counts,
)
from playerTypes.ppo_bot import (
    CRITIC_DIM,
    DMC_ENSEMBLE,
    STATE_DIM,
    PresidentCriticNet,
    PresidentPolicyNet,
    PresidentPPOBot,
    critic_input,
    opponent_hand_counts,
)
from utils.common import get_cached_traced_model
from utils.gameplay import iter_batched, run_tagged_games, seeded_game_gen
from utils.ppo import (
    build_batch,
    compute_gae,
    make_optimizer,
    masked_entropy,
    masked_log_softmax,
    ppo_update,
    predict_values,
    rollout_game_gen,
    train_critic,
)

CPU = torch.device("cpu")


def untrained_policy(seed=0):
    """Near-uniform over the legal moves, so sampled games wander widely."""
    torch.manual_seed(seed)
    return PresidentPolicyNet().eval()


class CheckedPPOBot(PresidentPPOBot):
    """Fails the test on any move outside the legal moves it was offered."""

    def get_move_gen(self, state):
        move = yield from super().get_move_gen(state)
        legal = state["legal_moves"] or [(0, 0, 0)]
        assert move in legal, f"illegal move {move}, legal {legal}"
        return move


@pytest.mark.parametrize("sample", [False, True])
def test_ppo_bot_only_plays_legal_moves(sample):
    model = untrained_policy()
    games = []
    for g in range(16):
        n = 4 + g % 4
        bots = {
            s: CheckedPPOBot(s, model, CPU, sample=sample, passer="bots/DMC.pt")
            for s in range(n)
        }
        games.append((g, seeded_game_gen(n, bots, [70 + g * 10 + r for r in range(3)])))
    assert len(run_tagged_games(games)) == 16


def test_greedy_ppo_bot_with_dmc_weights_plays_like_the_dmc_bot():
    """Covers the batched dispatch, the masking and the DMC passer end to end."""
    policy = get_cached_traced_model("bots/DMC.pt", PresidentPolicyNet)
    q_net = get_cached_traced_model("bots/DMC.pt", PresidentValueNet)
    results = []
    for bot_for in (
        lambda s: PresidentPPOBot(s, policy, CPU, passer="bots/DMC.pt"),
        lambda s: PresidentDMCBot(s, q_net, CPU),
    ):
        games = []
        for g in range(8):
            n = 4 + g % 4
            bots = {s: bot_for(s) for s in range(n)}
            games.append(
                (g, seeded_game_gen(n, bots, [900 + g * 10 + r for r in range(10)]))
            )
        results.append(run_tagged_games(games))
    assert results[0] == results[1]


def test_sampling_leaves_the_global_rngs_alone():
    random.seed(11)
    env = President(5, lightweight=True)
    env.full_reset()
    env.first_turn = False  # leading a fresh pile: any rank in hand is legal
    state = env._get_state(env.curr_turn)

    bot = PresidentPPOBot(
        env.curr_turn, untrained_policy(), CPU, sample=True, passer="bots/DMC.pt"
    )
    py_state, np_state = random.getstate(), np.random.get_state()
    moves = {bot.get_move(state) for _ in range(200)}

    assert random.getstate() == py_state
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    assert len(moves) > 1, f"one move sampled from {state['legal_moves']}"


def test_critic_sees_each_opponent_hand_in_the_actor_seat_order():
    random.seed(3)
    env = President(5, lightweight=True)
    env.full_reset()
    env.hands[4] = []  # an opponent who is already out

    counts = opponent_hand_counts(env, my_id=2)
    for row, p_id in enumerate([3, 4, 0, 1]):
        assert list(counts[row]) == list(_rank_counts(env.hands[p_id]))
    assert (counts[4:] == -1).all(), "absent seats are padded"

    x = torch.zeros(1, STATE_DIM)
    full = critic_input(x, torch.from_numpy(counts[None]))
    assert full.shape == (1, CRITIC_DIM)

    hands = full[0, STATE_DIM:].view(6, COUNT_BUCKETS * NUM_RANKS).numpy()
    rows = np.arange(COUNT_BUCKETS).reshape(-1, 1)
    for row, p_id in enumerate([3, 4, 0, 1]):
        expected = (rows < _rank_counts(env.hands[p_id])).ravel()
        assert np.array_equal(hands[row], expected.astype(np.float32))
    assert (hands[4:] == -1).all()


def test_recording_needs_the_env_for_the_critic():
    with pytest.raises(ValueError):
        PresidentPPOBot(0, untrained_policy(), CPU, record=True, passer="bots/DMC.pt")


def fake_batch(lengths, rewards, n_actions=125):
    n = sum(lengths)
    game = {
        "x": np.zeros((n, STATE_DIM), dtype=np.float32),
        "opp": np.zeros((n, 6, NUM_RANKS), dtype=np.int8),
        "legal": np.ones((n, n_actions), dtype=bool),
        "action": np.zeros(n, dtype=np.int64),
        "logp": np.zeros(n, dtype=np.float32),
        "lengths": np.array(lengths, dtype=np.int64),
        "rewards": np.array(rewards, dtype=np.float32),
        "version": 0,
    }
    return build_batch([game], CPU)


def test_gae_matches_a_hand_computed_round():
    batch = fake_batch([3, 1], [1.0, -0.5])
    values = torch.tensor([0.2, 0.5, 0.3, 0.1])

    # deltas 0.3, -0.2, 0.7 | -0.6; at lambda 0.5: 0.3 + 0.5 * (-0.2 + 0.5 * 0.7)
    adv, ret = compute_gae(values, batch, gamma=1.0, lam=0.5)
    assert torch.allclose(adv, torch.tensor([0.375, 0.15, 0.7, -0.6]))
    assert torch.allclose(ret, adv + values)

    adv, _ = compute_gae(values, batch, gamma=1.0, lam=1.0)
    assert torch.allclose(adv, batch["outcome"] - values), "lambda 1 is R - V"


def test_recorded_log_probs_match_the_trainer_so_ratios_start_at_one():
    random.seed(5)
    model = untrained_policy(seed=1)
    ensemble = get_cached_traced_model(DMC_ENSEMBLE, PresidentValueNet)
    games = list(
        iter_batched(
            lambda: rollout_game_gen(model, ensemble, [], [], CPU, version=7), 3, 3
        )
    )
    games = [g for g in games if g is not None]
    assert games and all(g["version"] == 7 for g in games)

    batch = build_batch(games, CPU)
    n = batch["x"].shape[0]
    assert int(batch["lengths"].sum()) == n
    assert batch["legal"][torch.arange(n), batch["action"]].all()
    assert ((batch["rewards"] >= -1) & (batch["rewards"] <= 1)).all()

    with torch.no_grad():
        log_probs = masked_log_softmax(model(batch["x"]), batch["legal"])
    recomputed = log_probs.gather(1, batch["action"][:, None]).squeeze(1)
    assert torch.allclose(recomputed, batch["logp"], atol=1e-5)


def test_masked_log_softmax_and_entropy_have_finite_gradients():
    torch.manual_seed(0)
    logits = torch.randn(64, 125, requires_grad=True)
    legal = torch.rand(64, 125) < 0.05
    legal[:, 0] = True
    action = torch.multinomial(legal.float(), 1).squeeze(1)

    log_probs = masked_log_softmax(logits, legal)
    entropy = masked_entropy(log_probs, legal)
    (-log_probs.gather(1, action[:, None]).mean() - entropy.mean()).backward()

    assert torch.isfinite(logits.grad).all()
    probs = log_probs.exp()
    manual = -(probs * torch.where(legal, log_probs, 0.0)).sum(1)
    assert torch.allclose(entropy, manual, atol=1e-6)
    assert (probs[~legal] == 0).all()


def test_one_ppo_iteration_stays_in_the_trust_region_and_fits_the_critic():
    random.seed(9)
    torch.manual_seed(0)
    actor = PresidentPolicyNet()
    ensemble = get_cached_traced_model(DMC_ENSEMBLE, PresidentValueNet)
    games = iter_batched(
        lambda: rollout_game_gen(actor.eval(), ensemble, [], [], CPU, 0), 2, 2
    )
    batch = build_batch([g for g in games if g is not None], CPU)

    critic = PresidentCriticNet()
    values = predict_values(critic, "full", batch)
    adv, returns = compute_gae(values, batch)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    stats = ppo_update(
        actor, make_optimizer(actor, 3e-5, 0.01), batch, adv, 0.1, 0.003, 1, 256, 1.0
    )
    assert stats["epochs"] == 1
    assert 0 <= stats["clip"] < 0.5 and abs(stats["kl"]) < 0.05

    critic_opt = make_optimizer(critic, 1e-4, 0.01)
    first = train_critic(critic, critic_opt, "full", batch, returns, 1, 256)
    for _ in range(5):
        last = train_critic(critic, critic_opt, "full", batch, returns, 1, 256)
    assert last < first
