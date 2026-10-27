import queue
import random
import functools
import numpy as np
import torch
from game import President
from playerTypes.player import Player
from playerTypes.baseline_bot import PresidentBaselineBot
from playerTypes.dmc_bot import (
    NUM_ACTIONS,
    STATE_DIM,
    PresidentDMCBot,
    PresidentValueNet,
)
from playerTypes.ppo_bot import (
    DMC_ENSEMBLE,
    PresidentPolicyNet,
    PresidentPPOBot,
    critic_input,
)
from .common import NUM_ROUNDS, get_cached_model, get_cached_traced_model
from .gameplay import play_round_gen, iter_batched
from .training import GRADIENT_CLIP, _concat_pinned, elite_subset

# Greedy unless gate 1 shows sampling plays better
EVAL_SAMPLE = False
PPO_CLASSES = (
    PresidentPolicyNet,
    functools.partial(PresidentPPOBot, sample=True) if EVAL_SAMPLE else PresidentPPOBot,
)

# Seats after seat 0 (always live), as cumulative thresholds of one roll
LIVE_SHARE = 0.50
ENSEMBLE_SHARE = 0.10
DMC_ELITE_SHARE = 0.15
AGGRESSIVE_SHARE = 0.05
PPO_ELITE_SHARE = 0.10  # the rest are Baseline bots

ELITE_MIN_BATCH = 8
ELITE_MAX_WAIT = 16


class TeacherPolicyNet(torch.nn.Module):
    """A Q-net played as a policy: logits = Q / tau, so a sampling PPO bot plays
    DMC's Boltzmann policy and a greedy one plays DMC's greedy move."""

    def __init__(self, q_net, tau):
        super().__init__()
        self.q_net = q_net
        self.tau = tau

    def forward(self, x):
        return self.q_net(x) / self.tau


def placement_reward(rank, num_players):
    """DMC's round reward: normalized placement in [-1, 1]."""
    max_score = num_players - 1
    return ((max_score - rank) / max_score) * 2 - 1


def rollout_game_gen(live_model, ensemble, dmc_elites, ppo_elites, device, version):
    """One game of 4-7 players. Each live seat's real decisions in a round form
    one episode, rewarded with its placement. Returns the packed episodes, or
    None when nothing was recorded."""
    num_players = random.randint(4, 7)
    env = President(num_players, lightweight=True)

    def live(seat):
        return PresidentPPOBot(
            seat,
            live_model,
            device,
            sample=True,
            record=True,
            critic_env=env,
            passer=ensemble,
        )

    bot_instances: dict[int, Player] = {0: live(0)}
    for seat in range(1, num_players):
        roll = random.random()
        threshold = LIVE_SHARE
        if roll < threshold:
            bot_instances[seat] = live(seat)
            continue
        threshold += ENSEMBLE_SHARE
        if roll < threshold:
            bot_instances[seat] = PresidentDMCBot(seat, ensemble, device)
            continue
        threshold += DMC_ELITE_SHARE
        if roll < threshold:
            model = random.choice(dmc_elites) if dmc_elites else ensemble
            bot_instances[seat] = PresidentDMCBot(seat, model, device)
            continue
        threshold += AGGRESSIVE_SHARE
        if roll < threshold:
            model = random.choice(dmc_elites) if dmc_elites else ensemble
            bot_instances[seat] = PresidentDMCBot(
                seat, model, device, profile="aggressive"
            )
            continue
        threshold += PPO_ELITE_SHARE
        if roll < threshold:
            if ppo_elites:
                bot_instances[seat] = PresidentPPOBot(
                    seat, random.choice(ppo_elites), device, passer=ensemble
                )
            else:  # No league yet, so a DMC elite stands in
                model = random.choice(dmc_elites) if dmc_elites else ensemble
                bot_instances[seat] = PresidentDMCBot(seat, model, device)
            continue
        bot_instances[seat] = PresidentBaselineBot(seat)

    episodes = []
    for round_idx in range(NUM_ROUNDS):
        completed = yield from play_round_gen(env, bot_instances, round_idx)
        if not completed:
            return None

        for rank, p_id in enumerate(env.out_order):
            bot = bot_instances[p_id]
            if isinstance(bot, PresidentPPOBot) and bot.record:
                if bot.trajectory:
                    episodes.append(
                        (bot.trajectory, placement_reward(rank, num_players))
                    )
                bot.trajectory = []
    return pack_episodes(episodes, version)


def pack_episodes(episodes, version):
    steps = [step for trajectory, _ in episodes for step in trajectory]
    if not steps:
        return None

    legal = np.zeros((len(steps), NUM_ACTIONS), dtype=bool)
    for row, (_, legal_idx, *_) in enumerate(steps):
        legal[row, legal_idx] = True

    return {
        "x": np.stack([s[0] for s in steps]),
        "opp": np.stack([s[4] for s in steps]),
        "legal": legal,
        "action": np.array([s[2] for s in steps], dtype=np.int64),
        "logp": np.array([s[3] for s in steps], dtype=np.float32),
        "lengths": np.array([len(t) for t, _ in episodes], dtype=np.int64),
        "rewards": np.array([r for _, r in episodes], dtype=np.float32),
        "version": version,
    }


def _trace(model, device, freeze=False):
    """Unfrozen traces share weights with model, so trainer updates show up."""
    with torch.no_grad():
        traced = torch.jit.trace(
            model.eval(), torch.zeros(8, STATE_DIM, device=device), check_trace=False
        )
        return torch.jit.freeze(traced) if freeze else traced


def actor_loop(
    worker_id,
    shared_model,
    model_version,
    device,
    dmc_elite_paths,
    ppo_elite_paths,
    stop_event,
    out_queue,
    games_in_flight,
    elites_per_worker,
    teacher_tau=None,
):
    """Plays rollout games for the trainer. The live seats play the shared actor,
    or with teacher_tau set, the DMC ensemble's Boltzmann policy."""
    torch.set_num_threads(1)
    device_model = None
    if teacher_tau is not None:
        q_net = get_cached_model(DMC_ENSEMBLE, PresidentValueNet, device)
        live_model = _trace(TeacherPolicyNet(q_net, teacher_tau), device, freeze=True)
    elif device.type == "cpu":
        # Traced straight from shared memory, so trainer updates show up as-is
        live_model = _trace(shared_model, device)
    else:
        # CUDA tensors can't be shared between processes on Windows, so keep a
        # device copy and refresh it from shared memory when the trainer steps
        device_model = PresidentPolicyNet().to(device)
        live_model = _trace(device_model, device)
    synced_version = -1

    ensemble = get_cached_traced_model(DMC_ENSEMBLE, PresidentValueNet, device=device)
    dmc_elites = [
        get_cached_traced_model(path, PresidentValueNet, device=device)
        for path in elite_subset(dmc_elite_paths, worker_id, elites_per_worker)
    ]
    ppo_elites = [
        get_cached_traced_model(path, PresidentPolicyNet, device=device)
        for path in elite_subset(ppo_elite_paths, worker_id, elites_per_worker)
    ]

    def make_game():
        nonlocal synced_version
        version = model_version.value

        if device_model is not None and version != synced_version:
            with torch.no_grad():
                for dst, src in zip(
                    device_model.parameters(), shared_model.parameters()
                ):
                    dst.copy_(src)
        synced_version = version
        return rollout_game_gen(
            live_model, ensemble, dmc_elites, ppo_elites, device, version
        )

    try:
        for result in iter_batched(
            make_game,
            float("inf"),
            games_in_flight,
            stop_event.is_set,
            ELITE_MIN_BATCH,
            ELITE_MAX_WAIT,
            device,
        ):
            if result is None:
                continue
            while not stop_event.is_set():
                try:
                    out_queue.put(result, timeout=1.0)
                    break
                except queue.Full:
                    continue
    except KeyboardInterrupt:
        pass
    finally:
        out_queue.cancel_join_thread()


def collect_games(out_queue, actors, n_games, version=None, max_lag=None):
    """Returns (games, lags, dropped): games older than max_lag versions are
    dropped, as their behaviour policy is too far from the one being trained."""
    games, lags, dropped = [], [], 0
    while len(games) < n_games:
        try:
            game = out_queue.get(timeout=5)
        except queue.Empty:
            if not all(a.is_alive() for a in actors):
                raise RuntimeError("An actor process died.")
            continue
        if version is not None:
            lag = version - game["version"]
            if lag > max_lag:
                dropped += 1
                continue
            lags.append(lag)
        games.append(game)
    return games, lags, dropped


def build_batch(games, device):
    pin = device.type == "cuda"

    def cat(key, dtype):
        arrays = [g[key] for g in games]
        return _concat_pinned(arrays, dtype, pin).to(device, non_blocking=True)

    batch = {
        "x": cat("x", torch.float32),
        "opp": cat("opp", torch.int8),
        "legal": cat("legal", torch.bool),
        "action": cat("action", torch.int64),
        "logp": cat("logp", torch.float32),
        "lengths": cat("lengths", torch.int64),
        "rewards": cat("rewards", torch.float32),
    }
    lengths = batch["lengths"]
    steps = torch.arange(int(lengths.max()), device=device)
    starts = torch.cumsum(lengths, 0) - lengths
    # [episodes, max length] views of the flat decisions, for the GAE recursion
    batch["valid"] = steps[None, :] < lengths[:, None]
    batch["index"] = torch.where(batch["valid"], starts[:, None] + steps[None, :], 0)
    episode_of = torch.repeat_interleave(
        torch.arange(lengths.numel(), device=device), lengths
    )
    batch["outcome"] = batch["rewards"][episode_of]
    return batch


def compute_gae(values, batch, gamma=1.0, lam=0.95):
    """values [N] at each decision -> (advantages, returns) [N]. The placement is
    the only reward and comes after the episode's last decision."""
    valid, rewards = batch["valid"], batch["rewards"]
    v = torch.where(valid, values[batch["index"]], 0.0)
    episodes, length = v.shape

    next_v = torch.zeros_like(v)
    next_v[:, :-1] = gamma * v[:, 1:]
    next_v[torch.arange(episodes, device=v.device), batch["lengths"] - 1] = rewards
    delta = torch.where(valid, next_v - v, 0.0)

    # Padding only follows an episode's end, where delta is 0, so the running
    # sum is 0 when it reaches the last decision
    adv = torch.zeros_like(v)
    running = torch.zeros(episodes, device=v.device)
    for t in reversed(range(length)):
        running = delta[:, t] + gamma * lam * running
        adv[:, t] = running

    advantages = adv[valid]
    return advantages, advantages + values


def masked_log_softmax(logits, legal):
    return torch.log_softmax(logits.float().masked_fill(~legal, float("-inf")), dim=1)


def masked_entropy(log_probs, legal):
    # Zeroing the illegal log-probs first keeps their -inf out of the gradient
    safe = log_probs.masked_fill(~legal, 0.0)
    return -(safe.exp() * safe).masked_fill(~legal, 0.0).sum(1)


def explained_variance(pred, target):
    var = target.var()
    return float(1.0 - (target - pred).var() / var) if var > 0 else float("nan")


def critic_features(name, batch, rows=slice(None)):
    """The full critic sees every opponent's hand, the own critic only the
    actor's state."""
    x = batch["x"][rows]
    return critic_input(x, batch["opp"][rows]) if name == "full" else x


@torch.no_grad()
def predict_values(critic, name, batch, chunk=16384):
    n = batch["x"].shape[0]
    out = torch.empty(n, device=batch["x"].device)
    critic.eval()
    with torch.autocast(device_type=out.device.type, dtype=torch.bfloat16):
        for i in range(0, n, chunk):
            rows = slice(i, i + chunk)
            out[rows] = critic(critic_features(name, batch, rows)).float()
    return out


def train_critic(critic, optimizer, name, batch, targets, epochs, mini_batch_size):
    n = targets.shape[0]
    critic.train()
    loss_sum = torch.zeros((), device=targets.device)
    n_steps = 0
    for _ in range(epochs):
        perm = torch.randperm(n, device=targets.device)
        for i in range(0, n, mini_batch_size):
            rows = perm[i : i + mini_batch_size]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=targets.device.type, dtype=torch.bfloat16):
                pred = critic(critic_features(name, batch, rows))
            loss = torch.nn.functional.mse_loss(pred.float(), targets[rows])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), GRADIENT_CLIP)
            optimizer.step()
            loss_sum += loss.detach()
            n_steps += 1
    return (loss_sum / max(n_steps, 1)).item()


def ppo_update(
    actor,
    optimizer,
    batch,
    advantages,
    clip,
    ent_coef,
    epochs,
    mini_batch_size,
    max_kl,
):
    """Clipped PPO on the actor, in full fp32: its logits are on a Q / tau scale,
    where bf16 or TF32 rounding would show up in the probability ratios. Stops
    the epochs early once a minibatch's approximate KL exceeds max_kl."""
    n = advantages.shape[0]
    actor.train()
    sums = {"pi": 0.0, "ent": 0.0, "kl": 0.0, "clip": 0.0}
    n_steps = 0
    epochs_done = 0.0

    for _ in range(epochs):
        perm = torch.randperm(n, device=advantages.device)
        n_minibatches = (n + mini_batch_size - 1) // mini_batch_size
        stopped = False

        for k, i in enumerate(range(0, n, mini_batch_size)):
            rows = perm[i : i + mini_batch_size]
            legal = batch["legal"][rows]
            log_probs = masked_log_softmax(actor(batch["x"][rows]), legal)
            new_logp = log_probs.gather(1, batch["action"][rows, None]).squeeze(1)
            log_ratio = new_logp - batch["logp"][rows]
            ratio = log_ratio.exp()
            adv = advantages[rows]

            surrogate = torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv)
            entropy = masked_entropy(log_probs, legal).mean()
            loss = -surrogate.mean() - ent_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), GRADIENT_CLIP)
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1) - log_ratio).mean().item()
                sums["pi"] += -surrogate.mean().item()
                sums["ent"] += entropy.item()
                sums["kl"] += approx_kl
                sums["clip"] += ((ratio - 1).abs() > clip).float().mean().item()
            n_steps += 1

            if approx_kl > max_kl:
                epochs_done += (k + 1) / n_minibatches
                stopped = True
                break
        if stopped:
            break
        epochs_done += 1

    stats = {key: value / max(n_steps, 1) for key, value in sums.items()}
    stats["epochs"] = epochs_done
    return stats
