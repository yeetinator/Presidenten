import os

# Avoid oversubscription
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
import copy
import glob
import time
import sys
import numpy as np
import torch.multiprocessing as mp
from playerTypes.dmc_bot import PresidentValueNet, STATE_DIM
from playerTypes.ppo_bot import DMC_ENSEMBLE, PresidentPolicyNet, PresidentCriticNet
from utils.common import get_cached_model
from utils.orchestrator import current_lr_scale
from utils.ppo import (
    DISTILLED_PATH,
    DMC_ELITES_DIR,
    HELDOUT_PATH,
    PPO_DIR,
    actor_loop,
    build_batch,
    collect_games,
    compute_gae,
    explained_variance,
    make_optimizer,
    masked_log_softmax,
    ppo_update,
    predict_values,
    train_critic,
)
from utils.training import save_snapshot, update_ema, shutdown

# "own" (Phase 4) takes advantages from the critic that sees only the actor's
# information; both critics are always trained, so "full" also logs ev_own
CRITIC = "full"
SNAPSHOT_DIR = PPO_DIR if CRITIC == "full" else f"{PPO_DIR}_own"

GAMES_PER_ITER = 384  # about 100k decisions
ITERS_PER_CYCLE = 312  # about 120k games, one DMC cycle
SAVE_SNAPSHOT_EVERY = 39
MILESTONE_SNAPSHOT_EVERY = 780
LEARNING_RATE = 3e-5
CRITIC_LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
EMA_DECAY = 0.987  # Per iteration: about 30k games, as DMC's 0.998 per batch
CLIP = 0.1
EPOCHS = 4
MINI_BATCH = 4096
MAX_KL = 0.015  # Stops the epochs early, 1.5x a 0.01 target
GAMMA = 1.0
LAMBDA = 0.95
MAX_LAG = 2  # Policy versions a game may lag the trainer by
NUM_WORKERS = 12
GAMES_IN_FLIGHT_PER_WORKER = 32
ELITES_PER_WORKER = 2

ENT_START = 0.003
ENT_END = 0.0005
ENT_DECAY_ITERS = 3000

INFERENCE_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_entropy_coef(batch_idx):
    frac = min(batch_idx / ENT_DECAY_ITERS, 1.0)
    return ENT_START * (ENT_END / ENT_START) ** frac


@torch.no_grad()
def heldout_agreement(actor, heldout, teacher_pick):
    """Share of held-out decisions where the actor's greedy move is the DMC
    ensemble's: how far PPO has moved from its teacher."""
    log_probs = masked_log_softmax(actor(heldout["x"]), heldout["legal"])
    return (log_probs.argmax(1) == teacher_pick).float().mean().item()


def run_training_loop(snapshot_dir=SNAPSHOT_DIR, max_iters=None):
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")  # the actor's logits need fp32
    os.makedirs(snapshot_dir, exist_ok=True)

    ctx = mp.get_context("spawn")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | advantages from the {CRITIC} critic")

    resume_path = os.path.join(snapshot_dir, "latest_model.pt")
    start_path = resume_path if os.path.exists(resume_path) else DISTILLED_PATH
    print(f"Starting from {start_path}...")
    checkpoint = torch.load(start_path, map_location=device)

    live_model = PresidentPolicyNet().to(device)
    live_model.load_state_dict(
        checkpoint.get("live_model_state_dict", checkpoint["model_state_dict"])
    )
    ema_model = copy.deepcopy(live_model).requires_grad_(False)
    ema_model.load_state_dict(checkpoint["model_state_dict"])
    shared_model = PresidentPolicyNet().to("cpu")
    shared_model.load_state_dict(live_model.state_dict())
    shared_model.share_memory()

    critics = {
        "full": PresidentCriticNet().to(device),
        "own": PresidentCriticNet(STATE_DIM).to(device),
    }
    critics["full"].load_state_dict(checkpoint["critic_state_dict"])
    critics["own"].load_state_dict(checkpoint["own_critic_state_dict"])
    critic_opts = {
        name: make_optimizer(net, CRITIC_LEARNING_RATE, WEIGHT_DECAY)
        for name, net in critics.items()
    }

    lr = LEARNING_RATE * current_lr_scale(snapshot_dir)
    optimizer = make_optimizer(live_model, lr, WEIGHT_DECAY)
    batch_idx = checkpoint["batch_idx"]
    if start_path == resume_path:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for name, opt in critic_opts.items():
            opt.load_state_dict(checkpoint[f"{name}_critic_optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = lr
    print(f"Learning rate: {lr:.2e} | critic {CRITIC_LEARNING_RATE:.0e}")

    heldout = build_batch([dict(np.load(HELDOUT_PATH))], device)
    teacher = get_cached_model(DMC_ENSEMBLE, PresidentValueNet, device)
    with torch.no_grad():
        q = teacher(heldout["x"]).masked_fill(~heldout["legal"], float("-inf"))
    teacher_pick = q.argmax(1)

    dmc_elites = glob.glob(f"{DMC_ELITES_DIR}/model_gen_*.pt")
    ppo_elites = glob.glob(f"{snapshot_dir}/elites/model_gen_*.pt")
    model_version = ctx.Value("i", 0, lock=False)  # bumped after each update
    stop_event = ctx.Event()
    out_queue = ctx.Queue(maxsize=GAMES_PER_ITER)
    actors = [
        ctx.Process(
            target=actor_loop,
            args=(
                i,
                shared_model,
                model_version,
                INFERENCE_DEVICE,
                dmc_elites,
                ppo_elites,
                stop_event,
                out_queue,
                GAMES_IN_FLIGHT_PER_WORKER,
                ELITES_PER_WORKER,
            ),
            daemon=True,
        )
        for i in range(NUM_WORKERS)
    ]
    for a in actors:
        a.start()

    print("Starting training loop. Press Ctrl+C to stop and save the model.")
    iters_done = batch_idx % ITERS_PER_CYCLE
    if max_iters is not None:  # a short run, e.g. a smoke test
        iters_done = ITERS_PER_CYCLE - max_iters
    window = {}
    rollout_times, train_times, lags = [], [], []
    dropped = kept = 0
    t_window_start = time.perf_counter()

    try:
        while iters_done < ITERS_PER_CYCLE:
            t0 = time.perf_counter()
            batch_idx += 1
            iters_done += 1

            games, game_lags, n_dropped = collect_games(
                out_queue, actors, GAMES_PER_ITER, model_version.value, MAX_LAG
            )
            t1 = time.perf_counter()
            lags += game_lags
            dropped += n_dropped
            kept += len(games)

            batch = build_batch(games, device)
            stats = {}
            returns = {}
            for name, critic in critics.items():
                values = predict_values(critic, name, batch)
                adv, returns[name] = compute_gae(values, batch, GAMMA, LAMBDA)
                stats[f"ev_{name}"] = explained_variance(values, batch["outcome"])
                if name == CRITIC:
                    advantages = adv

            stats["adv"] = advantages.std().item()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            stats.update(
                ppo_update(
                    live_model,
                    optimizer,
                    batch,
                    advantages,
                    CLIP,
                    get_entropy_coef(batch_idx),
                    EPOCHS,
                    MINI_BATCH,
                    MAX_KL,
                )
            )
            shared_model.load_state_dict(live_model.state_dict())
            model_version.value += 1

            for name, critic in critics.items():
                stats[f"v_{name}"] = train_critic(
                    critic,
                    critic_opts[name],
                    name,
                    batch,
                    returns[name],
                    EPOCHS,
                    MINI_BATCH,
                )
            # Warm up the decay when starting fresh from the distilled actor
            ema_decay = min(EMA_DECAY, (1 + batch_idx) / (10 + batch_idx))
            update_ema(ema_model, live_model, ema_decay)

            stats["n"] = batch["x"].shape[0]
            for key, value in stats.items():
                window.setdefault(key, []).append(value)
            rollout_times.append(t1 - t0)
            train_times.append(time.perf_counter() - t1)

            if batch_idx % SAVE_SNAPSHOT_EVERY == 0:
                t_now = time.perf_counter()
                avg_iter = (t_now - t_window_start) / SAVE_SNAPSHOT_EVERY
                t_window_start = t_now
                m = {key: np.mean(values) for key, values in window.items()}
                agree = heldout_agreement(ema_model, heldout, teacher_pick)
                drop_pct = 100.0 * dropped / max(dropped + kept, 1)

                print(
                    f"=========================================================================\n"
                    f"it {batch_idx} | pi {m['pi']:+.4f} v {m[f'v_{CRITIC}']:.4f} | "
                    f"ent {m['ent']:.3f} | kl {m['kl']:.4f} | clip {100 * m['clip']:.1f}% | "
                    f"ev {m['ev_full']:.3f} own {m['ev_own']:.3f} | ep {m['epochs']:.1f} | "
                    f"adv {m['adv']:.3f} | lag {np.mean(lags):.2f} drop {drop_pct:.1f}% | "
                    f"agree {100 * agree:.1f}% | {m['n'] / 1000:.0f}k dec | lr {lr:.1e} | "
                    f"roll {np.mean(rollout_times):.2f}s train {np.mean(train_times):.2f}s | "
                    f"{avg_iter:.2f}s/it"
                )
                window.clear()
                rollout_times.clear()
                train_times.clear()
                lags.clear()
                dropped = kept = 0

                snapshot_path = save_snapshot(
                    snapshot_dir,
                    batch_idx,
                    live_model,
                    ema_model,
                    optimizer,
                    resume_path,
                    MILESTONE_SNAPSHOT_EVERY,
                    extra={
                        "critic_state_dict": critics["full"].state_dict(),
                        "own_critic_state_dict": critics["own"].state_dict(),
                        **{
                            f"{name}_critic_optimizer_state_dict": opt.state_dict()
                            for name, opt in critic_opts.items()
                        },
                    },
                )
                print(
                    f"Saved model snapshot to {snapshot_path} and updated latest_model.pt\n"
                )
    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        shutdown(actors, stop_event, out_queue)


def main():
    # python -m ppo_loop.train_ppo [snapshot_dir] [max_iters]
    args = sys.argv[1:]
    run_training_loop(
        args[0] if len(args) > 0 else SNAPSHOT_DIR,
        int(args[1]) if len(args) > 1 else None,
    )


if __name__ == "__main__":
    main()
