import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
import glob
import random
import time
import numpy as np
import torch.multiprocessing as mp
from playerTypes.player import Player
from playerTypes.baseline_bot import PresidentBaselineBot
from playerTypes.dmc_bot import PresidentDMCBot, PresidentValueNet
from utils import (
    get_cached_traced_model,
    game_loop,
    train_on_batch,
    save_snapshot,
    handle_keyboard_interrupt,
)

BATCH_GAMES = 60
SAVE_SNAPSHOT_EVERY = 250
MILESTONE_SNAPSHOT_EVERY = 5000
LEARNING_RATE = 1e-4
NUM_WORKERS = 12

TEMP_START = 0.05
TEMP_END = 0.01
TEMP_DECAY_BATCHES = 20000

_TRACED_LIVE_MODEL = None
_ELITE_SNAPSHOTS = []


def init_train_worker(shared_model, elite_snapshots):
    global _TRACED_LIVE_MODEL, _ELITE_SNAPSHOTS
    torch.set_num_threads(1)

    _ELITE_SNAPSHOTS = elite_snapshots or []
    for path in _ELITE_SNAPSHOTS:
        get_cached_traced_model(path, PresidentValueNet)

    first_linear = next(
        m for m in shared_model.modules() if isinstance(m, torch.nn.Linear)
    )
    example = torch.zeros(8, first_linear.in_features, dtype=torch.float32)
    with torch.no_grad():
        _TRACED_LIVE_MODEL = torch.jit.trace(shared_model, example)


def get_temperature(batch_idx):
    frac = min(batch_idx / TEMP_DECAY_BATCHES, 1.0)
    return TEMP_START * (TEMP_END / TEMP_START) ** frac


def parallel_worker(temperature):
    """Run a single game in a separate process and return the training data."""
    return run_single_game(
        _TRACED_LIVE_MODEL, torch.device("cpu"), temperature, _ELITE_SNAPSHOTS
    )


def run_single_game(live_model, device, temperature, elite_snapshots=None):
    num_players = random.randint(4, 7)
    bot_instances: dict[int, Player] = {
        0: PresidentDMCBot(0, live_model, device, True, temperature)
    }

    for seat in range(1, num_players):
        roll = random.random()
        if roll < 0.65:
            bot_instances[seat] = PresidentDMCBot(
                seat, live_model, device, True, temperature
            )
        elif roll < 0.80 and elite_snapshots:
            snap_path = random.choice(elite_snapshots)
            snap_model = get_cached_traced_model(snap_path, PresidentValueNet)
            bot_instances[seat] = PresidentDMCBot(seat, snap_model, device)
        elif roll < 0.90 and elite_snapshots:
            snap_path = random.choice(elite_snapshots)
            snap_model = get_cached_traced_model(snap_path, PresidentValueNet)
            bot_instances[seat] = PresidentDMCBot(
                seat, snap_model, device, profile="aggressive"
            )
        else:
            bot_instances[seat] = PresidentBaselineBot(seat)

    x, a, y = game_loop(num_players, bot_instances, live_model)
    learners = [
        bot
        for bot in bot_instances.values()
        if isinstance(bot, PresidentDMCBot) and bot.training
    ]
    stats = (
        sum(b.n_decisions for b in learners),
        sum(b.n_explored for b in learners),
        sum(b.q_gap_sum for b in learners),
    )

    return x, a, y, stats


def run_training_loop(snapshot_dir="snapshots"):
    torch.set_num_threads(2)
    os.makedirs(snapshot_dir, exist_ok=True)

    ctx = mp.get_context("spawn")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    resume_path = os.path.join(snapshot_dir, "latest_model.pt")
    live_model = PresidentValueNet().to(device)
    shared_model = PresidentValueNet().to("cpu")
    shared_model.share_memory()
    shared_model.load_state_dict(live_model.state_dict())

    optimizer = torch.optim.Adam(live_model.parameters(), lr=LEARNING_RATE, fused=True)
    loss_fn = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.999988)

    if os.path.exists(resume_path):
        print(f"Resuming checkpoint from {resume_path}...")
        checkpoint = torch.load(resume_path, map_location=device)
        live_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        batch_idx = checkpoint["batch_idx"]
    else:
        batch_idx = 0

    print("Starting training loop. Press Ctrl+C to stop and save the model.")
    max_batches = 0 if batch_idx == 0 else batch_idx % 2000
    running_losses = []
    rollout_times = []
    train_times = []
    iter_times = []
    n_decisions = n_explored = 0
    q_gap_sum = 0.0
    elite_snapshots = glob.glob(os.path.join(snapshot_dir, "elites/model_gen_*.pt"))

    with ctx.Pool(
        processes=NUM_WORKERS,
        initializer=init_train_worker,
        initargs=(shared_model, elite_snapshots),
    ) as pool:
        try:
            temperature = get_temperature(batch_idx)
            tasks = [(temperature,) for _ in range(BATCH_GAMES)]
            pending = pool.starmap_async(parallel_worker, tasks, chunksize=1)

            while max_batches < 2000:
                t_iter_start = time.perf_counter()
                batch_idx += 1
                max_batches += 1
                temperature = get_temperature(batch_idx)

                try:
                    t0 = time.perf_counter()
                    game_data = pending.get(timeout=600)
                    t1 = time.perf_counter()
                except mp.TimeoutError:
                    print(f"\nBatch {batch_idx} timed out.")
                    tasks = [(temperature,) for _ in range(BATCH_GAMES)]
                    pending = pool.starmap_async(parallel_worker, tasks, chunksize=1)
                    continue

                next_tasks = [(temperature,) for _ in range(BATCH_GAMES)]
                pending = pool.starmap_async(parallel_worker, next_tasks, chunksize=1)

                all_x, all_a, all_y = [], [], []
                for x, a, y, (dec, expl, gap) in game_data:
                    n_decisions += dec
                    n_explored += expl
                    q_gap_sum += gap

                    if x.shape[0] == 0:
                        continue

                    all_x.append(x)
                    all_a.append(a)
                    all_y.append(y)

                if all_x:
                    avg_loss = train_on_batch(
                        all_x,
                        all_a,
                        all_y,
                        device,
                        live_model,
                        optimizer,
                        loss_fn,
                        shared_model,
                        scheduler,
                    )
                    t2 = time.perf_counter()
                    running_losses.append(avg_loss)
                    rollout_times.append(t1 - t0)
                    train_times.append(t2 - t1)

                    if batch_idx % 25 == 0:
                        rolling_loss = np.mean(running_losses)
                        avg_rollout = np.mean(rollout_times)
                        avg_train = np.mean(train_times)
                        avg_iter = np.mean(iter_times) if iter_times else 0.0
                        train_share = (avg_train / (avg_rollout + avg_train)) * 100
                        explore_pct = 100.0 * n_explored / max(n_decisions, 1)
                        mean_gap = q_gap_sum / max(n_explored, 1)

                        print(
                            f"=========================================================================\n"
                            f"BATCH {batch_idx:<5} | Rolling Loss: {rolling_loss:.6f} | "
                            f"LR: {scheduler.get_last_lr()[0]:.6f} | Temp: {temperature:.4f}\n"
                            f"Explored: {explore_pct:.1f}% of decisions | "
                            f"Mean Q-Gap when exploring: {mean_gap:.4f}\n"
                            f"Rollout: {avg_rollout:.2f}s | Train: {avg_train:.2f}s | "
                            f"Train Share: {train_share:.1f}% | Iter: {avg_iter:.2f}s\n"
                            f"========================================================================="
                        )
                        running_losses.clear()
                        rollout_times.clear()
                        train_times.clear()
                        iter_times.clear()
                        n_decisions = n_explored = 0
                        q_gap_sum = 0.0
                else:
                    print(f"Batch {batch_idx}: No training data collected.")

                if batch_idx % SAVE_SNAPSHOT_EVERY == 0:
                    snapshot_path = save_snapshot(
                        snapshot_dir,
                        batch_idx,
                        live_model,
                        optimizer,
                        scheduler,
                        resume_path,
                        MILESTONE_SNAPSHOT_EVERY,
                    )
                    print(
                        f"Saved model snapshot to {snapshot_path} and updated latest_model.pt."
                    )
                iter_times.append(time.perf_counter() - t_iter_start)
        except KeyboardInterrupt:
            handle_keyboard_interrupt(pool)


def main():
    run_training_loop()


if __name__ == "__main__":
    main()
