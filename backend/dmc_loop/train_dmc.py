import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
import copy
import glob
import random
import time
import sys
import queue
import numpy as np
import torch.multiprocessing as mp
from playerTypes.player import Player
from playerTypes.baseline_bot import PresidentBaselineBot
from playerTypes.dmc_bot import PresidentDMCBot, PresidentValueNet, STATE_DIM
from utils.common import get_cached_traced_model
from utils.gameplay import game_loop_gen, iter_batched
from utils.orchestrator import current_lr_scale
from utils.training import train_on_batch, save_snapshot, update_ema

BATCH_GAMES = 60
SAVE_SNAPSHOT_EVERY = 250
MILESTONE_SNAPSHOT_EVERY = 5000
LEARNING_RATE = 1e-4
EMA_DECAY = 0.998  # Per batch, so snapshots average roughly the last 500 batches
NUM_WORKERS = 12

TEMP_START = 0.05
TEMP_END = 0.01
TEMP_DECAY_BATCHES = 20000

RECORD_ELITES = True

GAMES_IN_FLIGHT_PER_WORKER = 32
ELITES_PER_WORKER = 2  # 0 = sample from all elites
QUEUE_GAMES = 2 * BATCH_GAMES
_CPU = torch.device("cpu")
ELITE_MIN_BATCH = 4
ELITE_MAX_WAIT = 16


def get_temperature(batch_idx):
    frac = min(batch_idx / TEMP_DECAY_BATCHES, 1.0)
    return TEMP_START * (TEMP_END / TEMP_START) ** frac


def training_game_gen(
    live_model, temperature, elite_snapshots=None, record_elites=True
):
    device = _CPU
    num_players = random.randint(4, 7)
    bot_instances: dict[int, Player] = {
        0: PresidentDMCBot(
            player_id=0,
            model=live_model,
            device=device,
            explore=True,
            temperature=temperature,
        )
    }

    for seat in range(1, num_players):
        roll = random.random()
        if roll < 0.65:
            bot_instances[seat] = PresidentDMCBot(
                player_id=seat,
                model=live_model,
                device=device,
                explore=True,
                temperature=temperature,
            )
        elif roll < 0.80 and elite_snapshots:
            snap_path = random.choice(elite_snapshots)
            snap_model = get_cached_traced_model(snap_path, PresidentValueNet)
            bot_instances[seat] = PresidentDMCBot(
                player_id=seat, model=snap_model, device=device, record=record_elites
            )
        elif roll < 0.90 and elite_snapshots:
            snap_path = random.choice(elite_snapshots)
            snap_model = get_cached_traced_model(snap_path, PresidentValueNet)
            bot_instances[seat] = PresidentDMCBot(
                player_id=seat, model=snap_model, device=device, profile="aggressive"
            )
        else:
            bot_instances[seat] = PresidentBaselineBot(seat)

    x, a, y = yield from game_loop_gen(num_players, bot_instances, lightweight=True)
    learners = [
        b
        for b in bot_instances.values()
        if isinstance(b, PresidentDMCBot) and b.explore
    ]
    stats = (
        sum(b.n_decisions for b in learners),
        sum(b.n_explored for b in learners),
        sum(b.q_gap_sum for b in learners),
    )

    return x, a, y, stats


def _trace_live(model):
    with torch.no_grad():
        return torch.jit.trace(model, torch.zeros(8, STATE_DIM), check_trace=False)


def elite_subset(elite_paths, worker_id, k):
    """Balanced when NUM_WORKERS * k is a multiple of len(elite_paths)."""
    elites = sorted(elite_paths)
    if not k or len(elites) <= k:
        return elites
    return [elites[(worker_id * k + j) % len(elites)] for j in range(k)]


def actor_loop(
    worker_id,
    shared_model,
    elite_paths,
    temperature,
    stop_event,
    out_queue,
    games_in_flight,
    elites_per_worker,
):
    torch.set_num_threads(1)
    live_model = _trace_live(shared_model)
    elites = elite_subset(elite_paths, worker_id, elites_per_worker)
    for path in elites:
        get_cached_traced_model(path, PresidentValueNet)

    def make_game():
        return training_game_gen(live_model, temperature.value, elites, RECORD_ELITES)

    try:
        for result in iter_batched(
            make_game,
            float("inf"),
            games_in_flight,
            stop_event.is_set,
            ELITE_MIN_BATCH,
            ELITE_MAX_WAIT,
        ):
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


def shutdown(actors, stop_event, out_queue):
    stop_event.set()
    deadline = time.time() + 10

    while any(a.is_alive() for a in actors) and time.time() < deadline:
        try:
            while True:
                out_queue.get_nowait()
        except (queue.Empty, OSError, EOFError):
            pass
        for a in actors:
            a.join(timeout=0.1)
    for a in actors:
        if a.is_alive():
            a.kill()


def run_training_loop(
    snapshot_dir="snapshots",
    games_in_flight=GAMES_IN_FLIGHT_PER_WORKER,
    elites_per_worker=ELITES_PER_WORKER,
):
    torch.set_num_threads(2)
    os.makedirs(snapshot_dir, exist_ok=True)

    ctx = mp.get_context("spawn")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    resume_path = os.path.join(snapshot_dir, "latest_model.pt")
    live_model = PresidentValueNet().to(device)
    shared_model = PresidentValueNet().to("cpu")
    shared_model.share_memory()

    optimizer = torch.optim.Adam(live_model.parameters(), lr=LEARNING_RATE, fused=True)
    loss_fn = torch.nn.MSELoss()

    if os.path.exists(resume_path):
        print(f"Resuming checkpoint from {resume_path}...")
        checkpoint = torch.load(resume_path, map_location=device)
        # Checkpoints from before EMA only hold the live weights
        live_model.load_state_dict(
            checkpoint.get("live_model_state_dict", checkpoint["model_state_dict"])
        )
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        batch_idx = checkpoint["batch_idx"]
    else:
        checkpoint = None
        batch_idx = 0

    ema_model = copy.deepcopy(live_model).requires_grad_(False)
    if checkpoint is not None:
        ema_model.load_state_dict(checkpoint["model_state_dict"])

    lr = LEARNING_RATE * current_lr_scale(snapshot_dir)
    for group in optimizer.param_groups:
        group["lr"] = lr
    print(f"Learning rate: {lr:.2e}")

    shared_model.load_state_dict(live_model.state_dict())

    print("Starting training loop. Press Ctrl+C to stop and save the model.")
    max_batches = 0 if batch_idx == 0 else batch_idx % 2000
    running_losses = []
    rollout_times = []
    train_times = []
    iter_times = []
    n_decisions = n_explored = 0
    q_gap_sum = 0.0
    elite_paths = glob.glob(os.path.join(snapshot_dir, "elites/model_gen_*.pt"))

    temperature = ctx.Value("d", get_temperature(batch_idx), lock=False)
    stop_event = ctx.Event()
    out_queue = ctx.Queue(maxsize=QUEUE_GAMES)
    actors = [
        ctx.Process(
            target=actor_loop,
            args=(
                i,
                shared_model,
                elite_paths,
                temperature,
                stop_event,
                out_queue,
                games_in_flight,
                elites_per_worker,
            ),
            daemon=True,
        )
        for i in range(NUM_WORKERS)
    ]
    for a in actors:
        a.start()

    try:
        while max_batches < 2000:
            t_iter_start = time.perf_counter()
            batch_idx += 1
            max_batches += 1

            t0 = time.perf_counter()
            game_data = []
            while len(game_data) < BATCH_GAMES:
                try:
                    game_data.append(out_queue.get(timeout=5))
                except queue.Empty:
                    if not all(a.is_alive() for a in actors):
                        raise RuntimeError("An actor process died.")
            t1 = time.perf_counter()
            temperature.value = get_temperature(batch_idx)

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
                )
                # Warm up the decay so the random init washes out quickly
                ema_decay = min(EMA_DECAY, (1 + batch_idx) / (10 + batch_idx))
                update_ema(ema_model, live_model, ema_decay)
                t2 = time.perf_counter()
                running_losses.append(avg_loss)
                rollout_times.append(t1 - t0)
                train_times.append(t2 - t1)

                if batch_idx % 250 == 0:
                    rolling_loss = np.mean(running_losses)
                    avg_rollout = np.mean(rollout_times)
                    avg_train = np.mean(train_times)
                    avg_iter = np.mean(iter_times) if iter_times else 0.0
                    explore_pct = 100.0 * n_explored / max(n_decisions, 1)

                    print(
                        f"=========================================================================\n"
                        f"b {batch_idx} | loss {rolling_loss:.4f} | "
                        f"lr {lr:.1e} | T {temperature.value:.3f} | expl {explore_pct:.1f}% | "
                        f"roll {avg_rollout:.2f}s train {avg_train:.2f}s | {avg_iter:.2f}s/it\n"
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
                    ema_model,
                    optimizer,
                    resume_path,
                    MILESTONE_SNAPSHOT_EVERY,
                )
                print(
                    f"Saved model snapshot to {snapshot_path} and updated latest_model.pt."
                )
            iter_times.append(time.perf_counter() - t_iter_start)
    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        shutdown(actors, stop_event, out_queue)


def main():
    # python -m dmc_loop.train_dmc [snapshot_dir] [games_in_flight] [elites_per_worker]
    args = sys.argv[1:]
    run_training_loop(
        args[0] if len(args) > 0 else "snapshots",
        int(args[1]) if len(args) > 1 else GAMES_IN_FLIGHT_PER_WORKER,
        int(args[2]) if len(args) > 2 else ELITES_PER_WORKER,
    )


if __name__ == "__main__":
    main()
