import glob
import os
import random
import torch
import torch.multiprocessing as mp
import numpy as np
from playerTypes.player import Player
from playerTypes.baseline_bot import PresidentBaselineBot
from playerTypes.dmc_bot import (
    PresidentDMCBot,
    PresidentValueNet,
    MasterDMCBot,
    MasterValueNet,
    MASTER_INPUT_DIM,
)
from utils import (
    init_worker,
    get_cached_model,
    prune_cache,
    game_loop,
    train_on_batch,
    save_snapshot,
    handle_keyboard_interrupt,
)

BATCH_GAMES = 52
SAVE_SNAPSHOT_EVERY = 250
LEARNING_RATE = 1e-4
NUM_WORKERS = 13


def parallel_worker(shared_model, epsilon, basic_elites, master_elites):
    active_paths = set(basic_elites + master_elites)

    for p in active_paths:
        net_cls = MasterValueNet if p in master_elites else PresidentValueNet
        get_cached_model(p, net_cls)

    prune_cache(active_paths)
    return run_single_game(
        shared_model, torch.device("cpu"), epsilon, basic_elites, master_elites
    )


def run_single_game(live_model, device, epsilon, basic_elites, master_elites):
    num_players = random.randint(4, 7)
    bot_instances: dict[int, Player] = {
        0: MasterDMCBot(0, live_model, device, True, epsilon)
    }

    for seat in range(1, num_players):
        roll = random.random()
        if roll < 0.60 and basic_elites:
            snap_path = random.choice(basic_elites)
            model = get_cached_model(snap_path, PresidentValueNet)
            bot_instances[seat] = PresidentDMCBot(seat, model, device)
        elif roll < 0.75 and basic_elites:
            snap_path = random.choice(basic_elites)
            model = get_cached_model(snap_path, PresidentValueNet)
            bot_instances[seat] = PresidentDMCBot(
                seat, model, device, profile="aggressive"
            )
        elif roll < 0.85:
            bot_instances[seat] = MasterDMCBot(seat, live_model, device, True, epsilon)
        elif roll < 0.90 and master_elites:
            snap_path = random.choice(master_elites)
            model = get_cached_model(snap_path, MasterValueNet)
            bot_instances[seat] = MasterDMCBot(seat, model, device)
        else:
            bot_instances[seat] = PresidentBaselineBot(seat)
    return game_loop(
        num_players, bot_instances, live_model, MASTER_INPUT_DIM, MasterDMCBot
    )


def run_training_loop(
    snapshot_dir="snapshots_master_dmc", basic_snapshot_dir="snapshots"
):
    os.makedirs(snapshot_dir, exist_ok=True)
    ctx = mp.get_context("spawn")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    resume_path = os.path.join(snapshot_dir, "latest_model.pt")
    live_model = MasterValueNet().to(device)
    shared_model = MasterValueNet().to("cpu")
    shared_model.share_memory()
    shared_model.load_state_dict(live_model.state_dict())
    optimizer = torch.optim.Adam(live_model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99998)

    if os.path.exists(resume_path):
        print(f"Resuming checkpoint from {resume_path}...")
        checkpoint = torch.load(resume_path, map_location=device)
        live_model.load_state_dict(checkpoint["model_state_dict"])
        shared_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        batch_idx = checkpoint["batch_idx"]
        epsilon = checkpoint["epsilon"]
    else:
        batch_idx = 0
        epsilon = 0.2

    print(f"Starting Master DMC training loop.")
    max_batches = 0 if batch_idx == 0 else batch_idx % 2000
    running_losses = []
    basic_elites = glob.glob(
        os.path.join(basic_snapshot_dir, "model_gen_*.pt")
    ) + glob.glob(os.path.join(basic_snapshot_dir, "elites/model_gen_*.pt"))
    master_elites = glob.glob(os.path.join(snapshot_dir, "elites/model_gen_*.pt"))

    with ctx.Pool(processes=NUM_WORKERS, initializer=init_worker) as pool:
        try:
            while max_batches < 2000:
                batch_idx += 1
                max_batches += 1
                epsilon = max(0.02, epsilon * 0.9997)
                tasks = [
                    (shared_model, epsilon, basic_elites, master_elites)
                    for _ in range(BATCH_GAMES)
                ]

                try:
                    results = pool.starmap_async(parallel_worker, tasks)
                    game_data = results.get(timeout=600)
                    all_x = [g[0] for g in game_data if len(g[0]) > 0]
                    all_y = [g[1] for g in game_data if len(g[1]) > 0]
                except mp.TimeoutError:
                    print(f"\nBatch {batch_idx} timed out.")
                    continue

                if all_x:
                    avg_loss = train_on_batch(
                        all_x,
                        all_y,
                        device,
                        live_model,
                        optimizer,
                        loss_fn,
                        shared_model,
                        scheduler,
                    )
                    running_losses.append(avg_loss)

                    if batch_idx % 25 == 0:
                        rolling_loss = np.mean(running_losses)
                        print(
                            f"=========================================================================\n"
                            f"MASTER BATCH {batch_idx:<5} | Loss: {rolling_loss:.6f} | "
                            f"LR: {scheduler.get_last_lr()[0]:.6f} | Epsilon: {epsilon:.4f}\n"
                            f"========================================================================="
                        )
                        running_losses.clear()
                if batch_idx % SAVE_SNAPSHOT_EVERY == 0:
                    snapshot_path = save_snapshot(
                        snapshot_dir,
                        batch_idx,
                        live_model,
                        optimizer,
                        scheduler,
                        epsilon,
                        resume_path,
                    )
                    print(f"Saved master model snapshot to {snapshot_path}.")
        except KeyboardInterrupt:
            handle_keyboard_interrupt(pool)


def main():
    run_training_loop()


if __name__ == "__main__":
    main()
