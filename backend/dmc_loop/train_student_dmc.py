import glob
import os
import random
import torch
import torch.multiprocessing as mp
import numpy as np
from game import President
from playerTypes.player import Player
from playerTypes.baseline_bot import PresidentBaselineBot
from playerTypes.dmc_bot import (
    PresidentDMCBot,
    PresidentValueNet,
    StudentDMCBot,
    MasterValueNet,
    INPUT_DIM,
)
from utils import (
    init_worker,
    get_cached_model,
    prune_cache,
    play_round,
    handle_keyboard_interrupt,
    NUM_ROUNDS,
)

BATCH_GAMES = 52
SAVE_SNAPSHOT_EVERY = 250
LEARNING_RATE = 1e-4
NUM_WORKERS = 13


def parallel_worker(shared_model, epsilon, basic_elites, student_elites):
    active_paths = set(basic_elites + student_elites)

    for p in active_paths:
        get_cached_model(p, PresidentValueNet)

    prune_cache(active_paths)
    return run_single_game(
        shared_model, torch.device("cpu"), epsilon, basic_elites, student_elites
    )


def run_single_game(live_model, device, epsilon, basic_elites, student_elites):
    num_players = random.randint(4, 7)
    master_model = get_cached_model("PLACEHOLDER", MasterValueNet)
    bot_instances: dict[int, Player] = {
        0: StudentDMCBot(
            0, live_model, device, True, epsilon, master_model=master_model
        )
    }

    for seat in range(1, num_players):
        roll = random.random()
        if roll < 0.55:
            bot_instances[seat] = StudentDMCBot(
                seat, live_model, device, True, epsilon, master_model=master_model
            )
        elif roll < 0.68 and student_elites:
            snap_path = random.choice(student_elites)
            model = get_cached_model(snap_path, PresidentValueNet)
            bot_instances[seat] = StudentDMCBot(seat, model, device)
        elif roll < 0.75 and student_elites:
            snap_path = random.choice(student_elites)
            model = get_cached_model(snap_path, PresidentValueNet)
            bot_instances[seat] = StudentDMCBot(
                seat, model, device, profile="aggressive"
            )
        elif roll < 0.85 and basic_elites:
            snap_path = random.choice(basic_elites)
            model = get_cached_model(snap_path, PresidentValueNet)
            bot_instances[seat] = PresidentDMCBot(seat, model, device)
        elif roll < 0.90 and basic_elites:
            snap_path = random.choice(basic_elites)
            model = get_cached_model(snap_path, PresidentValueNet)
            bot_instances[seat] = PresidentDMCBot(
                seat, model, device, profile="aggressive"
            )
        else:
            bot_instances[seat] = PresidentBaselineBot(seat)

    env = President(num_players)
    game_x, game_y, game_master_y = [], [], []

    for round_idx in range(NUM_ROUNDS):
        if not play_round(env, bot_instances, round_idx):
            for bot in bot_instances.values():
                if isinstance(bot, StudentDMCBot):
                    bot.trajectory.clear()
                    bot.master_trajectory.clear()
            return (
                np.empty((0, INPUT_DIM), dtype=np.float32),
                np.empty((0, 1), dtype=np.float32),
                np.empty((0, 1), dtype=np.float32),
            )

        max_possible_score = env.players - 1
        gamma = 1.0

        for rank, p_id in enumerate(env.out_order):
            bot = bot_instances[p_id]
            if isinstance(bot, StudentDMCBot):
                if bot.training and len(bot.trajectory) > 0 and bot.model == live_model:
                    round_score = max_possible_score - rank
                    normalized_score = (round_score / max_possible_score) * 2 - 1

                    for i, (features, master_val) in enumerate(
                        zip(reversed(bot.trajectory), reversed(bot.master_trajectory))
                    ):
                        game_x.append(features)
                        game_y.append([normalized_score * (gamma**i)])
                        game_master_y.append([master_val])

                bot.trajectory.clear()
                bot.master_trajectory.clear()
    return (
        np.array(game_x, dtype=np.float32),
        np.array(game_y, dtype=np.float32),
        np.array(game_master_y, dtype=np.float32),
    )


def run_training_loop(
    snapshot_dir="snapshots_student_dmc", basic_snapshot_dir="snapshots"
):
    os.makedirs(snapshot_dir, exist_ok=True)
    ctx = mp.get_context("spawn")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    resume_path = os.path.join(snapshot_dir, "latest_model.pt")
    live_model = PresidentValueNet().to(device)
    shared_model = PresidentValueNet().to("cpu")
    shared_model.share_memory()
    shared_model.load_state_dict(live_model.state_dict())
    optimizer = torch.optim.Adam(live_model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99995)

    if os.path.exists(resume_path):
        print(f"Resuming checkpoint from {resume_path}...")
        checkpoint = torch.load(resume_path, map_location=device)
        live_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        batch_idx = checkpoint["batch_idx"]
        epsilon = checkpoint["epsilon"]
        alpha = checkpoint["alpha"]
    else:
        batch_idx = 0
        epsilon = 0.2
        alpha = 0.3

    print(f"Starting Student DMC training loop.")
    max_batches = 0 if batch_idx == 0 else batch_idx % 2000
    running_losses = []
    basic_elites = glob.glob(
        os.path.join(basic_snapshot_dir, "model_gen_*.pt")
    ) + glob.glob(os.path.join(basic_snapshot_dir, "elites/model_gen_*.pt"))
    student_elites = glob.glob(os.path.join(snapshot_dir, "elites/model_gen_*.pt"))

    with ctx.Pool(processes=NUM_WORKERS, initializer=init_worker) as pool:
        try:
            while max_batches < 2000:
                batch_idx += 1
                max_batches += 1
                epsilon = max(0.02, epsilon * 0.9997)
                alpha = min(0.9, alpha / 0.99995)
                tasks = [
                    (shared_model, epsilon, basic_elites, student_elites)
                    for _ in range(BATCH_GAMES)
                ]

                try:
                    results = pool.starmap_async(parallel_worker, tasks)
                    game_data = results.get(timeout=600)
                    all_x = [g[0] for g in game_data if len(g[0]) > 0]
                    all_y = [g[1] for g in game_data if len(g[1]) > 0]
                    all_master_y = [g[2] for g in game_data if len(g[2]) > 0]
                except mp.TimeoutError:
                    print(f"\nBatch {batch_idx} timed out.")
                    continue

                if all_x:
                    merged_x = np.concatenate(all_x, axis=0)
                    merged_y = np.concatenate(all_y, axis=0)
                    merged_master_y = np.concatenate(all_master_y, axis=0)

                    x_tensor = torch.from_numpy(merged_x).to(device)
                    y_tensor = torch.from_numpy(merged_y).to(device)
                    master_y_tensor = torch.from_numpy(merged_master_y).to(device)

                    dataset_size = x_tensor.size(0)
                    mini_batch_size = 512
                    epochs = 4

                    live_model.train()
                    epoch_losses = []

                    for epoch in range(epochs):
                        perm = torch.randperm(dataset_size)
                        for i in range(0, dataset_size, mini_batch_size):
                            indices = perm[i : i + mini_batch_size]
                            batch_x, batch_y, batch_master_y = (
                                x_tensor[indices],
                                y_tensor[indices],
                                master_y_tensor[indices],
                            )

                            optimizer.zero_grad()
                            predictions = live_model(batch_x)
                            outcome_loss = loss_fn(predictions, batch_y)
                            distill_loss = loss_fn(predictions, batch_master_y)
                            loss = alpha * outcome_loss + (1 - alpha) * distill_loss
                            loss.backward()
                            optimizer.step()
                            epoch_losses.append(loss.item())

                    shared_model.load_state_dict(live_model.state_dict())
                    scheduler.step()
                    avg_loss = np.mean(epoch_losses)
                    running_losses.append(avg_loss)

                    if batch_idx % 25 == 0:
                        rolling_loss = np.mean(running_losses)
                        print(
                            f"=========================================================================\n"
                            f"STUDENT Batch {batch_idx:<5} | Loss: {rolling_loss:.6f} | "
                            f"Epsilon: {epsilon:.4f} | Alpha: {alpha:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}\n"
                            f"========================================================================="
                        )
                        running_losses.clear()
                if batch_idx % SAVE_SNAPSHOT_EVERY == 0:
                    snapshot_path = os.path.join(
                        snapshot_dir, f"model_gen_{batch_idx}.pt"
                    )
                    checkpoint = {
                        "batch_idx": batch_idx,
                        "model_state_dict": live_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epsilon": epsilon,
                        "alpha": alpha,
                    }
                    torch.save(checkpoint, snapshot_path)
                    torch.save(checkpoint, resume_path)
                    print(f"Saved student model snapshot to {snapshot_path}.")
        except KeyboardInterrupt:
            handle_keyboard_interrupt(pool)


def main():
    run_training_loop()


if __name__ == "__main__":
    main()
