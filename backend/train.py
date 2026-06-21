import torch
import glob
import random
import os
import concurrent.futures
import numpy as np
import multiprocessing
from playerTypes.baseline_bot import PresidentenBaselineBot
from playerTypes.dmc_bot import PresidentenDMCBot, PresidentenValueNet
from game import Presidenten

BATCH_GAMES = 50
ROUNDS_PER_GAME = 10
SAVE_SNAPSHOT_EVERY = 250
LEARNING_RATE = 1e-4
INPUT_DIM = 131
GRADIENT_CLIP = 1.0
NUM_WORKERS = 10

_GLOBAL_LIVE_MODEL = None
_SNAPSHOT_CACHE = {}


def init_worker():
    torch.set_num_threads(1)
    global _GLOBAL_LIVE_MODEL
    _GLOBAL_LIVE_MODEL = PresidentenValueNet(INPUT_DIM).to("cpu")
    _GLOBAL_LIVE_MODEL.eval()


def parallel_worker(model_state, epsilon, youngest_paths=None, oldest_paths=None):
    global _GLOBAL_LIVE_MODEL, _SNAPSHOT_CACHE
    if _GLOBAL_LIVE_MODEL is not None:
        _GLOBAL_LIVE_MODEL.load_state_dict(model_state)

    paths = (youngest_paths or []) + (oldest_paths or [])
    if paths:
        for path in paths:
            if path not in _SNAPSHOT_CACHE:
                snap_model = PresidentenValueNet(INPUT_DIM).to("cpu")
                checkpoint = torch.load(path, map_location="cpu")
                snap_model.load_state_dict(checkpoint["model_state_dict"])
                snap_model.eval()
                _SNAPSHOT_CACHE[path] = snap_model
    if len(_SNAPSHOT_CACHE) > 200:
        active_paths = set(paths)
        _SNAPSHOT_CACHE = {
            k: v for k, v in _SNAPSHOT_CACHE.items() if k in active_paths
        }
    return run_single_game(
        _GLOBAL_LIVE_MODEL,
        torch.device("cpu"),
        epsilon,
        youngest_paths,
        oldest_paths,
    )


def run_single_game(
    live_model, device, epsilon, youngest_paths=None, oldest_paths=None
):
    global _SNAPSHOT_CACHE
    num_players = random.randint(4, 7)
    bot_instances: dict[int, PresidentenDMCBot | PresidentenBaselineBot] = {}
    has_snapshots = bool(youngest_paths or oldest_paths)
    use_snapshot = has_snapshots and random.random() < 0.5
    snapshot_seats = (
        random.sample(range(num_players), k=num_players // 2) if use_snapshot else []
    )

    for seat in range(num_players):
        if seat in snapshot_seats:
            roll = random.random()
            if roll < 0.8 and youngest_paths:
                snap_path = random.choice(youngest_paths)
                snap_model = _SNAPSHOT_CACHE[snap_path]
                bot_instances[seat] = PresidentenDMCBot(
                    player_id=seat, model=snap_model, device=device, training=False
                )
            elif roll < 0.95 and oldest_paths:
                snap_path = random.choice(oldest_paths)
                snap_model = _SNAPSHOT_CACHE[snap_path]
                bot_instances[seat] = PresidentenDMCBot(
                    player_id=seat, model=snap_model, device=device, training=False
                )
            else:
                bot_instances[seat] = PresidentenBaselineBot(player_id=seat)
        else:
            bot_instances[seat] = PresidentenDMCBot(
                player_id=seat,
                model=live_model,
                device=device,
                training=True,
                epsilon=epsilon,
            )

    env = Presidenten(num_players)
    game_x, game_y = [], []

    for round_idx in range(ROUNDS_PER_GAME):
        state = env.full_reset(next_round=(round_idx > 0))
        if round_idx > 0:
            cards_to_pass = {}
            for p_id, role in env.roles.items():
                if role != "Citizen":
                    cards_to_pass[p_id] = bot_instances[p_id].choose_cards_to_pass(
                        env._get_state(p_id)
                    )

            env.exchange_cards(cards_to_pass)
            state = env._get_state(env.curr_turn)

        move_count = 0
        while not env.game_over:
            move_count += 1
            if move_count > 500:
                for bot in bot_instances.values():
                    if isinstance(bot, PresidentenDMCBot):
                        bot.trajectory.clear()
                return np.empty((0, INPUT_DIM), dtype=np.float32), np.empty(
                    (0, 1), dtype=np.float32
                )

            curr_player = env.curr_turn
            if curr_player is None:
                break

            chosen_move = bot_instances[curr_player].get_move(state, env)
            state, _ = env.step(curr_player, chosen_move)
        env.assign_roles()

        max_possible_score = env.players - 1
        for rank, p_id in enumerate(env.out_order):
            bot = bot_instances[p_id]
            if isinstance(bot, PresidentenDMCBot):
                if bot.training and len(bot.trajectory) > 0:
                    round_score = env.players - 1 - rank
                    normalized_score = (round_score / max_possible_score) * 2 - 1

                    for features in bot.trajectory:
                        game_x.append(features)
                        game_y.append([normalized_score])
                    bot.trajectory.clear()
    return np.array(game_x, dtype=np.float32), np.array(game_y, dtype=np.float32)


def main():
    os.makedirs("snapshots", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    resume_path = "snapshots/latest_model.pt"
    live_model = PresidentenValueNet(INPUT_DIM).to(device)
    optimizer = torch.optim.Adam(live_model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20000)

    if os.path.exists(resume_path):
        print(f"Resuming checkpoint from {resume_path}...")
        checkpoint = torch.load(resume_path, map_location=device)
        live_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        batch_idx = checkpoint["batch_idx"]
        epsilon = checkpoint["epsilon"]
    else:
        batch_idx = 0
        epsilon = 0.2
    print("Starting training loop. Press Ctrl+C to stop and save the model.")

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=NUM_WORKERS, initializer=init_worker
    ) as executor:
        try:
            while True:
                batch_idx += 1
                epsilon = max(0.05, epsilon * 0.9997)
                all_x, all_y = [], []
                current_weights = live_model.state_dict()
                snapshot_files = glob.glob("snapshots/model_gen_*.pt")
                youngest_files, oldest_files = None, None

                if snapshot_files:
                    snapshot_files.sort(
                        key=lambda x: int(
                            os.path.basename(x).split("_")[2].split(".")[0]
                        )
                    )
                    youngest_files = snapshot_files[-10:]
                    oldest_files = snapshot_files[:10]

                futures = [
                    executor.submit(
                        parallel_worker,
                        current_weights,
                        epsilon,
                        youngest_files,
                        oldest_files,
                    )
                    for _ in range(BATCH_GAMES)
                ]
                try:
                    for f in concurrent.futures.as_completed(futures, timeout=90.0):
                        game_x, game_y = f.result()
                        all_x.append(game_x)
                        all_y.append(game_y)
                except concurrent.futures.TimeoutError:
                    print(f"\nBatch {batch_idx} timed out.")
                    for f in futures:
                        f.cancel()
                    continue

                if all_x:
                    merged_x = np.concatenate(all_x, axis=0)
                    merged_y = np.concatenate(all_y, axis=0)

                    x_tensor = torch.from_numpy(merged_x).to(device)
                    y_tensor = torch.from_numpy(merged_y).to(device)

                    dataset_size = x_tensor.size(0)
                    mini_batch_size = 512
                    epochs = 4

                    live_model.train()
                    epoch_losses = []

                    for epoch in range(epochs):
                        permutation = torch.randperm(dataset_size)
                        for i in range(0, dataset_size, mini_batch_size):
                            indices = permutation[i : i + mini_batch_size]
                            batch_x, batch_y = x_tensor[indices], y_tensor[indices]

                            optimizer.zero_grad()
                            predictions = live_model(batch_x)
                            loss = loss_fn(predictions, batch_y)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                live_model.parameters(), GRADIENT_CLIP
                            )
                            optimizer.step()
                            epoch_losses.append(loss.item())

                    avg_loss = np.mean(epoch_losses)
                    scheduler.step()
                    print(
                        f"Batch {batch_idx}: Avg Loss = {avg_loss:.6f} | LR = {scheduler.get_last_lr()[0]:.6f} | "
                        f"Epsilon = {epsilon:.4f} | Total Move Rows = {dataset_size}"
                    )
                else:
                    print(f"Batch {batch_idx}: No training data collected.")

                if batch_idx % SAVE_SNAPSHOT_EVERY == 0:
                    snapshot_path = f"snapshots/model_gen_{batch_idx}.pt"
                    checkpoint = {
                        "batch_idx": batch_idx,
                        "model_state_dict": live_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epsilon": epsilon,
                    }
                    torch.save(checkpoint, snapshot_path)
                    torch.save(checkpoint, resume_path)
                    print(
                        f"Saved model snapshot to {snapshot_path} and updated latest_model.pt."
                    )
        except KeyboardInterrupt:
            print("Training interrupted.")
            executor.shutdown(wait=False, cancel_futures=True)
            active_workers = multiprocessing.active_children()
            for worker in active_workers:
                worker.kill()
            for worker in active_workers:
                worker.join(timeout=0.1)
            os._exit(0)


if __name__ == "__main__":
    main()
