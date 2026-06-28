import torch
import glob
import random
import os
import numpy as np
import multiprocessing
import torch.multiprocessing as mp
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

_SNAPSHOT_CACHE = {}
_SHARED_MODEL = None


def init_worker(shared_model):
    torch.set_num_threads(1)
    global _SHARED_MODEL
    _SHARED_MODEL = shared_model


def parallel_worker(epsilon, elite_snapshots=None):
    global _SHARED_MODEL, _SNAPSHOT_CACHE

    paths = elite_snapshots or []
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
    return run_single_game(_SHARED_MODEL, torch.device("cpu"), epsilon, elite_snapshots)


def run_single_game(live_model, device, epsilon, elite_snapshots=None):
    global _SNAPSHOT_CACHE
    num_players = random.randint(4, 7)
    bot_instances: dict[int, PresidentenDMCBot | PresidentenBaselineBot] = {}
    bot_instances[0] = PresidentenDMCBot(
        0, live_model, device, training=True, epsilon=epsilon
    )

    for seat in range(1, num_players):
        roll = random.random()
        if roll < 0.65:
            bot_instances[seat] = PresidentenDMCBot(
                player_id=seat,
                model=live_model,
                device=device,
                training=True,
                epsilon=epsilon,
            )
        elif roll < 0.80 and elite_snapshots:
            snap_path = random.choice(elite_snapshots)
            snap_model = _SNAPSHOT_CACHE[snap_path]
            bot_instances[seat] = PresidentenDMCBot(
                player_id=seat, model=snap_model, device=device, training=False
            )
        elif roll < 0.90 and elite_snapshots:
            snap_path = random.choice(elite_snapshots)
            snap_model = _SNAPSHOT_CACHE[snap_path]
            bot_instances[seat] = PresidentenDMCBot(
                player_id=seat,
                model=snap_model,
                device=device,
                training=True,
                epsilon=0.35,
            )
        else:
            bot_instances[seat] = PresidentenBaselineBot(player_id=seat)

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
        gamma = 0.95

        for rank, p_id in enumerate(env.out_order):
            bot = bot_instances[p_id]
            if isinstance(bot, PresidentenDMCBot):
                if bot.training and len(bot.trajectory) > 0 and bot.model == live_model:
                    round_score = env.players - 1 - rank
                    normalized_score = (round_score / max_possible_score) * 2 - 1

                    for i, features in enumerate(reversed(bot.trajectory)):
                        discounted_score = normalized_score * (gamma**i)
                        game_x.append(features)
                        game_y.append([discounted_score])
                bot.trajectory.clear()
    return np.array(game_x, dtype=np.float32), np.array(game_y, dtype=np.float32)


def main():
    os.makedirs("snapshots", exist_ok=True)

    ctx = mp.get_context("spawn")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    resume_path = "snapshots/latest_model.pt"
    live_model = PresidentenValueNet(INPUT_DIM).to(device)
    shared_model = PresidentenValueNet(INPUT_DIM).to("cpu")
    shared_model.share_memory()
    shared_model.load_state_dict(live_model.state_dict())

    optimizer = torch.optim.Adam(live_model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99995)

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
    max_batches = 0 if batch_idx == 0 else batch_idx % 2000
    running_losses = []
    log_interval = 20

    with ctx.Pool(
        processes=NUM_WORKERS, initializer=init_worker, initargs=(shared_model,)
    ) as pool:
        try:
            while max_batches < 2000:
                batch_idx += 1
                max_batches += 1
                epsilon = max(0.05, epsilon * 0.9997)
                all_x, all_y = [], []
                elite_paths = glob.glob("snapshots/elites/model_gen_*.pt")
                graduated_paths = glob.glob("snapshots/graduated/model_gen_*.pt")
                elite_snapshots = elite_paths + graduated_paths
                tasks = [
                    (
                        epsilon,
                        elite_snapshots,
                    )
                    for _ in range(BATCH_GAMES)
                ]

                try:
                    results = pool.starmap_async(parallel_worker, tasks)
                    game_data = results.get(timeout=600)

                    for game_x, game_y in game_data:
                        all_x.append(game_x)
                        all_y.append(game_y)
                except mp.TimeoutError:
                    print(f"\nBatch {batch_idx} timed out.")
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
                    shared_model.load_state_dict(live_model.state_dict())
                    scheduler.step()

                    running_losses.append(avg_loss)

                    if batch_idx % log_interval == 0:
                        rolling_loss = np.mean(running_losses)
                        print(
                            f"=========================================================================\n"
                            f"BATCH {batch_idx:<5} | Rolling Loss: {rolling_loss:.6f} | "
                            f"LR: {scheduler.get_last_lr()[0]:.6f} | Epsilon: {epsilon:.4f}\n"
                            f"========================================================================="
                        )
                        running_losses.clear()
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
            pool.terminate()
            active_workers = multiprocessing.active_children()
            for worker in active_workers:
                worker.kill()
            for worker in active_workers:
                worker.join(timeout=0.1)
            os._exit(1)


if __name__ == "__main__":
    main()
