import torch
import glob
import random
import os
import concurrent.futures
import numpy as np
from playerTypes.dmc_bot import PresidentenDMCBot, PresidentenValueNet
from game import Presidenten

BATCH_GAMES = 48
ROUNDS_PER_GAME = 10
SAVE_SNAPSHOT_EVERY = 500
LEARNING_RATE = 1e-4
INPUT_DIM = 91
GRADIENT_CLIP = 1.0
NUM_WORKERS = 12
_GLOBAL_LIVE_MODEL = None
_SNAPSHOT_CACHE = {}


def init_worker():
    torch.set_num_threads(1)
    global _GLOBAL_LIVE_MODEL
    _GLOBAL_LIVE_MODEL = PresidentenValueNet(INPUT_DIM).to("cpu")
    _GLOBAL_LIVE_MODEL.eval()


def parallel_worker(model_state, epsilon, snapshot_paths=None):
    global _GLOBAL_LIVE_MODEL, _SNAPSHOT_CACHE
    if _GLOBAL_LIVE_MODEL is not None:
        _GLOBAL_LIVE_MODEL.load_state_dict(model_state)

    local_league_models = []
    if snapshot_paths:
        for path in snapshot_paths:
            if path not in _SNAPSHOT_CACHE:
                snap_model = PresidentenValueNet(INPUT_DIM).to("cpu")
                checkpoint = torch.load(path, map_location="cpu")

                if "model_state_dict" in checkpoint:
                    snap_model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    snap_model.load_state_dict(checkpoint)

                snap_model.eval()
                _SNAPSHOT_CACHE[path] = snap_model
            local_league_models.append(_SNAPSHOT_CACHE[path])
    if snapshot_paths and len(_SNAPSHOT_CACHE) > 200:
        active_paths = set(snapshot_paths)
        _SNAPSHOT_CACHE = {
            k: v for k, v in _SNAPSHOT_CACHE.items() if k in active_paths
        }
    return run_single_game(
        _GLOBAL_LIVE_MODEL,
        torch.device("cpu"),
        epsilon,
        local_league_models if local_league_models else None,
    )


def run_single_game(live_model, device, epsilon, league_models=None):
    bot_instances: dict[int, PresidentenDMCBot] = {}
    use_snapshot = league_models is not None and random.random() < 0.5
    snapshot_seats = random.sample(range(4), k=2) if use_snapshot else []

    for seat in range(4):
        if seat in snapshot_seats and league_models:
            snap_model = random.choice(league_models)
            bot_instances[seat] = PresidentenDMCBot(
                player_id=seat, model=snap_model, device=device, training=False
            )
        else:
            bot_instances[seat] = PresidentenDMCBot(
                player_id=seat,
                model=live_model,
                device=device,
                training=True,
                epsilon=epsilon,
            )
            bot_instances[seat].trajectory = []

    env = Presidenten(players=4)
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

        while not env.game_over:
            curr_player = env.curr_turn
            if curr_player is None:
                break

            chosen_move = bot_instances[curr_player].get_move(state, env)
            state, _ = env.step(curr_player, chosen_move)
        env.assign_roles()

        max_possible_score = env.players - 1
        for rank, p_id in enumerate(env.out_order):
            bot = bot_instances[p_id]
            if bot.training and len(bot.trajectory) > 0:
                round_score = env.players - 1 - rank
                normalized_score = (round_score / max_possible_score) * 2 - 1

                for features in bot.trajectory:
                    game_x.append(features)
                    game_y.append([normalized_score])
                bot.trajectory.clear()
    return game_x, game_y


def main():
    os.makedirs("snapshots", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    resume_path = "snapshots/latest_model.pt"
    live_model = PresidentenValueNet(INPUT_DIM).to(device)
    optimizer = torch.optim.Adam(live_model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()

    if os.path.exists(resume_path):
        print(f"Resuming checkpoint from {resume_path}...")
        checkpoint = torch.load(resume_path, map_location=device)
        live_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
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
                selected_files = None

                if snapshot_files:
                    selected_files = random.sample(
                        snapshot_files, k=min(3, len(snapshot_files))
                    )

                futures = [
                    executor.submit(
                        parallel_worker, current_weights, epsilon, selected_files
                    )
                    for _ in range(BATCH_GAMES)
                ]
                for f in concurrent.futures.as_completed(futures):
                    game_x, game_y = f.result()
                    all_x.extend(game_x)
                    all_y.extend(game_y)

                if all_x:
                    x_tensor = torch.FloatTensor(np.array(all_x)).to(device)
                    y_tensor = torch.FloatTensor(np.array(all_y)).to(device)

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
                    print(
                        f"Batch {batch_idx}: Avg Loss = {avg_loss:.6f} | Epsilon = {epsilon:.4f} | Total Move Rows = {dataset_size}"
                    )
                else:
                    print(f"Batch {batch_idx}: No training data collected.")

                if batch_idx % SAVE_SNAPSHOT_EVERY == 0:
                    snapshot_path = f"snapshots/model_gen_{batch_idx}.pt"
                    checkpoint = {
                        "batch_idx": batch_idx,
                        "model_state_dict": live_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epsilon": epsilon,
                    }
                    torch.save(checkpoint, snapshot_path)
                    torch.save(checkpoint, resume_path)
                    print(
                        f"Saved model snapshot to {snapshot_path} and updated latest_model.pt."
                    )
        except KeyboardInterrupt:
            print("Training interrupted.")


if __name__ == "__main__":
    main()
