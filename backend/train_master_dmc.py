import glob
import multiprocessing
import os
import random
import torch
import torch.multiprocessing as mp
import numpy as np
from game import President
from playerTypes.baseline_bot import PresidentBaselineBot
from playerTypes.dmc_bot import PresidentDMCBot, PresidentValueNet
from playerTypes.master_dmc_bot import MasterDMCBot, MasterValueNet

BATCH_GAMES = 52
ROUNDS_PER_GAME = 10
SAVE_SNAPSHOT_EVERY = 250
LEARNING_RATE = 1e-4
GRADIENT_CLIP = 1.0
NUM_WORKERS = 13
INPUT_DIM_MASTER = 193
INPUT_DIM_BASIC = 115

_SNAPSHOT_CACHE = {}
_SHARED_MODEL = None


def init_worker(shared_model):
    torch.set_num_threads(1)
    global _SHARED_MODEL
    _SHARED_MODEL = shared_model


def get_cached_model(path, is_master=False):
    global _SNAPSHOT_CACHE
    if path not in _SNAPSHOT_CACHE:
        net_class = MasterValueNet if is_master else PresidentValueNet
        model = net_class().to("cpu")
        checkpoint = torch.load(path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        _SNAPSHOT_CACHE[path] = model
    return _SNAPSHOT_CACHE[path]


def parallel_worker(epsilon, basic_elites, master_elites):
    global _SHARED_MODEL, _SNAPSHOT_CACHE
    active_paths = set(basic_elites + master_elites)

    for p in active_paths:
        is_master = p in master_elites
        get_cached_model(p, is_master=is_master)

    if len(_SNAPSHOT_CACHE) > 200:
        _SNAPSHOT_CACHE = {
            k: v for k, v in _SNAPSHOT_CACHE.items() if k in active_paths
        }
    return run_single_game(
        _SHARED_MODEL, torch.device("cpu"), epsilon, basic_elites, master_elites
    )


def run_single_game(live_model, device, epsilon, basic_elites, master_elites):
    num_players = random.randint(4, 7)
    bot_instances: dict[int, PresidentDMCBot | PresidentBaselineBot] = {}
    bot_instances[0] = MasterDMCBot(0, live_model, device, True, epsilon)

    for seat in range(1, num_players):
        roll = random.random()
        if roll < 0.60 and basic_elites:
            snap_path = random.choice(basic_elites)
            model = get_cached_model(snap_path)
            bot_instances[seat] = PresidentDMCBot(seat, model, device)
        elif roll < 0.75 and basic_elites:
            snap_path = random.choice(basic_elites)
            model = get_cached_model(snap_path)
            bot_instances[seat] = PresidentDMCBot(
                seat, model, device, profile="aggressive"
            )
        elif roll < 0.90 and master_elites:
            snap_path = random.choice(master_elites)
            model = get_cached_model(snap_path, True)
            bot_instances[seat] = MasterDMCBot(seat, model, device)
        else:
            bot_instances[seat] = PresidentBaselineBot(seat)

    env = President(num_players)
    game_x, game_y = [], []

    for round_idx in range(ROUNDS_PER_GAME):
        state = env.full_reset(round_idx > 0)
        if round_idx > 0:
            cards_to_pass = {}
            for p_id, role in env.roles.items():
                if role != "Citizen":
                    cards_to_pass[p_id] = bot_instances[p_id].choose_cards_to_pass(
                        env._get_state(p_id), env
                    )

            for pair in env.role_pairs:
                env.exchange_cards(pair, cards_to_pass)
            state = env._get_state(env.curr_turn)

        move_count = 0
        while not env.game_over:
            move_count += 1
            if move_count > 500:
                for bot in bot_instances.values():
                    if isinstance(bot, PresidentDMCBot):
                        bot.trajectory.clear()
                return np.empty((0, INPUT_DIM_MASTER), dtype=np.float32), np.empty(
                    (0, 1), dtype=np.float32
                )

            curr_player = env.curr_turn
            if curr_player is None:
                break

            chosen_move = bot_instances[curr_player].get_move(state, env)
            state, _ = env.step(curr_player, chosen_move)

            if env.was_pile_reset:
                env.clear_pile()
        env.assign_roles()

        max_possible_score = env.players - 1
        gamma = 1.0

        for rank, p_id in enumerate(env.out_order):
            bot = bot_instances[p_id]
            if (
                isinstance(bot, MasterDMCBot)
                and bot.training
                and bot.model == live_model
            ):
                if len(bot.trajectory) > 0:
                    round_score = max_possible_score - rank
                    normalized_score = (round_score / max_possible_score) * 2 - 1

                    for i, features in enumerate(reversed(bot.trajectory)):
                        discounted_score = normalized_score * (gamma**i)
                        game_x.append(features)
                        game_y.append([discounted_score])
                bot.trajectory.clear()
    return np.array(game_x, dtype=np.float32), np.array(game_y, dtype=np.float32)


def run_training_loop(
    snapshot_dir="snapshots_master_dmc", basic_snapshot_dir="snapshots", decay=0.99998
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
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, decay)

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

    with ctx.Pool(
        processes=NUM_WORKERS, initializer=init_worker, initargs=(shared_model,)
    ) as pool:
        try:
            while max_batches < 2000:
                batch_idx += 1
                max_batches += 1
                epsilon = max(0.02, epsilon * 0.9997)
                basic_elites = glob.glob(
                    os.path.join(basic_snapshot_dir, "model_gen_*.pt")
                ) + glob.glob(os.path.join(basic_snapshot_dir, "elites/model_gen_*.pt"))
                master_elites = glob.glob(
                    os.path.join(snapshot_dir, "elites/model_gen_*.pt")
                )
                tasks = [
                    (epsilon, basic_elites, master_elites) for _ in range(BATCH_GAMES)
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
                        perm = torch.randperm(dataset_size)
                        for i in range(0, dataset_size, mini_batch_size):
                            indices = perm[i : i + mini_batch_size]
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
                    snapshot_path = os.path.join(
                        snapshot_dir, f"model_gen_{batch_idx}.pt"
                    )
                    checkpoint = {
                        "batch_idx": batch_idx,
                        "model_state_dict": live_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epsilon": epsilon,
                    }
                    torch.save(checkpoint, snapshot_path)
                    torch.save(checkpoint, resume_path)
                    print(f"Saved master model snapshot to {snapshot_path}.")
        except KeyboardInterrupt:
            print("Training interrupted.")
            pool.terminate()
            active_workers = multiprocessing.active_children()
            for worker in active_workers:
                worker.kill()
            for worker in active_workers:
                worker.join(timeout=0.1)
            os._exit(1)


def main():
    run_training_loop()


if __name__ == "__main__":
    main()
