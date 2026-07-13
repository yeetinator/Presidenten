import torch
import os
import multiprocessing
import random
import concurrent.futures
import numpy as np
from game import President
from playerTypes.dmc_bot import PresidentDMCBot, INPUT_DIM

NUM_ROUNDS = 10
GRADIENT_CLIP = 1.0
NUM_WORKERS = 12
K_FACTOR = 16.0

_SNAPSHOT_CACHE = {}


def init_worker():
    torch.set_num_threads(1)


def get_cached_model(path, model_cls):
    global _SNAPSHOT_CACHE

    if path not in _SNAPSHOT_CACHE:
        model = model_cls().to("cpu")
        checkpoint = torch.load(path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        _SNAPSHOT_CACHE[path] = model
    return _SNAPSHOT_CACHE[path]


def prune_cache(active_paths, max_size=200):
    global _SNAPSHOT_CACHE
    if len(_SNAPSHOT_CACHE) > max_size:
        _SNAPSHOT_CACHE = {
            k: v for k, v in _SNAPSHOT_CACHE.items() if k in active_paths
        }


def game_loop(
    num_players,
    bot_instances,
    live_model,
    input_dim=INPUT_DIM,
    bot_class=PresidentDMCBot,
):
    env = President(num_players)
    game_x, game_y = [], []

    for round_idx in range(NUM_ROUNDS):
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
                return np.empty((0, input_dim), dtype=np.float32), np.empty(
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
            if isinstance(bot, bot_class):
                if bot.training and len(bot.trajectory) > 0 and bot.model == live_model:
                    round_score = max_possible_score - rank
                    normalized_score = (round_score / max_possible_score) * 2 - 1

                    for i, features in enumerate(reversed(bot.trajectory)):
                        discounted_score = normalized_score * (gamma**i)
                        game_x.append(features)
                        game_y.append([discounted_score])
                bot.trajectory.clear()
    return np.array(game_x, dtype=np.float32), np.array(game_y, dtype=np.float32)


def eval_game_loop(
    num_players,
    match_seed,
    bot_instances,
    seat_assignments,
    rotation,
    slot_norm_scores,
):
    env = President(num_players)
    for round_idx in range(NUM_ROUNDS):
        round_seed = (match_seed * 1000 + round_idx) % (2**32 - 1)
        random.seed(round_seed)
        np.random.seed(round_seed)

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
                break

            curr_player = env.curr_turn
            if curr_player is None:
                break

            chosen_move = bot_instances[curr_player].get_move(state, env)
            state, _ = env.step(curr_player, chosen_move)

            if env.was_pile_reset:
                env.clear_pile()
        env.assign_roles()

    max_pos_score = (num_players - 1) * NUM_ROUNDS
    for seat, key in enumerate(seat_assignments):
        raw_score = env.scores[seat][0]
        norm_score = (raw_score / max_pos_score) * 2 - 1
        original_slot = (seat + rotation) % num_players
        slot_norm_scores[original_slot] += norm_score / num_players
    return slot_norm_scores


def train_on_batch(
    all_x, all_y, device, live_model, optimizer, loss_fn, shared_model, scheduler
):
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
            torch.nn.utils.clip_grad_norm_(live_model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            epoch_losses.append(loss.item())

    shared_model.load_state_dict(live_model.state_dict())
    scheduler.step()

    return np.mean(epoch_losses)


def save_snapshot(
    snapshot_dir, batch_idx, live_model, optimizer, scheduler, epsilon, resume_path
):
    snapshot_path = os.path.join(snapshot_dir, f"model_gen_{batch_idx}.pt")
    checkpoint = {
        "batch_idx": batch_idx,
        "model_state_dict": live_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epsilon": epsilon,
    }
    torch.save(checkpoint, snapshot_path)
    torch.save(checkpoint, resume_path)

    return snapshot_path


def handle_keyboard_interrupt(pool):
    print("Training interrupted.")
    pool.terminate()
    active_workers = multiprocessing.active_children()
    for worker in active_workers:
        worker.kill()
    for worker in active_workers:
        worker.join(timeout=0.1)
    os._exit(1)


def parse_batch_num(filepath):
    try:
        filename = os.path.basename(filepath)
        parts = filename.replace(".pt", "").split("_")
        return int(parts[-1])
    except (IndexError, ValueError):
        return 0


def update_pairwise_elo(ratings, sampled_keys, match_scores):
    n = len(sampled_keys)
    for i in range(n):
        for j in range(i + 1, n):
            m1, m2 = sampled_keys[i], sampled_keys[j]
            r1, r2 = ratings[m1], ratings[m2]
            e1 = 1.0 / (1.0 + 10.0 ** ((r2 - r1) / 400.0))

            if match_scores[i] > match_scores[j]:
                s1, s2 = 1.0, 0.0
            elif match_scores[i] < match_scores[j]:
                s1, s2 = 0.0, 1.0
            else:
                s1, s2 = 0.5, 0.5

            ratings[m1] += (K_FACTOR / (n - 1)) * (s1 - e1)
            ratings[m2] += (K_FACTOR / (n - 1)) * (s2 - (1.0 - e1))


def anchor_baseline_elo(ratings, baseline_key):
    if baseline_key in ratings:
        offset = 1000.0 - ratings[baseline_key]
        for key in ratings:
            ratings[key] += offset


def run_elo_tournament(
    active_pool, run_match_fn, match_tasks, baseline_key, snapshot_files
):
    ratings = {key: 1000.0 for key in active_pool}
    completed = 0

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=NUM_WORKERS, initializer=init_worker
    ) as executor:
        futures = [executor.submit(run_match_fn, task) for task in match_tasks]
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            sampled_keys, slot_norm_scores = future.result()
            update_pairwise_elo(ratings, sampled_keys, slot_norm_scores)

            if completed % 50 == 0 or completed == len(match_tasks):
                print(f" -> Completed {completed}/{len(match_tasks)} matches")

    anchor_baseline_elo(ratings, baseline_key)

    results = []
    for path in snapshot_files:
        batch_num = parse_batch_num(path)
        results.append(
            {
                "path": path,
                "batch": batch_num,
                "elo": round(ratings.get(path, 1000.0), 2),
            }
        )
    results.sort(key=lambda x: x["elo"], reverse=True)
    return ratings, results
