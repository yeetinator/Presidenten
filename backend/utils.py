import torch
import os
import multiprocessing
import random
import concurrent.futures
import sys
import glob
import json
import subprocess
import shutil
import time
import numpy as np
from game import President
from playerTypes.player import Player
from playerTypes.baseline_bot import PresidentBaselineBot
from playerTypes.dmc_bot import PresidentDMCBot, PresidentValueNet, INPUT_DIM

NUM_ROUNDS = 10
GRADIENT_CLIP = 1.0
NUM_WORKERS = 12
K_FACTOR = 16.0
BASELINE_KEY = "BASELINE_BOT"
PLATEAU_WINDOW = 4
PLATEAU_PATIENCE = 3
MIN_ELO_GAIN = 5.0

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


def play_round(env: President, bot_instances: dict[int, Player], round_idx: int):
    state = env.full_reset(round_idx > 0)
    if round_idx > 0:
        cards_to_pass = {}
        for p_id, role in env.roles.items():
            if role != "Citizen":
                cards_to_pass[p_id] = bot_instances[p_id].choose_cards_to_pass(
                    env._get_state(p_id)
                )

        for pair in env.role_pairs:
            env.exchange_cards(pair, cards_to_pass)
        state = env._get_state(env.curr_turn)

    move_count = 0
    while not env.game_over:
        move_count += 1
        if move_count > 500:
            return False

        curr_player = env.curr_turn
        if curr_player is None:
            break

        chosen_move = bot_instances[curr_player].get_move(state, env)
        state, _ = env.step(curr_player, chosen_move)

        if env.was_pile_reset:
            env.clear_pile()
    env.assign_roles()
    return True


def game_loop(
    num_players,
    bot_instances: dict[int, Player],
    live_model,
    input_dim=INPUT_DIM,
    bot_class=PresidentDMCBot,
):
    env = President(num_players)
    game_x, game_y = [], []

    for round_idx in range(NUM_ROUNDS):
        if not play_round(env, bot_instances, round_idx):
            for bot in bot_instances.values():
                if isinstance(bot, PresidentDMCBot):
                    bot.trajectory.clear()
            return np.empty((0, input_dim), dtype=np.float32), np.empty(
                (0, 1), dtype=np.float32
            )

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
    bot_instances: dict[int, Player],
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
                        env._get_state(p_id)
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


def run_duplicate_match(match_args):
    match_seed, num_players, sampled_keys, keys_set, net_class, bot_class = match_args
    device = torch.device("cpu")
    slot_norm_scores = [0.0] * num_players

    for rotation in range(num_players):
        seat_assignments = [
            sampled_keys[(i + rotation) % num_players] for i in range(num_players)
        ]
        bot_instances: dict[int, Player] = {}

        for seat, key in enumerate(seat_assignments):
            if key == BASELINE_KEY:
                bot_instances[seat] = PresidentBaselineBot(seat)
            elif key in keys_set and bot_class != PresidentDMCBot:
                model = get_cached_model(key, net_class)
                bot_instances[seat] = bot_class(seat, model, device)
            else:
                model = get_cached_model(key, PresidentValueNet)
                bot_instances[seat] = PresidentDMCBot(seat, model, device)
        slot_norm_scores = eval_game_loop(
            num_players,
            match_seed,
            bot_instances,
            seat_assignments,
            rotation,
            slot_norm_scores,
        )
    return sampled_keys, slot_norm_scores


def run_evaluation(
    snapshot_dir,
    name: str,
    net_class,
    bot_class,
    basic_snapshot_dir=None,
    gen_cycle=None,
):
    if gen_cycle is None:
        try:
            gen_cycle = int(sys.argv[1])
        except (IndexError, ValueError):
            gen_cycle = 1

    candidates = glob.glob(f"{snapshot_dir}/model_gen_*.pt")
    elites = glob.glob(f"{snapshot_dir}/elites/model_gen_*.pt")
    snapshot_files = list(set(candidates + elites))

    basic_elites = (
        glob.glob(f"{basic_snapshot_dir}/elites/model_gen_*.pt")
        + glob.glob(f"{basic_snapshot_dir}/model_gen_*.pt")
        if basic_snapshot_dir
        else []
    )
    if not snapshot_files:
        print(f"No {name} snapshot files found.")
        return

    keys_set = set(snapshot_files)
    active_pool = [BASELINE_KEY] + snapshot_files + basic_elites
    num_matches = get_num_matches(len(active_pool))

    print(
        f"Starting {name} Elo Tournament ({num_matches} Matches across mixed pool)..."
    )

    base_entropy = random.randint(1_000_000, 99_000_000)
    match_tasks = []

    for match_idx in range(num_matches):
        num_players = 4 + (match_idx % 4)
        match_seed = base_entropy + (gen_cycle * 10000) + match_idx
        sampled_snap = random.choice(snapshot_files)
        other_pool = [k for k in active_pool if k != sampled_snap]

        if len(other_pool) >= num_players - 1:
            sampled_others = random.sample(other_pool, num_players - 1)
        else:
            sampled_others = list(other_pool)
            while len(sampled_others) < num_players - 1:
                sampled_others.append(BASELINE_KEY)

        sampled_keys = [sampled_snap] + sampled_others
        random.shuffle(sampled_keys)

        match_tasks.append(
            (match_seed, num_players, sampled_keys, keys_set, net_class, bot_class)
        )

    ratings, results = run_elo_tournament(
        active_pool,
        run_duplicate_match,
        match_tasks,
        BASELINE_KEY,
        snapshot_files,
    )

    print("\n" + "=" * 65)
    print(f" RANK  | {name.upper()} BATCH GENERATION | ELO RATING vs BASIC FIELD")
    print("=" * 65)
    print(f" BASELINE CONTROL BOT     | Elo: {ratings[BASELINE_KEY]:.2f}")

    if basic_elites:
        best_basic_key = max(basic_elites, key=lambda k: ratings[k])
        best_basic_file = os.path.basename(best_basic_key)
        print(
            f" BEST BASIC ELITE BOT     | File: {best_basic_file:<15} | Elo: {ratings[best_basic_key]:.2f}"
        )
    print("-" * 65)

    for rank, res in enumerate(results, start=1):
        marker = " <- ELITE" if rank <= 8 else ""
        print(
            f" {rank:<5} {marker:<9} | Batch {res['batch']:<10} | Elo: {res['elo']:.2f}"
        )
    print("=" * 65)

    os.makedirs(f"{snapshot_dir}/evals", exist_ok=True)
    json_path = f"{snapshot_dir}/evals/evaluation_results_{gen_cycle}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)


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


def get_num_matches(pool_len: int):
    return ((pool_len * 60 + NUM_WORKERS - 1) // NUM_WORKERS) * NUM_WORKERS


def load_best_elo_history(snapshot_dir, up_to_cycle):
    history = []
    for cycle in range(1, up_to_cycle + 1):
        json_path = f"{snapshot_dir}/evals/evaluation_results_{cycle}.json"
        if not os.path.exists(json_path):
            continue

        with open(json_path) as f:
            results = json.load(f)

        if results:
            history.append(results[0]["elo"])
    return history


def should_stop_training(snapshot_dir, gen_cycle):
    elo_history = load_best_elo_history(snapshot_dir, gen_cycle)
    if len(elo_history) < PLATEAU_WINDOW * PLATEAU_PATIENCE:
        return False

    windows = [
        np.mean(elo_history[i : i + PLATEAU_WINDOW])
        for i in range(0, len(elo_history) - PLATEAU_WINDOW + 1, PLATEAU_WINDOW)
    ]
    recent = windows[-(PLATEAU_PATIENCE + 1) :]
    gains = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]

    print(
        f"  [early-stop check] last {len(recent)} smoothed elo windows: "
        f"{[round(w, 1) for w in recent]} | gains: {[round(g, 1) for g in gains]}"
    )
    return all(g < MIN_ELO_GAIN for g in gains)


def manage_league_files(snapshot_dir, gen_cycle=None):
    print("\n================ MANAGING LEAGUE POOL ================")

    json_path = f"{snapshot_dir}/evals/evaluation_results_{gen_cycle}.json"
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found. Skipping league management.")
        return

    with open(json_path, "r") as f:
        results = json.load(f)

    if not results:
        print("Error: No evaluation results found. Skipping league management.")
        return

    os.makedirs(f"{snapshot_dir}/elites", exist_ok=True)
    temp_dir = f"{snapshot_dir}/tmp"
    os.makedirs(temp_dir, exist_ok=True)

    top_8 = results[:8]
    print("Staging top 8 models")

    for rank, res in enumerate(top_8, start=1):
        batch = res["batch"]
        src_path = res["path"]

        if os.path.exists(src_path):
            shutil.copy2(src_path, f"{temp_dir}/elite_model_{batch}.pt")
            print(f"  -> Rank {rank}: Batch {batch:<5} staged (Elo: {res['elo']:.2f})")

    print(f"Purging old files")
    for old_file in glob.glob(f"{snapshot_dir}/elites/model_gen_*.pt"):
        os.remove(old_file)

    for snap in glob.glob(f"{snapshot_dir}/model_gen_*.pt"):
        os.remove(snap)

    print("Committing staged files")
    for staged_elite in glob.glob(f"{temp_dir}/elite_model_*.pt"):
        batch = staged_elite.split("_")[-1].split(".")[0]
        shutil.move(staged_elite, f"{snapshot_dir}/elites/model_gen_{batch}.pt")

    shutil.rmtree(temp_dir)
    print("League management completed.")


def run_step(module_name, args=None, dir_name=None):
    print(f"\n================ LAUNCHING {module_name.upper()} ================")
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    folder_name = dir_name if dir_name else os.path.basename(backend_dir)
    module_path = f"{folder_name}.{module_name}"

    cmd = [sys.executable, "-m", module_path]
    if args:
        cmd.extend(args)

    result = subprocess.run(cmd, capture_output=False, text=True, cwd=backend_dir)
    if result.returncode != 0:
        print(f"Error: {module_name} failed with return code {result.returncode}.")
        return False
    return True


def get_resume_cycle(snapshot_dir):
    resume_path = f"{snapshot_dir}/latest_model.pt"
    if os.path.exists(resume_path):
        try:
            checkpoint = torch.load(resume_path, map_location=torch.device("cpu"))
            batch_idx = checkpoint.get("batch_idx", 0)
            return (batch_idx // 2000) + 1
        except Exception as e:
            print(f"Error loading checkpoint from {resume_path}: {e}")
    return 1


def run_orchestrator(snapshot_dir, train_script, evaluate_script, dir_name):
    gen_cycle = get_resume_cycle(snapshot_dir)
    while True:
        print(f"\n=== STARTING LEAGUE GENERATION CYCLE {gen_cycle} ===")
        if not run_step(train_script, dir_name=dir_name):
            break
        if not run_step(evaluate_script, [str(gen_cycle)], dir_name):
            break

        manage_league_files(snapshot_dir, gen_cycle)

        if should_stop_training(snapshot_dir, gen_cycle):
            print(
                f"\n=== NO SUSTAINED ELO GAIN ACROSS {PLATEAU_PATIENCE} WINDOWS "
                f"OF {PLATEAU_WINDOW} CYCLES EACH -- STOPPING AT CYCLE {gen_cycle} ==="
            )
            break

        gen_cycle += 1
        time.sleep(5)
