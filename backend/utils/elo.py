import torch
import sys
import glob
import random
import os
import json
import concurrent.futures
import numpy as np
from playerTypes.player import Player
from playerTypes.baseline_bot import PresidentBaselineBot
from playerTypes.dmc_bot import PresidentDMCBot, PresidentValueNet
from .common import get_cached_traced_model, parse_batch_num, init_worker, NUM_WORKERS
from .gameplay import eval_game_loop

BASELINE_KEY = "BASELINE_BOT"
K_FACTOR = 16.0


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
                model = get_cached_traced_model(key, net_class)
                bot_instances[seat] = bot_class(seat, model, device)
            else:
                model = get_cached_traced_model(key, PresidentValueNet)
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
    snapshot_files = candidates + elites

    milestones_raw = get_stratified_milestones(snapshot_dir, max_milestones=5)
    existing_batches = {parse_batch_num(path) for path in snapshot_files}
    milestones = [
        path for path in milestones_raw if parse_batch_num(path) not in existing_batches
    ]

    basic_elites = (
        glob.glob(f"{basic_snapshot_dir}/elites/model_gen_*.pt")
        + glob.glob(f"{basic_snapshot_dir}/model_gen_*.pt")
        if basic_snapshot_dir
        else []
    )
    if not snapshot_files:
        print(f"No {name} snapshot files found.")
        return

    active_pool = [BASELINE_KEY] + snapshot_files + basic_elites + milestones
    keys_set = set(snapshot_files + milestones)
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


def get_stratified_milestones(snapshot_dir, max_milestones=5):
    all_milestones = sorted(
        glob.glob(f"{snapshot_dir}/milestones/model_gen_*.pt"), key=parse_batch_num
    )
    if len(all_milestones) <= max_milestones:
        return all_milestones

    indices = np.linspace(0, len(all_milestones) - 1, max_milestones, dtype=int)
    return [all_milestones[i] for i in set(indices)]
