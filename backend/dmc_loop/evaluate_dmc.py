import glob
import torch
import os
import json
import random
import sys
from playerTypes.player import Player
from playerTypes.baseline_bot import PresidentBaselineBot
from playerTypes.dmc_bot import PresidentValueNet, PresidentDMCBot
from utils import get_cached_model, eval_game_loop, run_elo_tournament

NUM_MATCHES = 300
BASELINE_KEY = "BASELINE_BOT"


def run_duplicate_match(match_args):
    match_seed, num_players, sampled_keys = match_args
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


def run_evaluation(snapshot_dir="snapshots", gen_cycle=None):
    if gen_cycle is None:
        try:
            gen_cycle = int(sys.argv[1])
        except (IndexError, ValueError):
            gen_cycle = 1

    candidate_files = glob.glob(f"{snapshot_dir}/model_gen_*.pt")
    elite_files = glob.glob(f"{snapshot_dir}/elites/model_gen_*.pt")
    all_snapshot_files = list(set(candidate_files + elite_files))

    if not all_snapshot_files:
        print("No snapshot files found.")
        return

    active_pool = [BASELINE_KEY] + all_snapshot_files

    print(
        f"Starting Elo Tournament ({NUM_MATCHES} Duplicate Matches across 4-7 Players)..."
    )
    print(f"Found {len(all_snapshot_files)} snapshots")

    base_entropy = random.randint(1_000_000, 99_000_000)
    match_tasks = []

    for match_idx in range(NUM_MATCHES):
        num_players = 4 + (match_idx % 4)
        match_seed = base_entropy + (gen_cycle * 10000) + match_idx

        if len(active_pool) >= num_players:
            sampled_keys = random.sample(active_pool, num_players)
        else:
            sampled_keys = list(active_pool)
            while len(sampled_keys) < num_players:
                sampled_keys.append(BASELINE_KEY)

        match_tasks.append((match_seed, num_players, sampled_keys))

    ratings, results = run_elo_tournament(
        active_pool, run_duplicate_match, match_tasks, BASELINE_KEY, all_snapshot_files
    )

    print("\n" + "=" * 65)
    print(f" RANK  | BATCH GENERATION | ELO RATING")
    print("=" * 65)
    print(f" BASELINE CONTROL BOT     | Elo: {ratings[BASELINE_KEY]:.2f}")
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


def main():
    run_evaluation()


if __name__ == "__main__":
    main()
