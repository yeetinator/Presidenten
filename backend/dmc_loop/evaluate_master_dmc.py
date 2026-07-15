import glob
import torch
import os
import json
import random
import sys
from playerTypes.dmc_bot import (
    PresidentValueNet,
    PresidentDMCBot,
    MasterDMCBot,
    MasterValueNet,
    PresidentBaselineBot,
)
from utils import get_cached_model, eval_game_loop, run_elo_tournament

NUM_MATCHES = 300
BASELINE_KEY = "BASELINE_BOT"


def run_duplicate_match(match_args):
    match_seed, num_players, sampled_keys, master_keys = match_args
    device = torch.device("cpu")
    slot_norm_scores = [0.0] * num_players

    for rotation in range(num_players):
        seat_assignments = [
            sampled_keys[(i + rotation) % num_players] for i in range(num_players)
        ]
        bot_instances = {}

        for seat, key in enumerate(seat_assignments):
            if key == BASELINE_KEY:
                bot_instances[seat] = PresidentBaselineBot(seat)
            elif key in master_keys:
                model = get_cached_model(key, MasterValueNet)
                bot_instances[seat] = MasterDMCBot(seat, model, device)
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
    snapshot_dir="snapshots_master_dmc", basic_snapshot_dir="snapshots", gen_cycle=None
):
    if gen_cycle is None:
        try:
            gen_cycle = int(sys.argv[1])
        except (IndexError, ValueError):
            gen_cycle = 1

    master_candidate = glob.glob(f"{snapshot_dir}/model_gen_*.pt")
    master_elites = glob.glob(f"{snapshot_dir}/elites/model_gen_*.pt")
    master_snapshot_files = list(set(master_candidate + master_elites))

    basic_elites = glob.glob(f"{basic_snapshot_dir}/elites/model_gen_*.pt") + glob.glob(
        f"{basic_snapshot_dir}/model_gen_*.pt"
    )
    if not master_snapshot_files:
        print("No Master snapshot files found.")
        return

    active_pool = [BASELINE_KEY] + master_snapshot_files + basic_elites
    master_keys_set = set(master_snapshot_files)

    print(
        f"Starting Master Elo Tournament ({NUM_MATCHES} Matches across mixed pool)..."
    )

    base_entropy = random.randint(1_000_000, 99_000_000)
    match_tasks = []

    for match_idx in range(NUM_MATCHES):
        num_players = 4 + (match_idx % 4)
        match_seed = base_entropy + (gen_cycle * 10000) + match_idx
        sampled_master = random.choice(master_snapshot_files)
        other_pool = [k for k in active_pool if k != sampled_master]

        if len(other_pool) >= num_players - 1:
            sampled_others = random.sample(other_pool, num_players - 1)
        else:
            sampled_others = list(other_pool)
            while len(sampled_others) < num_players - 1:
                sampled_others.append(BASELINE_KEY)

        sampled_keys = [sampled_master] + sampled_others
        random.shuffle(sampled_keys)

        match_tasks.append((match_seed, num_players, sampled_keys, master_keys_set))

    ratings, results = run_elo_tournament(
        active_pool,
        run_duplicate_match,
        match_tasks,
        BASELINE_KEY,
        master_snapshot_files,
    )

    print("\n" + "=" * 65)
    print(" RANK  | MASTER BATCH GENERATION | ELO RATING vs BASIC FIELD")
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


def main():
    run_evaluation()


if __name__ == "__main__":
    main()
