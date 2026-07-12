import glob
import concurrent.futures
import torch
import os
import json
import random
import sys
import numpy as np
from playerTypes.dmc_bot import PresidentValueNet, PresidentDMCBot
from playerTypes.master_dmc_bot import MasterDMCBot, MasterValueNet
from playerTypes.baseline_bot import PresidentBaselineBot
from game import President
from evaluate_dmc import anchor_baseline_elo, parse_batch_num, update_pairwise_elo

NUM_MATCHES = 300
NUM_ROUNDS = 10
NUM_WORKERS = 12
K_FACTOR = 16.0
BASELINE_KEY = "BASELINE_BOT"

_MODEL_CACHE = {}


def get_model(path, is_master=False):
    if path not in _MODEL_CACHE:
        device = torch.device("cpu")
        net_class = MasterValueNet if is_master else PresidentValueNet
        model = net_class().to(device)
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        _MODEL_CACHE[path] = model
    return _MODEL_CACHE[path]


def run_duplicate_match(match_args):
    match_seed, num_players, sampled_keys, master_keys = match_args
    torch.set_num_threads(1)
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
                model = get_model(key, is_master=True)
                bot_instances[seat] = MasterDMCBot(seat, model, device)
            else:
                model = get_model(key, is_master=False)
                bot_instances[seat] = PresidentDMCBot(seat, model, device)

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
    ratings = {key: 1000.0 for key in active_pool}

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

    completed = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(run_duplicate_match, task) for task in match_tasks]
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            sampled_keys, slot_norm_scores = future.result()
            update_pairwise_elo(ratings, sampled_keys, slot_norm_scores)

            if completed % 50 == 0 or completed == NUM_MATCHES:
                print(f" -> Completed {completed}/{NUM_MATCHES} matches")

    anchor_baseline_elo(ratings)
    results = []
    for path in master_snapshot_files:
        batch_num = parse_batch_num(path)
        results.append(
            {
                "path": path,
                "batch": batch_num,
                "elo": round(ratings.get(path, 1000.0), 2),
            }
        )
    results.sort(key=lambda x: x["elo"], reverse=True)

    print("\n" + "=" * 65)
    print(" RANK  | MASTER BATCH GENERATION | ELO RATING vs BASIC FIELD")
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
