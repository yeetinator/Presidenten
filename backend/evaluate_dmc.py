import glob
import concurrent.futures
import torch
import os
import json
import random
import sys
import numpy as np
from playerTypes.dmc_bot import PresidentValueNet, PresidentDMCBot
from playerTypes.baseline_bot import PresidentBaselineBot
from game import President

NUM_MATCHES = 300
NUM_ROUNDS = 10
NUM_WORKERS = 12
K_FACTOR = 16.0
BASELINE_KEY = "BASELINE_BOT"

_MODEL_CACHE = {}


def parse_batch_num(filepath):
    try:
        filename = os.path.basename(filepath)
        parts = filename.replace(".pt", "").split("_")
        return int(parts[-1])
    except (IndexError, ValueError):
        return 0


def get_model(path, net_class, input_dim):
    if path not in _MODEL_CACHE:
        device = torch.device("cpu")
        model = net_class(input_dim).to(device)
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        _MODEL_CACHE[path] = model
    return _MODEL_CACHE[path]


def run_duplicate_match(match_args):
    match_seed, num_players, sampled_keys, net_class, bot_class, input_dim = match_args
    torch.set_num_threads(1)
    device = torch.device("cpu")
    slot_norm_scores = [0.0] * num_players

    for rotation in range(num_players):
        seat_assignments = [
            sampled_keys[(i + rotation) % num_players] for i in range(num_players)
        ]
        bot_instances: dict[int, PresidentDMCBot | PresidentBaselineBot] = {}

        for seat, key in enumerate(seat_assignments):
            if key == BASELINE_KEY:
                bot_instances[seat] = PresidentBaselineBot(seat)
            else:
                model = get_model(key, net_class, input_dim)
                bot_instances[seat] = bot_class(seat, model, device)

        env = President(num_players)
        for round_idx in range(NUM_ROUNDS):
            round_seed = match_seed * 1000 + round_idx
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


def update_pairwise_elo(ratings, sampled_keys, match_scores, k_factor=K_FACTOR):
    n = len(sampled_keys)
    for i in range(n):
        for j in range(i + 1, n):
            m1, m2 = sampled_keys[i], sampled_keys[j]
            r1, r2 = ratings[m1], ratings[m2]
            e1 = 1.0 / (1.0 + 10.0 ** ((r2 - r1) / 400.0))
            e2 = 1.0 - e1

            if match_scores[i] > match_scores[j]:
                s1, s2 = 1.0, 0.0
            elif match_scores[i] < match_scores[j]:
                s1, s2 = 0.0, 1.0
            else:
                s1, s2 = 0.5, 0.5

            delta1 = (k_factor / (n - 1)) * (s1 - e1)
            delta2 = (k_factor / (n - 1)) * (s2 - e2)
            ratings[m1] += delta1
            ratings[m2] += delta2


def anchor_baseline_elo(ratings):
    if BASELINE_KEY in ratings:
        offset = 1000.0 - ratings[BASELINE_KEY]
        for key in ratings:
            ratings[key] += offset


def run_evaluation(
    net_class=PresidentValueNet,
    bot_class=PresidentDMCBot,
    input_dim=115,
    snapshot_dir="snapshots",
    gen_cycle=None,
):
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
    ratings = {}

    for key in active_pool:
        ratings[key] = 1000.0

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

        match_tasks.append(
            (match_seed, num_players, sampled_keys, net_class, bot_class, input_dim)
        )

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
    for path in all_snapshot_files:
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
    run_evaluation(
        net_class=PresidentValueNet,
        bot_class=PresidentDMCBot,
        input_dim=115,
        snapshot_dir="snapshots",
    )


if __name__ == "__main__":
    main()
