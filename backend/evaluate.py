import glob
import concurrent.futures
import torch
import os
import json
import random
import sys
import numpy as np
from playerTypes.dmc_bot import PresidentenValueNet, PresidentenDMCBot
from playerTypes.baseline_bot import PresidentenBaselineBot
from game import Presidenten

TOTAL_GAMES = 800
NUM_ROUNDS = 10
NUM_WORKERS = 8
INPUT_DIM = 115

try:
    gen_cycle = int(sys.argv[1]) * 10000
except (IndexError, ValueError):
    gen_cycle = 0


def evaluate_snapshot(snapshot_file):
    torch.set_num_threads(1)
    device = torch.device("cpu")

    try:
        batch_num = int(os.path.basename(snapshot_file).split("_")[2].split(".")[0])
    except (IndexError, ValueError):
        batch_num = 0

    model = PresidentenValueNet(INPUT_DIM).to(device)
    checkpoint = torch.load(snapshot_file, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    accumulated_norm_score = 0
    accumulated_wins = 0

    for game_idx in range(TOTAL_GAMES):
        curr_seed = gen_cycle + game_idx
        random.seed(curr_seed)
        np.random.seed(curr_seed)
        num_players = 4 + (game_idx % 4)
        bot_instances: dict[int, PresidentenDMCBot | PresidentenBaselineBot] = {
            0: PresidentenDMCBot(0, model, device)
        }

        for seat in range(1, num_players):
            bot_instances[seat] = PresidentenBaselineBot(seat)

        env = Presidenten(num_players)
        for round_idx in range(NUM_ROUNDS):
            state = env.full_reset(next_round=(round_idx > 0))
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
        raw_score = env.scores[0][0]
        norm_score = (raw_score / max_pos_score) * 2 - 1
        accumulated_norm_score += norm_score
        accumulated_wins += env.scores[0][1]

    avg_norm_score = accumulated_norm_score / TOTAL_GAMES

    return {
        "batch": batch_num,
        "wins": accumulated_wins,
        "avg_norm_score": avg_norm_score,
    }


def main():
    snapshot_files = glob.glob("snapshots/model_gen_*.pt") + glob.glob(
        "snapshots/elites/model_gen_*.pt"
    )

    if not snapshot_files:
        print("No snapshot files found.")
        return

    print(f"Found {len(snapshot_files)} snapshot files.")
    results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(evaluate_snapshot, snapshot): snapshot
            for snapshot in snapshot_files
        }
        completed = 0

        for future in concurrent.futures.as_completed(futures):
            completed += 1
            res = future.result()
            results.append(res)
            print(f" -> Completed {completed}/{len(snapshot_files)}: {res["batch"]}")

    results.sort(key=lambda x: (x["avg_norm_score"], x["wins"]), reverse=True)

    print("\n" + "=" * 65)
    print(f" RANK  | BATCH GENERATION | ROUND WINS | AVG NORM SCORE ")
    print("=" * 65)

    for rank, res in enumerate(results, start=1):
        marker = " <- BEST" if rank <= 10 else ""
        print(
            f" {rank:<5} {marker} | Batch {res['batch']:<10} | "
            f"{res['wins']:<10} | {res['avg_norm_score']:.4f}"
        )
    print("=" * 65)

    json_path = "snapshots/evaluation_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
