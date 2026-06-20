import glob
import concurrent.futures
import torch
import os
from playerTypes.dmc_bot import PresidentenValueNet, PresidentenDMCBot
from playerTypes.baseline_bot import PresidentenBaselineBot
from game import Presidenten

TOTAL_GAMES = 1000
NUM_ROUNDS = 10
NUM_WORKERS = 12  # Adjust based on your system's CPU cores and memory
INPUT_DIM = 119
ELITE_BATCHES = {""" 11250,
    11000,
    11750,
    8250,
    11500,
    10750,
    8750,
    10500,
    9750,
    9000,
    9500,
    5750, """}


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

    bot_instances = {
        0: PresidentenDMCBot(0, model, device),
        1: PresidentenBaselineBot(1),
        2: PresidentenBaselineBot(2),
        3: PresidentenBaselineBot(3),
    }

    accumulated_score = 0
    accumulated_wins = 0

    for _ in range(TOTAL_GAMES):
        env = Presidenten()
        for round_idx in range(NUM_ROUNDS):
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
            env.assign_roles()

        accumulated_score += env.scores[0][0]
        accumulated_wins += env.scores[0][1]

    total_rounds = TOTAL_GAMES * NUM_ROUNDS
    avg_score = accumulated_score / total_rounds

    return {
        "batch": batch_num,
        "total_score": accumulated_score,
        "wins": accumulated_wins,
        "avg_score_per_round": avg_score,
    }


def main():
    snapshot_pattern = "snapshots/model_gen_*.pt"
    snapshot_files = glob.glob(snapshot_pattern)

    if not snapshot_files:
        print("No snapshot files found.")
        return

    if len(ELITE_BATCHES) > 1:
        snapshot_files = [
            f
            for f in snapshot_files
            if any(str(batch) in os.path.basename(f) for batch in ELITE_BATCHES)
        ]

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

    results.sort(key=lambda x: (x["total_score"], x["wins"]), reverse=True)

    print("\n" + "=" * 65)
    print(f" RANK  | BATCH GENERATION | TOTAL POINTS | ROUND WINS | AVG PTS/RD ")
    print("=" * 65)

    for rank, res in enumerate(results, start=1):
        marker = " <- BEST" if rank <= 12 else ""
        print(
            f" {rank:<5} {marker} | Batch {res['batch']:<10} | {res['total_score']:<12} | "
            f"{res['wins']:<10} | {res['avg_score_per_round']:.3f}"
        )
    print("=" * 65)


if __name__ == "__main__":
    main()
