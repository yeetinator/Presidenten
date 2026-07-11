import concurrent.futures
import os
import random
import numpy as np
import torch

from game import President
from playerTypes.dmc_bot import PresidentDMCBot, PresidentValueNet
from playerTypes.master_dmc_bot import MasterDMCBot, MasterValueNet

TOTAL_GAMES = 792
NUM_ROUNDS = 10
NUM_WORKERS = 12

MASTER_CHECKPOINT = "playerTypes/best_master_dmc_25500.pt"
OLD_DMC_CHECKPOINT = "playerTypes/model_gen_48250.pt"


def run_benchmark_game(game_args):
    game_seed, num_players, master_model, old_dmc_model, device = game_args
    torch.set_num_threads(1)
    master_placements = [0] * num_players
    master_norm_score = 0.0
    old_dmc_avg_norm = 0.0
    master_wins = 0

    for rotation in range(num_players):
        bot_instances: dict[int, PresidentDMCBot] = {}
        for seat in range(num_players):
            if seat == rotation:
                bot_instances[seat] = MasterDMCBot(seat, master_model, device)
            else:
                bot_instances[seat] = PresidentDMCBot(seat, old_dmc_model, device)

        env = President(num_players)
        for round_idx in range(NUM_ROUNDS):
            round_seed = game_seed + round_idx
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
            master_rank = env.out_order.index(rotation)
            master_placements[master_rank] += 1

        max_pos_score = (num_players - 1) * NUM_ROUNDS
        master_raw_score = env.scores[rotation][0]
        master_norm_score += ((master_raw_score / max_pos_score) * 2 - 1) / num_players
        master_wins += env.scores[rotation][1]
        old_dmc_raw_scores = [
            env.scores[s][0] for s in range(num_players) if s != rotation
        ]
        old_dmc_avg_norm += (
            (np.mean(old_dmc_raw_scores) / max_pos_score) * 2 - 1
        ) / num_players

    match_rounds = NUM_ROUNDS * num_players
    return {
        "num_players": num_players,
        "master_norm_score": master_norm_score,
        "master_wins": master_wins,
        "old_dmc_avg_norm": old_dmc_avg_norm,
        "master_placements": master_placements,
        "match_rounds": match_rounds,
    }


def main():
    if not os.path.exists(MASTER_CHECKPOINT):
        print(f"Error: Master checkpoint not found at '{MASTER_CHECKPOINT}'")
        return

    if not os.path.exists(OLD_DMC_CHECKPOINT):
        print(f"Error: Old DMC checkpoint not found at '{OLD_DMC_CHECKPOINT}'")
        return

    print(f"Starting Head-to-Head Benchmark ({TOTAL_GAMES} Games)...")
    print(f" Master Bot Path : {MASTER_CHECKPOINT}")
    print(f" Old DMC Bot Path: {OLD_DMC_CHECKPOINT}\n")

    device = torch.device("cpu")
    master_model = MasterValueNet(input_dim=193).to(
        device
    )  # PresidentValueNet(input_dim=115).to(device)
    master_ckpt = torch.load(MASTER_CHECKPOINT, map_location=device)
    master_model.load_state_dict(master_ckpt["model_state_dict"])
    master_model.eval()

    old_dmc_model = PresidentValueNet(input_dim=115).to(device)
    old_ckpt = torch.load(OLD_DMC_CHECKPOINT, map_location=device)
    old_dmc_model.load_state_dict(old_ckpt["model_state_dict"])
    old_dmc_model.eval()

    base_entropy = random.randint(1_000_000, 99_000_000)
    game_tasks = []

    for game_idx in range(TOTAL_GAMES):
        num_players = 4 + (game_idx % 4)
        game_seed = base_entropy + game_idx
        game_tasks.append((game_seed, num_players, master_model, old_dmc_model, device))

    completed = 0
    results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(run_benchmark_game, task) for task in game_tasks]

        for future in concurrent.futures.as_completed(futures):
            completed += 1
            results.append(future.result())

            if completed % 100 == 0 or completed == TOTAL_GAMES:
                print(f" Completed {completed}/{TOTAL_GAMES} games...")

    avg_master_score = np.mean([r["master_norm_score"] for r in results])
    avg_old_dmc_score = np.mean([r["old_dmc_avg_norm"] for r in results])
    total_master_wins = sum(r["master_wins"] for r in results)
    total_rounds = sum(r["match_rounds"] for r in results)
    win_rate = (total_master_wins / total_rounds) * 100

    max_ranks = max(r["num_players"] for r in results)
    rank_counts = [0] * max_ranks
    rounds_per_rank = [0] * max_ranks

    for r in results:
        p_count = r["num_players"]
        match_rounds = r["match_rounds"]

        for rank_idx, count in enumerate(r["master_placements"]):
            rank_counts[rank_idx] += count

        for rank_idx in range(p_count):
            rounds_per_rank[rank_idx] += match_rounds

    print("\n" + "=" * 65)
    print("           MASTER BOT vs. OLD DMC BOT HEAD-TO-HEAD           ")
    print("=" * 65)
    print(f" Total Matches Evaluated : {TOTAL_GAMES} ({total_rounds} rounds)")
    print(f" Master Avg Norm Score   : {avg_master_score:+.4f}")
    print(f" Old DMC Avg Norm Score  : {avg_old_dmc_score:+.4f}")
    print(f" Score Delta (Master - Old): {avg_master_score - avg_old_dmc_score:+.4f}")
    print(
        f" Master Round Win Rate   : {win_rate:.2f}% ({total_master_wins}/{total_rounds})"
    )
    print("-" * 65)
    print(" Master Finish Position Breakdown:")
    rank_labels = [
        "1st (President)",
        "2nd (Vice-Pres)",
        "3rd",
        "4th",
        "5th",
        "6th",
        "7th (Scum)",
    ]
    for i in range(max_ranks):
        label = rank_labels[i] if i < len(rank_labels) else f"{i+1}th"
        actual_finishes = rank_counts[i]
        valid_rounds = rounds_per_rank[i]
        pct = (actual_finishes / valid_rounds) * 100
        print(
            f"   {label:<16}: {actual_finishes:<6} / {valid_rounds:<6} rounds ({pct:.2f}%)"
        )
    print("=" * 65)


if __name__ == "__main__":
    main()
