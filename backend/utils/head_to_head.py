import torch
import random
import multiprocessing
import concurrent.futures
import numpy as np
from game import President
from playerTypes.player import Player
from playerTypes.dmc_bot import PresidentDMCBot, PresidentValueNet
from .common import NUM_ROUNDS, NUM_WORKERS
from .gameplay import play_round

_H2H_SIDES = {}


def _init_h2h_worker(master_spec, old_spec):
    torch.set_num_threads(1)
    for name, (path, net_class, bot_class) in (
        ("master", master_spec),
        ("old", old_spec),
    ):
        model = net_class().to("cpu")
        checkpoint = torch.load(path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        _H2H_SIDES[name] = (model, bot_class)


def _play_h2h_game(game_args):
    game_seed, num_players = game_args
    device = torch.device("cpu")
    master_model, master_bot = _H2H_SIDES["master"]
    old_model, old_bot = _H2H_SIDES["old"]

    master_placements = [0] * num_players
    master_norm_score = 0.0
    old_avg_norm = 0.0
    master_wins = 0
    max_pos_score = (num_players - 1) * NUM_ROUNDS

    for rotation in range(num_players):
        bot_instances: dict[int, Player] = {
            seat: (
                master_bot(seat, master_model, device)
                if seat == rotation
                else old_bot(seat, old_model, device)
            )
            for seat in range(num_players)
        }

        env = President(num_players)
        for round_idx in range(NUM_ROUNDS):
            round_seed = game_seed + round_idx
            random.seed(round_seed)
            np.random.seed(round_seed)

            play_round(env, bot_instances, round_idx)
            master_placements[env.out_order.index(rotation)] += 1

        master_raw = env.scores[rotation][0]
        master_norm_score += ((master_raw / max_pos_score) * 2 - 1) / num_players
        master_wins += env.scores[rotation][1]
        old_raw = [env.scores[s][0] for s in range(num_players) if s != rotation]
        old_avg_norm += ((np.mean(old_raw) / max_pos_score) * 2 - 1) / num_players

    return {
        "num_players": num_players,
        "master_norm_score": master_norm_score,
        "old_avg_norm": old_avg_norm,
        "master_wins": master_wins,
        "master_placements": master_placements,
        "match_rounds": NUM_ROUNDS * num_players,
    }


def run_h2h(
    master_path,
    old_path,
    total_games,
    master_classes=(PresidentValueNet, PresidentDMCBot),
    old_classes=(PresidentValueNet, PresidentDMCBot),
    num_workers=NUM_WORKERS,
    progress_every=100,
):
    base_entropy = random.randint(1_000_000, 99_000_000)
    game_tasks = [
        (base_entropy + game_idx * NUM_ROUNDS, 4 + (game_idx % 4))
        for game_idx in range(total_games)
    ]

    results = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_h2h_worker,
        initargs=((master_path, *master_classes), (old_path, *old_classes)),
    ) as executor:
        futures = [executor.submit(_play_h2h_game, task) for task in game_tasks]
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            results.append(future.result())
            if progress_every and (
                completed % progress_every == 0 or completed == total_games
            ):
                print(f" -> Completed {completed}/{total_games} H2H games")

    master_scores = np.array([r["master_norm_score"] for r in results])
    old_scores = np.array([r["old_avg_norm"] for r in results])
    diffs = master_scores - old_scores
    mean_diff = float(np.mean(diffs))
    se_diff = float(np.std(diffs, ddof=1) / np.sqrt(len(diffs)))

    max_ranks = max(r["num_players"] for r in results)
    rank_counts = [0] * max_ranks
    rounds_per_rank = [0] * max_ranks
    for r in results:
        for rank_idx, count in enumerate(r["master_placements"]):
            rank_counts[rank_idx] += count
        for rank_idx in range(r["num_players"]):
            rounds_per_rank[rank_idx] += r["match_rounds"]

    return {
        "master_path": master_path,
        "old_path": old_path,
        "total_games": total_games,
        "total_rounds": sum(r["match_rounds"] for r in results),
        "master_avg": float(np.mean(master_scores)),
        "old_avg": float(np.mean(old_scores)),
        "delta": mean_diff,
        "ci_lo": mean_diff - 1.96 * se_diff,
        "ci_hi": mean_diff + 1.96 * se_diff,
        "t_stat": mean_diff / se_diff if se_diff > 0 else float("inf"),
        "master_wins": sum(r["master_wins"] for r in results),
        "rank_counts": rank_counts,
        "rounds_per_rank": rounds_per_rank,
    }


def format_h2h(s):
    rank_labels = [
        "1st (President)",
        "2nd (Vice-Pres)",
        "3rd",
        "4th",
        "5th",
        "6th",
        "7th (Scum)",
    ]
    verdict = (
        " -> CI includes 0: not distinguishable from no difference at this sample size."
        if s["ci_lo"] <= 0 <= s["ci_hi"]
        else " -> CI excludes 0: difference looks real, not just noise."
    )
    win_rate = 100.0 * s["master_wins"] / s["total_rounds"]
    lines = [
        "=" * 65,
        "                 HEAD-TO-HEAD BENCHMARK                 ",
        "=" * 65,
        f" Master Bot Path : {s['master_path']}",
        f" Old Bot Path    : {s['old_path']}",
        f" Total Matches Evaluated : {s['total_games']} ({s['total_rounds']} rounds)",
        f" Master Avg Norm Score   : {s['master_avg']:+.4f}",
        f" Old Avg Norm Score      : {s['old_avg']:+.4f}",
        f" Score Delta (Master - Old): {s['delta']:+.4f}",
        f" 95% CI on delta (paired, n={s['total_games']} games): "
        f"[{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}]  (t={s['t_stat']:+.2f})",
        verdict,
        f" Master Round Win Rate   : {win_rate:.2f}% ({s['master_wins']}/{s['total_rounds']})",
        "-" * 65,
        " Master Finish Position Breakdown:",
    ]
    for i, (count, rounds) in enumerate(zip(s["rank_counts"], s["rounds_per_rank"])):
        label = rank_labels[i] if i < len(rank_labels) else f"{i + 1}th"
        lines.append(
            f"   {label:<16}: {count:<6} / {rounds:<6} rounds ({100.0 * count / rounds:.2f}%)"
        )
    lines.append("=" * 65)
    return "\n".join(lines)
