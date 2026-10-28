import torch
import random
import multiprocessing
import concurrent.futures
import numpy as np
from playerTypes.player import Player
from playerTypes.dmc_bot import PresidentDMCBot, PresidentValueNet
from .common import get_cached_traced_model, NUM_ROUNDS, NUM_WORKERS
from .gameplay import seeded_game_gen, run_tagged_games

_H2H_SIDES = {}

# Early stopping (Haybittle-Peto): at each interim look, the match ends only if
# |delta / se| > H2H_INTERIM_Z. Such a strict bound barely changes the overall
# error rate, so the final look can keep the usual 95% interval.
H2H_INTERIM_LOOKS = (0.25, 0.5, 0.75)  # fractions of total_games
H2H_INTERIM_Z = 3.0
H2H_MIN_LOOK_GAMES = 100  # below this the standard error is too noisy to trust
H2H_GAMES_PER_TASK = 20


def _init_h2h_worker(master_spec, old_spec):
    torch.set_num_threads(1)
    for name, (path, net_class, bot_class) in (
        ("master", master_spec),
        ("old", old_spec),
    ):
        # No path: a bot without a model, built as bot_class(seat)
        model = None if path is None else get_cached_traced_model(path, net_class)
        _H2H_SIDES[name] = (model, bot_class)


def _make_bot(side, seat, device) -> Player:
    model, bot_class = _H2H_SIDES[side]
    return bot_class(seat) if model is None else bot_class(seat, model, device)


def _role_transitions(out_orders, seat, num_players):
    """How often the seat kept the top spot and got out of the bottom one, counted
    over the rounds it started in that role."""
    counts = {"top_rounds": 0, "kept_top": 0, "bottom_rounds": 0, "left_bottom": 0}
    for prev, curr in zip(out_orders, out_orders[1:]):
        prev_rank, rank = prev.index(seat), curr.index(seat)
        if prev_rank == 0:
            counts["top_rounds"] += 1
            counts["kept_top"] += rank == 0
        elif prev_rank == num_players - 1:
            counts["bottom_rounds"] += 1
            counts["left_bottom"] += rank != num_players - 1
    return counts


def _play_h2h_games(game_chunk):
    device = torch.device("cpu")

    games = []
    for g_idx, (game_seed, num_players) in enumerate(game_chunk):
        round_seeds = [game_seed + round_idx for round_idx in range(NUM_ROUNDS)]
        for rotation in range(num_players):
            bot_instances: dict[int, Player] = {
                seat: _make_bot("master" if seat == rotation else "old", seat, device)
                for seat in range(num_players)
            }
            games.append(
                (
                    (g_idx, rotation),
                    seeded_game_gen(num_players, bot_instances, round_seeds),
                )
            )

    finished = run_tagged_games(games)

    results = []
    for g_idx, (_, num_players) in enumerate(game_chunk):
        master_placements = [0] * num_players
        master_norm_score = 0.0
        old_avg_norm = 0.0
        master_wins = 0
        max_pos_score = (num_players - 1) * NUM_ROUNDS
        transitions = dict.fromkeys(
            ("top_rounds", "kept_top", "bottom_rounds", "left_bottom"), 0
        )

        for rotation in range(num_players):
            scores, out_orders = finished[(g_idx, rotation)]
            for out_order in out_orders:
                master_placements[out_order.index(rotation)] += 1
            for key, count in _role_transitions(
                out_orders, rotation, num_players
            ).items():
                transitions[key] += count

            master_raw = scores[rotation][0]
            master_norm_score += ((master_raw / max_pos_score) * 2 - 1) / num_players
            master_wins += scores[rotation][1]
            old_raw = [scores[s][0] for s in range(num_players) if s != rotation]
            old_avg_norm += ((np.mean(old_raw) / max_pos_score) * 2 - 1) / num_players

        results.append(
            {
                "num_players": num_players,
                "master_norm_score": master_norm_score,
                "old_avg_norm": old_avg_norm,
                "master_wins": master_wins,
                "master_placements": master_placements,
                "master_transitions": transitions,
                "match_rounds": NUM_ROUNDS * num_players,
            }
        )
    return results


def _mean_se(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return float(np.mean(values)), float("nan")
    return float(np.mean(values)), float(np.std(values, ddof=1) / np.sqrt(len(values)))


def _paired_delta(results):
    return _mean_se([r["master_norm_score"] - r["old_avg_norm"] for r in results])


def _interim_looks(total_games):
    # Multiples of 4 so every player count (4-7) is equally represented
    looks = {int(total_games * frac) // 4 * 4 for frac in H2H_INTERIM_LOOKS}
    return sorted(n for n in looks if H2H_MIN_LOOK_GAMES <= n < total_games)


def run_h2h(
    master_path,
    old_path,
    total_games,
    master_classes=(PresidentValueNet, PresidentDMCBot),
    old_classes=(PresidentValueNet, PresidentDMCBot),
    num_workers=NUM_WORKERS,
    progress_every=100,
    early_stop=True,
    base_seed=None,
    games_per_task=H2H_GAMES_PER_TASK,
):
    """Plays one master seat against old in every other seat. A path of None
    makes a side a bot without a model, built as bot_class(seat); slow bots
    such as ISMCTS want small games_per_task so every worker gets some."""
    # A fixed base_seed replays the same deals, e.g. to rerun a gate exactly
    base_entropy = (
        random.randint(1_000_000, 99_000_000) if base_seed is None else base_seed
    )
    game_tasks = [
        (base_entropy + game_idx * NUM_ROUNDS, 4 + (game_idx % 4))
        for game_idx in range(total_games)
    ]  # [(seed, num_players), ...]

    looks = _interim_looks(total_games) if early_stop else []
    results = [None] * total_games
    finished_prefix = 0  # games [0, finished_prefix) have all finished
    stopped_at = None

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_h2h_worker,
        initargs=((master_path, *master_classes), (old_path, *old_classes)),
    ) as executor:
        futures = {
            executor.submit(_play_h2h_games, game_tasks[start:end]): range(start, end)
            for start in range(0, total_games, games_per_task)
            for end in [min(start + games_per_task, total_games)]
        }
        completed = 0

        for future in concurrent.futures.as_completed(futures):
            for idx, result in zip(futures[future], future.result()):
                results[idx] = result
                completed += 1

                if progress_every and (
                    completed % progress_every == 0 or completed == total_games
                ):
                    print(f" -> Completed {completed}/{total_games} H2H games")

            # Looks use the first n games in submission order, not the first n to
            # finish (short 4-player games would otherwise be over-represented)
            while (
                finished_prefix < total_games and results[finished_prefix] is not None
            ):
                finished_prefix += 1
            reached = [n for n in looks if n <= finished_prefix]
            if not reached:
                continue
            looks = [n for n in looks if n > finished_prefix]

            # se ~ 0 means the diffs are float noise (e.g. a model against itself)
            mean_diff, se_diff = _paired_delta(results[: reached[-1]])
            if se_diff > 1e-9 and abs(mean_diff / se_diff) > H2H_INTERIM_Z:
                stopped_at = reached[-1]
                for pending in futures:
                    pending.cancel()
                break

    if stopped_at is not None:
        results = results[:stopped_at]
        print(f" -> H2H stopped early after {stopped_at}/{total_games} games")

    master_scores = np.array([r["master_norm_score"] for r in results])
    old_scores = np.array([r["old_avg_norm"] for r in results])
    mean_diff, se_diff = _paired_delta(results)
    _, se_master = _mean_se(master_scores)

    transitions = {
        key: sum(r["master_transitions"][key] for r in results)
        for key in results[0]["master_transitions"]
    }
    by_players = {}
    for num_players in sorted({r["num_players"] for r in results}):
        group = [r["master_norm_score"] for r in results if r["num_players"] == num_players]
        mean, se = _mean_se(group)
        by_players[num_players] = {"games": len(group), "master_avg": mean, "se": se}

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
        "total_games": len(results),
        "planned_games": total_games,
        "base_seed": base_entropy,
        "stopped_early": stopped_at is not None,
        "total_rounds": sum(r["match_rounds"] for r in results),
        "master_avg": float(np.mean(master_scores)),
        "master_se": se_master,
        "old_avg": float(np.mean(old_scores)),
        "delta": mean_diff,
        "ci_lo": mean_diff - 1.96 * se_diff,
        "ci_hi": mean_diff + 1.96 * se_diff,
        "t_stat": mean_diff / se_diff if se_diff > 0 else float("inf"),
        "master_wins": sum(r["master_wins"] for r in results),
        "rank_counts": rank_counts,
        "rounds_per_rank": rounds_per_rank,
        "last_place": sum(r["master_placements"][-1] for r in results),
        "transitions": transitions,
        "by_players": by_players,
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
        f" Total Matches Evaluated : {s['total_games']} ({s['total_rounds']} rounds)"
        + (
            f" - stopped early (planned {s['planned_games']})"
            if s.get("stopped_early")
            else ""
        ),
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
