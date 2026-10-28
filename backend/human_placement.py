"""Places a human among the bots, from rounds played in the web game's benchmark
mode (start the server with PRESIDENT_BENCHMARK_LOG=<file>).

Every logged round is played again from its start, with the same deal, roles and
opponents, but with a bot in the human's seat. The human's finish minus the bot's
finish on the same round leaves the luck of the deal out of the comparison.

python human_placement.py results/human_rounds.jsonl
python human_placement.py --dry-run Baseline --sessions 20 --bots DMC Random
    (a bot plays the human's seat in simulated sessions, to check the intervals)
"""

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import random
import numpy as np
from utils.common import NUM_ROUNDS, NUM_WORKERS, init_worker
from utils.gameplay import (
    play_round_gen,
    replay_round_gen,
    run_tagged_games,
    seed_deal,
)
from utils.ladder import BOTS, ISMCTS_WORKERS, STOCHASTIC, beat_rate, make_bot, norm_score
from game import President

RESULTS_PATH = "results/human_placement.json"
FIELD = "DMC"
STOCHASTIC_REPLAYS = 8
# Rounds are resampled in runs of this many, as one round's finish sets the
# roles, and so the exchange, of the next
BLOCK_ROUNDS = 5
BOOTSTRAPS = 20_000
TASK_ROUNDS = 64


def _replay_chunk(tasks):
    """tasks: [(tag, record, bot name)] -> {tag: the bot's score in the human's seat}"""
    games = []
    for tag, record, bot in tasks:
        human = record["human_seat"]
        bots = {
            seat: make_bot(bot if seat == human else label, seat)
            for seat, label in enumerate(record["seats"])
        }
        games.append(
            (
                tag,
                replay_round_gen(
                    record["num_players"],
                    bots,
                    record["round_idx"],
                    record["roles"],
                    record["seed"],
                ),
            )
        )
    records = {tag: record for tag, record, _ in tasks}
    return {
        tag: norm_score(
            out_order.index(records[tag]["human_seat"]), records[tag]["num_players"]
        )
        for tag, out_order in run_tagged_games(games).items()
    }


def replay_scores(records, bots):
    """{bot: each round's score with the bot in the human's seat}, averaged over
    STOCHASTIC_REPLAYS replays for bots that can play a round differently."""
    tasks = [
        ((bot, rep, i), record, bot)
        for bot in bots
        for rep in range(STOCHASTIC_REPLAYS if bot in STOCHASTIC else 1)
        for i, record in enumerate(records)
    ]
    # Slow ISMCTS rounds first and one per task, so they spread over the workers
    tasks.sort(key=lambda t: t[2] != "ISMCTS")
    chunks, start = [], 0
    while start < len(tasks):
        size = 1 if tasks[start][2] == "ISMCTS" else TASK_ROUNDS
        chunks.append(tasks[start : start + size])
        start += size

    workers = ISMCTS_WORKERS if "ISMCTS" in bots else NUM_WORKERS
    sums = {bot: np.zeros(len(records)) for bot in bots}
    counts = {bot: np.zeros(len(records)) for bot in bots}
    with concurrent.futures.ProcessPoolExecutor(
        workers, mp_context=multiprocessing.get_context("spawn"), initializer=init_worker
    ) as executor:
        done = 0
        for result in executor.map(_replay_chunk, chunks):
            for (bot, _, i), score in result.items():
                sums[bot][i] += score
                counts[bot][i] += 1
            done += 1
            if done % max(1, len(chunks) // 10) == 0:
                print(f" -> replayed {done}/{len(chunks)} chunks")
    return {bot: sums[bot] / counts[bot] for bot in bots}


def human_scores(records):
    return np.array(
        [
            norm_score(r["out_order"].index(r["human_seat"]), r["num_players"])
            for r in records
        ]
    )


def _blocks(records):
    keys = [(r["game"], r["round_idx"] // BLOCK_ROUNDS) for r in records]
    ids = {key: i for i, key in enumerate(dict.fromkeys(keys))}
    return np.array([ids[key] for key in keys]), len(ids)


def block_bootstrap(values, records, rng):
    """Bootstrap means of per-round values, resampling blocks of rounds."""
    block_of, n_blocks = _blocks(records)
    sums = np.bincount(block_of, weights=values, minlength=n_blocks)
    sizes = np.bincount(block_of, minlength=n_blocks)
    picks = rng.integers(0, n_blocks, size=(BOOTSTRAPS, n_blocks))
    return sums[picks].sum(1) / sizes[picks].sum(1)


def placement(records, bot_scores, seed=0):
    """Beat rates on these rounds, and the human minus each bot with a 95%
    interval and the share of bootstrap samples in which the human is ahead."""
    rng = np.random.default_rng(seed)
    human = human_scores(records)
    plain = beat_rate(block_bootstrap(human, records, rng))
    report = {
        "rounds": len(records),
        "human": {
            "beat_rate": float(beat_rate(human.mean())),
            # Without the replays, for comparison: the deal's luck stays in
            "plain_ci": [float(np.percentile(plain, 2.5)), float(np.percentile(plain, 97.5))],
        },
        "bots": {},
    }
    for bot, scores in bot_scores.items():
        # Beat rate differences are half the normalized ones
        diffs = (human - scores) / 2
        boot = block_bootstrap(diffs, records, rng)
        half = len(records) // 2
        report["bots"][bot] = {
            "beat_rate": float(beat_rate(scores.mean())),
            "diff": float(diffs.mean()),
            "diff_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "p_human_ahead": float(np.mean(boot > 0) + 0.5 * np.mean(boot == 0)),
            # A clear change between the halves would hint at learning the bots
            "diff_first_half": float(diffs[:half].mean()),
            "diff_second_half": float(diffs[half:].mean()),
        }
    return report


def format_placement(report):
    h = report["human"]
    lines = [
        f"Rounds: {report['rounds']}",
        f"Human beat rate {100 * h['beat_rate']:.1f}% "
        f"(without replays: 95% CI {100 * h['plain_ci'][0]:.1f} to {100 * h['plain_ci'][1]:.1f})",
        f"{'bot':<10}{'bot beat rate':>15}{'human - bot':>14}{'95% CI':>20}{'P(human ahead)':>17}",
    ]
    for bot, b in sorted(report["bots"].items(), key=lambda kv: kv[1]["beat_rate"]):
        lo, hi = b["diff_ci"]
        lines.append(
            f"{bot:<10}{100 * b['beat_rate']:>14.1f}%{100 * b['diff']:>+13.1f}"
            f"{f'[{100 * lo:+.1f}, {100 * hi:+.1f}]':>20}{100 * b['p_human_ahead']:>16.0f}%"
        )
    return "\n".join(lines)


def logged_game_gen(num_players, stand_in, game, num_rounds, field=FIELD):
    """A bot in seat 0 of a table of field bots, dealt and logged as in benchmark
    mode."""
    seats = ["Human"] + [field] * (num_players - 1)
    bots = {0: make_bot(stand_in, 0)} | {
        seat: make_bot(field, seat) for seat in range(1, num_players)
    }
    env = President(num_players, lightweight=True)
    records = []
    for round_idx in range(num_rounds):
        seed = game * 1000 + round_idx
        roles = dict(env.roles)
        seed_deal(env, seed)
        yield from play_round_gen(env, bots, round_idx)
        records.append(
            {
                "game": game,
                "round_idx": round_idx,
                "seed": seed,
                "num_players": num_players,
                "human_seat": 0,
                "seats": seats,
                "roles": roles,
                "out_order": list(env.out_order),
            }
        )
    return records


def _simulate_chunk(tasks):
    games = [
        (game, logged_game_gen(num_players, stand_in, game, NUM_ROUNDS))
        for game, num_players, stand_in in tasks
    ]
    return run_tagged_games(games)


def simulate_sessions(stand_in, sessions, rounds, seed):
    """Sessions of rounds with stand_in in the human's seat, table sizes 4-7 in turn."""
    games_per_session = rounds // NUM_ROUNDS
    tasks = [
        (seed + s * 1000 + g, 4 + g % 4, stand_in)
        for s in range(sessions)
        for g in range(games_per_session)
    ]
    with concurrent.futures.ProcessPoolExecutor(
        NUM_WORKERS, mp_context=multiprocessing.get_context("spawn"), initializer=init_worker
    ) as executor:
        chunks = [tasks[i : i + 16] for i in range(0, len(tasks), 16)]
        games = {}
        for result in executor.map(_simulate_chunk, chunks):
            games.update(result)

    session_games = [
        tasks[s * games_per_session : (s + 1) * games_per_session]
        for s in range(sessions)
    ]
    return [[r for game, *_ in part for r in games[game]] for part in session_games]


def dry_run(stand_in, bots, sessions, rounds):
    """Do the intervals from single sessions cover the value all sessions agree on?"""
    all_sessions = simulate_sessions(stand_in, sessions, rounds, random.randint(1, 10**6))
    records = [r for session in all_sessions for r in session]
    scores = replay_scores(records, bots)

    reports, start = [], 0
    for s, session in enumerate(all_sessions):
        part = {bot: v[start : start + len(session)] for bot, v in scores.items()}
        reports.append(placement(session, part, seed=s))
        start += len(session)
    pooled = placement(records, scores)

    print(f"\nStand-in '{stand_in}', {sessions} sessions of {rounds} rounds")
    plain = [r["human"]["plain_ci"] for r in reports]
    print(f" plain beat-rate interval: +/-{100 * np.mean([(hi - lo) / 2 for lo, hi in plain]):.1f} pp")
    for bot in bots:
        truth = pooled["bots"][bot]["diff"]
        cis = [r["bots"][bot]["diff_ci"] for r in reports]
        covered = sum(lo <= truth <= hi for lo, hi in cis)
        width = np.mean([(hi - lo) / 2 for lo, hi in cis])
        print(
            f" vs {bot:<9}: pooled diff {100 * truth:+.2f} pp, interval +/-{100 * width:.1f} pp, "
            f"covers pooled in {covered}/{sessions}"
        )
    return pooled, reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", nargs="?", default="results/human_rounds.jsonl")
    parser.add_argument("--bots", nargs="*", default=list(BOTS))
    parser.add_argument("--dry-run", metavar="BOT")
    parser.add_argument("--sessions", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--any-field", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        dry_run(args.dry_run, args.bots, args.sessions, args.rounds)
        return

    with open(args.log) as f:
        records = [json.loads(line) for line in f if line.strip()]
    kept = [
        r
        for r in records
        if args.any_field
        or all(label == FIELD for i, label in enumerate(r["seats"]) if i != r["human_seat"])
    ]
    print(f"{len(kept)} of {len(records)} logged rounds were at {FIELD} tables")

    report = placement(kept, replay_scores(kept, args.bots))
    print(format_placement(report))
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(report, f, indent=4)


if __name__ == "__main__":
    main()
