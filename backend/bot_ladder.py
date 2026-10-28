"""Ranks the bots: each one sits in one seat of a table of copies of a field bot,
and every game is replayed with it in every seat, so it gets every hand once.

python bot_ladder.py                   # every bot in every field, skips finished runs
python bot_ladder.py --only ISMCTS     # some bots
python bot_ladder.py --field DMC       # one field
python bot_ladder.py --force           # rerun finished runs
"""

import argparse
import json
import os
import time
from utils.head_to_head import run_h2h
from utils.ladder import BOTS, ISMCTS_WORKERS, beat_rate

RESULTS_PATH = "results/bot_ladder.json"
FIELDS = ("DMC", "Baseline")
GAMES = 800
# 6.5 s of thinking per ISMCTS round makes 120 games about 1.5 h on 8 cores
ISMCTS_GAMES = 120
# Every bot in a field plays the same deals
FIELD_SEEDS = {"DMC": 20261028, "Baseline": 20261029}


def load_results():
    if not os.path.exists(RESULTS_PATH):
        return {}
    with open(RESULTS_PATH) as f:
        return json.load(f)


def save_results(results):
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=4)


def run_entry(bot, field):
    is_ismcts = bot == "ISMCTS"
    path, *classes = BOTS[bot]
    field_path, *field_classes = BOTS[field]
    start = time.time()
    s = run_h2h(
        path,
        field_path,
        ISMCTS_GAMES if is_ismcts else GAMES,
        master_classes=tuple(classes),
        old_classes=tuple(field_classes),
        early_stop=False,
        base_seed=FIELD_SEEDS[field],
        progress_every=10 if is_ismcts else 200,
        **({"games_per_task": 1, "num_workers": ISMCTS_WORKERS} if is_ismcts else {}),
    )
    rounds = s["total_rounds"]
    t = s["transitions"]
    return {
        "bot": bot,
        "field": field,
        "games": s["total_games"],
        "rounds": rounds,
        "base_seed": s["base_seed"],
        "beat_rate": beat_rate(s["master_avg"]),
        # Half the normalized score's interval, as beat rate is (norm + 1) / 2
        "beat_rate_ci": 1.96 * s["master_se"] / 2,
        "by_players": {
            n: {
                "games": g["games"],
                "beat_rate": beat_rate(g["master_avg"]),
                "beat_rate_ci": 1.96 * g["se"] / 2,
            }
            for n, g in s["by_players"].items()
        },
        "first": s["rank_counts"][0] / rounds,
        "last": s["last_place"] / rounds,
        "kept_top": t["kept_top"] / max(1, t["top_rounds"]),
        "left_bottom": t["left_bottom"] / max(1, t["bottom_rounds"]),
        "transitions": t,
        "seconds": round(time.time() - start),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=list(BOTS))
    parser.add_argument("--field", nargs="*", default=list(FIELDS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    results = load_results()
    for field in args.field:
        for bot in args.only:
            key = f"{bot} vs {field}"
            # A bot among copies of itself is level with the table by symmetry,
            # but how often it ends first or last is still worth knowing
            if key in results and not args.force:
                continue
            print(f"\n=== {key} ===")
            results[key] = entry = run_entry(bot, field)
            save_results(results)
            print(
                f" -> beat rate {100 * entry['beat_rate']:.1f}% "
                f"+/- {100 * entry['beat_rate_ci']:.1f} "
                f"({entry['games']} games, {entry['seconds']} s)"
            )


if __name__ == "__main__":
    main()
