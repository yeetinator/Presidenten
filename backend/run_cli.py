from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PACKAGE_NAME = "presidenten_game"
PACKAGE_DIR = Path(__file__).resolve().parent / "game"


def _load_package():
    if PACKAGE_NAME in sys.modules:
        return

    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        PACKAGE_DIR / "__init__.py",
        submodule_search_locations=[str(PACKAGE_DIR)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load local game package.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = module
    spec.loader.exec_module(module)


_load_package()

from game.cli_utils import get_settings
from game.tournament import (
    create_players,
    game_parallelism,
    play_presidenten_game,
    print_scores,
    search_parallelism,
    update_final_scores,
)

TOTAL_GAMES = 1000
NUM_WORKERS = 10


def main():
    parallelism, assign_p, has_human, num_players, num_rounds, dmc_paths = (
        get_settings()
    )
    master_scores = {p_id: (0, 0) for p_id in range(num_players)}

    if parallelism == "g":
        master_scores = game_parallelism(
            assign_p,
            num_players,
            num_rounds,
            dmc_paths,
            TOTAL_GAMES,
            NUM_WORKERS,
        )
    elif parallelism == "s":
        master_scores = search_parallelism(
            assign_p,
            has_human,
            num_players,
            num_rounds,
            dmc_paths,
            TOTAL_GAMES,
            NUM_WORKERS,
        )
    else:
        assigned_players = create_players(assign_p, 400, dmc_paths)
        for idx in range(TOTAL_GAMES):
            if idx % 10 == 0:
                print(f"\n=== GAME {idx+1} ===\n")
            round_scores = play_presidenten_game(
                idx,
                num_players,
                num_rounds,
                assigned_players,
                assign_p,
                has_human=has_human,
            )
            master_scores = update_final_scores(master_scores, round_scores)

    print_scores(master_scores, assign_p, num_players, num_rounds, TOTAL_GAMES)


if __name__ == "__main__":
    main()
