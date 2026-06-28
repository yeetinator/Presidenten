from __future__ import annotations

import os
from typing import Any, Callable, TypeVar, overload

from .types import PlayerType

T = TypeVar("T")


@overload
def get_val_input(
    prompt: str,
    cast_type: Callable[[str], T],
    valid_choices: Any = None,
    delimiter: None = None,
) -> T: ...


@overload
def get_val_input(
    prompt: str,
    cast_type: Callable[[str], T],
    valid_choices: Any = None,
    delimiter: str = ...,
) -> list[T]: ...


def get_val_input(prompt, cast_type, valid_choices=None, delimiter=None):
    while True:
        try:
            raw = input(prompt).strip()
            if delimiter:
                val = [cast_type(t.strip()) for t in raw.split(delimiter) if t.strip()]
            else:
                val = cast_type(raw)

            if valid_choices is not None:
                if callable(valid_choices):
                    if not valid_choices(val):
                        raise ValueError()
                else:
                    if delimiter:
                        if not all(item in valid_choices for item in val):
                            raise ValueError()
                    elif val not in valid_choices:
                        raise ValueError()
            return val
        except ValueError:
            print("Invalid input. Please try again.")


def get_settings():
    assign_p = {}
    dmc_paths = {}

    num_players = get_val_input(
        "Number of players (4-7): ", int, valid_choices={4, 5, 6, 7}
    )
    num_rounds = get_val_input("Number of rounds: ", int)

    for p_id in range(num_players):
        prompt = (
            f"Player {p_id} - 0: Human, 1: Random, 2: Baseline, 3: ISMCTS, 4: DMC: "
        )
        raw_choice = get_val_input(prompt, int, valid_choices={0, 1, 2, 3, 4})
        assign_p[p_id] = PlayerType(raw_choice)

    has_human = PlayerType.HUMAN in assign_p.values()

    choices = {"g", "s"} if not has_human else {"s"}
    prompt = "Search or game parallelism? (g/s): " if not has_human else ""
    parallelism = (
        get_val_input(prompt, str, valid_choices=choices).lower() if prompt else "s"
    )

    dmc_count = list(assign_p.values()).count(PlayerType.DMC)
    if dmc_count > 0:
        use_best_model = get_val_input(
            f"{dmc_count} DMC Bot(s) detected. Use best model? (y/n): ",
            str,
            valid_choices={"y", "n"},
        ).lower()

        dmc_p_ids = [
            p_id for p_id, p_type in assign_p.items() if p_type == PlayerType.DMC
        ]
        BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        if use_best_model == "y":
            dmc_paths = {
                p_id: os.path.join(BACKEND_DIR, "playerTypes", "best_model_27500.pt")
                for p_id in dmc_p_ids
            }
        else:

            def val_dmc_indices(indices):
                if len(indices) != dmc_count:
                    return False
                return all(
                    os.path.isfile(
                        os.path.join(BACKEND_DIR, "snapshots", f"model_gen_{idx}.pt")
                    )
                    for idx in indices
                )

            prompt = "Enter batch indices (comma-separated, e.g., 1000,2000): "
            indices = get_val_input(prompt, int, val_dmc_indices, ",")
            paths = [
                os.path.join(BACKEND_DIR, "snapshots", f"model_gen_{idx}.pt")
                for idx in indices
            ]
            dmc_paths = dict(zip(dmc_p_ids, paths))

    return parallelism, assign_p, has_human, num_players, num_rounds, dmc_paths
