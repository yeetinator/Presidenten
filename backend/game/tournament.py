from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING

import torch

from .engine import Presidenten
from .types import PlayerType

if TYPE_CHECKING:
    from playerTypes.human import HumanPlayer
    from playerTypes.random_bot import PresidentenRandomBot
    from playerTypes.baseline_bot import PresidentenBaselineBot
    from playerTypes.ismcts_bot import PresidentenISMCTSBot
    from playerTypes.dmc_bot import PresidentenDMCBot


def create_players(assign_p: dict[int, PlayerType], iterations=400, dmc_paths=None):
    from playerTypes.human import HumanPlayer
    from playerTypes.random_bot import PresidentenRandomBot
    from playerTypes.baseline_bot import PresidentenBaselineBot
    from playerTypes.ismcts_bot import PresidentenISMCTSBot
    from playerTypes.dmc_bot import PresidentenDMCBot, PresidentenValueNet

    assigned_players: dict[
        int,
        HumanPlayer
        | PresidentenRandomBot
        | PresidentenBaselineBot
        | PresidentenISMCTSBot
        | PresidentenDMCBot,
    ] = {}
    for p_id, p_type in assign_p.items():
        if p_type == PlayerType.HUMAN:
            assigned_players[p_id] = HumanPlayer(p_id)
        elif p_type == PlayerType.RANDOM:
            assigned_players[p_id] = PresidentenRandomBot(p_id)
        elif p_type == PlayerType.BASELINE:
            assigned_players[p_id] = PresidentenBaselineBot(p_id)
        elif p_type == PlayerType.ISMCTS:
            assigned_players[p_id] = PresidentenISMCTSBot(p_id, iterations)
        elif p_type == PlayerType.DMC:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dmc_model = PresidentenValueNet().to(device)

            if dmc_paths and p_id in dmc_paths:
                snap = torch.load(dmc_paths[p_id], map_location=device)
                dmc_model.load_state_dict(snap["model_state_dict"])

            dmc_model.eval()
            assigned_players[p_id] = PresidentenDMCBot(p_id, dmc_model, device)
    return assigned_players


def play_presidenten_game(
    game_id,
    num_players,
    num_rounds,
    assigned_players: dict[
        int,
        HumanPlayer
        | PresidentenRandomBot
        | PresidentenBaselineBot
        | PresidentenISMCTSBot
        | PresidentenDMCBot,
    ],
    assign_p: dict[int, PlayerType],
    parallelism="g",
    has_human=False,
    executor=None,
):
    ismcts_ids: set[int] = {
        p_id for p_id, p_type in assign_p.items() if p_type == PlayerType.ISMCTS
    }
    env = Presidenten(players=num_players, verbose=has_human)

    for idx in range(num_rounds):
        state = env.full_reset(next_round=(idx > 0))
        if has_human:
            print(f"=== ROUND {idx+1} ===")
            print("Player Roles for this Round:")

            if idx == 0:
                role_items = sorted(env.roles.items())
            else:
                role_order = {role: i for i, role in enumerate(env._get_roles())}
                role_items = sorted(
                    env.roles.items(),
                    key=lambda item: (role_order[item[1]], item[0]),
                )

            for p_id, role in role_items:
                print(f" -> Player {p_id}: {role}")
            print("-" * 50, "\n")

        if idx > 0:
            cards_to_pass = {}
            for p_id, role in env.roles.items():
                if role != "Citizen":
                    if has_human:
                        print(f"Player {p_id} ({role}) is choosing cards...")
                    cards_to_pass[p_id] = assigned_players[p_id].choose_cards_to_pass(
                        env._get_state(p_id)
                    )

            env.exchange_cards(cards_to_pass)
            state = env._get_state(env.curr_turn)

        while not env.game_over:
            curr_p_id = env.curr_turn
            if curr_p_id is None:
                break
            curr_p_type = assigned_players[curr_p_id]

            if curr_p_id in ismcts_ids:
                assert curr_p_type.__class__.__name__ == "PresidentenISMCTSBot"
                chosen_move = curr_p_type.get_move(
                    state,
                    env,
                    executor,
                    parallelism,
                )
            else:
                chosen_move = curr_p_type.get_move(state, env)

            if has_human and not state["is_finish_prompt"]:
                p_name = assign_p[curr_p_id].name if assign_p[curr_p_id] else "Unknown"
                print(
                    f"\nPlayer {curr_p_id} ({state['my_role']}, {p_name}) chose: "
                    f"{Presidenten.visualize_move(chosen_move)}\n"
                )
                if curr_p_type.__class__.__name__ != "HumanPlayer":
                    input("Press Enter to continue...\n")

            state, _ = env.step(curr_p_id, chosen_move)
            if env.was_pile_reset:
                env.clear_pile()

        env.assign_roles()
        if has_human:
            print(
                f"Round {idx+1} Complete! Finishing Order: {env.out_order}. "
                f"Players who finished with a 2: {env.ended_2}. Scores: {env.scores}\n"
            )
            input("Press Enter to continue...\n")
    return env.scores


worker_players: (
    dict[
        int,
        HumanPlayer
        | PresidentenRandomBot
        | PresidentenBaselineBot
        | PresidentenISMCTSBot
        | PresidentenDMCBot,
    ]
    | None
) = None


def init_worker(assign_p, iterations, dmc_paths):
    global worker_players
    worker_players = create_players(assign_p, iterations, dmc_paths)


def worker_game_task(game_id, num_players, num_rounds, assign_p):
    global worker_players
    assert worker_players is not None, "Worker players not initialized"
    return play_presidenten_game(
        game_id,
        num_players,
        num_rounds,
        worker_players,
        assign_p,
        "g",
    )


def update_final_scores(master_scores, round_scores):
    for p_id in master_scores:
        master_scores[p_id] = (
            master_scores[p_id][0] + round_scores[p_id][0],
            master_scores[p_id][1] + round_scores[p_id][1],
        )
    return master_scores


def game_parallelism(
    assign_p,
    num_players,
    num_rounds,
    dmc_paths,
    total_games,
    num_workers,
):
    print(f"Starting Tournament: {total_games} games, {num_rounds} rounds each.")
    print(f"Deploying across {num_workers} parallel game workers...\n")

    master_scores = {p_id: (0, 0) for p_id in range(num_players)}
    iters = 400

    with ProcessPoolExecutor(
        num_workers, initializer=init_worker, initargs=(assign_p, iters, dmc_paths)
    ) as executor:
        futures = [
            executor.submit(
                worker_game_task,
                idx,
                num_players,
                num_rounds,
                assign_p,
            )
            for idx in range(total_games)
        ]
        for i, f in enumerate(futures):
            master_scores = update_final_scores(master_scores, f.result())
            print(f" -> Game {i+1}/{total_games} finished processing.")
    return master_scores


def search_parallelism(
    assign_p,
    has_human,
    num_players,
    num_rounds,
    dmc_paths,
    total_games,
    num_workers,
):
    master_scores = {p_id: (0, 0) for p_id in range(num_players)}
    iters = 1200 + 200 * (num_players - 4)
    assigned_players = create_players(assign_p, iters, dmc_paths)

    with ProcessPoolExecutor(num_workers) as shared_executor:
        for idx in range(total_games):
            if idx % 10 == 0:
                print(f"\n=== GAME {idx+1} ===\n")
            round_scores = play_presidenten_game(
                idx,
                num_players,
                num_rounds,
                assigned_players,
                assign_p,
                "s",
                has_human,
                shared_executor,
            )
            master_scores = update_final_scores(master_scores, round_scores)
    return master_scores


def print_scores(
    scores, assign_p: dict[int, PlayerType], num_players, num_rounds, total_games
):
    print("\n" + "=" * 60)
    print(f"=== FINAL SCORES: {total_games} Games | {num_rounds} Rounds Each ===")
    print("=" * 60)

    for p_id in sorted(scores, key=lambda x: scores[x][0], reverse=True):
        p_name = assign_p[p_id].name if assign_p[p_id] else "Unknown"
        avg_finish_pos = num_players - (scores[p_id][0] / (total_games * num_rounds))
        win_rate = scores[p_id][1] / (total_games * num_rounds) * 100
        avg_norm_score = (
            scores[p_id][0] / (total_games * num_rounds * (num_players - 1))
        ) * 2 - 1

        print(
            f"Player {p_id} ({p_name}): "
            f"Average Finishing Position: {avg_finish_pos:.2f} | "
            f"Win Rate: {win_rate:.2f}% | "
            f"Average Normalized Score: {avg_norm_score:.2f}"
        )
    print("=" * 60)
