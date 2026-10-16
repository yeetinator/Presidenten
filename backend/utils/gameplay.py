import numpy as np
import random
from game import President
from playerTypes.player import Player
from playerTypes.dmc_bot import PresidentDMCBot, PresidentValueNet, INPUT_DIM
from .common import NUM_ROUNDS


def play_round(env: President, bot_instances: dict[int, Player], round_idx: int):
    state = env.full_reset(round_idx > 0)
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

    completed = True
    move_count = 0
    while not env.game_over:
        move_count += 1
        if move_count > 500:
            completed = False
            break

        curr_player = env.curr_turn
        if curr_player is None:
            break

        chosen_move = bot_instances[curr_player].get_move(state, env)
        state, _ = env.step(curr_player, chosen_move)

        if env.was_pile_reset:
            env.clear_pile()
    env.assign_roles()
    return completed


def game_loop(
    num_players,
    bot_instances: dict[int, Player],
    input_dim=INPUT_DIM,
    bot_class=PresidentDMCBot,
):
    env = President(num_players)
    game_x, game_a, game_y = [], [], []

    def _empty():
        return (
            np.empty((0, input_dim), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0, 1), dtype=np.float32),
        )

    for round_idx in range(NUM_ROUNDS):
        if not play_round(env, bot_instances, round_idx):
            for bot in bot_instances.values():
                if isinstance(bot, PresidentDMCBot):
                    bot.trajectory.clear()
            return _empty()

        max_possible_score = env.players - 1
        gamma = 0.997

        for rank, p_id in enumerate(env.out_order):
            bot = bot_instances[p_id]
            if isinstance(bot, bot_class):
                if bot.record and len(bot.trajectory) > 0:
                    round_score = max_possible_score - rank
                    normalized_score = (round_score / max_possible_score) * 2 - 1

                    for i, (features, action_idx) in enumerate(
                        reversed(bot.trajectory)
                    ):
                        discounted_score = normalized_score * (gamma**i)
                        game_x.append(features)
                        game_a.append(action_idx)
                        game_y.append([discounted_score])
                bot.trajectory.clear()
    if not game_x:
        return _empty()
    return (
        np.array(game_x, dtype=np.float32),
        np.array(game_a, dtype=np.int64),
        np.array(game_y, dtype=np.float32),
    )


def eval_game_loop(
    num_players,
    match_seed,
    bot_instances: dict[int, Player],
    seat_assignments,
    rotation,
    slot_norm_scores,
):
    env = President(num_players)
    for round_idx in range(NUM_ROUNDS):
        round_seed = (match_seed * 1000 + round_idx) % (2**32 - 1)
        random.seed(round_seed)
        np.random.seed(round_seed)
        play_round(env, bot_instances, round_idx)

    max_pos_score = (num_players - 1) * NUM_ROUNDS
    for seat, key in enumerate(seat_assignments):
        raw_score = env.scores[seat][0]
        norm_score = (raw_score / max_pos_score) * 2 - 1
        original_slot = (seat + rotation) % num_players
        slot_norm_scores[original_slot] += norm_score / num_players
    return slot_norm_scores
