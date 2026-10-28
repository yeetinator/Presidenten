import random
import torch
import numpy as np
from game import President
from playerTypes.player import Player
from playerTypes.dmc_bot import PresidentDMCBot, INPUT_DIM
from .common import NUM_ROUNDS

EVAL_CONCURRENCY = 128
EVAL_MIN_BATCH = 4  # park a model's requests until it has this many...
EVAL_MAX_WAIT = 8  # ...or a game has waited this many ticks


def play_round_gen(env: President, bot_instances: dict[int, Player], round_idx: int):
    state = env.full_reset(round_idx > 0)
    if round_idx > 0:
        cards_to_pass = {}
        for p_id, role in env.roles.items():
            # Low roles' picks are ignored by the exchange, but bots like ISMCTS
            # note what they handed over, as in the web game
            if role != "Citizen":
                bot = bot_instances[p_id]
                p_state = env._get_state(p_id)

                # Bots with *_gen methods hand their forwards to the caller to batch
                if hasattr(bot, "choose_cards_to_pass_gen"):
                    cards_to_pass[p_id] = yield from bot.choose_cards_to_pass_gen(
                        p_state
                    )
                else:
                    cards_to_pass[p_id] = bot.choose_cards_to_pass(p_state)

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

        bot = bot_instances[curr_player]
        if hasattr(bot, "get_move_gen"):
            chosen_move = yield from bot.get_move_gen(state)
        elif getattr(bot, "needs_env", False):
            chosen_move = bot.get_move(state, env)
        else:
            chosen_move = bot.get_move(state)
        state, _ = env.step(curr_player, chosen_move)

        if env.was_pile_reset:
            env.clear_pile()
    env.assign_roles()
    return completed


def _empty(input_dim):
    return (
        np.empty((0, input_dim), dtype=np.float32),
        np.empty((0,), dtype=np.int64),
        np.empty((0, 1), dtype=np.float32),
    )


def game_loop_gen(
    num_players,
    bot_instances: dict[int, Player],
    input_dim=INPUT_DIM,
    bot_class=PresidentDMCBot,
    lightweight=False,
):
    env = President(num_players, lightweight=lightweight)
    game_x, game_a, game_y = [], [], []

    for round_idx in range(NUM_ROUNDS):
        completed = yield from play_round_gen(env, bot_instances, round_idx)
        if not completed:
            for bot in bot_instances.values():
                if isinstance(bot, PresidentDMCBot):
                    bot.trajectory.clear()
            return _empty(input_dim)

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
        return _empty(input_dim)
    return (
        np.array(game_x, dtype=np.float32),
        np.array(game_a, dtype=np.int64),
        np.array(game_y, dtype=np.float32),
    )


def seeded_game_gen(num_players, bot_instances, round_seeds):
    """Play len(round_seeds) rounds in one env, seeding the deal of each round.

    Returns (scores, out_orders): env.scores after the last round and the
    finishing order of every round.
    """

    env = President(num_players, lightweight=True)
    out_orders = []
    for round_idx, round_seed in enumerate(round_seeds):
        random.seed(round_seed)
        yield from play_round_gen(env, bot_instances, round_idx)
        out_orders.append(list(env.out_order))
    return dict(env.scores), out_orders


def seed_deal(env: President, seed):
    """Make the next deal depend on seed alone. The deck is shuffled in place, so
    it is put back in order first, or the deal would depend on earlier rounds."""
    env.deck.sort()
    random.seed(seed)


def replay_round_gen(num_players, bot_instances, round_idx, roles, seed):
    """Play one round of a logged game again from its start: the deal of seed
    and, after the first round, the roles before the card exchange.

    Returns the finishing order."""

    env = President(num_players, lightweight=True)
    if round_idx > 0:
        env.roles = {int(p_id): role for p_id, role in roles.items()}
        env.first_turn = False
    seed_deal(env, seed)
    yield from play_round_gen(env, bot_instances, round_idx)
    return list(env.out_order)


def _tagged(tag, game):
    result = yield from game
    return tag, result


def run_tagged_games(
    games,
    concurrency=EVAL_CONCURRENCY,
    min_batch=EVAL_MIN_BATCH,
    max_wait=EVAL_MAX_WAIT,
):
    """games: list of (tag, game generator). Returns {tag: game result}."""
    queue = iter([_tagged(tag, game) for tag, game in games])
    return dict(
        iter_batched(
            lambda: next(queue), len(games), concurrency, None, min_batch, max_wait
        )
    )


def iter_batched(
    make_game,
    n_games,
    concurrency,
    should_stop=None,
    min_batch=1,
    max_wait=0,
    device="cpu",
):
    results = []
    started = 0
    pending = []  # list of [gen, model, x, ticks_waited]

    def start_games():
        nonlocal started
        if should_stop is not None and should_stop():
            pending.clear()
            return

        while started < n_games and len(pending) < concurrency:
            gen = make_game()
            started += 1

            try:
                model, x = next(gen)
                pending.append([gen, model, x, 0])
            except StopIteration as stop:
                results.append(stop.value)

    start_games()
    yield from results
    results.clear()

    while pending:
        # Group requests by model -> one forward per distinct model
        groups = {}
        for slot, (_, model, x, waited) in enumerate(pending):
            entry = groups.get(id(model))
            if entry is None:
                groups[id(model)] = [model, [slot], [x], waited]
            else:
                entry[1].append(slot)
                entry[2].append(x)
                entry[3] = max(entry[3], waited)

        run = list(groups.values())
        if min_batch > 1 and len(run) > 1:
            biggest = max(run, key=lambda g: len(g[1]))
            run = [
                g
                for g in run
                if g is biggest or len(g[1]) >= min_batch or g[3] >= max_wait
            ]

        replies = [None] * len(pending)
        with torch.inference_mode():
            for model, slots, xs, _ in run:
                batch = xs[0] if len(xs) == 1 else np.concatenate(xs)
                q = model(torch.from_numpy(batch).to(device)).cpu().numpy()
                offset = 0

                for slot, x in zip(slots, xs):
                    rows = x.shape[0]
                    replies[slot] = q[offset : offset + rows]
                    offset += rows

        still_pending = []
        for entry, q in zip(pending, replies):
            if q is None:  # parked this tick
                entry[3] += 1
                still_pending.append(entry)
                continue
            try:
                model, x = entry[0].send(q)
                still_pending.append([entry[0], model, x, 0])
            except StopIteration as stop:
                results.append(stop.value)

        pending = still_pending
        yield from results
        results.clear()
        start_games()
        yield from results
        results.clear()
