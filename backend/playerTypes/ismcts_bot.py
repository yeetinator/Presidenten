import math
import random
import time
from collections import Counter
from concurrent.futures import Executor, ProcessPoolExecutor
from game import President
from .player import Player
from .baseline_bot import PresidentBaselineBot

PASS = (0, 0, 0)
HIGH_ROLES = {"President", "Vice-President", "Secretary"}
# Jump-ins are decided quickly: the web game can hold up a human's move for them
FINISH_TIME_LIMIT = 0.25


class ISMCTSNode:
    __slots__ = ("children", "visits", "score", "avail")

    def __init__(self):
        self.children: dict[tuple[int, tuple[int, int, int]], ISMCTSNode] = {}
        self.visits = 0
        self.score = 0.0
        self.avail = 0  # Times this move was legal when its parent was selected from


def _no_state(p_id, reset_view=False):
    # Shadows President._get_turn_state on simulation copies, so step() skips
    # building a state dict nobody reads
    return None


def _candidate_moves(legal_moves):
    # Using more 2s as wildcards than needed for a (card, count) only wastes 2s
    best = {}
    for move in legal_moves:
        key = (move[0], move[1])
        if key not in best or move[2] < best[key][2]:
            best[key] = move
    return list(best.values())


class ISMCTSSearch:
    def __init__(
        self,
        real_env: President,
        player_id: int,
        hand: list[int],
        opp_hand_counts: dict[int, int],
        guaranteed: dict[int, list[int]],
        ceilings: dict[int, tuple[int, set[int]]],
        exploration=0.7,
        miss_chance=0.01,
        burn_in=100,
        thinning=2,
        seed=None,
    ):
        self.env = real_env
        self.me = player_id
        self.players = real_env.players
        self.exploration = exploration
        self.rng = random.Random(seed)
        self.policy = PresidentBaselineBot(player_id)
        self.my_hand = list(hand)

        hidden = [0] * 16
        for card in range(3, 16):
            hidden[card] = 4 - real_env._history_flat_counts[card]
        for card in hand:
            hidden[card] -= 1

        self.need = {p: count for p, count in opp_hand_counts.items() if count > 0}
        self.guaranteed: dict[int, list[int]] = {}

        for p, cards in guaranteed.items():
            if p not in self.need:
                continue
            kept = []
            for card in cards:
                if hidden[card] > 0 and len(kept) < self.need[p]:
                    hidden[card] -= 1
                    kept.append(card)
            self.guaranteed[p] = kept
        # Ranks the exchange partner can hold besides the guaranteed cards: what we
        # gave them, and nothing above the lowest card they passed us
        self.eligible = {
            p: {card for card in range(3, 16) if card <= ceiling or card in gave}
            for p, (ceiling, gave) in ceilings.items()
            if p in self.need
        }
        self.pool = [card for card in range(3, 16) for _ in range(hidden[card])]
        self.determinizations: list[dict[int, list[int]]] = []

        self.template = dict(real_env.__dict__)
        self.template.update(
            verbose=False,
            track_suits=False,
            lightweight=True,
            _get_turn_state=_no_state,
        )

        # Opponents' moves this round tell us about their hands: deals are drawn
        # from a Markov chain that favours deals under which the policy would
        # have made the same moves
        self.history = real_env.history[:]
        self.played_by: dict[int, list[int]] = {p: [] for p in range(self.players)}
        for p, (card, count, twos) in self.history:
            self.played_by[p] += [card] * (count - twos) + [15] * twos

        self.fixed = {p: len(self.guaranteed.get(p, ())) for p in self.need}
        self.swappable = [p for p, count in self.need.items() if count > self.fixed[p]]
        self.infer = bool(self.history) and len(self.swappable) > 1
        self.log_miss = math.log(miss_chance)
        self.burn_in = burn_in
        self.thinning = thinning
        self.chain: dict[int, list[int]] | None = None
        self.chain_misses = 0
        # Written to by replays, never read
        self.scratch_passed = [[0] * 16 for _ in range(self.players)]

    def _determinize(self):
        if not self.infer:
            return self._sample()

        steps = self.thinning
        if self.chain is None:
            self.chain = self._sample()
            self.chain_misses = self._misses(self.chain)
            steps = self.burn_in

        for _ in range(steps):
            self._chain_step(self.chain)
        return {p: hand[:] for p, hand in self.chain.items()}

    def _chain_step(self, hands: dict[int, list[int]]):
        """Metropolis step: swap a card between two opponents, keep the swap with
        probability miss_chance ** (extra moves it fails to explain)."""
        rng = self.rng
        q, r = rng.sample(self.swappable, 2)
        i = rng.randrange(self.fixed[q], len(hands[q]))
        j = rng.randrange(self.fixed[r], len(hands[r]))
        a, b = hands[q][i], hands[r][j]

        if a == b or (
            (q in self.eligible and b not in self.eligible[q])
            or (r in self.eligible and a not in self.eligible[r])
        ):
            return

        limit = self.chain_misses + math.log(1.0 - rng.random()) / self.log_miss
        hands[q][i], hands[r][j] = b, a
        misses = self._misses(hands, limit)

        if misses <= limit:
            self.chain_misses = misses
        else:
            hands[q][i], hands[r][j] = a, b

    def _misses(self, hands: dict[int, list[int]], limit=math.inf):
        """Replays this round as if dealt `hands` and counts the opponents' moves the
        policy would not have made, stopping once there are more than `limit`."""
        me = self.me
        sim = President.__new__(President)
        sim.__dict__.update(self.template)
        sim.hands = {p: hands[p] + self.played_by[p] for p in range(self.players)}
        sim.history = []
        sim._history_flat_counts = [0] * 16
        sim._passed_by_player = sim._passed_multi_by_player = self.scratch_passed
        sim.playing = set(range(self.players))
        sim.out_order = []
        sim.ended_2 = []
        sim.game_over = False
        sim.first_turn = self.env.round == 1
        sim.clear_pile()
        misses = 0

        def decline_finishes(until):
            # Declining a jump-in isn't recorded, so whoever the sampled hands give
            # one to must have passed on it
            nonlocal misses
            while sim.pending_finish:
                queue = sim.pending_finish["queue"]
                if until(queue):
                    return
                finisher = queue[0][2]
                if finisher != me and (
                    self._policy_move(sim, finisher, sim.get_legal_moves(finisher))
                    != PASS
                ):
                    misses += 1
                self._play(sim, finisher, PASS)

        try:
            for p_id, move in self.history:
                decline_finishes(
                    lambda queue: queue[0][2] == p_id
                    and (move[0], move[1], p_id) in queue
                )
                if p_id != me and (
                    self._policy_move(sim, p_id, sim.get_legal_moves(p_id)) != move
                ):
                    misses += 1
                if misses > limit:
                    return misses
                self._play(sim, p_id, move)
        except ValueError:
            return math.inf  # The sampled hands can't have made these moves

        pending = self.env.pending_finish
        if not pending or pending["queue"][0][2] == me:
            decline_finishes(lambda queue: queue[0][2] == me)
        return misses

    def _sample(self):
        pool = self.pool[:]
        self.rng.shuffle(pool)
        hands = {p: [] for p in range(self.players)}
        hands[self.me] = self.my_hand

        for p, ranks in self.eligible.items():
            hand = hands[p] = self.guaranteed.get(p, [])[:]
            needed = self.need[p] - len(hand)
            rest = []

            for card in pool:
                if needed and card in ranks:
                    hand.append(card)
                    needed -= 1
                else:
                    rest.append(card)
            # The constraint can't be met, so fill up with whatever is left
            hand.extend(rest[:needed])
            pool = rest[needed:]

        i = 0
        for p, count in self.need.items():
            if p in self.eligible:
                continue
            hand = hands[p] = self.guaranteed.get(p, [])[:]
            needed = count - len(hand)
            hand.extend(pool[i : i + needed])
            i += needed
        return hands

    def _clone(self, hands: dict[int, list[int]]) -> President:
        env = self.env
        sim = President.__new__(President)
        sim.__dict__.update(self.template)
        sim.hands = {p: hand[:] for p, hand in hands.items()}
        sim.history = []
        sim._history_flat_counts = env._history_flat_counts[:]
        sim._passed_by_player = [row[:] for row in env._passed_by_player]
        sim._passed_multi_by_player = [row[:] for row in env._passed_multi_by_player]
        sim.playing = set(env.playing)
        sim.passed = set(env.passed)
        sim.out_order = env.out_order[:]
        sim.ended_2 = env.ended_2[:]
        sim.pile = env.pile[:]

        pending = env.pending_finish
        if pending:
            queue = pending["queue"]
            if queue[0][2] == self.me:
                # Only our own jump-in is known, so who else could finish is
                # worked out from the sampled hands
                card_val, count, twos = env.last_move
                options = [
                    option
                    for option in (
                        sim._finishing_option(card_val, count - twos, env.pile_leader),
                        (
                            sim._finishing_option(15, twos, env.pile_leader)
                            if twos
                            else None
                        ),
                    )
                    if option and option[0] >= queue[0][0]
                ]
                queue = sorted(options)
            sim.pending_finish = {**pending, "queue": list(queue)}
            # The web game can ask for our normal move while someone else may
            # still jump in; playing lets their chance lapse
            while sim.pending_finish and sim.pending_finish["queue"][0][2] != self.me:
                self._play(sim, sim.pending_finish["queue"][0][2], PASS)
        return sim

    @staticmethod
    def _play(sim: President, p_id, move):
        sim.step(p_id, move)
        if sim.was_pile_reset:
            sim.clear_pile()

    def _policy_move(self, sim: President, p_id, legal_moves):
        if len(legal_moves) == 1:
            return legal_moves[0]

        pending = sim.pending_finish
        hands = sim.hands
        return self.policy.get_move(
            {
                "hand": hands[p_id],
                "legal_moves": legal_moves,
                "opp_hand_counts": {p: len(h) for p, h in hands.items() if p != p_id},
                "is_finish_prompt": bool(pending and pending["queue"][0][2] == p_id),
                "last_move": sim.last_move,
                "first_turn": sim.first_turn,
                "history_vector": sim._history_flat_counts[3:],
            }
        )

    def _reward(self, sim: President):
        order = sim.out_order + [
            p for p in range(self.players) if p not in sim.out_order
        ]
        for p in reversed(sim.ended_2):
            order.remove(p)
            order.append(p)
        return (self.players - 1 - order.index(self.me)) / (self.players - 1)

    def _select(self, node: ISMCTSNode, moves, ordered=False):
        """UCB over our moves; returns (move, child, expanded)."""
        me = self.me
        children = node.children
        untried = []

        for move in moves:
            child = children.get((me, move))
            if child is None:
                untried.append(move)
            else:
                child.avail += 1

        if untried:
            move = untried[0] if ordered else self.rng.choice(untried)
            child = children[(me, move)] = ISMCTSNode()
            child.avail = 1
            return move, child, True

        def ucb(move):
            child = children[(me, move)]
            return child.score / child.visits + self.exploration * math.sqrt(
                math.log(child.avail) / child.visits
            )

        move = max(moves, key=ucb)
        return move, children[(me, move)], False

    def _iterate(self, root: ISMCTSNode, root_moves):
        me = self.me
        move, node, expanded = self._select(root, root_moves, ordered=True)
        path = [root, node]

        # The n-th visit of every root move uses the n-th determinization, so the
        # moves are compared on the same deals (common random numbers)
        idx = node.visits
        while len(self.determinizations) <= idx:
            self.determinizations.append(self._determinize())
        sim = self._clone(self.determinizations[idx])
        self._play(sim, me, move)

        while not sim.game_over:
            if me in sim.out_order and me not in sim.ended_2:
                break  # Our rank can't change anymore

            p_id = sim.curr_turn
            if p_id is None:
                break
            legal_moves = sim.get_legal_moves(p_id)
            if not legal_moves:
                break

            if expanded:
                move = self._policy_move(sim, p_id, legal_moves)
            elif p_id == me:
                move, node, expanded = self._select(node, _candidate_moves(legal_moves))
                path.append(node)
            else:
                move = self._policy_move(sim, p_id, legal_moves)
                pending = sim.pending_finish
                # Declining a jump-in happens out of sight of the other players
                if not (move == PASS and pending and pending["queue"][0][2] == p_id):
                    child = node.children.get((p_id, move))
                    if child is None:
                        child = node.children[(p_id, move)] = ISMCTSNode()
                    node = child
                    path.append(node)
            self._play(sim, p_id, move)

        reward = self._reward(sim)
        for node in path:
            node.visits += 1
            node.score += reward

    def run(self, root_moves, time_limit, max_iterations=None):
        root = ISMCTSNode()
        deadline = time.perf_counter() + time_limit
        iterations = 0
        while (max_iterations is None or iterations < max_iterations) and (
            time.perf_counter() < deadline
        ):
            self._iterate(root, root_moves)
            iterations += 1

        return {
            move: (child.visits, child.score)
            for (_, move), child in root.children.items()
        }


def _execute_search(
    search_args: dict, root_moves, time_limit, max_iterations, seed=None
):
    search = ISMCTSSearch(**search_args, seed=seed)
    return search.run(root_moves, time_limit, max_iterations)


class PresidentISMCTSBot(Player):
    needs_env = True  # Searches from the real game, not just its own view of it

    def __init__(self, player_id, iterations=None, time_limit=1.0):
        self.player_id = player_id
        self.iterations = iterations  # Optional cap on top of the time limit
        self.time_limit = time_limit
        self.baseline = PresidentBaselineBot(player_id)
        self._pending_exchange = None  # (role, hand before exchange, cards given)
        self._exchange = None  # What the last exchange revealed about the partner

    def get_move(
        self,
        state: dict,
        real_env: President,
        executor: Executor | None = None,
        parallelism="g",
        num_workers=10,
    ) -> tuple[int, int, int]:
        start = time.perf_counter()
        legal_moves = state["legal_moves"]
        if not legal_moves:
            return PASS
        if len(legal_moves) == 1:
            return legal_moves[0]

        # Try the baseline's move first, then the rest in its order of preference
        candidates = _candidate_moves(legal_moves)
        preference = [self.baseline.get_move(state)] + self.baseline.get_ranked_moves(
            state
        )
        rank = {}

        for move in preference:
            rank.setdefault(move, len(rank))

        candidates.sort(key=lambda move: rank.get(move, len(rank)))
        if len(candidates) == 1:
            return candidates[0]

        guaranteed, ceilings = self._exchange_constraints(state, real_env)
        search_args = {
            "real_env": real_env,
            "player_id": self.player_id,
            "hand": state["hand"],
            "opp_hand_counts": state["opp_hand_counts"],
            "guaranteed": guaranteed,
            "ceilings": ceilings,
        }
        time_limit = self.time_limit

        if state["is_finish_prompt"]:
            time_limit = min(time_limit, FINISH_TIME_LIMIT)
        time_left = time_limit - (time.perf_counter() - start)

        if parallelism == "s" and num_workers > 1:
            # Leave room for sending the game to the workers and back
            time_left = max(0.01, time_left - 0.1)
            cap = self.iterations and max(1, self.iterations // num_workers)
            if executor is None:
                with ProcessPoolExecutor(max_workers=num_workers) as local_executor:
                    results = self._run_workers(
                        local_executor,
                        search_args,
                        candidates,
                        time_left,
                        cap,
                        num_workers,
                    )
            else:
                results = self._run_workers(
                    executor, search_args, candidates, time_left, cap, num_workers
                )
        else:
            # The last iteration can run a little past the deadline
            time_left = max(0.01, time_left - 0.02)
            results = [
                _execute_search(search_args, candidates, time_left, self.iterations)
            ]

        total_stats: dict[tuple[int, int, int], list[float]] = {}
        for worker_stats in results:
            for move, (visits, score) in worker_stats.items():
                stats = total_stats.setdefault(move, [0, 0.0])
                stats[0] += visits
                stats[1] += score

        if not total_stats:
            return candidates[0]
        # Most visited move, ties broken by the higher average score
        return max(
            total_stats,
            key=lambda move: (
                total_stats[move][0],
                total_stats[move][1] / max(1, total_stats[move][0]),
            ),
        )

    def _run_workers(
        self, executor: Executor, search_args, candidates, time_limit, cap, num_workers
    ):
        futures = [
            executor.submit(_execute_search, search_args, candidates, time_limit, cap)
            for _ in range(num_workers)
        ]
        return [future.result() for future in futures]

    def _exchange_constraints(self, state: dict, real_env: President):
        """Cards the exchange partner must still hold, and if they passed us their
        highest cards, the lowest of those with the ranks we gave them."""
        hand = Counter(state["hand"])

        if self._pending_exchange is not None:
            role, before, gave = self._pending_exchange
            self._pending_exchange = None
            kept = Counter(before) - Counter(gave)
            # We can't have played yet, so our hand is what the exchange left us
            if (
                role == state["my_role"]
                and sum(hand.values()) == len(before)
                and not kept - hand
            ):
                partner = self._exchange_partner(state)
                if partner is not None:
                    self._exchange = {
                        "round": real_env.round,
                        "partner": partner,
                        "high": role in HIGH_ROLES,
                        "gave": Counter(gave),
                        "received": list((hand - kept).elements()),
                        "start_hand": hand,
                    }

        info = self._exchange
        # Our hand only shrinks during a round, anything else is a stale exchange
        if info is None or info["round"] != real_env.round or hand - info["start_hand"]:
            return {}, {}

        played = real_env._history_flat_counts
        we_played = info["start_hand"] - hand
        partner = info["partner"]
        guaranteed = {
            partner: [
                card
                for card, count in info["gave"].items()
                for _ in range(max(0, count - (played[card] - we_played[card])))
            ]
        }
        ceilings = {}
        if info["high"] and info["received"]:
            ceilings[partner] = (min(info["received"]), set(info["gave"]))
        return guaranteed, ceilings

    def _exchange_partner(self, state: dict):
        role = state["my_role"]
        for high_role, low_role, _ in state["role_pairs"]:
            partner_role = (
                low_role
                if role == high_role
                else high_role if role == low_role else None
            )
            if partner_role is not None:
                return next(
                    (p for p, r in state["player_roles"].items() if r == partner_role),
                    None,
                )
        return None

    def choose_cards_to_pass(self, state: dict) -> list[int]:
        role = state["my_role"]
        self._exchange = None
        count = next(
            (
                c
                for high_role, low_role, c in state["role_pairs"]
                if role in (high_role, low_role)
            ),
            0,
        )
        if not count:
            self._pending_exchange = None
            return []

        if role in HIGH_ROLES:
            cards = self.baseline.choose_cards_to_pass(state)
            self._pending_exchange = (role, list(state["hand"]), list(cards))
            return cards

        # Low roles have to pass their highest cards
        self._pending_exchange = (
            role,
            list(state["hand"]),
            sorted(state["hand"], reverse=True)[:count],
        )
        return []
