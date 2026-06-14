import math
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from game import Presidenten
from playerTypes.baseline_bot import PresidentenBaselineBot


class ISMCTSNode:
    def __init__(self, move=None, parent=None, player_id=None):
        self.move = move
        self.parent = parent
        self.player_id = player_id
        self.children = []
        self.visits = 0
        self.score = 0.0

    def select_child(self, legal_moves_set, player_id, exploration_weight=1.41):
        if self.visits == 0:
            return random.choice(
                [
                    child
                    for child in self.children
                    if child.player_id == player_id and child.move in legal_moves_set
                ]
            )

        log_total = math.log(self.visits)
        best_score = -float("inf")
        best_child = None

        for child in self.children:
            if child.player_id == player_id and child.move in legal_moves_set:
                score = (child.score / child.visits) + exploration_weight * math.sqrt(
                    log_total / child.visits
                )
                if score > best_score:
                    best_score = score
                    best_child = child
        return best_child


def _execute_mcts_batch(player_id, iterations, real_env):
    bot = PresidentenISMCTSBot(player_id, iterations)
    return bot.run_search_batch(real_env)


class PresidentenISMCTSBot:
    def __init__(self, player_id, iterations=200):
        self.player_id = player_id
        self.iterations = iterations

    def get_move(
        self,
        state: dict,
        real_env=None,
        executor=None,
        num_workers=4,
        parallelism="g",
    ):
        legal_moves = state["legal_moves"]
        total_stats = {}

        if not legal_moves:
            return (0, 0, 0)
        if len(legal_moves) == 1:
            return legal_moves[0]
        if state["last_move"] == (0, 0, 0) and state["first_turn"]:
            return PresidentenBaselineBot(player_id=self.player_id).get_ranked_moves(
                state
            )[0]

        if parallelism == "s":
            iterations_per_worker = max(1, self.iterations // num_workers)

            if num_workers == 1:
                results = [self.run_search_batch(real_env)]
            elif executor is not None:
                futures = [
                    executor.submit(
                        _execute_mcts_batch,
                        self.player_id,
                        iterations_per_worker,
                        real_env,
                    )
                    for _ in range(num_workers)
                ]
                results = [future.result() for future in futures]
            else:
                with ProcessPoolExecutor(max_workers=num_workers) as local_executor:
                    futures = [
                        local_executor.submit(
                            _execute_mcts_batch,
                            self.player_id,
                            iterations_per_worker,
                            real_env,
                        )
                        for _ in range(num_workers)
                    ]
                    results = [future.result() for future in futures]

            for worker_stats in results:
                for move, stats in worker_stats.items():
                    if move not in total_stats:
                        total_stats[move] = {"visits": 0, "score": 0.0}
                    total_stats[move]["visits"] += stats["visits"]
                    total_stats[move]["score"] += stats["score"]
        else:
            total_stats = self.run_search_batch(real_env)

        legal_root_moves = {
            move: total_stats[move] for move in total_stats if move in legal_moves
        }
        if not legal_root_moves:
            return random.choice(legal_moves)

        return max(
            legal_root_moves,
            key=lambda move: legal_root_moves[move]["visits"],
        )

    def run_search_batch(self, real_env):
        root = ISMCTSNode()
        rollout_bots = {
            p: PresidentenBaselineBot(player_id=p) for p in range(real_env.players)
        }
        known_cards = list(real_env.hands[self.player_id])
        for _, move in real_env.history:
            if move != (0, 0, 0):
                card_val, count, twos = move
                known_cards.extend([card_val] * (count - twos))
                known_cards.extend([15] * twos)

        known_counts = Counter(known_cards)
        base_hidden_pool = []

        for card_val in range(3, 16):
            unaccounted = 4 - known_counts[card_val]
            if unaccounted > 0:
                base_hidden_pool.extend([card_val] * unaccounted)

        pending_finish_cards = {p: [] for p in range(real_env.players)}
        if real_env.pending_finish:
            for card, count, p in real_env.pending_finish["queue"]:
                if p != self.player_id:
                    pending_finish_cards[p].extend([card] * count)
                    for _ in range(count):
                        if card in base_hidden_pool:
                            base_hidden_pool.remove(card)

        for _ in range(self.iterations):
            sim_env = self._determinize_environment(
                real_env, base_hidden_pool, pending_finish_cards
            )
            curr_node = root

            while not sim_env.game_over:
                curr_player = sim_env.curr_turn
                sim_legal_moves = sim_env.get_legal_moves(curr_player)

                if not sim_legal_moves:
                    break

                tried_moves_set = {
                    child.move
                    for child in curr_node.children
                    if child.player_id == curr_player
                }
                untried_moves = [m for m in sim_legal_moves if m not in tried_moves_set]

                if untried_moves:
                    sim_state = sim_env._get_state(curr_player)
                    ranked_moves = rollout_bots[curr_player].get_ranked_moves(sim_state)
                    chosen_move = next(
                        (m for m in ranked_moves if m in untried_moves),
                        random.choice(untried_moves),
                    )

                    new_node = ISMCTSNode(
                        move=chosen_move,
                        parent=curr_node,
                        player_id=curr_player,
                    )
                    curr_node.children.append(new_node)
                    curr_node = new_node
                    sim_env.step(curr_player, chosen_move)
                    break
                else:
                    next_node = curr_node.select_child(
                        set(sim_legal_moves), curr_player
                    )
                    if next_node is None:
                        break

                    curr_node = next_node
                    sim_env.step(curr_player, curr_node.move)

            while not sim_env.game_over:
                curr_player = sim_env.curr_turn
                sim_legal_moves = sim_env.get_legal_moves(curr_player)

                if not sim_legal_moves:
                    break

                sim_state = sim_env._get_state(curr_player)
                chosen_move = rollout_bots[curr_player].get_move(sim_state)
                sim_env.step(curr_player, chosen_move)

            for p in range(sim_env.players):
                if p not in sim_env.out_order:
                    sim_env.out_order.append(p)

            node = curr_node
            while node is not None:
                node.visits += 1
                if node.player_id is not None:
                    rank = sim_env.out_order.index(node.player_id)
                    normalized_score = (sim_env.players - 1 - rank) / (
                        sim_env.players - 1
                    )
                    node.score += normalized_score
                node = node.parent

        return {
            child.move: {
                "visits": child.visits,
                "score": child.score,
            }
            for child in root.children
        }

    def _deal_hidden_cards(
        self, real_env, hidden_pool, pending_finish_cards, opp_hand_counts
    ):
        hands = {}
        pool_pointer = 0

        for p in range(real_env.players):
            if p == self.player_id:
                continue

            base_hand_size = max(0, opp_hand_counts[p] - len(pending_finish_cards[p]))
            hand = hidden_pool[pool_pointer : pool_pointer + base_hand_size]
            pool_pointer += base_hand_size

            if pending_finish_cards[p]:
                hand.extend(pending_finish_cards[p])
            hands[p] = sorted(hand)
        return hands

    def _is_valid_hand(
        self,
        p,
        assigned_cards,
        real_env,
        pile_card,
        pile_count,
        history_vector,
        starting_cards,
        opp_hand_counts,
    ):
        if p not in real_env.passed or pile_card == 0:
            return True

        hand_counts = Counter(assigned_cards)
        for card_val, count in hand_counts.items():
            if card_val <= pile_card or count < pile_count:
                continue

            history_index = card_val - 3
            if (
                count == 3
                and 0 <= history_index < len(history_vector)
                and history_vector[history_index] == 0
                and card_val >= 10
                and any(hand_count <= 3 for hand_count in opp_hand_counts.values())
            ):
                continue

            card_diff = card_val - pile_card if pile_card != 0 else 0
            junk_count = sum(1 for c in assigned_cards if c < 8)

            if (
                (card_val >= 14 or card_diff > 4)
                and len(assigned_cards) > starting_cards * 0.5
                and junk_count >= 2
                and pile_card < 14
            ):
                continue

            return False
        return True

    def _determinize_environment(
        self, real_env, base_hidden_pool, pending_finish_cards
    ):
        sim_env = Presidenten(players=real_env.players, verbose=False)

        sim_env.last_move = real_env.last_move
        sim_env.pile_leader = real_env.pile_leader
        sim_env.passed = real_env.passed.copy()
        sim_env.playing = real_env.playing.copy()
        sim_env.first_turn = real_env.first_turn
        sim_env.curr_turn = real_env.curr_turn
        sim_env.history = real_env.history.copy()
        sim_env.out_order = real_env.out_order.copy()
        sim_env.ended_2 = real_env.ended_2.copy()
        sim_env.roles = real_env.roles.copy()
        sim_env.round = real_env.round
        sim_env.scores = real_env.scores.copy()
        sim_env.game_over = real_env.game_over
        sim_env.pending_finish = (
            {
                "queue": real_env.pending_finish["queue"].copy(),
                "resume_turn": real_env.pending_finish["resume_turn"],
                "pile_reset": real_env.pending_finish["pile_reset"],
            }
            if real_env.pending_finish
            else None
        )
        sim_env.hands[self.player_id] = list(real_env.hands[self.player_id])

        history_vector = real_env._get_state(self.player_id)["history_vector"]
        pile_card, pile_count, _ = real_env.last_move
        starting_cards = math.ceil(52 / real_env.players)
        max_attempts = 40
        opp_hand_counts = real_env._get_state(self.player_id)["opp_hand_counts"]

        for _ in range(max_attempts):
            hidden_pool = list(base_hidden_pool)
            random.shuffle(hidden_pool)

            hands = self._deal_hidden_cards(
                real_env, hidden_pool, pending_finish_cards, opp_hand_counts
            )
            if all(
                self._is_valid_hand(
                    p,
                    hand,
                    real_env,
                    pile_card,
                    pile_count,
                    history_vector,
                    starting_cards,
                    opp_hand_counts,
                )
                for p, hand in hands.items()
            ):
                for p, hand in hands.items():
                    sim_env.hands[p] = hand
                return sim_env

        hidden_pool = list(base_hidden_pool)
        random.shuffle(hidden_pool)

        for p, hand in self._deal_hidden_cards(
            real_env, hidden_pool, pending_finish_cards, opp_hand_counts
        ).items():
            sim_env.hands[p] = hand
        return sim_env

    def choose_cards_to_pass(self, state):
        return PresidentenBaselineBot(player_id=self.player_id).choose_cards_to_pass(
            state
        )
