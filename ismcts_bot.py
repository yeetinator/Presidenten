import math
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from game import Presidenten
from baseline_bot import PresidentenBaselineBot


class ISMCTSNode:
    def __init__(self, move=None, parent=None, player_id=None):
        self.move = move
        self.parent = parent
        self.player_id = player_id
        self.children = []
        self.visits = 0
        self.wins = 0.0

    def select_child(self, legal_moves, player_id, exploration_weight=1.41):
        legal_children = [
            child
            for child in self.children
            if child.move in legal_moves and child.player_id == player_id
        ]
        if not legal_children:
            return None

        if self.visits == 0:
            return random.choice(legal_children)

        log_total = math.log(self.visits)
        best_score = -float("inf")
        best_child = None

        for child in legal_children:
            if child.visits == 0:
                score = float("inf")
            else:
                score = (child.wins / child.visits) + exploration_weight * math.sqrt(
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
        self, state: dict, real_env=None, executor=None, num_workers=4, parallelism="s"
    ):
        legal_moves = state["legal_moves"]
        total_visits = {}

        if not legal_moves:
            return (0, 0, 0)
        if len(legal_moves) == 1:
            return legal_moves[0]

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

            for worker_visits in results:
                for move, visits in worker_visits.items():
                    total_visits[move] = total_visits.get(move, 0) + visits
        else:
            total_visits = self.run_search_batch(real_env)

        legal_root_moves = {
            move: visits for move, visits in total_visits.items() if move in legal_moves
        }
        if not legal_root_moves:
            return random.choice(legal_moves)
        return max(legal_root_moves, key=lambda move: legal_root_moves[move])

    def run_search_batch(self, real_env):
        root = ISMCTSNode()
        for _ in range(self.iterations):
            sim_env = self._determinize_environment(real_env)
            curr_node = root

            while not sim_env.game_over:
                curr_player = sim_env.curr_turn
                sim_legal_moves = sim_env.get_legal_moves(curr_player)

                if not sim_legal_moves:
                    break

                tried_moves = [
                    child.move
                    for child in curr_node.children
                    if child.player_id == curr_player
                ]
                untried_moves = [m for m in sim_legal_moves if m not in tried_moves]

                if untried_moves:
                    if curr_player != self.player_id:
                        rollout_bot = PresidentenBaselineBot(player_id=curr_player)
                        sim_state = sim_env._get_state(curr_player)
                        smart_move = rollout_bot.get_move(sim_state)

                        if smart_move in untried_moves:
                            chosen_move = smart_move
                        else:
                            chosen_move = random.choice(untried_moves)
                    else:
                        chosen_move = random.choice(untried_moves)

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
                    next_node = curr_node.select_child(sim_legal_moves, curr_player)
                    if next_node is None:
                        break

                    curr_node = next_node
                    sim_env.step(curr_player, curr_node.move)

            while not sim_env.game_over:
                curr_player = sim_env.curr_turn
                sim_legal_moves = sim_env.get_legal_moves(curr_player)

                if not sim_legal_moves:
                    break

                rollout_bot = PresidentenBaselineBot(player_id=curr_player)
                sim_state = sim_env._get_state(curr_player)
                chosen_move = rollout_bot.get_move(sim_state)
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
                    node.wins += normalized_score
                node = node.parent

        return {child.move: child.visits for child in root.children}

    def _determinize_environment(self, real_env):
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

        if real_env.pending_finish:
            sim_env.pending_finish = {
                "queue": real_env.pending_finish["queue"].copy(),
                "resume_turn": real_env.pending_finish["resume_turn"],
                "pile_reset": real_env.pending_finish["pile_reset"],
            }
        else:
            sim_env.pending_finish = None

        known_cards = list(real_env.hands[self.player_id])
        sim_env.hands[self.player_id] = known_cards.copy()

        for _, move in real_env.history:
            if move != (0, 0, 0):
                card_val, count, twos = move
                known_cards.extend([card_val] * (count - twos))
                known_cards.extend([15] * twos)

        total_deck_counts = Counter(real_env.deck)
        known_counts = Counter(known_cards)
        hidden_pool = []

        for card_val, total_count in total_deck_counts.items():
            unaccounted = total_count - known_counts[card_val]
            if unaccounted > 0:
                hidden_pool.extend([card_val] * unaccounted)

        pending_finish_cards = {p: [] for p in range(real_env.players)}
        if real_env.pending_finish:
            for card, count, p in real_env.pending_finish["queue"]:
                if p != self.player_id:
                    pending_finish_cards[p].extend([card] * count)
                    for _ in range(count):
                        if card in hidden_pool:
                            hidden_pool.remove(card)

        random.shuffle(hidden_pool)
        pool_pointer = 0

        for p in range(real_env.players):
            if p == self.player_id:
                continue

            revealed_count = len(pending_finish_cards[p])
            opp_hand_size = len(real_env.hands[p])
            base_hand_size = max(0, opp_hand_size - revealed_count)

            sim_env.hands[p] = hidden_pool[pool_pointer : pool_pointer + base_hand_size]
            sim_env.hands[p].sort()
            pool_pointer += base_hand_size

            if pending_finish_cards[p]:
                sim_env.hands[p].extend(pending_finish_cards[p])
                sim_env.hands[p].sort()

        return sim_env
