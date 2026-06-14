from collections import Counter
from itertools import combinations
import math


class PresidentenBaselineBot:
    def __init__(self, player_id):
        self.player_id = player_id

    def get_ranked_moves(self, state: dict, **kwargs):
        hand = state["hand"]
        legal_moves = state["legal_moves"]

        if not legal_moves:
            return []

        if state.get("is_finish_prompt", False):
            jump_in_action = [m for m in legal_moves if m != (0, 0, 0)]
            if not jump_in_action:
                return [(0, 0, 0)]

            chosen_jump = jump_in_action[0]
            finish_card, _, _ = chosen_jump

            if finish_card != 15:
                return [chosen_jump]

            unique_vals = set(hand)
            if len(unique_vals) == 2:
                return [chosen_jump]
            return [(0, 0, 0)]

        unique_vals = set(hand)
        hand_counts = Counter(hand)

        playable_moves = [m for m in legal_moves if m != (0, 0, 0)]
        if not playable_moves:
            return [(0, 0, 0)]

        if len(unique_vals) == 2 and 15 in unique_vals:
            two_moves = [m for m in playable_moves if m[0] == 15]
            if two_moves:
                two_moves.sort(key=lambda x: x[1])
                return two_moves

        if state["last_move"] == (0, 0, 0) and state["first_turn"]:
            playable_moves.sort(key=lambda x: (x[2], x[0], -x[1]))
            return playable_moves

        history_vector = state.get("history_vector", [])
        filtered_moves = playable_moves

        for card_value, count in hand_counts.items():
            history_index = card_value - 3
            if (
                count == 3
                and 0 <= history_index < len(history_vector)
                and history_vector[history_index] == 0
                and card_value >= 10
                and any(
                    hand_count <= 3 for hand_count in state["opp_hand_counts"].values()
                )
            ):
                candidate_moves = [
                    m for m in filtered_moves if not (m[0] == card_value and m[1] == 3)
                ]
                if candidate_moves:
                    filtered_moves = candidate_moves
        playable_moves = filtered_moves

        low_card_moves = [m for m in playable_moves if m[0] < 10 and m[2] == 0]
        high_card_moves = [m for m in playable_moves if m[0] >= 10]

        if low_card_moves:
            low_card_moves.sort(key=lambda x: (hand_counts[x[0]], x[0], -x[1]))

        if high_card_moves:
            high_card_moves.sort(key=lambda x: (x[2], hand_counts[x[0]], x[0], -x[1]))

        ranked = low_card_moves + high_card_moves
        return ranked or [legal_moves[0]]

    def get_move(self, state: dict, *args, **kwargs):
        legal_moves = state["legal_moves"]

        if len(legal_moves) == 1:
            return legal_moves[0]

        ranked_moves = self.get_ranked_moves(state, *args, **kwargs)
        if not ranked_moves:
            return (0, 0, 0)

        best_move = ranked_moves[0]
        if best_move == (0, 0, 0):
            return best_move

        hand = state["hand"]
        card_val, _, twos = best_move
        card_diff = (
            card_val - state["last_move"][0] if state["last_move"][0] != 0 else 0
        )
        num_players = len(state["opp_hand_counts"]) + 1
        starting_cards = math.ceil(52 / num_players)
        junk_count = sum(1 for c in hand if c < 8)

        if (
            card_val >= 10
            and (card_val >= 14 or twos > 0 or card_diff > 4)
            and len(hand) > starting_cards * 0.5
            and junk_count >= 2
            and state["last_move"][0] < 14
            and (0, 0, 0) in legal_moves
        ):
            return (0, 0, 0)
        return best_move

    def choose_cards_to_pass(self, state):
        _, _, count = next(
            (
                (hr, lr, c)
                for hr, lr, c in state["role_pairs"]
                if hr == state["my_role"]
            ),
            (None, None, 0),
        )
        hand = state["hand"]
        low_cards = sorted([c for c in hand if c < 10])
        high_cards = sorted([c for c in hand if c >= 10])

        def find_best_selection(pool, needed):
            if len(pool) <= needed:
                return list(pool)

            best_selection = None
            min_remaining_singles = float("inf")
            best_val_tuple = (float("inf"),) * needed

            for indices in combinations(range(len(pool)), needed):
                selection = [pool[i] for i in indices]
                remaining = [pool[i] for i in range(len(pool)) if i not in indices]
                rem_counts = Counter(remaining)
                singles_count = sum(1 for c in rem_counts.values() if c == 1)
                selected_sorted = tuple(sorted(selection))

                if singles_count < min_remaining_singles:
                    min_remaining_singles = singles_count
                    best_val_tuple = selected_sorted
                    best_selection = selection
                elif (
                    singles_count == min_remaining_singles
                    and selected_sorted < best_val_tuple
                ):
                    best_val_tuple = selected_sorted
                    best_selection = selection

            return best_selection or pool[:needed]

        cards_to_pass = []

        low_to_take = min(len(low_cards), count)
        if low_to_take > 0:
            chosen_low = find_best_selection(low_cards, low_to_take)
            cards_to_pass.extend(chosen_low)
            count -= low_to_take

        if count > 0:
            chosen_high = find_best_selection(high_cards, count)
            cards_to_pass.extend(chosen_high)
        return cards_to_pass
