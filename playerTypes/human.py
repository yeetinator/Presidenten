from game import Presidenten
from collections import Counter


class HumanPlayer:
    def __init__(self, player_id):
        self.player_id = player_id

    def get_move(self, state: dict, *args, **kwargs):
        legal_moves = state["legal_moves"]
        if not legal_moves:
            return (0, 0, 0)

        print(f"\nPlayer {self.player_id} ({state["my_role"]}), it's your turn!")
        print(f"Your hand: {Presidenten.visualize_hand(state["hand"])}")
        print(f"Last move: {Presidenten.visualize_move(state["last_move"])}")

        print("Legal moves: ")
        for idx, move in enumerate(legal_moves):
            print(f"  {idx}: {Presidenten.visualize_move(move)}")

        move_idx = int(input("Enter move index: "))
        while move_idx < 0 or move_idx >= len(legal_moves):
            print("Invalid move index. Try again.\n")
            move_idx = int(input("Enter move index: "))
        chosen_move = legal_moves[move_idx]

        return chosen_move

    def choose_cards_to_pass(self, state):
        hr, _, count = next(
            (
                (hr, lr, c)
                for hr, lr, c in state["role_pairs"]
                if hr == state["my_role"] or lr == state["my_role"]
            ),
            (None, None, 0),
        )
        if not state["my_role"] in {"President", "Vice-President", "Secretary"}:
            print(
                f"\nA ({state["my_role"]}) must pass his highest {count} cards to the {hr}."
            )
            highest_cards = state["hand"][-count:] if count > 0 else []
            print(f"You must pass: {Presidenten.visualize_hand(highest_cards)}")
            return []

        print(f"\nChoose {count} cards to give away!")
        print(f"Your hand: {Presidenten.visualize_hand(state["hand"])}")
        card_indices = input(
            "Enter card values to pass (comma-separated, duplicates allowed): "
        )
        indices = [
            int(idx.strip()) for idx in card_indices.split(",") if idx.strip().isdigit()
        ]

        if len(indices) != count:
            print(f"You must choose exactly {count} cards to pass. Try again.\n")
            return self.choose_cards_to_pass(state)

        indices_counts = Counter(indices)
        hand_counts = Counter(state["hand"])

        for card, selected_count in indices_counts.items():
            if hand_counts[card] < selected_count:
                print(f"You don't have enough {card}s to pass. Try again.\n")
                return self.choose_cards_to_pass(state)

        return indices
