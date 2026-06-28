from game import Presidenten, get_val_input
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

        move_idx = get_val_input(
            "Enter move index: ", int, lambda x: 0 <= x < len(legal_moves)
        )
        chosen_move = legal_moves[move_idx]

        return chosen_move

    def choose_cards_to_pass(self, state: dict):
        my_role: str = state["my_role"]
        hr, _, count = next(
            (
                (hr, lr, c)
                for hr, lr, c in state["role_pairs"]
                if hr == my_role or lr == my_role
            ),
            (None, None, 0),
        )

        if my_role not in {"President", "Vice-President", "Secretary"}:
            if count > 0:
                print(
                    f"\nAs a {my_role}, your highest {count} card(s) will be passed to the {hr}."
                )
                highest_cards = state["hand"][-count:]
                print(f"Cards passed: {Presidenten.visualize_hand(highest_cards)}")
            return []

        def parse_card(card: str):
            card = card.strip().upper()
            mapping = {"J": 11, "Q": 12, "K": 13, "A": 14, "2": 15}
            if card in mapping:
                return mapping[card]
            return int(card)

        def val_hand_selection(chosen):
            if len(chosen) != count:
                print(f"Selection Error: You must select exactly {count} cards.")
                return False

            chosen_counts = Counter(chosen)
            hand_counts = Counter(state["hand"])

            for card, selected_count in chosen_counts.items():
                if hand_counts[card] < selected_count:
                    card_name = Presidenten.visualize_card(card)
                    print(f"Selection Error: You don't have enough {card_name}s.")

                    return False
            return True

        print(f"\n=== {my_role.upper()} CARD EXCHANGE ===")
        print(f"Your hand: {Presidenten.visualize_hand(state["hand"])}")

        prompt = (
            f"Enter {count} card values to pass (comma-separated, duplicates allowed): "
        )
        return get_val_input(prompt, parse_card, val_hand_selection, ",")
