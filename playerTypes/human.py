from game import Presidenten


class HumanPlayer:
    def __init__(self, player_id):
        self.player_id = player_id

    def get_move(self, state: dict, *args, **kwargs):
        legal_moves = state["legal_moves"]
        if not legal_moves:
            return (0, 0, 0)

        print(f"Player {self.player_id} ({state["my_role"]}), it's your turn!")
        print(f"Your hand: {Presidenten.visualize_hand(state["hand"])}")
        print(f"Last move: {Presidenten.visualize_move(state["last_move"])}")

        print("Legal moves: ")
        for idx, move in enumerate(legal_moves):
            print(f"  {idx}: {Presidenten.visualize_move(move)}")

        move_idx = int(input("Enter move index: "))
        chosen_move = legal_moves[move_idx]

        return chosen_move
