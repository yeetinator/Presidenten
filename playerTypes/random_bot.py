import random


class PresidentenRandomBot:
    def __init__(self, player_id):
        self.player_id = player_id

    def get_move(self, state: dict, *args, **kwargs):
        legal_moves = state["legal_moves"]
        if not legal_moves:
            return (0, 0, 0)
        return random.choice(legal_moves)

    def choose_cards_to_pass(self, state):
        if not state["my_role"] in {"President", "Vice-President", "Secretary"}:
            return []

        _, _, count = next(
            (
                (hr, lr, c)
                for hr, lr, c in state["role_pairs"]
                if hr == state["my_role"]
            ),
            (None, None, 0),
        )
        return random.sample(state["hand"], count)
