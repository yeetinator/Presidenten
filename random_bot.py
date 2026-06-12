import random


class PresidentenRandomBot:
    def __init__(self, player_id):
        self.player_id = player_id

    def get_move(self, state: dict, **kwargs):
        legal_moves = state["legal_moves"]
        if not legal_moves:
            return (0, 0, 0)
        return random.choice(legal_moves)
