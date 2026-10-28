import random
from .player import Player


class PresidentRandomBot(Player):
    def __init__(self, player_id, seed=None):
        super().__init__(player_id)
        # Own generator: drawing from the global random would tie the moves to the
        # seeded deals of benchmarks
        self.rng = random.Random(seed)

    def get_move(self, state: dict, *args, **kwargs) -> tuple[int, int, int]:
        legal_moves = state["legal_moves"]
        if not legal_moves:
            return (0, 0, 0)
        return self.rng.choice(legal_moves)

    def choose_cards_to_pass(self, state: dict) -> list[int]:
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
        return self.rng.sample(state["hand"], count)
