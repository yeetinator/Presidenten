from abc import ABC, abstractmethod


class Player(ABC):
    def __init__(self, player_id: int):
        self.player_id = player_id

    @abstractmethod
    def get_move(self, state: dict, *args, **kwargs) -> tuple[int, int, int]:
        pass

    @abstractmethod
    def choose_cards_to_pass(self, state: dict) -> list[int]:
        pass
