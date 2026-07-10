import numpy as np
from game import President
from playerTypes.dmc_bot import (
    PresidentDMCBot,
    PresidentValueNet,
    vectorize_state,
)


class MasterValueNet(PresidentValueNet):
    def __init__(self, input_dim=193):
        super().__init__(input_dim)

    def forward(self, x):
        return self.net(x)


class MasterDMCBot(PresidentDMCBot):
    def vectorize_state(self, state: dict, env: President | None = None):
        if env is None:
            num_players = len(state["opp_hand_counts"]) + 1
            return vectorize_state(state, num_players)
        return vectorize_master_state(state, env, env.players)


def vectorize_master_state(state, env: President, num_players=4):
    student_vec = vectorize_state(state, num_players)
    my_id = [p for p in range(num_players) if p not in state["opp_hand_counts"]][0]
    MAX_PLAYERS = 7
    opp_hands_matrix = np.full((MAX_PLAYERS - 1, 13), -1.0, dtype=np.float32)

    for i in range(1, MAX_PLAYERS):
        slot = i - 1
        if i < num_players:
            opp_id = (my_id + i) % num_players
            opp_hands_matrix[slot, :] = 0.0
            for card in env.hands[opp_id]:
                opp_hands_matrix[slot, card - 3] += 1.0 / 4.0

    opp_hands_vec = opp_hands_matrix.flatten()  # 78
    return np.concatenate([student_vec, opp_hands_vec])
