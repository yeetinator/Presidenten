import torch
import torch.nn as nn
import numpy as np
import random

from game import Presidenten
from playerTypes.baseline_bot import PresidentenBaselineBot


class PresidentenValueNet(nn.Module):
    def __init__(self, input_dim=91):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x)


class PresidentenDMCBot:
    def __init__(
        self, player_id, model: PresidentenValueNet, device, training=False, epsilon=0.2
    ):
        self.player_id = player_id
        self.model = model.to(device)
        self.device = device
        self.trajectory = []
        self.training = training
        self.epsilon = epsilon

    def get_move(self, state, env: Presidenten, *args, **kwargs):
        legal_moves = state["legal_moves"]
        if not legal_moves:
            return (0, 0, 0)

        if len(legal_moves) == 1:
            return legal_moves[0]
        else:
            features = [
                vectorize_state_action(state, move, env.players) for move in legal_moves
            ]
            if self.training and random.random() < self.epsilon:
                best_idx = random.randint(0, len(legal_moves) - 1)
            else:
                features_tensor = torch.FloatTensor(np.array(features)).to(self.device)
                with torch.no_grad():
                    output_tensor: torch.Tensor = self.model(features_tensor)
                    q_values = output_tensor.squeeze(-1).cpu().numpy()
                best_idx = np.argmax(q_values)
            chosen_move = legal_moves[best_idx]

            if self.training:
                self.trajectory.append(features[best_idx])
            return chosen_move

    def choose_cards_to_pass(self, state):
        if not state["my_role"] in {"President", "Vice-President", "Secretary"}:
            return []
        return PresidentenBaselineBot(player_id=self.player_id).choose_cards_to_pass(
            state
        )


def vectorize_state_action(state, move, num_players=4):
    ROLE_MAP = {
        "President": 1.0,
        "Vice-President": 0.8,
        "Secretary": 0.6,
        "Citizen": 0.5,
        "Clerk": 0.4,
        "High-Scum": 0.2,
        "Scum": 0.0,
    }

    opp_hand_counts: dict[int, int] = state["opp_hand_counts"]
    state_player_roles: dict[int, str] = state["player_roles"]

    hand_vector = np.zeros(13, dtype=np.float32)
    for card in state["hand"]:
        hand_vector[card - 3] += 1.0
    hand_vector /= 4.0  # Normalize to [0, 1] range

    history_vector = np.array(state["history_vector"], dtype=np.float32) / 4.0

    last_move_vector = np.zeros(15, dtype=np.float32)
    p_card, p_count, p_twos = state["last_move"]
    if p_card != 0:
        last_move_vector[p_card - 3] = 1.0
        last_move_vector[13] = float(p_count) / 4.0
        last_move_vector[14] = float(p_twos) / 4.0

    my_id = [p for p in range(num_players) if p not in opp_hand_counts][0]
    opp_counts = [float(len(state["hand"])) / 13.0]
    player_roles = []
    pile_status = []

    for i in range(num_players):
        opp_id = (my_id + i) % num_players
        if i > 0:
            opp_counts.append(float(opp_hand_counts[opp_id]) / 13.0)

        role_str = state_player_roles[opp_id]
        player_roles.append(ROLE_MAP.get(role_str, 0.5))

        is_active = 1.0 if opp_id in state["active_players"] else 0.0
        has_passed = 1.0 if opp_id in state["passed"] else 0.0
        pile_status.extend([is_active, has_passed])

    opp_vector = np.array(opp_counts, dtype=np.float32)
    role_vector = np.array(player_roles, dtype=np.float32)
    status_vector = np.array(pile_status, dtype=np.float32)

    action_window = np.zeros(16, dtype=np.float32)
    recent_history = state["history"][-4:]

    for idx, (act_p_id, act_move) in enumerate(recent_history):
        rel_pos = float((act_p_id - my_id) % num_players) / float(num_players)
        m_card, m_count, m_twos = act_move
        norm_card = float(m_card - 3) / 12.0 if m_card != 0 else 0.0
        offset = idx * 4
        action_window[offset] = rel_pos
        action_window[offset + 1] = norm_card
        action_window[offset + 2] = float(m_count) / 4.0
        action_window[offset + 3] = float(m_twos) / 4.0

    misc_vector = np.array(
        [
            1.0 if state["is_finish_prompt"] else 0.0,
            1.0 if state["first_turn"] else 0.0,
        ],
        dtype=np.float32,
    )

    state_vector = np.concatenate(
        [
            hand_vector,  # 13
            history_vector,  # 13
            last_move_vector,  # 15
            opp_vector,  # 4
            role_vector,  # 4
            status_vector,  # 8
            action_window,  # 16
            misc_vector,  # 2
        ]
    )

    action_vector = np.zeros(16, dtype=np.float32)
    m_card, m_count, m_twos = move
    if m_card == 0:
        action_vector[15] = 1.0
    else:
        action_vector[m_card - 3] = 1.0
        action_vector[13] = float(m_count) / 4.0
        action_vector[14] = float(m_twos) / 4.0
    return np.concatenate([state_vector, action_vector])
