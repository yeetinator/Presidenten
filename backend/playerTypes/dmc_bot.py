import torch
import torch.nn as nn
import numpy as np
import random
import itertools
from collections import Counter
from playerTypes.baseline_bot import PresidentBaselineBot
from game import President


class PresidentValueNet(nn.Module):
    def __init__(self, input_dim=115):
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


class PresidentDMCBot:
    def __init__(
        self,
        player_id,
        model: PresidentValueNet,
        device,
        training=False,
        epsilon=0.2,
        profile=None,
    ):
        self.player_id = player_id
        self.model = model.to(device)
        self.device = device
        self.trajectory = []
        self.training = training
        self.epsilon = epsilon
        self.profile = profile

    def get_move(self, state: dict, env: President, *args, **kwargs):
        legal_moves = state["legal_moves"]
        if not legal_moves:
            return (0, 0, 0)

        if len(legal_moves) == 1:
            return legal_moves[0]
        else:
            state_vec = vectorize_state(state, env.players)
            features = [
                np.concatenate([state_vec, vectorize_move(move)])
                for move in legal_moves
            ]
            if self.training and random.random() < self.epsilon:
                best_idx = random.randint(0, len(legal_moves) - 1)
            else:
                features_tensor = torch.FloatTensor(np.array(features)).to(self.device)
                with torch.no_grad():
                    output_tensor: torch.Tensor = self.model(features_tensor)
                    q_values = output_tensor.squeeze(-1).cpu().numpy()

                if self.profile == "aggressive":
                    non_pass_indices = [
                        i for i, m in enumerate(legal_moves) if m != (0, 0, 0)
                    ]
                    if non_pass_indices:
                        best_idx = max(non_pass_indices, key=lambda i: q_values[i])
                    else:
                        best_idx = np.argmax(q_values)
                else:
                    best_idx = np.argmax(q_values)
            chosen_move = legal_moves[best_idx]

            if self.training:
                self.trajectory.append(features[best_idx])
            return chosen_move

    def choose_cards_to_pass(self, state: dict):
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
        hand = state["hand"]
        possible_passes = list(set(itertools.combinations(hand, count)))
        num_players = len(state["opp_hand_counts"]) + 1
        all_hypo_features = []
        combo_map = []

        for combo_idx, pass_combo in enumerate(possible_passes):
            hypothetical_state = state.copy()
            hypo_hand = list(hand)

            for c in pass_combo:
                hypo_hand.remove(c)

            hypothetical_state["hand"] = hypo_hand
            hand_counts = Counter(hypo_hand)

            if not hand_counts:
                continue

            hypo_state_vec = vectorize_state(hypothetical_state, num_players)
            for card, count_held in hand_counts.items():
                for play_count in range(1, count_held + 1):
                    move = (card, play_count, 0)
                    features = np.concatenate([hypo_state_vec, vectorize_move(move)])
                    all_hypo_features.append(features)
                    combo_map.append(combo_idx)

        if not all_hypo_features:
            return (
                list(possible_passes[0])
                if possible_passes
                else PresidentBaselineBot(self.player_id).choose_cards_to_pass(state)
            )

        features_tensor = torch.FloatTensor(np.array(all_hypo_features)).to(self.device)
        with torch.no_grad():
            q_values = self.model(features_tensor).squeeze(-1).cpu().numpy()

        combo_scores = {i: [] for i in range(len(possible_passes))}
        for q_val, combo_idx in zip(q_values, combo_map):
            combo_scores[combo_idx].append(q_val)

        best_pass = None
        best_value = -float("inf")

        for combo_idx, scores in combo_scores.items():
            if scores:
                avg_value = np.mean(scores)
                if avg_value > best_value:
                    best_value = avg_value
                    best_pass = possible_passes[combo_idx]
        return list(best_pass) if best_pass else []


def vectorize_state(state, num_players=4):
    ROLE_MAP = {
        "President": 1.0,
        "Vice-President": 0.8,
        "Secretary": 0.6,
        "Citizen": 0.5,
        "Clerk": 0.4,
        "High-Scum": 0.2,
        "Scum": 0.0,
    }
    MAX_PLAYERS = 7

    opp_hand_counts: dict[int, int] = state["opp_hand_counts"]
    state_player_roles: dict[int, str] = state["player_roles"]

    hand_vector = np.zeros(13, dtype=np.float32)
    for card in state["hand"]:
        hand_vector[card - 3] += 1.0
    hand_vector /= 4.0  # Normalize to [0, 1] range

    history_vector = np.array(state["history_vector"], dtype=np.float32) / 4.0

    unseen_vector = np.zeros(13, dtype=np.float32)
    for i in range(3, 16):
        total_seen = hand_vector[i - 3] + history_vector[i - 3]
        unseen_vector[i - 3] = max(0.0, 1.0 - total_seen)

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

    for i in range(MAX_PLAYERS):
        if i < num_players:
            opp_id = (my_id + i) % num_players
            if i > 0:
                opp_counts.append(float(opp_hand_counts[opp_id]) / 13.0)

            role_str = state_player_roles[opp_id]
            player_roles.append(ROLE_MAP.get(role_str, 0.5))

            is_active = 1.0 if opp_id in state["active_players"] else 0.0
            has_passed = 1.0 if opp_id in state["passed"] else 0.0
            pile_status.extend([is_active, has_passed])
        else:
            if i > 0:
                opp_counts.append(-1.0)
            player_roles.append(-1.0)
            pile_status.extend([-1.0, -1.0])

    opp_vector = np.array(opp_counts, dtype=np.float32)
    role_vector = np.array(player_roles, dtype=np.float32)
    status_vector = np.array(pile_status, dtype=np.float32)

    pile_vector = np.zeros(15, dtype=np.float32)
    if state["pile_leader"] is not None:
        rel_leader_pos = float((state["pile_leader"] - my_id) % num_players) / float(
            num_players
        )
    else:
        rel_leader_pos = -1.0

    for card in state["cards_in_pile"]:
        pile_vector[card - 3] += 0.25

    pile_vector[13] = len(state["cards_in_pile"]) / 4.0
    pile_vector[14] = rel_leader_pos

    misc_vector = np.array(
        [
            1.0 if state["is_finish_prompt"] else 0.0,
            1.0 if state["first_turn"] else 0.0,
        ],
        dtype=np.float32,
    )

    return np.concatenate(
        [
            hand_vector,  # 13
            history_vector,  # 13
            unseen_vector,  # 13
            last_move_vector,  # 15
            opp_vector,  # 7
            role_vector,  # 7
            status_vector,  # 14
            pile_vector,  # 15
            misc_vector,  # 2
        ]
    )


def vectorize_move(move):
    action_vector = np.zeros(16, dtype=np.float32)
    m_card, m_count, m_twos = move
    if m_card == 0:
        action_vector[15] = 1.0
    else:
        action_vector[m_card - 3] = 1.0
        action_vector[13] = float(m_count) / 4.0
        action_vector[14] = float(m_twos) / 4.0
    return action_vector
