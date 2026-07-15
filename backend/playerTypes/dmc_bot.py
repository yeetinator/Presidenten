import torch
import torch.nn as nn
import numpy as np
import random
import itertools
from collections import Counter
from .player import Player
from .baseline_bot import PresidentBaselineBot
from game import President

NUM_RANKS = 13
COUNT_BUCKETS = 4
MAX_PLAYERS = 7

HAND_DIM = NUM_RANKS * COUNT_BUCKETS
HISTORY_DIM = NUM_RANKS
UNSEEN_DIM = NUM_RANKS * COUNT_BUCKETS
LAST_MOVE_DIM = NUM_RANKS + COUNT_BUCKETS + COUNT_BUCKETS
OPP_DIM = MAX_PLAYERS
ROLE_DIM = MAX_PLAYERS
STATUS_DIM = MAX_PLAYERS * 2
PILE_DIM = NUM_RANKS + 2
MISC_DIM = 2

STATE_DIM = (
    HAND_DIM
    + HISTORY_DIM
    + UNSEEN_DIM
    + LAST_MOVE_DIM
    + OPP_DIM
    + ROLE_DIM
    + STATUS_DIM
    + PILE_DIM
    + MISC_DIM
)
MOVE_DIM = NUM_RANKS + COUNT_BUCKETS + COUNT_BUCKETS + 1
INPUT_DIM = STATE_DIM + MOVE_DIM

MASTER_OPP_DIM = (MAX_PLAYERS - 1) * NUM_RANKS
MASTER_VEC_DIM = STATE_DIM + MASTER_OPP_DIM
MASTER_INPUT_DIM = MASTER_VEC_DIM + MOVE_DIM


def to_thermometer(counts):
    matrix = np.zeros((COUNT_BUCKETS, NUM_RANKS), dtype=np.float32)
    for rank_idx, count in enumerate(counts):
        c = min(max(int(count), 0), COUNT_BUCKETS)
        if c > 0:
            matrix[:c, rank_idx] = 1.0
    return matrix.flatten()


def vectorize_hand_state(hand, history_vector):
    hand_counts = np.zeros(NUM_RANKS, dtype=np.int32)
    for card in hand:
        hand_counts[card - 3] += 1

    hand_vector = to_thermometer(hand_counts)
    history_counts = np.array(history_vector, dtype=np.int32)
    unseen_counts = np.maximum(0, COUNT_BUCKETS - (hand_counts + history_counts))
    unseen_vector = to_thermometer(unseen_counts)

    return hand_vector, unseen_vector


def vectorize_invariant_state(state, num_players=4):
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

    history_vector = np.array(state["history_vector"], dtype=np.float32) / COUNT_BUCKETS

    last_move_vector = np.zeros(LAST_MOVE_DIM, dtype=np.float32)
    p_card, p_count, p_twos = state["last_move"]
    if p_card != 0:
        last_move_vector[p_card - 3] = 1.0
        last_move_vector[NUM_RANKS : NUM_RANKS + p_count] = 1.0
        last_move_vector[
            NUM_RANKS + COUNT_BUCKETS : NUM_RANKS + COUNT_BUCKETS + p_twos
        ] = 1.0

    my_id = [p for p in range(num_players) if p not in opp_hand_counts][0]
    opp_counts = []
    player_roles = []
    pile_status = []

    for i in range(MAX_PLAYERS):
        if i < num_players:
            opp_id = (my_id + i) % num_players
            if i > 0:
                opp_counts.append(float(opp_hand_counts[opp_id]) / NUM_RANKS)

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

    pile_vector = np.zeros(PILE_DIM, dtype=np.float32)
    if state["pile_leader"] is not None:
        rel_leader_pos = float((state["pile_leader"] - my_id) % num_players) / float(
            num_players
        )
    else:
        rel_leader_pos = -1.0

    for card in state["cards_in_pile"]:
        pile_vector[card - 3] += 1.0 / COUNT_BUCKETS

    pile_vector[NUM_RANKS] = len(state["cards_in_pile"]) / COUNT_BUCKETS
    pile_vector[NUM_RANKS + 1] = rel_leader_pos

    misc_vector = np.array(
        [
            1.0 if state["is_finish_prompt"] else 0.0,
            1.0 if state["first_turn"] else 0.0,
        ],
        dtype=np.float32,
    )

    return {
        "history_vector": history_vector,
        "last_move_vector": last_move_vector,
        "opp_vector": opp_vector,
        "role_vector": role_vector,
        "status_vector": status_vector,
        "pile_vector": pile_vector,
        "misc_vector": misc_vector,
    }


def vectorize_state(state, num_players=4):
    hand_vec, unseen_vec = vectorize_hand_state(state["hand"], state["history_vector"])
    inv = vectorize_invariant_state(state, num_players)
    opp_vec = np.concatenate(
        [[float(len(state["hand"])) / NUM_RANKS], inv["opp_vector"]]
    )

    return np.concatenate(
        [
            hand_vec,  # 52
            inv["history_vector"],  # 13
            unseen_vec,  # 52
            inv["last_move_vector"],  # 21
            opp_vec,  # 7
            inv["role_vector"],  # 7
            inv["status_vector"],  # 14
            inv["pile_vector"],  # 15
            inv["misc_vector"],  # 2
        ]
    )


def vectorize_master_state(state, env: President, num_players=4):
    student_vec = vectorize_state(state, num_players)
    my_id = [p for p in range(num_players) if p not in state["opp_hand_counts"]][0]
    MAX_PLAYERS = 7
    opp_hands_matrix = np.full((MAX_PLAYERS - 1, NUM_RANKS), -1.0, dtype=np.float32)

    for i in range(1, MAX_PLAYERS):
        slot = i - 1
        if i < num_players:
            opp_id = (my_id + i) % num_players
            opp_hands_matrix[slot, :] = 0.0
            for card in env.hands[opp_id]:
                opp_hands_matrix[slot, card - 3] += 1.0 / COUNT_BUCKETS

    opp_hands_vec = opp_hands_matrix.flatten()  # 78
    return np.concatenate([student_vec, opp_hands_vec])


def vectorize_move(move):
    action_vector = np.zeros(MOVE_DIM, dtype=np.float32)
    m_card, m_count, m_twos = move
    if m_card == 0:
        action_vector[MOVE_DIM - 1] = 1.0
    else:
        action_vector[m_card - 3] = 1.0
        action_vector[NUM_RANKS : NUM_RANKS + m_count] = 1.0
        action_vector[
            NUM_RANKS + COUNT_BUCKETS : NUM_RANKS + COUNT_BUCKETS + m_twos
        ] = 1.0
    return action_vector


class PresidentValueNet(nn.Module):
    def __init__(self, input_dim=INPUT_DIM):
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


class MasterValueNet(nn.Module):
    def __init__(self, input_dim=MASTER_INPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x)


class PresidentDMCBot(Player):
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

    def vectorize_state(self, state: dict):
        return vectorize_state(state, len(state["opp_hand_counts"]) + 1)

    def get_move(self, state: dict, *args, **kwargs) -> tuple[int, int, int]:
        legal_moves = state["legal_moves"]
        if not legal_moves:
            return (0, 0, 0)

        if len(legal_moves) == 1:
            return legal_moves[0]
        else:
            state_vec = self.vectorize_state(state)
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
        hand = state["hand"]
        possible_passes = sorted(list(set(itertools.combinations(hand, count))))
        all_hypo_features = []
        combo_map = []
        num_players = len(state["opp_hand_counts"]) + 1
        inv = vectorize_invariant_state(state, num_players)
        hypo_hand_count_norm = float(len(hand) - count) / 13.0
        opp_vec = np.concatenate([[hypo_hand_count_norm], inv["opp_vector"]])

        for combo_idx, pass_combo in enumerate(possible_passes):
            hypo_hand = list(hand)
            for c in pass_combo:
                hypo_hand.remove(c)

            hand_counts = Counter(hypo_hand)
            if not hand_counts:
                continue

            hand_vec, unseen_vec = vectorize_hand_state(
                hypo_hand, state["history_vector"]
            )
            hypo_state_vec = np.concatenate(
                [
                    hand_vec,
                    inv["history_vector"],
                    unseen_vec,
                    inv["last_move_vector"],
                    opp_vec,
                    inv["role_vector"],
                    inv["status_vector"],
                    inv["pile_vector"],
                    inv["misc_vector"],
                ]
            )

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


class MasterDMCBot(PresidentDMCBot):
    def vectorize_state(self, state: dict, env: President | None = None):
        if env is None:
            num_players = len(state["opp_hand_counts"]) + 1
            return vectorize_state(state, num_players)
        return vectorize_master_state(state, env, env.players)
