import torch
import torch.nn as nn
import numpy as np
import random
import itertools
import math
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

MASTER_OPP_DIM = (MAX_PLAYERS - 1) * HAND_DIM
MASTER_VEC_DIM = STATE_DIM + MASTER_OPP_DIM
MASTER_INPUT_DIM = MASTER_VEC_DIM + MOVE_DIM

_THERM_ROWS = np.arange(COUNT_BUCKETS, dtype=np.int32).reshape(-1, 1)


def to_thermometer(counts):
    c = np.asarray(counts, dtype=np.int32)
    matrix = (_THERM_ROWS < c[None, :]).astype(np.float32)
    return matrix.ravel()


def vectorize_moves(moves):
    n = len(moves)
    M = np.zeros((n, MOVE_DIM), dtype=np.float32)

    for i, (card, count, twos) in enumerate(moves):
        if card == 0:
            M[i, MOVE_DIM - 1] = 1.0
        else:
            M[i, card - 3] = 1.0
            M[i, NUM_RANKS : NUM_RANKS + count] = 1.0
            M[i, NUM_RANKS + COUNT_BUCKETS : NUM_RANKS + COUNT_BUCKETS + twos] = 1.0
    return M


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


def vectorize_master_state(state, num_players, env: President):
    student_vec = vectorize_state(state, num_players)
    my_id = [p for p in range(num_players) if p not in state["opp_hand_counts"]][0]
    MAX_PLAYERS = 7
    opp_hands_matrix = np.full((MAX_PLAYERS - 1, HAND_DIM), -1.0, dtype=np.float32)

    for i in range(1, MAX_PLAYERS):
        slot = i - 1
        if i < num_players:
            opp_id = (my_id + i) % num_players
            opp_counts = np.zeros(NUM_RANKS, dtype=np.int32)

            for card in env.hands[opp_id]:
                opp_counts[card - 3] += 1
            opp_hand_vector = to_thermometer(opp_counts)
            opp_hands_matrix[slot, :] = opp_hand_vector

    opp_hands_vec = opp_hands_matrix.flatten()  # 312
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
            nn.Linear(512, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 1),
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

    def vectorize_state(self, state: dict, *args, **kwargs):
        return vectorize_state(state, len(state["opp_hand_counts"]) + 1)

    def get_move(self, state: dict, *args, **kwargs) -> tuple[int, int, int]:
        legal_moves = state["legal_moves"]
        if not legal_moves:
            return (0, 0, 0)

        if len(legal_moves) == 1:
            return legal_moves[0]
        else:
            state_vec = self.vectorize_state(state, env=kwargs.get("env", None)).astype(
                np.float32
            )
            move_block = vectorize_moves(legal_moves)
            state_block = np.broadcast_to(state_vec, (len(legal_moves), state_vec.size))
            features = np.hstack([state_block, move_block])

            if self.training and random.random() < self.epsilon:
                best_idx = random.randint(0, len(legal_moves) - 1)
            else:
                features_tensor = torch.from_numpy(features).to(self.device)
                with torch.no_grad():
                    q_values = self.model(features_tensor).squeeze(-1).cpu().numpy()

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
                self.trajectory.append(features[best_idx].copy())
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
        no_15 = [c for c in hand if c != 15]
        total_combos = math.comb(len(no_15), count)

        if total_combos <= 35:
            possible_passes = sorted(list(set(itertools.combinations(no_15, count))))
        else:
            low_cards = [c for c in no_15 if c <= 10]
            low_high_cards = [c for c in no_15 if 10 < c <= 12]
            high_cards = [c for c in no_15 if 12 < c]

            if len(low_cards) >= count:
                forced = []
                pool = low_cards
                needed = count
            elif len(low_cards) + len(low_high_cards) >= count:
                forced = low_cards
                pool = low_high_cards
                needed = count - len(forced)
            else:
                forced = low_cards + low_high_cards
                pool = high_cards
                needed = count - len(forced)

            unique_combos = set(itertools.combinations(pool, needed))
            possible_passes = [sorted(forced + list(c)) for c in unique_combos]

        base_counts = np.zeros(NUM_RANKS, dtype=np.int32)
        for c in hand:
            base_counts[c - 3] += 1

        history_counts = np.array(state["history_vector"], dtype=np.int32)
        num_players = len(state["opp_hand_counts"]) + 1
        inv = vectorize_invariant_state(state, num_players)
        hypo_hand_count_norm = float(len(hand) - count) / 13.0
        opp_vec = np.concatenate([[hypo_hand_count_norm], inv["opp_vector"]]).astype(
            np.float32
        )

        inv_tail = np.concatenate(
            [
                inv["last_move_vector"],
                opp_vec,
                inv["role_vector"],
                inv["status_vector"],
                inv["pile_vector"],
                inv["misc_vector"],
            ]
        )
        hist_vec = inv["history_vector"]

        combo_state_vecs = []
        combo_moves_lists = []
        combo_indices = []

        for combo_idx, pass_combo in enumerate(possible_passes):
            counts = base_counts.copy()
            for c in pass_combo:
                counts[c - 3] -= 1
            if not np.any(counts):
                continue

            hand_vec = to_thermometer(counts)
            unseen_counts = np.maximum(0, COUNT_BUCKETS - (counts + history_counts))
            unseen_vec = to_thermometer(unseen_counts)
            hypo_state_vec = np.concatenate(
                [hand_vec, hist_vec, unseen_vec, inv_tail]
            ).astype(np.float32)

            moves_for_combo = [
                (rank_idx + 3, play_count, 0)
                for rank_idx, count_held in enumerate(counts)
                if count_held > 0
                for play_count in range(1, count_held + 1)
            ]
            if not moves_for_combo:
                continue

            combo_state_vecs.append(hypo_state_vec)
            combo_moves_lists.append(moves_for_combo)
            combo_indices.append(combo_idx)

        if not combo_state_vecs:
            return (
                list(possible_passes[0])
                if possible_passes
                else PresidentBaselineBot(self.player_id).choose_cards_to_pass(state)
            )

        all_moves = [m for moves in combo_moves_lists for m in moves]
        move_block = vectorize_moves(all_moves)

        repeats = [len(moves) for moves in combo_moves_lists]
        state_block = np.repeat(
            np.array(combo_state_vecs, dtype=np.float32), repeats, axis=0
        )
        features = np.hstack([state_block, move_block])
        combo_map = np.repeat(combo_indices, repeats)

        features_tensor = torch.from_numpy(features).to(self.device)
        with torch.no_grad():
            q_values = self.model(features_tensor).squeeze(-1).cpu().numpy()

        combo_scores: dict[int, list] = {}
        for q_val, combo_idx in zip(q_values, combo_map):
            combo_scores.setdefault(int(combo_idx), []).append(q_val)

        best_pass = None
        best_value = -float("inf")
        K = 3

        for combo_idx, scores in combo_scores.items():
            if scores:
                sorted_scores = sorted(scores, reverse=True)
                top_k_avg = np.mean(sorted_scores[: min(K, len(sorted_scores))])
                if top_k_avg > best_value:
                    best_value = top_k_avg
                    best_pass = possible_passes[combo_idx]
        return list(best_pass) if best_pass else []


class MasterDMCBot(PresidentDMCBot):
    def vectorize_state(self, state: dict, env: President | None = None):
        if env is None:
            num_players = len(state["opp_hand_counts"]) + 1
            return vectorize_state(state, num_players)
        return vectorize_master_state(state, env.players, env)


class StudentDMCBot(PresidentDMCBot):
    def __init__(
        self,
        player_id,
        model,
        device,
        training=False,
        epsilon=0.2,
        profile=None,
        master_model=None,
    ):
        super().__init__(player_id, model, device, training, epsilon, profile)
        self.master_model = master_model
        self.master_trajectory = []

    def get_move(self, state, env=None, *args, **kwargs) -> tuple[int, int, int]:
        traj_len_before = len(self.trajectory)
        chosen_move = super().get_move(state)

        if (
            self.training
            and self.master_model is not None
            and env is not None
            and len(self.trajectory) > traj_len_before
        ):
            master_state_vec = vectorize_master_state(state, env.players, env)
            master_feat = np.concatenate(
                [master_state_vec, vectorize_move(chosen_move)]
            )

            with torch.no_grad():
                master_q = self.master_model(
                    torch.FloatTensor(master_feat).unsqueeze(0)
                ).item()
            master_q = max(-1.0, min(1.0, master_q))
            self.master_trajectory.append(master_q)
        return chosen_move
