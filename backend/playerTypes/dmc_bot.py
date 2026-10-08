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
PASS_CAP = 4.0  # Passes per rank
ROLE_MAP = {
    "President": 1.0,
    "Vice-President": 0.8,
    "Secretary": 0.6,
    "Citizen": 0.5,
    "Clerk": 0.4,
    "High-Scum": 0.2,
    "Scum": 0.0,
}

HAND_DIM = NUM_RANKS * COUNT_BUCKETS
HISTORY_DIM = NUM_RANKS
UNSEEN_DIM = NUM_RANKS * COUNT_BUCKETS
PASS_PLAYER_DIM = MAX_PLAYERS * NUM_RANKS
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
    + PASS_PLAYER_DIM
    + LAST_MOVE_DIM
    + OPP_DIM
    + ROLE_DIM
    + STATUS_DIM
    + PILE_DIM
    + MISC_DIM
)
INPUT_DIM = STATE_DIM

MOVE_DIM = NUM_RANKS + COUNT_BUCKETS + COUNT_BUCKETS + 1

ACTION_MOVES: list[tuple[int, int, int]] = [(0, 0, 0)]
for _card in range(3, 15):
    for _count in range(1, COUNT_BUCKETS + 1):
        for _twos in range(_count):
            ACTION_MOVES.append((_card, _count, _twos))
for _count in range(1, COUNT_BUCKETS + 1):
    ACTION_MOVES.append((15, _count, 0))

NUM_ACTIONS = len(ACTION_MOVES)  # 125
PASS_ACTION = 0

_ACTION_LOOKUP = np.full(16 * 25, -1, dtype=np.int64)
for _idx, (_c, _n, _t) in enumerate(ACTION_MOVES):
    _ACTION_LOOKUP[_c * 25 + _n * 5 + _t] = _idx

_THERM_ROWS = np.arange(COUNT_BUCKETS, dtype=np.int32).reshape(-1, 1)

_MOVE_TABLE = np.zeros((16 * 25, MOVE_DIM), dtype=np.float32)
for card in [0] + list(range(3, 16)):
    for count in range(5):
        for twos in range(5):
            v = _MOVE_TABLE[card * 25 + count * 5 + twos]
            if card == 0:
                v[MOVE_DIM - 1] = 1.0
            else:
                v[card - 3] = 1.0
                v[NUM_RANKS : NUM_RANKS + count] = 1.0
                v[NUM_RANKS + COUNT_BUCKETS : NUM_RANKS + COUNT_BUCKETS + twos] = 1.0


MASTER_OPP_DIM = (MAX_PLAYERS - 1) * HAND_DIM
MASTER_VEC_DIM = STATE_DIM + MASTER_OPP_DIM
MASTER_INPUT_DIM = MASTER_VEC_DIM + MOVE_DIM


def action_indices(moves) -> np.ndarray:
    return _ACTION_LOOKUP[[c * 25 + n * 5 + t for c, n, t in moves]]


def to_thermometer(counts):
    c = np.asarray(counts, dtype=np.int32)
    matrix = (_THERM_ROWS < c[None, :]).astype(np.float32)
    return matrix.ravel()


def vectorize_hand_state(hand, history_vector):
    hand_counts = np.zeros(NUM_RANKS, dtype=np.int32)
    for card in hand:
        hand_counts[card - 3] += 1

    hand_vector = to_thermometer(hand_counts)
    history_counts = np.array(history_vector, dtype=np.int32)
    unseen_counts = np.maximum(0, COUNT_BUCKETS - (hand_counts + history_counts))
    unseen_vector = to_thermometer(unseen_counts)

    return hand_vector, unseen_vector


def _seat_relative_matrix(rows, num_players, my_id, scale, cap=None):
    matrix = np.full((MAX_PLAYERS, NUM_RANKS), -1.0, dtype=np.float32)
    for i in range(num_players):
        matrix[i] = rows[(my_id + i) % num_players]

    live = matrix[:num_players]
    if cap is not None:
        np.clip(live, 0.0, cap, out=live)
    live /= scale
    return matrix


def vectorize_invariant_state(state, num_players=4, my_id=0):
    opp_hand_counts: dict[int, int] = state["opp_hand_counts"]
    state_player_roles: dict[int, str] = state["player_roles"]

    history_vector = np.array(state["history_vector"], dtype=np.float32) / COUNT_BUCKETS

    pass_player_vector = _seat_relative_matrix(
        state["passed_by_player"], num_players, my_id, PASS_CAP, cap=PASS_CAP
    ).ravel()

    p_card, p_count, p_twos = state["last_move"]
    last_move_vector = _MOVE_TABLE[p_card * 25 + p_count * 5 + p_twos, :LAST_MOVE_DIM]

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
        "pass_player_vector": pass_player_vector,
        "last_move_vector": last_move_vector,
        "opp_vector": opp_vector,
        "role_vector": role_vector,
        "status_vector": status_vector,
        "pile_vector": pile_vector,
        "misc_vector": misc_vector,
    }


def invariant_tail(inv, opp_vector):
    return np.concatenate(
        [
            inv["pass_player_vector"],
            inv["last_move_vector"],
            opp_vector,
            inv["role_vector"],
            inv["status_vector"],
            inv["pile_vector"],
            inv["misc_vector"],
        ]
    ).astype(np.float32)


def vectorize_state(state, num_players=4, my_id=0):
    hand_vec, unseen_vec = vectorize_hand_state(state["hand"], state["history_vector"])
    inv = vectorize_invariant_state(state, num_players, my_id)
    opp_vec = np.concatenate(
        [[float(len(state["hand"])) / NUM_RANKS], inv["opp_vector"]]
    )

    return np.concatenate(
        [
            hand_vec,  # 52
            inv["history_vector"],  # 13
            unseen_vec,  # 52
            invariant_tail(inv, opp_vec),
        ]
    ).astype(np.float32)


class PresidentValueNet(nn.Module):
    def __init__(self, input_dim=STATE_DIM, n_actions=NUM_ACTIONS, width=1024):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.LayerNorm(width),
            nn.LeakyReLU(0.1),
            nn.Linear(width, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.1),
        )
        self.head = nn.Linear(512, n_actions)
        nn.init.uniform_(self.head.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        return self.head(self.trunk(x))


class PresidentDMCBot(Player):
    def __init__(
        self,
        player_id,
        model: PresidentValueNet,
        device,
        training=False,
        temperature=0.05,
        profile=None,
    ):
        self.player_id = player_id
        self.model = model.to(device)
        self.device = device
        self.trajectory = []
        self.training = training
        self.temperature = temperature
        self.profile = profile

        self.n_decisions = 0
        self.n_explored = 0
        self.q_gap_sum = 0.0

    def vectorize_state(self, state: dict, *args, **kwargs):
        return vectorize_state(state, len(state["opp_hand_counts"]) + 1, self.player_id)

    def _q_values(self, state_vec: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(state_vec).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.model(tensor).squeeze(0).cpu().numpy()

    def get_move(self, state: dict, *args, **kwargs) -> tuple[int, int, int]:
        legal_moves = state["legal_moves"]
        if not legal_moves:
            return (0, 0, 0)
        if len(legal_moves) == 1:
            return legal_moves[0]

        state_vec = self.vectorize_state(state, env=kwargs.get("env", None))
        legal_idx = action_indices(legal_moves)
        q_legal = self._q_values(state_vec)[legal_idx]

        if self.profile == "aggressive":
            non_pass = legal_idx != PASS_ACTION
            if non_pass.any():
                q_legal = np.where(non_pass, q_legal, -np.inf)

        greedy = int(np.argmax(q_legal))

        if self.training and self.temperature > 0:
            choice = self._boltzmann_sample(q_legal)
            self.n_decisions += 1
            if choice != greedy:
                self.n_explored += 1
                self.q_gap_sum += float(q_legal[greedy] - q_legal[choice])
        else:
            choice = greedy

        if self.training:
            self.trajectory.append((state_vec, int(legal_idx[choice])))
        return legal_moves[choice]

    def _boltzmann_sample(self, q_legal: np.ndarray) -> int:
        weights = np.exp((q_legal - q_legal.max()) / self.temperature)
        cdf = np.cumsum(weights)
        idx = int(np.searchsorted(cdf, random.random() * cdf[-1], side="right"))
        return min(idx, len(weights) - 1)

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
        inv = vectorize_invariant_state(state, num_players, self.player_id)

        hypo_hand_count_norm = float(len(hand) - count) / 13.0
        opp_vec = np.concatenate([[hypo_hand_count_norm], inv["opp_vector"]])
        inv_tail = invariant_tail(inv, opp_vec)
        hist_vec = inv["history_vector"]

        combo_state_vecs = []
        combo_action_idx = []
        combo_indices = []

        for combo_idx, pass_combo in enumerate(possible_passes):
            counts = base_counts.copy()
            for c in pass_combo:
                counts[c - 3] -= 1
            if not np.any(counts):
                continue

            moves_for_combo = [
                (rank_idx + 3, play_count, 0)
                for rank_idx, count_held in enumerate(counts)
                if count_held > 0
                for play_count in range(1, count_held + 1)
            ]
            if not moves_for_combo:
                continue

            hand_vec = to_thermometer(counts)
            unseen_counts = np.maximum(0, COUNT_BUCKETS - (counts + history_counts))
            unseen_vec = to_thermometer(unseen_counts)
            hypo_state_vec = np.concatenate(
                [hand_vec, hist_vec, unseen_vec, inv_tail]
            ).astype(np.float32)

            combo_state_vecs.append(hypo_state_vec)
            combo_action_idx.append(action_indices(moves_for_combo))
            combo_indices.append(combo_idx)

        if not combo_state_vecs:
            return (
                list(possible_passes[0])
                if possible_passes
                else PresidentBaselineBot(self.player_id).choose_cards_to_pass(state)
            )

        batch = torch.from_numpy(np.stack(combo_state_vecs)).to(self.device)
        with torch.no_grad():
            q_values = self.model(batch).cpu().numpy()

        best_pass = None
        best_value = -float("inf")
        K = 3

        for row, (combo_idx, idxs) in enumerate(zip(combo_indices, combo_action_idx)):
            scores = q_values[row, idxs]
            k = min(K, scores.shape[0])
            top_k_avg = float(np.mean(-np.sort(-scores)[:k]))
            if top_k_avg > best_value:
                best_value = top_k_avg
                best_pass = possible_passes[combo_idx]
        return list(best_pass) if best_pass else []


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
