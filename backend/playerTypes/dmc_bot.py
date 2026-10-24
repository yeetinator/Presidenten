import torch
import torch.nn as nn
import numpy as np
import random
import itertools
import math
from .player import Player
from .baseline_bot import PresidentBaselineBot

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
PASS_MULTI_DIM = MAX_PLAYERS * NUM_RANKS
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
    + PASS_MULTI_DIM
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

_HIGH_ROLES = {"President", "Vice-President", "Secretary"}
_PASS_SCALE = 1.0 / PASS_CAP
_PASS_LUT = (np.minimum(np.arange(256), PASS_CAP) * _PASS_SCALE).astype(np.float32)
_PAD = [[-1.0] * k for k in range(MAX_PLAYERS + 1)]
_ROWS = np.arange(COUNT_BUCKETS).reshape(-1, 1)

O_HAND = 0
O_HIST = O_HAND + NUM_RANKS * COUNT_BUCKETS  # 52
O_UNSEEN = O_HIST + NUM_RANKS  # 65
O_PASS = O_UNSEEN + NUM_RANKS * COUNT_BUCKETS  # 117
O_PASS_MULTI = O_PASS + MAX_PLAYERS * NUM_RANKS  # 208
O_LAST = O_PASS_MULTI + MAX_PLAYERS * NUM_RANKS  # 299
O_OPP = O_LAST + LAST_MOVE_DIM  # 320
O_TAIL_SMALL_END = O_OPP + 3 * MAX_PLAYERS + MAX_PLAYERS  # 348 (opp + role + status)
O_PILE = O_TAIL_SMALL_END  # 348
O_MISC = O_PILE + NUM_RANKS + 2  # 363
assert O_MISC + 2 == STATE_DIM

_SEAT_ORDER = {
    n: [[(me + i) % n for i in range(n)] for me in range(n)]
    for n in range(1, MAX_PLAYERS + 1)
}

_PLAY_IDX = np.array(
    [
        [_ACTION_LOOKUP[(r + 3) * 25 + k * 5] for k in range(1, COUNT_BUCKETS + 1)]
        for r in range(NUM_RANKS)
    ],
    dtype=np.int64,
)

_SLOT = np.arange(COUNT_BUCKETS)


def action_indices(moves) -> np.ndarray:
    return _ACTION_LOOKUP[[c * 25 + n * 5 + t for c, n, t in moves]]


def vectorize_state(state, my_id=0):
    out = np.empty(STATE_DIM, dtype=np.float32)
    hand = state["hand"]
    hand_counts = _rank_counts(hand)
    history = np.array(state["history_vector"])

    out[O_HAND:O_HIST] = (_ROWS < hand_counts).ravel()
    out[O_HIST:O_UNSEEN] = history * (1.0 / COUNT_BUCKETS)
    unseen = np.maximum(0, COUNT_BUCKETS - (hand_counts + history))
    out[O_UNSEEN:O_PASS] = (_ROWS < unseen).ravel()
    _fill_tail(out, state, my_id, len(hand))
    return out


def possible_passes_for(state):
    _, _, count = next(
        ((hr, lr, c) for hr, lr, c in state["role_pairs"] if hr == state["my_role"]),
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
    return count, possible_passes


def _rank_counts(cards):
    if not cards:
        return np.zeros(NUM_RANKS, dtype=np.int64)
    return np.bincount(cards, minlength=16)[3:]


def _fill_tail(out, state, my_id, own_hand_len):
    opp_hand_counts = state["opp_hand_counts"]
    n = len(opp_hand_counts) + 1
    order = _SEAT_ORDER[n][my_id]
    live = n * NUM_RANKS

    # Seat-relative pass matrices, clipped to [0, PASS_CAP] and scaled.
    # Counts are small non-negative ints, so bytes() packs the rows at C speed
    # and one table lookup gives min(k, cap) / cap.
    pbp = state["passed_by_player"]
    pbm = state["passed_multi_by_player"]

    try:
        packed = b"".join(
            [bytes(pbp[i]) for i in order] + [bytes(pbm[i]) for i in order]
        )
        vals = _PASS_LUT[np.frombuffer(packed, dtype=np.uint8)]
    except ValueError:  # a count > 255 cannot be packed into a byte
        vals = np.concatenate(
            [np.array(pbp)[order].ravel(), np.array(pbm)[order].ravel()]
        )
        vals = (np.minimum(vals, PASS_CAP) * _PASS_SCALE).astype(np.float32)

    if n == MAX_PLAYERS:
        out[O_PASS:O_LAST] = vals
    else:
        out[O_PASS : O_PASS + live] = vals[:live]
        out[O_PASS + live : O_PASS_MULTI] = -1.0
        out[O_PASS_MULTI : O_PASS_MULTI + live] = vals[live:]
        out[O_PASS_MULTI + live : O_LAST] = -1.0

    p_card, p_count, p_twos = state["last_move"]
    out[O_LAST:O_OPP] = _MOVE_TABLE[p_card * 25 + p_count * 5 + p_twos, :LAST_MOVE_DIM]

    # opp counts (own hand first), roles, pile status, pile, misc: one Python
    # list and a single slice assignment instead of several small numpy writes
    roles = state["player_roles"]
    active = state["active_players"]
    passed = state["passed"]
    pad = _PAD[MAX_PLAYERS - n]
    tail = [own_hand_len / NUM_RANKS]
    tail += [opp_hand_counts[pid] / NUM_RANKS for pid in order[1:]]
    tail += pad
    tail += [ROLE_MAP.get(roles[pid], 0.5) for pid in order]
    tail += pad

    for pid in order:
        tail.append(1.0 if pid in active else 0.0)
        tail.append(1.0 if pid in passed else 0.0)

    tail += pad
    tail += pad

    pile = state["cards_in_pile"]
    pile_counts = [0.0] * NUM_RANKS

    for card in pile:
        pile_counts[card - 3] += 1.0 / COUNT_BUCKETS

    tail += pile_counts
    tail.append(len(pile) / COUNT_BUCKETS)
    leader = state["pile_leader"]
    tail.append(float((leader - my_id) % n) / float(n) if leader is not None else -1.0)
    tail.append(1.0 if state["is_finish_prompt"] else 0.0)
    tail.append(1.0 if state["first_turn"] else 0.0)
    out[O_OPP:] = tail


def pass_candidates(state, my_id, count, possible_passes):
    hand = state["hand"]
    base_counts = _rank_counts(hand)
    history = np.array(state["history_vector"])

    m = len(possible_passes)
    if m == 0:
        return None

    combo_counts = np.zeros((m, 16), dtype=np.int64)
    for row, combo in enumerate(possible_passes):
        for c in combo:
            combo_counts[row, c] += 1

    counts = base_counts[None, :] - combo_counts[:, 3:]  # [M, 13]
    keep = counts.any(axis=1)

    if not keep.all():
        counts = counts[keep]

    combo_indices = np.flatnonzero(keep)
    if combo_indices.size == 0:
        return None

    k = counts.shape[0]
    x = np.empty((k, STATE_DIM), dtype=np.float32)
    x[:, O_HAND:O_HIST] = (_ROWS[None, :, :] < counts[:, None, :]).reshape(k, -1)
    unseen = np.maximum(0, COUNT_BUCKETS - (counts + history[None, :]))
    x[:, O_UNSEEN:O_PASS] = (_ROWS[None, :, :] < unseen[:, None, :]).reshape(k, -1)

    # Shared part (history + tail) computed once and broadcast
    proto = np.empty(STATE_DIM, dtype=np.float32)
    proto[O_HIST:O_UNSEEN] = history * (1.0 / COUNT_BUCKETS)
    _fill_tail(proto, state, my_id, len(hand) - count)
    x[:, O_HIST:O_UNSEEN] = proto[O_HIST:O_UNSEEN]
    x[:, O_PASS:] = proto[O_PASS:]
    return x, counts, combo_indices


def pick_best_pass(q_values, counts, combo_indices, possible_passes, top_k=3):
    rows = counts.shape[0]
    scores = q_values[:, _PLAY_IDX]  # [rows, 13, 4]
    valid = _SLOT[None, None, :] < counts[:, :, None]
    scores = np.where(valid, scores, np.float32(-np.inf)).reshape(rows, -1)
    top = -np.sort(-scores, axis=1)[:, :top_k]
    n_valid = counts.sum(axis=1)

    if (n_valid >= top_k).all():
        avgs = np.mean(top, axis=1)
    else:
        avgs = np.empty(rows, dtype=np.float32)
        for row in range(rows):
            k = min(top_k, int(n_valid[row]))
            avgs[row] = np.mean(top[row, :k])

    best_pass = possible_passes[combo_indices[int(np.argmax(avgs))]]
    return list(best_pass) if best_pass else []


def run_sync(gen, device):
    """Drive a request generator to the end with unbatched forwards."""
    try:
        model, x = next(gen)
        while True:
            with torch.inference_mode():
                q = model(torch.from_numpy(x).to(device)).cpu().numpy()
            model, x = gen.send(q)
    except StopIteration as stop:
        return stop.value


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


class PresidentEnsembleNet(nn.Module):
    """Averages the Q-values of value nets. Only nets from independently
    trained runs help: checkpoints from one run play almost identically."""

    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, x):
        return torch.stack([m(x) for m in self.members]).mean(0)


class PresidentDMCBot(Player):
    def __init__(
        self,
        player_id,
        model: nn.Module,
        device,
        explore=False,
        temperature=0.05,
        profile=None,
        record=None,
    ):
        if record is None:
            record = explore
        if record and profile is not None:
            raise ValueError(
                "Bots with an enabled profile must not record trajectories."
            )

        self.player_id = player_id
        # .to() walks every submodule: skip it when the
        # weights are already on the target device
        first_param = next(model.parameters(), None)
        on_device = first_param is not None and first_param.device == torch.device(
            device
        )
        self.model = model if on_device else model.to(device)
        self.device = device
        self.trajectory = []
        self.explore = explore
        self.record = record
        self.temperature = temperature
        self.profile = profile

        self.n_decisions = 0
        self.n_explored = 0
        self.q_gap_sum = 0.0

    def get_move_gen(self, state: dict):
        legal_moves = state["legal_moves"]
        if not legal_moves:
            return (0, 0, 0)
        if len(legal_moves) == 1:
            return legal_moves[0]

        state_vec = vectorize_state(state, self.player_id)
        legal_idx = action_indices(legal_moves)
        q = yield (self.model, state_vec[None, :])
        q_legal = q[0][legal_idx]

        if self.profile == "aggressive":
            non_pass = legal_idx != PASS_ACTION
            if non_pass.any():
                q_legal = np.where(non_pass, q_legal, -np.inf)

        greedy = int(np.argmax(q_legal))

        if self.explore and self.temperature > 0:
            choice = self._boltzmann_sample(q_legal)
            self.n_decisions += 1

            if choice != greedy:
                self.n_explored += 1
                self.q_gap_sum += float(q_legal[greedy] - q_legal[choice])
        else:
            choice = greedy

        if self.record:
            self.trajectory.append((state_vec, int(legal_idx[choice])))
        return legal_moves[choice]

    def _boltzmann_sample(self, q_legal: np.ndarray) -> int:
        weights = np.exp((q_legal - q_legal.max()) / self.temperature)
        cdf = np.cumsum(weights)
        idx = int(np.searchsorted(cdf, random.random() * cdf[-1], side="right"))
        return min(idx, len(weights) - 1)

    def choose_cards_to_pass_gen(self, state: dict):
        if state["my_role"] not in _HIGH_ROLES:
            return []

        count, possible_passes = possible_passes_for(state)
        built = pass_candidates(state, self.player_id, count, possible_passes)

        if built is None:
            return (
                list(possible_passes[0])
                if possible_passes
                else PresidentBaselineBot(self.player_id).choose_cards_to_pass(state)
            )

        x, counts, combo_indices = built
        q = yield (self.model, x)
        return pick_best_pass(q, counts, combo_indices, possible_passes)

    def get_move(self, state: dict, *args, **kwargs):
        return run_sync(self.get_move_gen(state), self.device)

    def choose_cards_to_pass(self, state: dict, *args, **kwargs):
        return run_sync(self.choose_cards_to_pass_gen(state), self.device)
