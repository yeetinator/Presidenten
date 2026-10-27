import numpy as np
import torch
from .player import Player
from .dmc_bot import (
    COUNT_BUCKETS,
    MAX_PLAYERS,
    NUM_RANKS,
    STATE_DIM,
    _SEAT_ORDER,
    _rank_counts,
    action_indices,
    run_sync,
    vectorize_state,
    PresidentDMCBot,
    PresidentValueNet,
)
from utils.common import get_cached_traced_model

DMC_ENSEMBLE = ("bots/DMC.pt", "bots/DMCW.pt")
OPP_SEATS = MAX_PLAYERS - 1
OPP_HANDS_DIM = OPP_SEATS * COUNT_BUCKETS * NUM_RANKS
CRITIC_DIM = STATE_DIM + OPP_HANDS_DIM


class PresidentPolicyNet(PresidentValueNet):
    """The value net's layers with the head read as move logits. Sharing the
    layout lets DMC weights load as a starting point."""


class PresidentCriticNet(PresidentValueNet):
    """Expected round placement from the actor's state plus every opponent's
    hand. Training only: it sees what the actor can't."""

    def __init__(self, input_dim=CRITIC_DIM, width=1024):
        super().__init__(input_dim, 1, width)

    def forward(self, x):
        return super().forward(x).squeeze(-1)


def opponent_hand_counts(env, my_id):
    """Rank counts of every opponent's hand in the actor's seat order, -1 rows
    for absent seats. This is hidden information, only ever fed to the critic."""
    out = np.full((OPP_SEATS, NUM_RANKS), -1, dtype=np.int8)
    for row, p_id in enumerate(_SEAT_ORDER[env.players][my_id][1:]):
        out[row] = _rank_counts(env.hands[p_id])
    return out


def critic_input(x, opp_counts):
    """Torch: actor states [B, STATE_DIM] and opponent counts [B, 6, 13] ->
    critic inputs [B, CRITIC_DIM]. Hands use the actor's own-hand encoding."""
    counts = opp_counts.to(x.device, torch.float32).unsqueeze(2)  # [B, 6, 1, 13]
    slots = torch.arange(COUNT_BUCKETS, device=x.device, dtype=torch.float32)
    hands = (slots.view(1, 1, -1, 1) < counts).to(x.dtype)  # [B, 6, 4, 13]
    hands = torch.where(counts < 0, -1.0, hands)
    return torch.cat([x, hands.flatten(1)], dim=1)


class PresidentPPOBot(Player):
    def __init__(
        self,
        player_id,
        model: torch.nn.Module,
        device,
        sample=False,
        record=False,
        critic_env=None,
        passer=DMC_ENSEMBLE,
        seed=None,
    ):
        if record and critic_env is None:
            raise ValueError("Recording bots need critic_env for the critic inputs.")

        self.player_id = player_id
        first_param = next(model.parameters(), None)
        on_device = first_param is not None and first_param.device == torch.device(
            device
        )
        self.model = model if on_device else model.to(device)
        self.device = device
        self.sample = sample
        self.record = record
        self.critic_env = critic_env
        # Own generator: drawing from the global random would tie the samples
        # to the seeded deals of benchmarks
        self.rng = np.random.default_rng(seed)
        self.trajectory = []

        # Passing is settled on the DMC passer, and the same passer on both sides
        # of a head-to-head leaves only the play to compare
        if isinstance(passer, torch.nn.Module):
            passer_model = passer
        else:
            passer_model = get_cached_traced_model(
                passer, PresidentValueNet, device=device
            )
        self.passer = PresidentDMCBot(player_id, passer_model, device)

    def get_move_gen(self, state: dict):
        legal_moves = state["legal_moves"]
        if not legal_moves:
            return (0, 0, 0)
        if len(legal_moves) == 1:
            return legal_moves[0]

        state_vec = vectorize_state(state, self.player_id)
        legal_idx = action_indices(legal_moves)
        logits = yield (self.model, state_vec[None, :])
        legal_logits = logits[0][legal_idx].astype(np.float64)

        log_probs = legal_logits - legal_logits.max()
        log_probs -= np.log(np.exp(log_probs).sum())

        if self.sample:
            choice = int(self.rng.choice(len(log_probs), p=np.exp(log_probs)))
        else:
            choice = int(np.argmax(legal_logits))

        if self.record:
            self.trajectory.append(
                (
                    state_vec,
                    legal_idx,
                    int(legal_idx[choice]),
                    float(log_probs[choice]),
                    opponent_hand_counts(self.critic_env, self.player_id),
                )
            )
        return legal_moves[choice]

    def choose_cards_to_pass_gen(self, state: dict):
        return (yield from self.passer.choose_cards_to_pass_gen(state))

    def get_move(self, state: dict, *args, **kwargs):
        return run_sync(self.get_move_gen(state), self.device)

    def choose_cards_to_pass(self, state: dict, *args, **kwargs):
        return run_sync(self.choose_cards_to_pass_gen(state), self.device)
