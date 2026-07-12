import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from playerTypes.dmc_bot import vectorize_state


def generate_action_space():
    actions = [(0, 0, 0)]
    for card in range(3, 15):
        for total_count in range(1, 5):
            for twos in range(0, total_count):
                actions.append((card, total_count, twos))

    for total_count in range(1, 5):
        actions.append((15, total_count, 0))

    move_to_idx = {move: idx for idx, move in enumerate(actions)}
    idx_to_move = {idx: move for idx, move in enumerate(actions)}

    return move_to_idx, idx_to_move, len(actions)


MOVE_TO_IDX, IDX_TO_MOVE, ACTION_DIM = generate_action_space()


class PresidentActorCritic(nn.Module):
    def __init__(self, actor_dim=115, critic_dim=193, action_dim=ACTION_DIM):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(actor_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(critic_dim, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 1),
        )

    def forward_actor(self, obs, mask):
        logits = self.actor(obs)
        logits = logits.masked_fill(~mask, -1e9)
        return Categorical(logits=logits)

    def forward_critic(self, privileged_state):
        return self.critic(privileged_state)


class PresidentPPOBot:
    def __init__(
        self,
        player_id,
        model: PresidentActorCritic,
        device,
        deterministic: bool = False,
    ):
        self.player_id = player_id
        self.model = model
        self.device = device
        self.deterministic = deterministic

    def get_action_mask(self, legal_moves):
        mask = np.zeros(ACTION_DIM, dtype=bool)
        for move in legal_moves:
            if move in MOVE_TO_IDX:
                mask[MOVE_TO_IDX[move]] = True
        return mask

    def get_move(self, state: dict, mask_np: np.ndarray | None = None, *args, **kwargs):
        if mask_np is None:
            mask_np = self.get_action_mask(state["legal_moves"])

        obs = (
            torch.FloatTensor(vectorize_state(state, len(state["opp_hand_counts"]) + 1))
            .unsqueeze(0)
            .to(self.device)
        )
        mask = torch.BoolTensor(mask_np).unsqueeze(0).to(self.device)

        with torch.no_grad():
            dist = self.model.forward_actor(obs, mask)
            if self.deterministic:
                action_idx = torch.argmax(dist.probs).item()
            else:
                action_idx = dist.sample().item()
        return IDX_TO_MOVE[int(action_idx)]
