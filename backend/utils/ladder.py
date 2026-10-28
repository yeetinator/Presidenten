import torch
from playerTypes.player import Player
from playerTypes.random_bot import PresidentRandomBot
from playerTypes.baseline_bot import PresidentBaselineBot
from playerTypes.ismcts_bot import PresidentISMCTSBot
from playerTypes.dmc_bot import PresidentDMCBot, PresidentValueNet
from playerTypes.ppo_bot import DMC_ENSEMBLE, PresidentPolicyNet, PresidentPPOBot
from .common import get_cached_traced_model

# The final PPO model (snapshots aren't in the repo)
PPO_PATH = "snapshots_ppo/elites/model_gen_4680.pt"

# name -> (model path, net class, bot class), as run_h2h takes a side. Without
# a path the bot is built as bot_class(seat)
BOTS = {
    "Random": (None, None, PresidentRandomBot),
    "Baseline": (None, None, PresidentBaselineBot),
    "ISMCTS": (None, None, PresidentISMCTSBot),
    "PPO": (PPO_PATH, PresidentPolicyNet, PresidentPPOBot),
    "DMC": (DMC_ENSEMBLE, PresidentValueNet, PresidentDMCBot),
}
# Bots that can play the same round differently, so replays are averaged
STOCHASTIC = {"Random", "ISMCTS"}
# ISMCTS thinks for a fixed time, so it gets one physical core per process
ISMCTS_WORKERS = 8


def make_bot(name, seat) -> Player:
    path, net_class, bot_class = BOTS[name]
    if path is None:
        return bot_class(seat)
    model = get_cached_traced_model(path, net_class)
    return bot_class(seat, model, torch.device("cpu"))


def norm_score(rank, num_players):
    """A finish as a score from -1 (last) to +1 (first)."""
    return (num_players - 1 - rank) / (num_players - 1) * 2 - 1


def beat_rate(norm):
    """The share of opponents finished ahead of: 0.5 is level with the table."""
    return (norm + 1) / 2
