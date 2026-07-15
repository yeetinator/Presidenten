from playerTypes.ppo_bot import PresidentActorCritic, PresidentPPOBot
from utils import run_evaluation


def main():
    run_evaluation(
        "snapshots_ppo", "PPO", PresidentActorCritic, PresidentPPOBot, "snapshots"
    )


if __name__ == "__main__":
    main()
