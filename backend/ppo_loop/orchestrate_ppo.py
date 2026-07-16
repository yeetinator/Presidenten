from utils import run_orchestrator


def main():
    run_orchestrator("snapshots_ppo", "train_ppo", "evaluate_ppo", "ppo_loop")


if __name__ == "__main__":
    main()
