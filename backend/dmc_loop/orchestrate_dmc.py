from utils import run_orchestrator


def main():
    run_orchestrator("snapshots", "train_dmc", "evaluate_dmc", "dmc_loop")


if __name__ == "__main__":
    main()
