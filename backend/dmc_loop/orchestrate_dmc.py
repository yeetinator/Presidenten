from utils import run_orchestrator


def main():
    run_orchestrator("snapshots", "train_dmc", "evaluate_dmc")


if __name__ == "__main__":
    main()
