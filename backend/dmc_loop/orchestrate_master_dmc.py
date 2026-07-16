from utils import run_orchestrator


def main():
    run_orchestrator(
        "snapshots_master_dmc", "train_master_dmc", "evaluate_master_dmc", "dmc_loop"
    )


if __name__ == "__main__":
    main()
