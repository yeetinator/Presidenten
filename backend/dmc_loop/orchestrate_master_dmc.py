from dmc_loop.orchestrate_dmc import run_orchestrator


def main():
    run_orchestrator(
        snapshot_dir="snapshots_master_dmc",
        train_script="train_master_dmc",
        evaluate_script="evaluate_master_dmc",
    )


if __name__ == "__main__":
    main()
