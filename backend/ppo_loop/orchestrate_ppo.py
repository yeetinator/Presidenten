from dmc_loop.orchestrate_dmc import run_orchestrator


def main():
    run_orchestrator(
        snapshot_dir="snapshots_ppo",
        train_script="train_ppo",
        evaluate_script="evaluate_ppo",
    )


if __name__ == "__main__":
    main()
