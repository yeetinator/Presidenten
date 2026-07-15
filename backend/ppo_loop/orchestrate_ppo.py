from dmc_loop.orchestrate_dmc import run_orchestrator


def main():
    run_orchestrator(
        snapshot_dir="snapshots_ppo",
        train_script="train_ppo.py",
        evaluate_script="evaluate_ppo.py",
    )


if __name__ == "__main__":
    main()
