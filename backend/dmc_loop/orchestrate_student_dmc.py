from utils import run_orchestrator


def main():
    run_orchestrator(
        "snapshots_student_dmc", "train_student_dmc", "evaluate_student_dmc", "dmc_loop"
    )


if __name__ == "__main__":
    main()
