from playerTypes.dmc_bot import PresidentValueNet, StudentDMCBot
from utils import run_evaluation


def main():
    run_evaluation(
        "snapshots_student_dmc",
        "Student",
        PresidentValueNet,
        StudentDMCBot,
        "snapshots",
    )


if __name__ == "__main__":
    main()
