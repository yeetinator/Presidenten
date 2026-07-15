from playerTypes.dmc_bot import PresidentValueNet, PresidentDMCBot
from utils import run_evaluation


def main():
    run_evaluation("snapshots", "Basic", PresidentValueNet, PresidentDMCBot)


if __name__ == "__main__":
    main()
