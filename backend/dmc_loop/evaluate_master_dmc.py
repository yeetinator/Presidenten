from playerTypes.dmc_bot import MasterDMCBot, MasterValueNet
from utils import run_evaluation


def main():
    run_evaluation(
        "snapshots_master_dmc", "Master", MasterValueNet, MasterDMCBot, "snapshots"
    )


if __name__ == "__main__":
    main()
