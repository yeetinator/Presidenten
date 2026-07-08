from playerTypes.master_dmc_bot import MasterDMCBot, MasterValueNet
from evaluate_dmc import run_evaluation


def main():
    run_evaluation(
        net_class=MasterValueNet,
        bot_class=MasterDMCBot,
        input_dim=193,
        snapshot_dir="snapshots_master_dmc",
    )


if __name__ == "__main__":
    main()
