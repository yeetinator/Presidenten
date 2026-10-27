from ppo_loop.train_ppo import SNAPSHOT_DIR
from utils.elo import run_evaluation
from utils.ppo import DMC_DIR, PPO_CLASSES


def main():
    # The frozen DMC elites join the pool as fixed anchors
    run_evaluation(SNAPSHOT_DIR, "PPO", *PPO_CLASSES, basic_snapshot_dir=DMC_DIR)


if __name__ == "__main__":
    main()
