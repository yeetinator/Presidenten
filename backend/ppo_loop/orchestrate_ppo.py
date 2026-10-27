from ppo_loop.gates_ppo import check_gate2
from ppo_loop.train_ppo import ITERS_PER_CYCLE, SNAPSHOT_DIR
from utils.orchestrator import run_orchestrator
from utils.ppo import PPO_CLASSES


def main():
    run_orchestrator(
        SNAPSHOT_DIR,
        "train_ppo",
        "evaluate_ppo",
        "ppo_loop",
        classes=PPO_CLASSES,
        batches_per_cycle=ITERS_PER_CYCLE,
        after_cycle=check_gate2,
    )


if __name__ == "__main__":
    main()
