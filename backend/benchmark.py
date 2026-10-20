"""from playerTypes.dmc_bot_copy import (
    PresidentDMCBot as CopyBot,
    PresidentValueNet as CopyValueNet,
)"""

from utils.head_to_head import run_h2h, format_h2h

# TOTAL_GAMES = 792
TOTAL_GAMES = 1200

MASTER_PATH = "snapshots/bench_refs/cycle_21.pt"
OLD_PATH = "snapshots/bench_refs/cycle_15.pt"


if __name__ == "__main__":
    report = format_h2h(
        run_h2h(
            MASTER_PATH,
            OLD_PATH,
            TOTAL_GAMES,
            # old_classes=(CopyValueNet, CopyBot),
        )
    )
    print(report)
