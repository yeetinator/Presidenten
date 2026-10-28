import numpy as np
import pytest

from human_placement import _replay_chunk, logged_game_gen, placement
from playerTypes.baseline_bot import PresidentBaselineBot
from playerTypes.ismcts_bot import PresidentISMCTSBot
from utils.gameplay import seeded_game_gen
from utils.head_to_head import run_h2h
from utils.ladder import norm_score


def run_plain(game):
    """Run a game generator whose bots never ask for a forward."""
    try:
        next(game)
    except StopIteration as stop:
        return stop.value
    raise AssertionError("a bot asked for a model forward")


def test_ismcts_plays_through_the_shared_round_loop():
    bots = {0: PresidentISMCTSBot(0, time_limit=0.01)} | {
        seat: PresidentBaselineBot(seat) for seat in range(1, 5)
    }
    scores, out_orders = run_plain(seeded_game_gen(5, bots, [11, 12, 13]))

    assert len(out_orders) == 3
    assert all(sorted(order) == list(range(5)) for order in out_orders)
    assert sum(points for points, _ in scores.values()) == 3 * (4 + 3 + 2 + 1)


def test_h2h_takes_bots_without_a_model():
    baseline = (None, PresidentBaselineBot)
    s = run_h2h(
        None,
        None,
        4,
        master_classes=baseline,
        old_classes=baseline,
        num_workers=2,
        early_stop=False,
        progress_every=0,
        base_seed=5,
    )
    # The same deterministic bot in every seat is level with the table
    assert s["delta"] == pytest.approx(0.0, abs=1e-12)
    assert set(s["by_players"]) == {4, 5, 6, 7}
    assert s["transitions"]["top_rounds"] > 0


@pytest.mark.parametrize("players", [4, 6, 7])
def test_replay_reproduces_logged_rounds(players):
    records = run_plain(logged_game_gen(players, "Baseline", 3, 6, field="Baseline"))
    tasks = [(i, record, "Baseline") for i, record in enumerate(records)]
    replayed = _replay_chunk(tasks)

    # Later rounds start from the logged roles, so the exchange must match too
    for i, record in enumerate(records):
        logged = norm_score(record["out_order"].index(0), players)
        assert replayed[i] == logged


def test_placement_of_a_stand_in_equal_to_the_bot():
    records = run_plain(logged_game_gen(5, "Baseline", 9, 10, field="Baseline"))
    tasks = [(i, record, "Baseline") for i, record in enumerate(records)]
    replayed = _replay_chunk(tasks)
    scores = np.array([replayed[i] for i in range(len(records))])

    report = placement(records, {"Baseline": scores})
    bot = report["bots"]["Baseline"]
    assert bot["diff"] == 0.0
    assert bot["diff_ci"] == [0.0, 0.0]
    assert bot["p_human_ahead"] == 0.5
