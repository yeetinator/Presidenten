import pytest

import main

# The real delays are tuned for watching the table; tests run them 100x faster
SPEEDUP = 100


@pytest.fixture(autouse=True)
def fast_delays(monkeypatch):
    for name in (
        "MESSAGE_DELAY",
        "EXCHANGE_DELAY",
        "BOT_THINK_DELAY",
        "REVEAL_DELAY",
        "PASS_DELAY",
    ):
        monkeypatch.setattr(main, name, getattr(main, name) / SPEEDUP)
    # Long enough that an answered jump-in never times out on a slow machine
    monkeypatch.setattr(main, "JUMP_IN_WINDOW", 0.3)
