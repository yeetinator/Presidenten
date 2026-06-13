"""Compatibility entrypoint that re-exports the real Presidenten web app.

The browser-facing game logic lives in app.py; this module keeps the 8010
launch path aligned with that implementation instead of the old demo arena.
"""

from __future__ import annotations

from app import app
