from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any, Callable, Protocol, cast


@dataclass
class GameMessage:
    role: str
    content: str


class Bot(Protocol):
    name: str

    def respond(self, history: list[GameMessage], user_message: str) -> str: ...


@dataclass
class CallableBot:
    name: str
    handler: Callable[..., Any]

    def respond(self, history: list[GameMessage], user_message: str) -> str:
        try:
            return str(self.handler(history, user_message))
        except TypeError:
            return str(self.handler(user_message))


@dataclass
class EchoBot:
    name: str = "Echo Bot"

    def respond(self, history: list[GameMessage], user_message: str) -> str:
        return f"You said: {user_message}"


@dataclass
class TacticalBot:
    name: str = "Tactical Bot"

    def respond(self, history: list[GameMessage], user_message: str) -> str:
        prompt = user_message.strip().lower()
        if not prompt:
            return "I need a move or command."
        if any(word in prompt for word in {"attack", "press", "push"}):
            return "I counter aggressively and force the pace."
        if any(word in prompt for word in {"defend", "block", "hold"}):
            return "I probe for weaknesses while you hold position."
        if any(word in prompt for word in {"draw", "trade", "exchange"}):
            return "I accept the exchange and look for tempo."
        return "I adapt and keep the position balanced."


def _coerce_bot(candidate: object) -> Bot | None:
    if hasattr(candidate, "respond") and hasattr(candidate, "name"):
        return cast(Bot, candidate)
    if callable(candidate):
        name = getattr(candidate, "name", getattr(candidate, "__name__", "Bot"))
        return CallableBot(name=name, handler=candidate)
    return None


def load_external_bots() -> list[Bot]:
    module_name = os.getenv("PRESIDENTEN_BOTS_MODULE")
    if not module_name:
        return []

    try:
        module = importlib.import_module(module_name)
    except Exception:
        return []

    raw_bots: object = []
    if hasattr(module, "get_bots"):
        raw_bots = module.get_bots()
    elif hasattr(module, "BOTS"):
        raw_bots = module.BOTS

    if not isinstance(raw_bots, list):
        return []

    bots: list[Bot] = []
    for candidate in raw_bots:
        bot = _coerce_bot(candidate)
        if bot is not None:
            bots.append(bot)
    return bots


DEFAULT_BOTS: list[Bot] = load_external_bots() or [EchoBot(), TacticalBot()]


def list_bots() -> list[str]:
    return [bot.name for bot in DEFAULT_BOTS]


def get_bot(name: str) -> Bot:
    for bot in DEFAULT_BOTS:
        if bot.name == name:
            return bot
    return DEFAULT_BOTS[0]
