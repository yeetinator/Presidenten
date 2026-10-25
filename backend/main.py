import asyncio
import functools
import itertools
from dataclasses import dataclass, field
from pathlib import Path

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from game import President, PlayerType
from game.engine import MAPPING_SV, MAPPING_VS
from playerTypes.player import Player
from playerTypes.random_bot import PresidentRandomBot
from playerTypes.baseline_bot import PresidentBaselineBot
from playerTypes.dmc_bot import (
    PresidentDMCBot,
    PresidentEnsembleNet,
    PresidentValueNet,
)

app = FastAPI(title="President Game Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MESSAGE_DELAY = 1.5
EXCHANGE_DELAY = 1.2
BOT_THINK_DELAY = 0.5
JUMP_IN_WINDOW = 1.5
REVEAL_DELAY = 2
PASS_DELAY = 1.0

MIN_PLAYERS = 4
MAX_PLAYERS = 7

PLAYER_TYPE_LABELS = {
    PlayerType.HUMAN: "Human",
    PlayerType.RANDOM: "Random",
    PlayerType.BASELINE: "Baseline",
    PlayerType.DMC: "DMC",
}

DISCONNECT_MESSAGE_TYPE = "__DISCONNECT__"

# Every prompt for player input gets a fresh id that its reply must echo
prompt_ids = itertools.count(1)


@dataclass
class GameTimings:
    message_delay: float = MESSAGE_DELAY
    exchange_delay: float = EXCHANGE_DELAY
    bot_think_delay: float = BOT_THINK_DELAY
    jump_in_window: float = JUMP_IN_WINDOW
    reveal_delay: float = REVEAL_DELAY
    pass_delay: float = PASS_DELAY

    def set_fast_forward(self):
        self.message_delay = 0.75
        self.bot_think_delay = 0.25
        self.pass_delay = 0.5

    def reset(self):
        self.message_delay = MESSAGE_DELAY
        self.exchange_delay = EXCHANGE_DELAY
        self.bot_think_delay = BOT_THINK_DELAY
        self.jump_in_window = JUMP_IN_WINDOW
        self.reveal_delay = REVEAL_DELAY
        self.pass_delay = PASS_DELAY


@dataclass
class GameSession:
    env: President | None = None
    assigned_players: dict[int, Player] = field(default_factory=dict)
    assign_p: dict[int, PlayerType] = field(default_factory=dict)
    human_id: int = 0
    timings: GameTimings = field(default_factory=GameTimings)
    ready_event: asyncio.Event = field(default_factory=asyncio.Event)
    next_round_event: asyncio.Event = field(default_factory=asyncio.Event)
    dealt_cards_event: asyncio.Event = field(default_factory=asyncio.Event)


def _parse_helper(suits):
    return [
        (
            MAPPING_SV[r]
            if (r := suit if len(suit) == 1 else suit[:-1]) in MAPPING_SV
            else int(r)
        )
        for suit in suits
    ]


def parse_suits_to_move(suits):
    if not suits:
        return (0, 0, 0)

    vals = _parse_helper(suits)
    count = len(vals)
    non_twos = [v for v in vals if v != 15]
    card_val = non_twos[0] if non_twos else 15
    twos_used = count - len(non_twos) if non_twos else 0

    return (card_val, count, twos_used)


def parse_suits_to_selection(suits):
    if not suits:
        return []
    return _parse_helper(suits)


def parse_moves_to_suits(moves: list):
    suits_moves = []

    for card_val, count, twos_used in moves:
        if card_val == 0:
            continue

        suit = MAPPING_VS.get(card_val, str(card_val))
        suits_moves.append([suit] * (count - twos_used) + ["2"] * twos_used)
    return suits_moves


def get_finish_suits(suits, state):
    if not suits:
        return []
    return [card for card in state["suited_hand"] if card.startswith(suits[0])]


def make_json_serializable(state: dict):
    clean_state = state.copy()
    if "passed" in clean_state and isinstance(clean_state["passed"], set):
        clean_state["passed"] = list(clean_state["passed"])

    if "active_players" in clean_state and isinstance(
        clean_state["active_players"], set
    ):
        clean_state["active_players"] = list(clean_state["active_players"])
    return clean_state


def enrich_state(
    state: dict, assign_p: dict[int, PlayerType], *, accepts_moves: bool = False
):
    clean_state = make_json_serializable(state)
    clean_state["player_types"] = {
        p_id: get_player_type_label(player_type)
        for p_id, player_type in assign_p.items()
    }
    # Only offer moves while the server is actually waiting for one, so the client
    # can't queue a move from a state that is about to change (e.g. pile clearing)
    legal_moves = state["legal_moves"] if accepts_moves else []
    clean_state["legal_moves"] = legal_moves
    clean_state["can_pass"] = (0, 0, 0) in legal_moves
    clean_state["legal_moves_suits"] = parse_moves_to_suits(legal_moves)

    return clean_state


def get_exchange_count(env: President, role: str) -> int:
    for high_role, low_role, count in env.role_pairs:
        if role == high_role or role == low_role:
            return count
    return 0


def get_player_type_label(player_type: PlayerType) -> str:
    return PLAYER_TYPE_LABELS.get(player_type, "Unknown")


async def websocket_router(
    websocket: WebSocket,
    play_queue: asyncio.Queue,
    control_queue: asyncio.Queue,
    disconnect_event: asyncio.Event,
):
    try:
        while True:
            message = await websocket.receive_json()
            message_type = message.get("type")

            if message_type in {"PLAY_MOVE", "EXCHANGE_CARDS"}:
                await play_queue.put(message)
            else:
                await control_queue.put(message)
    except WebSocketDisconnect:
        pass
    finally:
        disconnect_event.set()
        sentinel = {"type": DISCONNECT_MESSAGE_TYPE}
        await play_queue.put(sentinel)
        await control_queue.put(sentinel)


async def next_queued_message(
    input_queue: asyncio.Queue,
    pending_messages: list[dict],
    disconnect_event: asyncio.Event,
    *,
    allowed_types: set[str] | None = None,
    timeout: float | None = None,
):
    if disconnect_event.is_set():
        raise WebSocketDisconnect

    if allowed_types is not None:
        for idx, pending_message in enumerate(pending_messages):
            if pending_message.get("type") in allowed_types:
                return pending_messages.pop(idx)

    while True:
        if disconnect_event.is_set():
            raise WebSocketDisconnect

        if timeout is None:
            raw_data = await input_queue.get()
        else:
            raw_data = await asyncio.wait_for(input_queue.get(), timeout=timeout)

        if raw_data.get("type") == DISCONNECT_MESSAGE_TYPE:
            raise WebSocketDisconnect

        if allowed_types is None or raw_data.get("type") in allowed_types:
            return raw_data

        pending_messages.append(raw_data)


async def wait_for_reply(
    input_queue: asyncio.Queue,
    pending_messages: list[dict],
    disconnect_event: asyncio.Event,
    message_type: str,
    prompt_id: int,
    *,
    timeout: float | None = None,
) -> dict:
    """Next `message_type` message answering prompt `prompt_id`; replies to earlier
    prompts (e.g. a jump-in sent after its window closed) are dropped."""
    loop = asyncio.get_running_loop()
    deadline = None if timeout is None else loop.time() + timeout

    while True:
        remaining = None if deadline is None else max(0.0, deadline - loop.time())
        message = await next_queued_message(
            input_queue,
            pending_messages,
            disconnect_event,
            allowed_types={message_type},
            timeout=remaining,
        )
        if message.get("prompt_id") == prompt_id:
            return message


def is_offered_move(move: tuple, suits: list[str], prompt_state: dict) -> bool:
    """Whether the prompt offered this move and the suit cards are in the hand."""
    if move not in prompt_state["legal_moves"]:
        return False

    hand = prompt_state["suited_hand"]
    ranks = {v for v in _parse_helper(suits) if v != 15}
    return (
        len(set(suits)) == len(suits)
        and all(card in hand for card in suits)
        and len(ranks) <= 1
    )


def parse_start_game(message: dict) -> dict[int, PlayerType]:
    """Seat -> player type from a START_GAME message; raises ValueError if invalid."""
    num_players = message.get("num_players")
    player_types = message.get("player_types")

    if type(num_players) is not int or not MIN_PLAYERS <= num_players <= MAX_PLAYERS:
        raise ValueError(f"A game needs {MIN_PLAYERS} to {MAX_PLAYERS} players.")
    if not isinstance(player_types, list) or len(player_types) != num_players:
        raise ValueError("Every seat needs a player type.")

    try:
        seats = [PlayerType(player_type) for player_type in player_types]
    except ValueError:
        raise ValueError("Unknown player type.") from None

    if any(seat not in PLAYER_TYPE_LABELS for seat in seats):
        raise ValueError("That player type isn't available in the web game.")
    if seats[0] != PlayerType.HUMAN or seats.count(PlayerType.HUMAN) != 1:
        raise ValueError("Seat 0 must be the only human player.")

    return dict(enumerate(seats))


async def control_listener(
    websocket: WebSocket,
    control_queue: asyncio.Queue,
    session: GameSession,
    disconnect_event: asyncio.Event,
):
    while not disconnect_event.is_set():
        try:
            control_message = await next_queued_message(
                control_queue,
                [],
                disconnect_event,
                allowed_types={
                    "START_GAME",
                    "NEXT_ROUND",
                    "FAST_FORWARD",
                    "DEALT_CARDS",
                },
            )
        except WebSocketDisconnect:
            return

        message_type = control_message.get("type")

        if message_type == "FAST_FORWARD":
            session.timings.set_fast_forward()
            continue

        if message_type == "START_GAME":
            try:
                if session.env is not None:
                    raise ValueError("A game is already running.")
                assign_p = parse_start_game(control_message)
            except ValueError as e:
                await websocket.send_json({"type": "START_REJECTED", "message": str(e)})
                continue

            session.assign_p = assign_p
            session.env = President(len(assign_p), track_suits=True)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            session.assigned_players, session.human_id = build_assigned_players(
                session.assign_p, device
            )
            session.env.full_reset(next_round=False)
            session.ready_event.set()
            continue

        if message_type == "NEXT_ROUND":
            session.next_round_event.set()
            continue

        if message_type == "DEALT_CARDS":
            session.dealt_cards_event.set()


async def wait_for_event(event: asyncio.Event, disconnect_event: asyncio.Event):
    if not event.is_set():
        waiters = {
            asyncio.create_task(event.wait()),
            asyncio.create_task(disconnect_event.wait()),
        }
        try:
            await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for waiter in waiters:
                waiter.cancel()

    if not event.is_set():
        raise WebSocketDisconnect


# Models from independently trained runs, averaged into one ensemble
DMC_MODEL_FILES = ("DMC.pt", "DMCW.pt")


@functools.cache  # loaded once per device and shared by every game (inference only)
def load_dmc_model(device: torch.device) -> PresidentEnsembleNet:
    bots_dir = Path(__file__).resolve().parent / "bots"
    members = []

    for file_name in DMC_MODEL_FILES:
        model_path = bots_dir / file_name
        model = PresidentValueNet().to(device)
        try:
            load_path = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(load_path["model_state_dict"])
        except Exception as e:
            print(
                f"Warning: Could not load weights from {model_path}. Reason: {e}. Leaving it out of the ensemble."
            )
            continue
        members.append(model)

    if not members:
        print("Warning: No DMC weights could be loaded. Using untrained model instead.")
        members.append(PresidentValueNet().to(device))

    return PresidentEnsembleNet(members).eval()


def build_assigned_players(
    assign_p: dict[int, PlayerType], device: torch.device
) -> tuple[dict[int, Player], int]:
    assigned_players: dict[int, Player] = {}
    human_id = 0

    for p_id, p_type in assign_p.items():
        if p_type == PlayerType.HUMAN:
            human_id = p_id
        elif p_type == PlayerType.RANDOM:
            assigned_players[p_id] = PresidentRandomBot(p_id)
        elif p_type == PlayerType.BASELINE:
            assigned_players[p_id] = PresidentBaselineBot(p_id)
        elif p_type == PlayerType.DMC:
            assigned_players[p_id] = PresidentDMCBot(
                p_id, load_dmc_model(device), device
            )

    return assigned_players, human_id


def finisher(env: President) -> int | None:
    """The player who may jump in to finish the quad on the pile, if any."""
    return env.pending_finish["queue"][0][2] if env.pending_finish else None


@dataclass
class Table:
    """A game in progress and the connection it is played over."""

    session: GameSession
    env: President  # session.env, which is set once the game has started
    websocket: WebSocket
    input_queue: asyncio.Queue
    pending_messages: list[dict]
    disconnect_event: asyncio.Event

    @property
    def human_id(self) -> int:
        return self.session.human_id

    @property
    def timings(self) -> GameTimings:
        return self.session.timings

    def bot(self, p_id: int) -> Player:
        return self.session.assigned_players[p_id]

    def is_human(self, p_id: int) -> bool:
        return self.session.assign_p[p_id] == PlayerType.HUMAN

    def view(self, state: dict, *, accepts_moves: bool = False) -> dict:
        return enrich_state(state, self.session.assign_p, accepts_moves=accepts_moves)

    async def show_state(self):
        """Send the human's view of the table (offering no moves)."""
        state = self.view(self.env._get_state(self.human_id))
        await self.websocket.send_json({"type": "STATE_UPDATE", "state": state})

    async def clear_pile_and_show(self):
        self.env.clear_pile()
        await self.show_state()

    async def ask(
        self, message: dict, reply_type: str, *, timeout: float | None = None
    ) -> dict:
        """Send a prompt with a fresh id and wait for the reply that echoes it."""
        prompt_id = next(prompt_ids)
        await self.websocket.send_json({**message, "prompt_id": prompt_id})
        return await wait_for_reply(
            self.input_queue,
            self.pending_messages,
            self.disconnect_event,
            reply_type,
            prompt_id,
            timeout=timeout,
        )


async def run_exchange_phase(table: Table):
    env, human_id = table.env, table.human_id
    cards_to_pass: dict[int | str, list[int]] = {}
    human_role = env.roles[human_id]
    required_cards = get_exchange_count(env, human_role)
    can_choose = human_role in {"President", "Vice-President", "Secretary"}

    for p_id, role in env.roles.items():
        if role == "Citizen" or p_id == human_id:
            continue

        state = env._get_state(p_id)
        bot = table.bot(p_id)
        cards_to_pass[p_id] = await asyncio.to_thread(bot.choose_cards_to_pass, state)

    reply = await table.ask(
        {
            "type": "EXCHANGE_PROMPT",
            "state": table.view(env._get_state(human_id)),
            "required_cards": required_cards,
            "can_choose": can_choose,
        },
        "EXCHANGE_CARDS",
    )
    suits = reply.get("suits", []) if can_choose else []
    cards_to_pass[human_id] = parse_suits_to_selection(suits)

    for pair in env.role_pairs:
        env.exchange_cards(pair, cards_to_pass)
        await table.show_state()
        await asyncio.sleep(table.timings.exchange_delay)


async def play_round(table: Table):
    env = table.env
    while not env.game_over:
        curr_id = env.curr_turn
        if curr_id is None:
            break

        if env.pending_finish:  # the last play can still be finished into a quad
            if curr_id == table.human_id or finisher(env) == table.human_id:
                curr_id = await offer_human_jump_in(table, curr_id)
                if curr_id is None:
                    continue
            else:
                await play_bot_finish(table, curr_id)
                continue

        if table.is_human(curr_id):
            await play_human_turn(table)
        else:
            await play_bot_turn(table, curr_id)

    await end_round(table)


async def offer_human_jump_in(table: Table, curr_id: int) -> int | None:
    """Give the human the jump-in window. Returns None when that settles the turn, or
    whose turn it is instead when the window closes unused."""
    env, human_id, timings = table.env, table.human_id, table.timings
    jump_state = env._get_state(human_id)
    try:
        reply = await table.ask(
            {
                "type": "JUMP_IN_PROMPT",
                "state": table.view(jump_state, accepts_moves=True),
                "message": "JUMP IN!",
            },
            "PLAY_MOVE",
            timeout=timings.jump_in_window,
        )
    except asyncio.TimeoutError:
        pending = env.pending_finish
        if pending:
            if (
                curr_id == env._get_next_active_player(pending["resume_turn"])
                and pending["resume_played"]
            ):
                pending["resume_turn"] = curr_id

            curr_id = pending["resume_turn"]
            pending["resume_played"] = True
        return curr_id

    if not reply.get("jump", True):
        return curr_id

    suits = get_finish_suits(reply.get("suits", []), jump_state)
    move = parse_suits_to_move(suits)
    if not is_offered_move(move, suits, jump_state):
        return None  # ask again with a fresh prompt

    env.step(human_id, move, suits)
    await table.show_state()
    await asyncio.sleep(timings.message_delay)

    if env.was_pile_reset:
        env.clear_pile()
        if env.curr_turn != human_id:
            await table.show_state()
    return None


async def play_bot_finish(table: Table, bot_id: int):
    """The bot that could finish the quad on the pile decides whether to jump in."""
    env, timings = table.env, table.timings
    state = env._get_state(bot_id)
    await asyncio.sleep(timings.bot_think_delay)

    move = await asyncio.to_thread(table.bot(bot_id).get_move, state, env)
    env.step(bot_id, move)
    if move != (0, 0, 0):
        await table.show_state()
        await asyncio.sleep(timings.message_delay)

        if env.was_pile_reset:
            await table.clear_pile_and_show()
    elif not env.pending_finish:
        await asyncio.sleep(timings.pass_delay)
    if env.was_pile_reset:
        await table.clear_pile_and_show()
        await asyncio.sleep(timings.message_delay)


async def play_human_turn(table: Table):
    env, human_id, timings = table.env, table.human_id, table.timings
    may_still_jump = bool(env.pending_finish)  # right after an unused jump-in window
    prompt_state = env._get_state(human_id, only_finish=not may_still_jump)
    reply = await table.ask(
        {
            "type": "STATE_UPDATE",
            "state": table.view(prompt_state, accepts_moves=True),
            "clearJump": not may_still_jump,
        },
        "PLAY_MOVE",
    )

    suits = reply.get("suits", [])
    is_jump = bool(reply.get("jump"))
    if is_jump:
        suits = get_finish_suits(suits, prompt_state)
    move = parse_suits_to_move(suits)

    # A finish option can only be taken as a jump, never as a normal play
    finish_moves = (
        set(env.get_legal_moves(human_id)) - {(0, 0, 0)}
        if prompt_state["is_finish_prompt"]
        else set()
    )
    if not is_offered_move(move, suits, prompt_state) or (
        move != (0, 0, 0) and (move in finish_moves) != is_jump
    ):
        return  # ask again with a fresh prompt

    if not is_jump and finisher(env) == human_id:
        # Playing normally declines the jump-in, but a bot may still finish first
        env.step(human_id, (0, 0, 0))
        bot_id = finisher(env)
        if bot_id is not None:
            if bot_id == human_id:
                env.step(human_id, (0, 0, 0))
            else:
                bot_state = env._get_state(bot_id)
                bot_move = await asyncio.to_thread(
                    table.bot(bot_id).get_move, bot_state, env
                )
                if bot_move != (0, 0, 0):
                    env.curr_turn = bot_id
                    return
                env.step(bot_id, (0, 0, 0))

    env.step(human_id, move, suits)
    await table.show_state()

    if env.was_pile_reset and not env.pending_finish and not env.game_over:
        await asyncio.sleep(timings.message_delay)
        await table.clear_pile_and_show()
    else:
        await asyncio.sleep(
            timings.message_delay if move != (0, 0, 0) else timings.pass_delay
        )


async def play_bot_turn(table: Table, bot_id: int):
    env, human_id, timings = table.env, table.human_id, table.timings
    move = await asyncio.to_thread(
        table.bot(bot_id).get_move, env._get_state(bot_id), env
    )

    if move != (0, 0, 0) and finisher(env) == human_id:
        # The bot played before the human jumped in, so the finish lapses
        env.step(human_id, (0, 0, 0))

        if env.pending_finish:
            if finisher(env) == human_id:
                env.step(human_id, (0, 0, 0))
            else:
                return
    env.step(bot_id, move)
    if env.curr_turn != human_id or env.was_pile_reset:
        await table.show_state()

    if env.was_pile_reset and not env.game_over:
        await asyncio.sleep(timings.message_delay)

        env.clear_pile()
        if env.curr_turn == human_id:
            return  # the human's prompt shows the cleared pile

        await table.show_state()
        await asyncio.sleep(timings.message_delay)
    elif not env.pending_finish and not env.game_over and env.curr_turn != human_id:
        await asyncio.sleep(
            timings.message_delay if move != (0, 0, 0) else timings.pass_delay
        )


async def end_round(table: Table):
    env, timings = table.env, table.timings
    last_active_player = list(env.playing)[0] if env.playing else None
    if last_active_player is None:
        for p_id in range(env.players):
            if env.hands[p_id]:
                last_active_player = p_id
                break

    if last_active_player is not None:
        await asyncio.sleep(timings.message_delay)
        if last_active_player != table.human_id:
            await table.websocket.send_json(
                {"type": "REVEAL_BOT", "seat": last_active_player}
            )
            await asyncio.sleep(timings.reveal_delay)

    env.assign_roles()
    await table.websocket.send_json(
        {
            "type": "ROUND_OVER",
            "scores": env.scores,
            "roles": env.roles,
            "out_order": env.out_order,
        }
    )


async def deal_hands(table: Table):
    session = table.session
    session.dealt_cards_event.clear()  # every deal waits for its own animation
    await table.websocket.send_json(
        {
            "type": "DEAL_CARDS",
            "state": table.view(table.env._get_state(table.human_id)),
        }
    )
    await wait_for_event(session.dealt_cards_event, table.disconnect_event)
    session.timings.reset()


async def run_connection(
    websocket: WebSocket,
    play_queue: asyncio.Queue,
    control_queue: asyncio.Queue,
    disconnect_event: asyncio.Event,
):
    session = GameSession()
    control_task = asyncio.create_task(
        control_listener(websocket, control_queue, session, disconnect_event)
    )

    try:
        await wait_for_event(session.ready_event, disconnect_event)

        if session.env is None:
            return
        table = Table(session, session.env, websocket, play_queue, [], disconnect_event)

        await deal_hands(table)
        if table.env.curr_turn != table.human_id:
            await asyncio.sleep(table.timings.message_delay)
        await play_round(table)

        while not disconnect_event.is_set():
            await wait_for_event(session.next_round_event, disconnect_event)
            session.next_round_event.clear()

            if not table.env.game_over:
                continue

            table.env.full_reset(next_round=True)
            await deal_hands(table)
            await run_exchange_phase(table)
            await table.show_state()
            await asyncio.sleep(table.timings.message_delay)
            await play_round(table)
    except WebSocketDisconnect:
        return
    except RuntimeError:
        # uvicorn raises this instead of WebSocketDisconnect when we send after the client left
        if disconnect_event.is_set():
            return
        raise
    finally:
        control_task.cancel()
        await asyncio.gather(control_task, return_exceptions=True)


@app.websocket("/ws/game")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    play_queue: asyncio.Queue = asyncio.Queue()
    control_queue: asyncio.Queue = asyncio.Queue()
    disconnect_event = asyncio.Event()

    router_task = asyncio.create_task(
        websocket_router(websocket, play_queue, control_queue, disconnect_event)
    )
    game_task = asyncio.create_task(
        run_connection(websocket, play_queue, control_queue, disconnect_event)
    )

    try:
        await asyncio.gather(router_task, game_task)
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise
    finally:
        router_task.cancel()
        game_task.cancel()
        await asyncio.gather(router_task, game_task, return_exceptions=True)
