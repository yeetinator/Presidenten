from __future__ import annotations
import random
import torch
import os
from typing import Callable, overload, TypeVar, Any, TYPE_CHECKING
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from enum import IntEnum

if TYPE_CHECKING:
    from playerTypes.human import HumanPlayer
    from playerTypes.random_bot import PresidentenRandomBot
    from playerTypes.baseline_bot import PresidentenBaselineBot
    from playerTypes.ismcts_bot import PresidentenISMCTSBot
    from playerTypes.dmc_bot import PresidentenDMCBot


class PlayerType(IntEnum):
    HUMAN = 0
    RANDOM = 1
    BASELINE = 2
    ISMCTS = 3
    DMC = 4

    @property
    def name(self):
        return {
            PlayerType.HUMAN: "Human",
            PlayerType.RANDOM: "Random Bot",
            PlayerType.BASELINE: "Baseline Bot",
            PlayerType.ISMCTS: "ISMCTS Bot",
            PlayerType.DMC: "DMC Bot",
        }[self]


T = TypeVar("T")


@overload
def get_val_input(
    prompt: str,
    cast_type: Callable[[str], T],
    valid_choices: Any = None,
    delimiter: None = None,
) -> T: ...
@overload
def get_val_input(
    prompt: str,
    cast_type: Callable[[str], T],
    valid_choices: Any = None,
    delimiter: str = ...,
) -> list[T]: ...
def get_val_input(prompt, cast_type, valid_choices=None, delimiter=None):
    while True:
        try:
            raw = input(prompt).strip()
            if delimiter:
                val = [cast_type(t.strip()) for t in raw.split(delimiter) if t.strip()]
            else:
                val = cast_type(raw)

            if valid_choices is not None:
                if callable(valid_choices):
                    if not valid_choices(val):
                        raise ValueError()
                else:
                    if delimiter:
                        if not all(item in valid_choices for item in val):
                            raise ValueError()
                    elif val not in valid_choices:
                        raise ValueError()
            return val
        except ValueError:
            print(f"Invalid input. Please try again.")


class Presidenten:
    def __init__(self, players=4, verbose=False):
        if players < 4:
            raise ValueError("Presidenten requires at least 4 players.")

        self.players = players
        self.verbose = verbose

        # 3 to Ace (14), plus 2 (15)
        self.deck = [rank for rank in range(3, 16) for _ in range(4)]
        self.hands = {p_id: [] for p_id in range(players)}

        self.scores = {
            p_id: (0, 0) for p_id in range(players)
        }  # (total_score, rounds_won)

        self.roles = {p_id: "Citizen" for p_id in range(players)}
        self.out_order = []  # Track finishing order for role assignment
        self.round = 0
        self.ended_2 = []  # Track players who finished with a 2 for role assignment

        self.history = []  # [(P_id, move), ...]
        self.last_move = (0, 0, 0)  # (card_value, count, twos_used)
        self.pile = []  # Cards currently on the pile

        self.pile_leader = None  # P_id of the player who last played to the pile
        self.passed = set()  # Players who have passed in the current pile
        self.playing = set(range(players))
        self.first_turn = True
        self.curr_turn = None  # P_id of the current player
        self.clubs_3_holder = None
        self.game_over = False
        self.pending_finish = None  # {"queue": [(card, count, player_id), ...], "resume_turn": player_id, "pile_reset": bool}

        self.role_pairs = []
        if self.players >= 4:
            self.role_pairs.append(("President", "Scum", 3 if self.players >= 6 else 2))
            self.role_pairs.append(
                ("Vice-President", "High-Scum", 2 if self.players >= 6 else 1)
            )
        if self.players >= 6:
            self.role_pairs.append(("Secretary", "Clerk", 1))

    def _get_roles(self):
        if self.players == 4:
            return ["President", "Vice-President", "High-Scum", "Scum"]
        elif self.players == 5:
            return ["President", "Vice-President", "Citizen", "High-Scum", "Scum"]
        elif self.players == 6:
            return [
                "President",
                "Vice-President",
                "Secretary",
                "Clerk",
                "High-Scum",
                "Scum",
            ]
        else:
            return (
                ["President", "Vice-President", "Secretary"]
                + ["Citizen"] * (self.players - 6)
                + ["Clerk", "High-Scum", "Scum"]
            )

    def assign_roles(self):
        if not self.out_order:
            return

        for p_id in range(
            self.players
        ):  # Add any remaining players who haven't finished
            if p_id not in self.out_order:
                self.out_order.append(p_id)

        # Move players who finished with a 2 to the end of the order
        for p_id in reversed(self.ended_2):
            if p_id in self.out_order:
                self.out_order.remove(p_id)
                self.out_order.append(p_id)

        roles = self._get_roles()
        for rank, p_id in enumerate(self.out_order):
            self.roles[p_id] = roles[rank]
            self.scores[p_id] = (
                self.scores[p_id][0] + self.players - 1 - rank,
                self.scores[p_id][1] + (1 if rank == 0 else 0),
            )

    def exchange_cards(self, cards_to_pass: dict[int | str, list[int]] | None = None):
        role_to_player = {role: p_id for p_id, role in self.roles.items()}

        # Makes sure no cards are exchanged back and forth in the same round
        staged_outgoing = {p_id: [] for p_id in range(self.players)}
        self.exchange_log = {}

        def pick_cards(p_id, count, highest=False, allow_custom=True):
            hand = self.hands[p_id]
            if allow_custom and cards_to_pass is not None:
                if p_id in cards_to_pass:
                    chosen = list(cards_to_pass[p_id])
                else:
                    chosen = list(cards_to_pass.get(self.roles[p_id], []))

                if len(chosen) != count:
                    raise ValueError(
                        f"Player {p_id} must exchange {count} card(s), got {len(chosen)}."
                    )

                chosen_counts = Counter(chosen)
                hand_counts = Counter(hand)

                for card, selected_count in chosen_counts.items():
                    if hand_counts[card] < selected_count:
                        raise ValueError(
                            f"Player {p_id} cannot exchange {selected_count} copy/copies of {card}."
                        )
                return chosen

            sorted_hand = sorted(hand, reverse=highest)
            return sorted_hand[:count]

        for high_role, low_role, count in self.role_pairs:
            high_player = role_to_player.get(high_role)
            low_player = role_to_player.get(low_role)

            if high_player is None or low_player is None:
                continue

            high_cards = pick_cards(
                high_player, count, highest=False, allow_custom=True
            )
            low_cards = pick_cards(low_player, count, highest=True, allow_custom=False)

            staged_outgoing[high_player].extend(high_cards)
            staged_outgoing[low_player].extend(low_cards)

            self.exchange_log[high_player] = {
                "pair": low_player,
                "role_type": "high",
                "gave": list(high_cards),
                "received": list(low_cards),
            }
            self.exchange_log[low_player] = {
                "pair": high_player,
                "role_type": "low",
                "gave": list(low_cards),
                "received": list(high_cards),
            }

        for p_id, cards in staged_outgoing.items():
            for card in cards:
                self.hands[p_id].remove(card)

        for high_role, low_role, _ in self.role_pairs:
            high_player = role_to_player.get(high_role)
            low_player = role_to_player.get(low_role)

            if high_player is None or low_player is None:
                continue

            self.hands[high_player].extend(staged_outgoing[low_player])
            self.hands[low_player].extend(staged_outgoing[high_player])

            if self.verbose:
                print(
                    f"\nExchanging cards between {high_player} ({high_role}) and {low_player} ({low_role})"
                )

        for p_id in self.hands:
            self.hands[p_id].sort()

    def full_reset(self, next_round=False):
        if next_round:
            self.round += 1
        else:
            self.roles = {p_id: "Citizen" for p_id in range(self.players)}
            self.round = 1
            self.scores = {p_id: (0, 0) for p_id in range(self.players)}

        random.shuffle(self.deck)
        self.hands = {p_id: [] for p_id in range(self.players)}

        for i, card in enumerate(self.deck):
            p_id = i % self.players
            self.hands[p_id].append(card)

        for p_id in self.hands:
            self.hands[p_id].sort()

        self.history = []
        self.last_move = (0, 0, 0)
        self.pile = []
        self.pile_leader = None
        self.passed = set()
        self.game_over = False
        self.playing = set(range(self.players))
        self.out_order = []
        self.ended_2 = []
        self.pending_finish = None
        self.exchange_log = {}

        if next_round:
            scum_player = [p_id for p_id, role in self.roles.items() if role == "Scum"][
                0
            ]
            self.curr_turn = scum_player if scum_player is not None else None
        else:
            self.curr_turn = random.choice(
                [
                    p_id for p_id, hand in self.hands.items() if 3 in hand
                ]  # 3 of Clubs starts
            )
            self.clubs_3_holder = self.curr_turn
            self.first_turn = True
        return self._get_state(self.curr_turn)

    def _get_state(self, p_id):
        flat_history_cards = []
        for _, move in self.history:
            if move != (0, 0, 0):
                card_val, count, twos = move
                flat_history_cards.extend([card_val] * (count - twos))
                flat_history_cards.extend([15] * twos)
        history_counts = Counter(flat_history_cards)

        return {
            "hand": self.hands[p_id].copy(),
            "legal_moves": self.get_legal_moves(p_id),
            "my_role": self.roles[p_id],
            "last_move": self.last_move,
            "opp_hand_counts": {
                p: len(self.hands[p]) for p in range(self.players) if p != p_id
            },
            "passed": self.passed.copy(),
            "active_players": self.playing.copy(),
            "first_turn": self.first_turn,
            "clubs_3_holder": self.clubs_3_holder,
            "history": self.history.copy(),
            "player_roles": self.roles.copy(),
            "history_vector": [history_counts[rank] for rank in range(3, 16)],
            "is_finish_prompt": bool(
                self.pending_finish and self.pending_finish["queue"][0][2] == p_id
            ),
            "round": self.round,
            "scores": self.scores.copy(),
            "role_pairs": self.role_pairs.copy(),
            "cards_in_pile": self.pile.copy(),
            "pile_leader": self.pile_leader,
        }

    def get_legal_moves(self, p_id):
        if self.pending_finish:
            finish_card, finish_count, finish_player = self.pending_finish["queue"][0]
            if p_id == finish_player:
                return [(0, 0, 0), (finish_card, finish_count, 0)]
            return []

        hand = self.hands[p_id]

        # Can't pass on an empty pile
        legal_moves = [(0, 0, 0)] if self.last_move[0] != 0 else []
        counts = Counter(hand)
        num_twos = counts[15]
        pile_card, pile_count, _ = self.last_move

        for card, count in counts.items():
            if self.first_turn and card != 3:
                continue  # First turn must play a 3

            if card > pile_card:
                for c in range(1, count + 1):
                    if card != 15:
                        for t in range(num_twos + 1):  # Combinations with wildcard 2
                            # No more than 4 cards at a time, and must beat the pile count
                            if 1 <= c + t <= 4 and c + t >= pile_count:
                                legal_moves.append((card, c + t, t))
                    else:
                        if c >= pile_count:
                            legal_moves.append((card, c, 0))
        return legal_moves

    def _remove_cards(self, p_id, card_val, count):
        for _ in range(count):
            self.hands[p_id].remove(card_val)

    def _finishing_option(self, card, played_count, p_id):
        if played_count >= 4 or any(item[1][0] == card for item in self.history[:-1]):
            return None

        players_with_card = {  # If multiple players have the card, it's impossible for it to be the finishing move
            p: hand
            for p, hand in self.hands.items()
            if p != p_id and hand.count(card) + played_count == 4
        }
        if len(players_with_card) != 1:
            return None

        p_id, hand = players_with_card.popitem()
        return (
            card,
            hand.count(card),
            p_id,
        )

    def handle_pending_finish(self, move, p_id):
        if not self.pending_finish:
            return

        queue: list = self.pending_finish["queue"]
        card_val, count, _ = queue[0]

        if move == (0, 0, 0):
            # Current AI/Player DECLINED the jump-in. Resume normal play.
            queue.pop(0)  # Remove the declined option
            if queue:  # Next option's player gets the chance
                self.curr_turn = queue[0][2]
            else:
                resume_turn = self.pending_finish["resume_turn"]
                was_pile_reset = self.pending_finish["pile_reset"]
                self.pending_finish = None
                self.curr_turn = resume_turn

                if was_pile_reset:
                    self._pile_reset()
        else:
            if self.verbose:
                print(
                    f"\nJUMP IN! Player {p_id} finishes the last move with [{self.visualize_move(move)}]"
                )

            self._remove_cards(p_id, card_val, count)
            self.history.append((p_id, move))

            if not self.hands[p_id] and p_id not in self.out_order:
                if self.verbose:
                    print(f"Player {p_id} is out!\n")
                    input("Press Enter to continue...\n")

                self.out_order.append(p_id)
                self.ended_2.append(p_id) if card_val == 15 else None

                if p_id in self.playing:
                    self.playing.remove(p_id)

            self.game_over = self._is_game_over()
            self._pile_reset()
            self.curr_turn = p_id
            self.pending_finish = None

    def handle_finishing(
        self, card_val, rcount, p_id, twos, temp_next_turn, pile_reset
    ):
        options = []
        if option := self._finishing_option(card_val, rcount, p_id):
            options.append(option)

        if twos:
            if option := self._finishing_option(15, twos, p_id):
                options.append(option)

        if not options:
            return False

        options.sort(key=lambda x: x[0])  # Prioritize lower card
        self.pending_finish = {
            "queue": options,
            "resume_turn": temp_next_turn,
            "pile_reset": pile_reset,
        }
        self.curr_turn = options[0][2]

        return True

    def _pile_reset(self):
        if self.game_over:
            return

        self.last_move = (0, 0, 0)
        self.passed = set()
        self.curr_turn = self._get_next_active_player(
            self.pile_leader, ignore_passed=True, include_start=True
        )
        self.pile_leader = None
        self.pile = []

    def _is_game_over(self):
        active_players = [p_id for p_id in range(self.players) if self.hands[p_id]]
        return len(active_players) <= 1

    def _get_next_active_player(self, p_id, ignore_passed=False, include_start=False):
        curr = p_id if include_start else (p_id + 1) % self.players
        for _ in range(self.players):
            if curr in self.playing and (ignore_passed or curr not in self.passed):
                return curr
            curr = (curr + 1) % self.players
        return p_id

    def step(self, p_id, move):
        card_val, count, twos = move
        pile_reset = False

        if self.pending_finish:
            self.handle_pending_finish(move, p_id)
            return self._get_state(self.curr_turn), self.game_over

        if move == (0, 0, 0):
            self.passed.add(p_id)
        else:
            self.last_move = move
            self.pile_leader = p_id
            rcount = count - twos

            self._remove_cards(p_id, card_val, rcount)
            self._remove_cards(p_id, 15, twos)

        if self.pile_leader is not None:
            pile_reset = all(
                p == self.pile_leader or p in self.passed for p in self.playing
            )  # The pile resets if all other active players have passed

        if card_val == 15:
            pile_reset = True  # Playing a 2 always resets the pile

        self.history.append((p_id, move))
        self.pile.extend([card_val] * (count - twos))
        self.pile.extend([15] * twos)
        self.first_turn = False

        if not self.hands[p_id] and p_id not in self.out_order:
            if self.verbose:
                print(f"Player {p_id} is out!\n")
                input("Press Enter to continue...\n")

            self.out_order.append(p_id)
            self.ended_2.append(p_id) if card_val == 15 or twos > 0 else None

            if p_id in self.playing:
                self.playing.remove(p_id)
            self.game_over = self._is_game_over()

        if self.game_over:
            self.curr_turn = p_id
            return self._get_state(self.curr_turn), self.game_over

        if pile_reset:
            temp_next_turn = self._get_next_active_player(
                self.pile_leader, ignore_passed=True, include_start=True
            )
        else:
            temp_next_turn = self._get_next_active_player(p_id)

        if move != (0, 0, 0) and not self.game_over:
            finish = self.handle_finishing(
                card_val, rcount, p_id, twos, temp_next_turn, pile_reset
            )
            if finish:
                return self._get_state(self.curr_turn), self.game_over

        if pile_reset:
            self._pile_reset()
        else:
            self.curr_turn = temp_next_turn
        return self._get_state(self.curr_turn), self.game_over

    @staticmethod
    def visualize_hand(hand):
        return [Presidenten.visualize_card(card) for card in hand]

    @staticmethod
    def visualize_card(card):
        FACE_NAMES = {
            11: "J",
            12: "Q",
            13: "K",
            14: "A",
            15: "2",
        }
        return FACE_NAMES.get(card, str(card))

    @staticmethod
    def visualize_move(move):
        if move == (0, 0, 0):
            return "Pass"

        card_val, count, twos = move
        card_name = Presidenten.visualize_card(card_val)

        if twos:
            rcount = count - twos
            return f"{count}x {card_name} (Using {rcount}x {card_name} + {twos}x 2)"
        return f"{count}x {card_name}"


def get_settings():
    assign_p = {}
    dmc_paths = {}

    num_players = get_val_input(
        "Number of players (4-7): ", int, valid_choices={4, 5, 6, 7}
    )
    num_rounds = get_val_input("Number of rounds: ", int)

    for p_id in range(num_players):
        prompt = (
            f"Player {p_id} - 0: Human, 1: Random, 2: Baseline, 3: ISMCTS, 4: DMC: "
        )
        raw_choice = get_val_input(prompt, int, valid_choices={0, 1, 2, 3, 4})
        assign_p[p_id] = PlayerType(raw_choice)

    has_human = PlayerType.HUMAN in assign_p.values()
    parallelism = None

    choices = {"g", "s"} if not has_human else {"s"}
    prompt = "Search or game parallelism? (g/s): " if not has_human else ""
    parallelism = (
        get_val_input(prompt, str, valid_choices=choices).lower() if prompt else "s"
    )

    dmc_count = list(assign_p.values()).count(PlayerType.DMC)
    if dmc_count > 0:

        use_best_model = get_val_input(
            f"{dmc_count} DMC Bot(s) detected. Use best model? (y/n): ",
            str,
            valid_choices={"y", "n"},
        ).lower()

        dmc_p_ids = [
            p_id for p_id, p_type in assign_p.items() if p_type == PlayerType.DMC
        ]

        if use_best_model == "y":
            dmc_paths = {
                p_id: "backend/playerTypes/best_model.pt" for p_id in dmc_p_ids
            }
        else:

            def val_dmc_indices(indices):
                if len(indices) != dmc_count:
                    return False
                return all(
                    os.path.isfile(f"snapshots/model_gen_{idx}.pt") for idx in indices
                )

            prompt = "Enter batch indices (comma-separated, e.g., 1000,2000): "
            indices = get_val_input(prompt, int, val_dmc_indices, ",")
            paths = [f"snapshots/model_gen_{idx}.pt" for idx in indices]
            dmc_paths = dict(zip(dmc_p_ids, paths))
    return parallelism, assign_p, has_human, num_players, num_rounds, dmc_paths


def create_players(assign_p: dict[int, PlayerType], iterations=400, dmc_paths=None):
    from playerTypes.human import HumanPlayer
    from playerTypes.random_bot import PresidentenRandomBot
    from playerTypes.baseline_bot import PresidentenBaselineBot
    from playerTypes.ismcts_bot import PresidentenISMCTSBot
    from playerTypes.dmc_bot import PresidentenDMCBot, PresidentenValueNet

    assigned_players: dict[
        int,
        HumanPlayer
        | PresidentenRandomBot
        | PresidentenBaselineBot
        | PresidentenISMCTSBot
        | PresidentenDMCBot,
    ] = {}
    for p_id, p_type in assign_p.items():
        if p_type == PlayerType.HUMAN:
            assigned_players[p_id] = HumanPlayer(p_id)
        elif p_type == PlayerType.RANDOM:
            assigned_players[p_id] = PresidentenRandomBot(p_id)
        elif p_type == PlayerType.BASELINE:
            assigned_players[p_id] = PresidentenBaselineBot(p_id)
        elif p_type == PlayerType.ISMCTS:
            assigned_players[p_id] = PresidentenISMCTSBot(p_id, iterations)
        elif p_type == PlayerType.DMC:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dmc_model = PresidentenValueNet().to(device)

            if dmc_paths and p_id in dmc_paths:
                snap = torch.load(dmc_paths[p_id], map_location=device)
                dmc_model.load_state_dict(snap["model_state_dict"])

            dmc_model.eval()
            assigned_players[p_id] = PresidentenDMCBot(p_id, dmc_model, device)
    return assigned_players


def play_presidenten_game(
    game_id,
    num_players,
    num_rounds,
    assigned_players: dict[
        int,
        HumanPlayer
        | PresidentenRandomBot
        | PresidentenBaselineBot
        | PresidentenISMCTSBot
        | PresidentenDMCBot,
    ],
    assign_p: dict[int, PlayerType],
    parallelism="g",
    has_human=False,
    executor=None,
):
    ismcts_ids: set[int] = {
        p_id for p_id, p_type in assign_p.items() if p_type == PlayerType.ISMCTS
    }
    env = Presidenten(players=num_players, verbose=has_human)

    for idx in range(num_rounds):
        state = env.full_reset(next_round=(idx > 0))
        if has_human:
            print(f"=== ROUND {idx+1} ===")
            print("Player Roles for this Round:")

            if idx == 0:
                role_items = sorted(env.roles.items())
            else:
                role_order = {role: i for i, role in enumerate(env._get_roles())}
                role_items = sorted(
                    env.roles.items(),
                    key=lambda item: (role_order[item[1]], item[0]),
                )

            for p_id, role in role_items:
                print(f" -> Player {p_id}: {role}")
            print("-" * 50, "\n")

        if idx > 0:
            cards_to_pass = {}
            for p_id, role in env.roles.items():
                if role != "Citizen":
                    if has_human:
                        print(f"Player {p_id} ({role}) is choosing cards...")
                    cards_to_pass[p_id] = assigned_players[p_id].choose_cards_to_pass(
                        env._get_state(p_id)
                    )

            env.exchange_cards(cards_to_pass)
            state = env._get_state(env.curr_turn)

        while not env.game_over:
            curr_p_id = env.curr_turn
            if curr_p_id is None:
                break
            curr_p_type = assigned_players[curr_p_id]

            if curr_p_id in ismcts_ids:
                assert curr_p_type.__class__.__name__ == "PresidentenISMCTSBot"
                chosen_move = curr_p_type.get_move(
                    state,
                    env,
                    executor,
                    parallelism,
                )
            else:
                chosen_move = curr_p_type.get_move(state, env)

            if has_human and not state["is_finish_prompt"]:
                p_name = assign_p[curr_p_id].name if assign_p[curr_p_id] else "Unknown"
                print(
                    f"\nPlayer {curr_p_id} ({state["my_role"]}, {p_name}) chose: "
                    f"{Presidenten.visualize_move(chosen_move)}\n"
                )
                if curr_p_type.__class__.__name__ != "HumanPlayer":
                    input("Press Enter to continue...\n")
            state, _ = env.step(curr_p_id, chosen_move)

        env.assign_roles()
        if has_human:
            print(
                f"Round {idx+1} Complete! Finishing Order: {env.out_order}. "
                f"Players who finished with a 2: {env.ended_2}. Scores: {env.scores}\n"
            )
            input("Press Enter to continue...\n")
    return env.scores


worker_players: (
    dict[
        int,
        HumanPlayer
        | PresidentenRandomBot
        | PresidentenBaselineBot
        | PresidentenISMCTSBot
        | PresidentenDMCBot,
    ]
    | None
) = None


def init_worker(assign_p, iterations, dmc_paths):
    global worker_players
    worker_players = create_players(assign_p, iterations, dmc_paths)


def worker_game_task(game_id, num_players, num_rounds, assign_p):
    global worker_players
    assert worker_players is not None, "Worker players not initialized"
    return play_presidenten_game(
        game_id,
        num_players,
        num_rounds,
        worker_players,
        assign_p,
        "g",
    )


def game_parallelism(
    assign_p,
    num_players,
    num_rounds,
    dmc_paths,
    total_games,
    num_workers,
):
    print(f"Starting Tournament: {total_games} games, {num_rounds} rounds each.")
    print(f"Deploying across {num_workers} parallel game workers...\n")

    master_scores = {p_id: (0, 0) for p_id in range(num_players)}
    iters = 400

    with ProcessPoolExecutor(
        num_workers, initializer=init_worker, initargs=(assign_p, iters, dmc_paths)
    ) as executor:
        futures = [
            executor.submit(
                worker_game_task,
                idx,
                num_players,
                num_rounds,
                assign_p,
            )
            for idx in range(total_games)
        ]
        for i, f in enumerate(futures):
            master_scores = update_final_scores(master_scores, f.result())
            print(f" -> Game {i+1}/{total_games} finished processing.")
    return master_scores


def search_parallelism(
    assign_p,
    has_human,
    num_players,
    num_rounds,
    dmc_paths,
    total_games,
    num_workers,
):
    master_scores = {p_id: (0, 0) for p_id in range(num_players)}
    iters = 1200 + 200 * (num_players - 4)
    assigned_players = create_players(assign_p, iters, dmc_paths)

    with ProcessPoolExecutor(num_workers) as shared_executor:
        for idx in range(total_games):
            if idx % 10 == 0:
                print(f"\n=== GAME {idx+1} ===\n")
            round_scores = play_presidenten_game(
                idx,
                num_players,
                num_rounds,
                assigned_players,
                assign_p,
                "s",
                has_human,
                shared_executor,
            )
            master_scores = update_final_scores(master_scores, round_scores)
    return master_scores


def update_final_scores(master_scores, round_scores):
    for p_id in master_scores:
        master_scores[p_id] = (
            master_scores[p_id][0] + round_scores[p_id][0],
            master_scores[p_id][1] + round_scores[p_id][1],
        )
    return master_scores


def print_scores(scores, assign_p: dict[int, PlayerType], num_players, num_rounds):
    print("\n" + "=" * 60)
    print(f"=== FINAL SCORES: {TOTAL_GAMES} Games | {num_rounds} Rounds Each ===")
    print("=" * 60)

    for p_id in sorted(scores, key=lambda x: scores[x][0], reverse=True):
        p_name = assign_p[p_id].name if assign_p[p_id] else "Unknown"
        avg_finish_pos = num_players - (scores[p_id][0] / (TOTAL_GAMES * num_rounds))
        win_rate = scores[p_id][1] / (TOTAL_GAMES * num_rounds) * 100
        avg_norm_score = (
            scores[p_id][0] / (TOTAL_GAMES * num_rounds * (num_players - 1))
        ) * 2 - 1

        print(
            f"Player {p_id} ({p_name}): "
            f"Average Finishing Position: {avg_finish_pos:.2f} | "
            f"Win Rate: {win_rate:.2f}% | "
            f"Average Normalized Score: {avg_norm_score:.2f}"
        )
    print("=" * 60)


def main():
    parallelism, assign_p, has_human, num_players, num_rounds, dmc_paths = (
        get_settings()
    )
    master_scores = {p_id: (0, 0) for p_id in range(num_players)}

    if parallelism == "g":
        master_scores = game_parallelism(
            assign_p,
            num_players,
            num_rounds,
            dmc_paths,
            TOTAL_GAMES,
            NUM_WORKERS,
        )
    elif parallelism == "s":
        master_scores = search_parallelism(
            assign_p,
            has_human,
            num_players,
            num_rounds,
            dmc_paths,
            TOTAL_GAMES,
            NUM_WORKERS,
        )
    else:
        assigned_players = create_players(assign_p, 400, dmc_paths)
        for idx in range(TOTAL_GAMES):
            if idx % 10 == 0:
                print(f"\n=== GAME {idx+1} ===\n")
            round_scores = play_presidenten_game(
                idx,
                num_players,
                num_rounds,
                assigned_players,
                assign_p,
                has_human=has_human,
            )
            master_scores = update_final_scores(master_scores, round_scores)
    print_scores(master_scores, assign_p, num_players, num_rounds)


if __name__ == "__main__":
    TOTAL_GAMES = 996
    NUM_WORKERS = 12  # Adjust based on your system's CPU cores and memory
    main()
