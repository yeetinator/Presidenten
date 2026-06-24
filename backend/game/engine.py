from __future__ import annotations

import random
from collections import Counter


class Presidenten:
    def __init__(self, players=4, verbose=False):
        if players < 4:
            raise ValueError("Presidenten requires at least 4 players.")

        self.players = players
        self.verbose = verbose

        # 3 to Ace (14), plus 2 (15)
        self.deck = [rank for rank in range(3, 16) for _ in range(4)]
        self.hands = {p_id: [] for p_id in range(players)}
        self.suited_hands: dict[int, list[str]] = {
            p_id: [] for p_id in range(players)
        }  # For future suit-based features

        self.scores = {
            p_id: (0, 0) for p_id in range(players)
        }  # (total_score, rounds_won)

        self.roles = {p_id: "Citizen" for p_id in range(players)}
        self.out_order = []  # Track finishing order for role assignment
        self.round = 0
        self.ended_2 = []  # Track players who finished with a 2 for role assignment

        self.history = []  # [(P_id, move), ...]
        self.last_move = (0, 0, 0)  # (card_value, count, twos_used)
        self.suit_last_move = []
        self.pile = []  # Cards currently on the pile
        self.was_pile_reset = False

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

    def _get_suit_prefix(self, card_val):
        mapping = {10: "T", 11: "J", 12: "Q", 13: "K", 14: "A", 15: "2"}
        return mapping.get(card_val, str(card_val))

    def exchange_cards(
        self,
        cards_to_pass: dict[int | str, list[int]] | None = None,
    ):
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
            chosen = sorted_hand[:count]

            return chosen

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
        self.suited_hands = self.gen_suited_hands()

    def gen_suited_hands(self):
        suited_hands = {p_id: [] for p_id in range(self.players)}
        suits = ["C", "D", "H", "S"]
        card_map = {"10": "T", "11": "J", "12": "Q", "13": "K", "14": "A", "15": "2"}
        pool: dict[int, list[str]] = {}

        for card in range(3, 16):
            c_str = card_map.get(str(card), str(card))
            pool[card] = [f"{c_str}{s}" for s in suits]
            random.shuffle(pool[card])

        if self.first_turn and self.clubs_3_holder is not None:
            suited_hands[self.clubs_3_holder].append("3C")
            pool[3].remove("3C")

        for p_id, hand in self.hands.items():
            for card in hand:
                if self.first_turn and p_id == self.clubs_3_holder and card == 3:
                    continue
                if pool[card]:
                    suited_hands[p_id].append(pool[card].pop())
        return suited_hands

    def full_reset(self, next_round=False):
        random.shuffle(self.deck)
        self.hands = {p_id: [] for p_id in range(self.players)}
        rng_offset = random.randint(0, self.players - 1)

        for i, card in enumerate(self.deck):
            p_id = (i + rng_offset) % self.players
            self.hands[p_id].append(card)

        for p_id in self.hands:
            self.hands[p_id].sort()

        if next_round:
            self.round += 1
            scum_player = [p_id for p_id, role in self.roles.items() if role == "Scum"][
                0
            ]
            self.curr_turn = scum_player if scum_player is not None else None
        else:
            self.roles = {p_id: "Citizen" for p_id in range(self.players)}
            self.round = 1
            self.scores = {p_id: (0, 0) for p_id in range(self.players)}

            self.curr_turn = random.choice(
                [
                    p_id for p_id, hand in self.hands.items() if 3 in hand
                ]  # 3 of Clubs starts
            )
            self.clubs_3_holder = self.curr_turn
            self.first_turn = True

        self.suited_hands = self.gen_suited_hands()

        self.history = []
        self.last_move = (0, 0, 0)
        self.suit_last_move = []
        self.pile = []
        self.was_pile_reset = False
        self.pile_leader = None
        self.passed = set()
        self.game_over = False
        self.playing = set(range(self.players))
        self.out_order = []
        self.ended_2 = []
        self.pending_finish = None
        self.exchange_log = {}

        return self._get_state(self.curr_turn)

    def _get_state(self, p_id, *, reset_view=False):
        last_move = (0, 0, 0) if reset_view else self.last_move
        suit_last_move = [] if reset_view else self.suit_last_move
        cards_in_pile = [] if reset_view else self.pile
        passed = set() if reset_view else self.passed
        pile_leader = None if reset_view else self.pile_leader

        flat_history_cards = []
        for _, move in self.history:
            if move != (0, 0, 0):
                card_val, count, twos = move
                flat_history_cards.extend([card_val] * (count - twos))
                flat_history_cards.extend([15] * twos)
        history_counts = Counter(flat_history_cards)

        return {
            "hand": self.hands[p_id].copy(),
            "suited_hand": self.suited_hands[p_id].copy(),
            "legal_moves": self.get_legal_moves(p_id, last_move=last_move),
            "my_role": self.roles[p_id],
            "last_move": last_move,
            "suit_last_move": suit_last_move.copy(),
            "curr_turn": self.curr_turn,
            "opp_hand_counts": {
                p: len(self.hands[p]) for p in range(self.players) if p != p_id
            },
            "passed": passed.copy(),
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
            "cards_in_pile": cards_in_pile.copy(),
            "pile_leader": pile_leader,
        }

    def get_legal_moves(self, p_id, last_move=None):
        if self.pending_finish:
            finish_card, finish_count, finish_player = self.pending_finish["queue"][0]
            if p_id == finish_player:
                return [(0, 0, 0), (finish_card, finish_count, 0)]
            return []

        hand = self.hands[p_id]
        current_last_move = self.last_move if last_move is None else last_move

        # Can't pass on an empty pile
        legal_moves = [(0, 0, 0)] if current_last_move[0] != 0 else []
        counts = Counter(hand)
        num_twos = counts[15]
        pile_card, pile_count, _ = current_last_move

        for card, count in counts.items():
            if self.first_turn and card != 3:
                continue  # First turn must play a 3

            if card > pile_card:
                for c in range(1, count + 1):
                    if card != 15:
                        for t in range(num_twos + 1):  # Combinations with wildcard 2
                            if 1 <= c + t <= 4 and c + t >= pile_count:
                                legal_moves.append((card, c + t, t))
                    else:
                        if c >= pile_count:
                            legal_moves.append((card, c, 0))
        return legal_moves

    def _remove_cards(self, p_id, card_val, count, suits):
        for _ in range(count):
            self.hands[p_id].remove(card_val)

        prefix = self._get_suit_prefix(card_val)
        if suits:
            for suit in suits:
                if suit.startswith(prefix) and suit in self.suited_hands[p_id]:
                    self.suited_hands[p_id].remove(suit)
        else:
            for _ in range(count):
                card = next(
                    (sc for sc in self.suited_hands[p_id] if sc.startswith(prefix)),
                    None,
                )
                if card:
                    self.suited_hands[p_id].remove(card)
                    self.suit_last_move.append(card)

    def _finishing_option(self, card, played_count, p_id):
        if played_count >= 4 or any(item[1][0] == card for item in self.history[:-1]):
            return None

        players_with_card = {
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

    def handle_pending_finish(self, move, p_id, suits):
        if not self.pending_finish:
            return

        queue: list = self.pending_finish["queue"]
        card_val, count, _ = queue[0]

        if move == (0, 0, 0):
            queue.pop(0)
            if queue:
                self.curr_turn = queue[0][2]
            else:
                resume_turn = self.pending_finish["resume_turn"]
                was_pile_reset = self.pending_finish["pile_reset"]
                self.pending_finish = None
                self.curr_turn = resume_turn
                self.was_pile_reset = was_pile_reset
        else:
            if self.verbose:
                print(
                    f"\nJUMP IN! Player {p_id} finishes the last move with [{self.visualize_move(move)}]"
                )

            if suits:
                self.suit_last_move.extend(suits)

            self._remove_cards(p_id, card_val, count, suits)
            self.history.append((p_id, move))
            self.last_move = move

            if not self.hands[p_id] and p_id not in self.out_order:
                self.out_order.append(p_id)
                self.ended_2.append(p_id) if card_val == 15 else None

                if p_id in self.playing:
                    self.playing.remove(p_id)

            self.game_over = self._is_game_over()
            self.pile.extend([card_val] * count)
            self.curr_turn = p_id
            self.pending_finish = None
            self.was_pile_reset = True

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

        options.sort(key=lambda x: x[0])
        self.pending_finish = {
            "queue": options,
            "resume_turn": temp_next_turn,
            "pile_reset": pile_reset,
        }
        self.curr_turn = options[0][2]

        return True

    def clear_pile(self):
        self.suit_last_move = []
        self.last_move = (0, 0, 0)
        self.passed = set()
        self.pile = []
        self.pile_leader = None
        self.was_pile_reset = False

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

    def step(self, p_id, move, suits=None):
        card_val, count, twos = move
        pile_reset = False

        if self.pending_finish:
            self.handle_pending_finish(move, p_id, suits)
            return (
                self._get_state(self.curr_turn, reset_view=self.was_pile_reset),
                self.game_over,
            )

        if move == (0, 0, 0):
            self.passed.add(p_id)
        else:
            self.last_move = move
            self.suit_last_move = suits if suits else []
            self.pile_leader = p_id
            rcount = count - twos

            self._remove_cards(p_id, card_val, rcount, suits)
            self._remove_cards(p_id, 15, twos, suits)

        if self.pile_leader is not None:
            pile_reset = all(
                p == self.pile_leader or p in self.passed for p in self.playing
            )

        if card_val == 15:
            pile_reset = True

        self.history.append((p_id, move))
        self.pile.extend([card_val] * (count - twos))
        self.pile.extend([15] * twos)
        self.first_turn = False

        if not self.hands[p_id] and p_id not in self.out_order:
            self.out_order.append(p_id)
            self.ended_2.append(p_id) if card_val == 15 or twos > 0 else None

            if p_id in self.playing:
                self.playing.remove(p_id)
            self.game_over = self._is_game_over()

        if self.game_over:
            self.curr_turn = p_id
            self.was_pile_reset = pile_reset

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
                self.was_pile_reset = False
                return self._get_state(self.curr_turn), self.game_over

        self.curr_turn = temp_next_turn
        self.was_pile_reset = pile_reset

        return (
            self._get_state(self.curr_turn, reset_view=self.was_pile_reset),
            self.game_over,
        )

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
