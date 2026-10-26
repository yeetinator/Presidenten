from __future__ import annotations

import random

MAPPING_VS = {10: "T", 11: "J", 12: "Q", 13: "K", 14: "A", 15: "2"}
MAPPING_SV = {"T": 10, "J": 11, "Q": 12, "K": 13, "A": 14, "2": 15}
SUITS = ["C", "D", "H", "S"]


class President:
    def __init__(self, players=4, verbose=False, track_suits=False, lightweight=False):
        if players < 4:
            raise ValueError("President requires at least 4 players.")

        self.players = players
        self.verbose = verbose
        self.track_suits = track_suits  # For front-end visualization

        # For headless rollouts: step() returns only {"legal_moves": ...} when the
        # next player has at most one legal move (bots return it without reading
        # anything else). _get_state() itself always returns the full state.
        self.lightweight = lightweight and not track_suits

        # 3 to Ace (14), plus 2 (15)
        self.deck = [rank for rank in range(3, 16) for _ in range(4)]

        self.curr_turn = None  # P_id of the current player
        self.clubs_3_holder = None
        self._reset_state(new_game=True)

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

        for p_id in range(self.players):
            # Add any remaining players who haven't finished
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

    def _get_card_value(self, suit_str):
        # No .get(r, int(r)) because int(r) will error if r is not an integer.
        r = suit_str[:-1]
        return MAPPING_SV[r] if r in MAPPING_SV else int(r)

    def exchange_cards(
        self,
        pair: tuple[str, str, int],
        cards_to_pass: dict[int | str, list[int]] | None = None,
    ):
        role_to_player = {role: p_id for p_id, role in self.roles.items()}

        # Makes sure no cards are exchanged back and forth in the same round
        staged_outgoing = {p_id: [] for p_id in range(self.players)}
        if self.track_suits:
            staged_suits = {p_id: [] for p_id in range(self.players)}

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
                return chosen

            sorted_hand = sorted(hand, reverse=highest)
            return sorted_hand[:count]

        def suit_exchange(player, cards):
            for card_val in cards:
                prefix = MAPPING_VS.get(card_val, str(card_val))
                suit_card = next(
                    (sc for sc in self.suited_hands[player] if sc.startswith(prefix)),
                    None,
                )
                if suit_card:
                    self.suited_hands[player].remove(suit_card)
                    staged_suits[player].append(suit_card)

        high_role, low_role, count = pair
        high_player = role_to_player.get(high_role)
        low_player = role_to_player.get(low_role)

        if high_player is not None and low_player is not None:
            high_cards = pick_cards(high_player, count, False, True)
            low_cards = pick_cards(low_player, count, True, False)

            staged_outgoing[high_player].extend(high_cards)
            staged_outgoing[low_player].extend(low_cards)

            if self.track_suits:
                suit_exchange(high_player, high_cards)
                suit_exchange(low_player, low_cards)

            for card in staged_outgoing[high_player]:
                self.hands[high_player].remove(card)
            for card in staged_outgoing[low_player]:
                self.hands[low_player].remove(card)

            self.hands[high_player].extend(staged_outgoing[low_player])
            self.hands[low_player].extend(staged_outgoing[high_player])

            if self.track_suits:
                self.suited_hands[high_player].extend(staged_suits[low_player])
                self.suited_hands[low_player].extend(staged_suits[high_player])

            if self.verbose:
                print(
                    f"\nExchanging cards between {high_player} ({high_role}) and {low_player} ({low_role})"
                )

        for p_id in range(self.players):
            self.hands[p_id].sort()
            if self.track_suits:
                self.suited_hands[p_id].sort(key=self._get_card_value)

    def gen_suited_hands(self):
        suited_hands = {p_id: [] for p_id in range(self.players)}
        pool: dict[int, list[str]] = {}

        for card in range(3, 16):
            c_str = MAPPING_VS.get(card, str(card))
            pool[card] = [f"{c_str}{s}" for s in SUITS]
            random.shuffle(pool[card])

        if self.first_turn and self.clubs_3_holder is not None:
            suited_hands[self.clubs_3_holder].append("3C")
            pool[3].remove("3C")

        for p_id, hand in self.hands.items():
            skipped_first_3 = False
            for card in hand:
                if (
                    self.first_turn
                    and p_id == self.clubs_3_holder
                    and card == 3
                    and not skipped_first_3
                ):
                    skipped_first_3 = True
                    continue
                if pool[card]:
                    suited_hands[p_id].append(pool[card].pop())
        return suited_hands

    def full_reset(self, next_round=False):
        self._reset_state(new_game=not next_round)

        random.shuffle(self.deck)
        rng_offset = random.randint(0, self.players - 1)

        for i, card in enumerate(self.deck):
            p_id = (i + rng_offset) % self.players
            self.hands[p_id].append(card)

        for p_id in self.hands:
            self.hands[p_id].sort()

        if next_round:
            self.round += 1
            self.curr_turn = next(
                (p_id for p_id, role in self.roles.items() if role == "Scum"), None
            )
        else:
            self.clubs_3_holder = self.curr_turn = random.choice(
                # 3 of Clubs starts
                [p_id for p_id, hand in self.hands.items() if 3 in hand]
            )

        if self.track_suits:
            self.suited_hands = self.gen_suited_hands()

        return self._get_state(self.curr_turn)

    def _reset_state(self, new_game):
        if new_game:
            self.roles = {p_id: "Citizen" for p_id in range(self.players)}
            # (total_score, rounds_won)
            self.scores = {p_id: (0, 0) for p_id in range(self.players)}
            self.round = 1
            self.first_turn = True

        self.hands = {p_id: [] for p_id in range(self.players)}
        if self.track_suits:
            self.suited_hands: dict[int, list[str]] = {
                p_id: [] for p_id in range(self.players)
            }

        self._history_flat_counts = [0] * 16  # Count of each card played
        # Count of each card passed by each player
        self._passed_by_player = [[0] * 16 for _ in range(self.players)]
        # Same but only on 2x+ cards
        self._passed_multi_by_player = [[0] * 16 for _ in range(self.players)]
        self.playing = set(range(self.players))  # Players with cards
        self.out_order = []  # Track finishing order for role assignment
        self.ended_2 = []  # Track players who finished with a 2 for role assignment
        self.game_over = False
        self.clear_pile()

    def _record_history_move(self, p_id, move):
        if move == (0, 0, 0):
            pile_card, pile_count, _ = self.last_move
            if pile_card:
                self._passed_by_player[p_id][pile_card] += 1
                if pile_count > 1:
                    self._passed_multi_by_player[p_id][pile_card] += 1
        else:
            card_val, count, twos = move
            self._history_flat_counts[card_val] += count - twos
            self._history_flat_counts[15] += twos

    def _get_turn_state(self, p_id, *, reset_view=False):
        if not self.lightweight:
            return self._get_state(p_id, reset_view=reset_view)

        last_move = (0, 0, 0) if reset_view else self.last_move
        legal_moves = self.get_legal_moves(p_id, last_move)

        if len(legal_moves) <= 1:
            return {"legal_moves": legal_moves}
        return self._get_state(p_id, reset_view=reset_view, legal_moves=legal_moves)

    def _get_state(self, p_id, *, reset_view=False, only_finish=True, legal_moves=None):
        last_move = (0, 0, 0) if reset_view else self.last_move
        if self.track_suits:
            suit_last_move = [] if reset_view else self.suit_last_move

        cards_in_pile = [] if reset_view else self.pile
        passed = set() if reset_view else self.passed
        pile_leader = None if reset_view else self.pile_leader

        state = {
            "hand": self.hands[p_id].copy(),
            "legal_moves": (
                legal_moves
                if legal_moves is not None
                else self.get_legal_moves(p_id, last_move, only_finish)
            ),
            "my_role": self.roles[p_id],
            "last_move": last_move,
            "curr_turn": self.curr_turn,
            "opp_hand_counts": {
                p: len(self.hands[p]) for p in range(self.players) if p != p_id
            },
            "passed": passed.copy(),
            "active_players": self.playing.copy(),
            "first_turn": self.first_turn,
            "player_roles": self.roles.copy(),
            "history_vector": self._history_flat_counts[3:],
            "passed_by_player": [
                self._passed_by_player[p][3:] for p in range(self.players)
            ],
            "passed_multi_by_player": [
                self._passed_multi_by_player[p][3:] for p in range(self.players)
            ],
            "is_finish_prompt": bool(
                self.pending_finish and self.pending_finish["queue"][0][2] == p_id
            ),
            "resume_turn": (
                self.pending_finish["resume_turn"] if self.pending_finish else None
            ),
            "role_pairs": self.role_pairs.copy(),
            "cards_in_pile": cards_in_pile.copy(),
            "pile_leader": pile_leader,
        }
        if self.track_suits:
            state["suited_hand"] = self.suited_hands[p_id].copy()
            state["opp_suited_hands"] = {
                p: self.suited_hands[p].copy() for p in range(self.players) if p != p_id
            }
            state["suit_last_move"] = suit_last_move.copy()
        return state

    def get_legal_moves(self, p_id, last_move=None, only_finish=True):
        finish_moves = [(0, 0, 0)]
        if self.pending_finish and self.pending_finish["queue"][0][2] == p_id:
            for finish_card, finish_count, finish_player in self.pending_finish[
                "queue"
            ]:
                if p_id == finish_player:
                    finish_moves.append((finish_card, finish_count, 0))
            if only_finish:
                return finish_moves

        counts = [0] * 16
        for card in self.hands[p_id]:
            counts[card] += 1
        num_twos = counts[15]

        # First turn must play a 3
        if self.first_turn:
            num_threes = counts[3]
            return [
                (3, c + t, t)
                for c in range(1, num_threes + 1)
                for t in range(min(num_twos, 4 - c) + 1)
            ]

        curr_last_move = self.last_move if last_move is None else last_move
        pile_card, pile_count, _ = curr_last_move
        # Can't pass on an empty pile
        legal_moves = finish_moves if pile_card != 0 else []

        min_card = max(3, pile_card + 1)
        for card in range(min_card, 15):
            count = counts[card]
            if not count:
                continue

            for c in range(1, count + 1):
                t_min = max(0, pile_count - c)
                t_max = min(num_twos, 4 - c)

                for t in range(t_min, t_max + 1):  # Combinations with wildcard 2
                    legal_moves.append((card, c + t, t))

        if num_twos and pile_card < 15:
            for c in range(max(1, pile_count), num_twos + 1):
                legal_moves.append((15, c, 0))

        return legal_moves

    def _remove_cards(self, p_id, card_val, count, suits):
        for _ in range(count):
            self.hands[p_id].remove(card_val)

        if self.track_suits:
            prefix = MAPPING_VS.get(card_val, str(card_val))
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
        if played_count >= 4 or self._history_flat_counts[card] != played_count:
            return None

        needed = 4 - played_count
        for p, hand in self.hands.items():
            if p != p_id and hand.count(card) == needed:
                return (card, needed, p)
        return None

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
                self.curr_turn = self.pending_finish["resume_turn"]
                self.was_pile_reset = self.pending_finish["pile_reset"]
                self.pending_finish = None
        else:
            if self.verbose:
                print(
                    f"\nJUMP IN! Player {p_id} finishes the last move with [{self.visualize_move(move)}]"
                )

            if self.track_suits and suits:
                self.suit_last_move.extend(suits)

            self._remove_cards(p_id, card_val, count, suits)
            self._record_history_move(p_id, move)
            self.last_move = move

            if not self.hands[p_id] and p_id not in self.out_order:
                self.out_order.append(p_id)
                self.ended_2.append(p_id) if card_val == 15 else None

                if p_id in self.playing:
                    self.playing.remove(p_id)

            self.game_over = self._is_game_over()
            self.pile.extend([card_val] * count)
            # The jumper leads the new pile, or the next player if those were their last cards
            self.curr_turn = self._get_next_active_player(
                p_id, ignore_passed=True, include_start=True
            )
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
            "resume_played": False,
        }
        self.curr_turn = options[0][2]

        return True

    def clear_pile(self):
        if self.track_suits:
            self.suit_last_move = []

        self.last_move = (0, 0, 0)  # (card_value, count, twos_used)
        self.passed = set()  # Players who have passed in the current pile
        self.pile = []  # Cards currently on the pile
        self.pile_leader = None  # P_id of the player who last played to the pile
        self.pending_finish = None  # {"queue": [(card, count, player_id), ...], "resume_turn": player_id, "pile_reset": bool}
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
        self.curr_turn = p_id
        card_val, count, twos = move
        pile_reset = False

        if self.pending_finish and self.pending_finish["queue"][0][2] == p_id:
            self.handle_pending_finish(move, p_id, suits)
            return (
                self._get_turn_state(self.curr_turn, reset_view=self.was_pile_reset),
                self.game_over,
            )

        if move == (0, 0, 0):
            self.passed.add(p_id)
        else:
            self.last_move = move
            if self.track_suits:
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

        self._record_history_move(p_id, move)
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

            return self._get_turn_state(self.curr_turn), self.game_over

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
                return self._get_turn_state(self.curr_turn), self.game_over

        self.curr_turn = temp_next_turn
        self.was_pile_reset = pile_reset

        return (
            self._get_turn_state(self.curr_turn, reset_view=self.was_pile_reset),
            self.game_over,
        )

    @staticmethod
    def visualize_hand(hand):
        return [President.visualize_card(card) for card in hand]

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
        card_name = President.visualize_card(card_val)

        if twos:
            rcount = count - twos
            return f"{count}x {card_name} (Using {rcount}x {card_name} + {twos}x 2)"
        return f"{count}x {card_name}"
