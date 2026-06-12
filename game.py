import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor


class Presidenten:
    FACE_NAMES = {
        11: "J",
        12: "Q",
        13: "K",
        14: "A",
        15: "2",
    }

    def __init__(self, players=4, verbose=False):
        if players < 4:
            raise ValueError("Presidenten requires at least 4 players.")

        self.players = players
        self.verbose = verbose

        # 3 to Ace (14), plus 2 (15)
        self.deck = [rank for rank in range(3, 16) for _ in range(4)]
        self.hands = {i: [] for i in range(players)}

        self.scores = {i: 0 for i in range(players)}

        self.roles = {i: "Citizen" for i in range(players)}
        self.out_order = []  # Track finishing order for role assignment
        self.round = 0
        self.ended_2 = []  # Track players who finished with a 2 for role assignment

        self.history = []  # [(P_id, move), ...]
        self.last_move = (0, 0, 0)  # (card_value, count, twos_used)
        self.pile_leader = 0  # P_id of the player who last played to the pile
        self.passed = set()  # Players who have passed in the current pile
        self.playing = set(range(players))
        self.first_turn = True
        self.curr_turn = 0  # P_id of the current player
        self.game_over = False
        self.pending_finish = None  # {"queue": [(card, count, player_id), ...], "resume_turn": player_id, "pile_reset": bool}

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

        for p in range(self.players):  # Add any remaining players who haven't finished
            if p not in self.out_order:
                self.out_order.append(p)

        # Move players who finished with a 2 to the end of the order
        for p in reversed(self.ended_2):
            if p in self.out_order:
                self.out_order.remove(p)
                self.out_order.append(p)

        roles = self._get_roles()
        for rank, player_id in enumerate(self.out_order):
            self.roles[player_id] = roles[rank]
            self.scores[player_id] += self.players - 1 - rank

    def exchange_cards(self, cards_to_pass: dict[int | str, list[int]] | None = None):
        role_pairs = []

        if self.players >= 4:
            role_pairs.append(("President", "Scum", 3 if self.players >= 6 else 2))
            role_pairs.append(
                ("Vice-President", "High-Scum", 2 if self.players >= 6 else 1)
            )

        if self.players >= 6:
            role_pairs.append(("Secretary", "Clerk", 1))

        role_to_player = {role: player_id for player_id, role in self.roles.items()}

        # Makes sure no cards are exchanged back and forth in the same round
        staged_outgoing = {player_id: [] for player_id in range(self.players)}

        def pick_cards(player_id, count, highest=False, allow_custom=True):
            hand = self.hands[player_id]
            if allow_custom and cards_to_pass is not None:
                if player_id in cards_to_pass:
                    chosen = list(cards_to_pass[player_id])
                else:
                    chosen = list(cards_to_pass.get(self.roles[player_id], []))

                if len(chosen) != count:
                    raise ValueError(
                        f"Player {player_id} must exchange {count} card(s), got {len(chosen)}."
                    )

                chosen_counts = Counter(chosen)
                hand_counts = Counter(hand)

                for card, selected_count in chosen_counts.items():
                    if hand_counts[card] < selected_count:
                        raise ValueError(
                            f"Player {player_id} cannot exchange {selected_count} copy/copies of {card}."
                        )
                return chosen

            sorted_hand = sorted(hand, reverse=highest)
            return sorted_hand[:count]

        for high_role, low_role, count in role_pairs:
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

        for player_id, cards in staged_outgoing.items():
            for card in cards:
                self.hands[player_id].remove(card)

        for high_role, low_role, _ in role_pairs:
            high_player = role_to_player.get(high_role)
            low_player = role_to_player.get(low_role)

            if high_player is None or low_player is None:
                continue

            self.hands[high_player].extend(staged_outgoing[low_player])
            self.hands[low_player].extend(staged_outgoing[high_player])

            if self.verbose:
                print(f"Exchanging cards between {high_player} and {low_player}:")
                print(
                    f" -> Player {high_player} gives: {self.visualize_hand(staged_outgoing[high_player])}"
                )
                print(
                    f" -> Player {low_player} gives: {self.visualize_hand(staged_outgoing[low_player])}\n"
                )

        for player_id in self.hands:
            self.hands[player_id].sort()

    def full_reset(self, next_round=False):
        if next_round:
            self.round += 1
        else:
            self.roles = {i: "Citizen" for i in range(self.players)}
            self.round = 1
            self.scores = {i: 0 for i in range(self.players)}

        random.shuffle(self.deck)
        self.hands = {i: [] for i in range(self.players)}

        for i, card in enumerate(self.deck):
            player_id = i % self.players
            self.hands[player_id].append(card)

        for player_id in self.hands:
            self.hands[player_id].sort()

        self.history = []
        self.last_move = (0, 0, 0)
        self.pile_leader = 0
        self.passed = set()
        self.game_over = False
        self.playing = set(range(self.players))
        self.out_order = []
        self.ended_2 = []
        self.pending_finish = None

        if next_round:
            scum_player = [p for p, role in self.roles.items() if role == "Scum"][0]
            self.curr_turn = scum_player if scum_player else 0
        else:
            self.curr_turn = random.choice(
                [p for p, hand in self.hands.items() if 3 in hand]  # 3 of Clubs starts
            )
            self.first_turn = True
        return self._get_state(self.curr_turn)

    def _get_state(self, player_id):
        flat_history_cards = []
        for _, move in self.history:
            if move != (0, 0, 0):
                card_val, count, twos = move
                flat_history_cards.extend([card_val] * (count - twos))
                flat_history_cards.extend([15] * twos)
        history_counts = Counter(flat_history_cards)

        return {
            "hand": self.hands[player_id].copy(),
            "legal_moves": self.get_legal_moves(player_id),
            "my_role": self.roles[player_id],
            "last_move": self.last_move,
            "opp_hand_counts": {
                p: len(self.hands[p]) for p in range(self.players) if p != player_id
            },
            "passed": self.passed.copy(),
            "active_players": self.playing.copy(),
            "first_turn": self.first_turn,
            "history": self.history.copy(),
            "player_roles": self.roles.copy(),
            "history_vector": [history_counts[rank] for rank in range(3, 16)],
            "is_finish_prompt": self.pending_finish
            and self.pending_finish["queue"][0][2] == player_id,
            "round": self.round,
            "scores": self.scores.copy(),
        }

    def get_legal_moves(self, player_id):
        if self.pending_finish:
            finish_card, finish_count, finish_player = self.pending_finish["queue"][0]
            if player_id == finish_player:
                return [(0, 0, 0), (finish_card, finish_count, 0)]
            return []
        hand = self.hands[player_id]

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

    def _remove_cards(self, player_id, card_val, count):
        for _ in range(count):
            self.hands[player_id].remove(card_val)

    def _finishing_option(self, card, played_count, player_id):
        if played_count >= 4:
            return None

        if any(item[1][0] == card for item in self.history[:-1]):
            return None  # If the card has been played before, it's impossible for it to be the finishing move

        players_with_card = {  # If multiple players have the card, it's impossible for it to be the finishing move
            p: hand
            for p, hand in self.hands.items()
            if p != player_id and hand.count(card) + played_count == 4
        }
        if len(players_with_card) != 1:
            return None

        player, hand = players_with_card.popitem()
        return (
            card,
            hand.count(card),
            player,
        )

    def handle_pending_finish(self, move, player_id):
        if not self.pending_finish:
            return

        card_val, count, _ = self.pending_finish["queue"][0]

        if move == (0, 0, 0):
            # Current AI/Player DECLINED the jump-in. Resume normal play.
            self.pending_finish["queue"].pop(0)  # Remove the declined option
            if self.verbose:
                print(f"Player {player_id} declines the jump-in.")

            if self.pending_finish["queue"]:  # Next option's player gets the chance
                self.curr_turn = self.pending_finish["queue"][0][2]
            else:
                resume_turn = self.pending_finish["resume_turn"]
                was_pile_reset = self.pending_finish["pile_reset"]
                if self.verbose:
                    print(f"Resuming normal play with Player {resume_turn}.")

                self.pending_finish = None
                self.curr_turn = resume_turn

                if was_pile_reset:
                    self._pile_reset()
        else:
            if self.verbose:
                print(
                    f"JUMP IN! Player {player_id} finishes the last move with [{self.visualize_move(move)}]"
                )

            self._remove_cards(player_id, card_val, count)
            self.history.append((player_id, move))

            if not self.hands[player_id] and player_id not in self.out_order:
                if self.verbose:
                    print(f"Player {player_id} is out!\n")

                self.out_order.append(player_id)
                self.ended_2.append(player_id) if card_val == 15 else None

                if player_id in self.playing:
                    self.playing.remove(player_id)

            self.game_over = self._is_game_over()
            self._pile_reset()
            self.curr_turn = player_id
            self.pending_finish = None

    def handle_finishing(
        self, card_val, rcount, player_id, twos, temp_next_turn, pile_reset
    ):
        options = []
        if option := self._finishing_option(card_val, rcount, player_id):
            options.append(option)

        if twos:
            if option := self._finishing_option(15, twos, player_id):
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
        self.pile_leader = 0

    def _is_game_over(self):
        active_players = [p for p in range(self.players) if self.hands[p]]
        return len(active_players) <= 1

    def _get_next_active_player(
        self, from_player, ignore_passed=False, include_start=False
    ):
        curr = from_player if include_start else (from_player + 1) % self.players
        for _ in range(self.players):
            if curr in self.playing and (ignore_passed or curr not in self.passed):
                return curr
            curr = (curr + 1) % self.players

        return from_player

    def step(self, player_id, move):
        card_val, count, twos = move
        pile_reset = False

        if self.pending_finish:
            self.handle_pending_finish(move, player_id)
            return self._get_state(self.curr_turn), self.game_over

        if move == (0, 0, 0):
            self.passed.add(player_id)
            if self.pile_leader is not None:
                pile_reset = all(
                    p == self.pile_leader or p in self.passed for p in self.playing
                )  # The pile resets if all other active players have passed
        else:
            self.last_move = move
            self.pile_leader = player_id
            rcount = count - twos

            self._remove_cards(player_id, card_val, rcount)
            self._remove_cards(player_id, 15, twos)

        if card_val == 15:
            pile_reset = True  # Playing a 2 always resets the pile

        self.history.append((player_id, move))
        self.first_turn = False

        if not self.hands[player_id] and player_id not in self.out_order:
            if self.verbose:
                print(f"Player {player_id} is out!\n")

            self.out_order.append(player_id)
            self.ended_2.append(player_id) if card_val == 15 else None

            if player_id in self.playing:
                self.playing.remove(player_id)
            self.game_over = self._is_game_over()

        if self.game_over:
            self.curr_turn = player_id
            return self._get_state(self.curr_turn), self.game_over

        if pile_reset:
            temp_next_turn = self._get_next_active_player(
                self.pile_leader, ignore_passed=True, include_start=True
            )
        else:
            temp_next_turn = self._get_next_active_player(player_id)

        if move != (0, 0, 0) and not self.game_over:
            finish = self.handle_finishing(
                card_val, rcount, player_id, twos, temp_next_turn, pile_reset
            )
            if finish:
                return self._get_state(self.curr_turn), self.game_over

        if pile_reset:
            self._pile_reset()
        else:
            self.curr_turn = temp_next_turn
        return self._get_state(self.curr_turn), self.game_over

    def visualize_hand(self, hand):
        return [self.visualize_card(card) for card in hand]

    def visualize_card(self, card):
        return self.FACE_NAMES.get(card, str(card))

    def visualize_move(self, move):
        if move == (0, 0, 0):
            return "Pass"

        card_val, count, twos = move
        card_name = self.visualize_card(card_val)

        if twos:
            rcount = count - twos
            return f"{count}x {card_name} (Using {rcount}x {card_name} + {twos}x 2)"
        return f"{count}x {card_name}"


if __name__ == "__main__":
    from baseline_bot import PresidentenBaselineBot
    from ismcts_bot import PresidentenISMCTSBot

    setting = input(
        "0: random play, 1: baseline bots, 2: player vs bots, 3: ismcts vs baseline bots: "
    )
    has_human = setting == "2"

    GAMES_TO_PLAY = 100
    ROUNDS_TO_PLAY = 10
    NUM_PLAYERS = 4
    HUMAN_ID = 0
    ISMCTS_ID = 0
    NUM_WORKERS = 8
    ISMCTS_ITERATIONS = (400 + 200 * (NUM_PLAYERS - 4)) * NUM_WORKERS
    env = Presidenten(players=NUM_PLAYERS, verbose=has_human)
    final_score = {i: 0 for i in range(NUM_PLAYERS)}

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as shared_executor:
        if setting == "3":
            bots = {}
            bots[0] = PresidentenISMCTSBot(
                player_id=ISMCTS_ID, iterations=ISMCTS_ITERATIONS
            )

            for i in range(1, NUM_PLAYERS):
                bots[i] = PresidentenBaselineBot(i)
        else:
            bots = [PresidentenBaselineBot(i) for i in range(NUM_PLAYERS)]

        for game_idx in range(GAMES_TO_PLAY):
            print(f"\n=== GAME {game_idx+1} ===")
            for round_idx in range(ROUNDS_TO_PLAY):
                print(f"\n=== ROUND {round_idx + 1} ===")
                state = env.full_reset(next_round=(round_idx > 0))

                if has_human:
                    print("Player Roles for this Round:")
                if round_idx == 0:
                    role_items = sorted(env.roles.items())
                else:
                    role_order = {
                        role: idx for idx, role in enumerate(env._get_roles())
                    }
                    role_items = sorted(
                        env.roles.items(),
                        key=lambda item: (role_order[item[1]], item[0]),
                    )

                if has_human:
                    for p_id, role in role_items:
                        print(f" -> Player {p_id}: {role}")
                    print("-" * 50, "\n")

                if round_idx > 0:
                    env.exchange_cards()
                    state = env._get_state(env.curr_turn)

                while not env.game_over:
                    curr_player = env.curr_turn
                    legal_moves = env.get_legal_moves(curr_player)

                    if not legal_moves:
                        chosen_move = (0, 0, 0)
                    elif setting == "0":
                        chosen_move = random.choice(legal_moves)
                    elif has_human and curr_player == HUMAN_ID:
                        print(
                            f"Player {curr_player} ({env.roles[curr_player]}) Hand: {env.visualize_hand(state['hand'])}"
                        )
                        print(f"Last Move: {env.visualize_move(state['last_move'])}")
                        print("Legal moves:")

                        for idx, move in enumerate(legal_moves):
                            print(f"  {idx}: {env.visualize_move(move)}")

                        move_idx = int(input("Enter move index: "))
                        chosen_move = legal_moves[move_idx]
                    elif setting == "3" and curr_player == ISMCTS_ID:
                        if has_human:
                            print(
                                f"Player {curr_player} ({env.roles[curr_player]}) is thinking..."
                            )
                        chosen_move = bots[curr_player].get_move(state, env, executor=shared_executor)  # type: ignore
                    else:
                        bot_instance = bots[curr_player]
                        chosen_move = bot_instance.get_move(state)  # Bot's turn

                    if has_human:
                        print(
                            f"Player {curr_player} ({env.roles[curr_player]}) Hand:",
                            env.visualize_hand(state["hand"]),
                        )
                        print(
                            f"Player {curr_player} plays [{env.visualize_move(chosen_move)}]\n"
                        )

                    state, game_over = env.step(curr_player, chosen_move)

                for p in range(env.players):
                    if p not in env.out_order:
                        env.out_order.append(p)

                env.assign_roles()  # Assign roles and update scores at the end of the round
                print(
                    f"\nRound {round_idx + 1} Complete! Finishing Order: {env.out_order}. Players who finished with a 2: {env.ended_2}. Scores: {env.scores}"
                )
                if has_human:
                    input("Press Enter to continue to the next round...")

            if has_human:
                break

            final_score = {
                p: final_score[p] + env.scores[p] for p in range(env.players)
            }

        print(f"\n=== Final Scores after {GAMES_TO_PLAY} Games ===")
        for p in range(env.players):
            print(f"Player {p}: {final_score[p]} points")
