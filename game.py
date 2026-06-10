import random
from collections import Counter


class Presidenten:
    FACE_NAMES = {
        11: "J",
        12: "Q",
        13: "K",
        14: "A",
        15: "2",
    }

    def __init__(self, players=4):
        self.players = players
        self.deck = [
            rank for rank in range(3, 16) for _ in range(4)
        ]  # 3 to Ace (14), plus 2 (15)
        self.hands = {i: [] for i in range(players)}
        self.history = []
        self.last_move = (0, 0, 0)  # (card_value, count, twos_used)
        self.passed = set()
        self.playing = set(range(players))
        self.first_turn = True
        self.game_over = False

    def full_reset(self):
        random.shuffle(self.deck)
        for i, card in enumerate(self.deck):
            player_id = i % self.players
            self.hands[player_id].append(card)

        for player_id in self.hands:
            self.hands[player_id].sort()

        self.history = []
        self.last_move = (0, 0, 0)
        self.curr_turn = random.choice(
            [p for p, hand in self.hands.items() if 3 in hand]
        )
        self.passed = set()
        self.first_turn = True
        self.game_over = False
        self.playing = set(range(self.players))

        return self._get_state(self.curr_turn)

    def _get_state(self, player_id):
        return {
            "hand": self.hands[player_id].copy(),
            "last_move": self.last_move,
            "opp_hand_counts": {
                p: len(self.hands[p]) for p in range(self.players) if p != player_id
            },
            "legal_moves": self.get_legal_moves(player_id),
            "history": self.history.copy(),
            "passed": self.passed.copy(),
            "first_turn": self.first_turn,
        }

    def get_legal_moves(self, player_id):
        hand = self.hands[player_id]
        legal_moves = (
            [(0, 0, 0)] if self.last_move[0] != 0 else []
        )  # Can only pass if there's a pile to beat
        counts = Counter(hand)
        num_twos = counts[15]

        pile_card, pile_count, _ = self.last_move
        hand_size = len(hand)

        for card, count in counts.items():
            if self.first_turn and card != 3:
                continue  # First turn must play a 3
            if card > pile_card:
                for c in range(1, count + 1):
                    if card != 15:
                        for t in range(num_twos + 1):
                            if 1 <= c + t <= 4 and c + t >= pile_count:
                                if hand_size - (c + t) == 0 and t > 0:
                                    continue  # Can't use twos to finish if it would end the game
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
            return None

        players_with_card = {
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

    def handle_finishing(self, card_val, rcount, twos, player_id):
        options = []

        if option := self._finishing_option(card_val, rcount, player_id):
            options.append(option)

        if twos:
            if option := self._finishing_option(15, twos, player_id):
                options.append(option)

        if not options:
            return None

        chosen = min(options, key=lambda x: x[0])  # Prioritize lower card
        vis_finish_options = [self.visualize_move((o[0], o[1], 0)) for o in options]
        print(
            f"\nJUMP IN! Player {chosen[2]} finishes the last move with [{self.visualize_move((chosen[0], chosen[1], 0))}] out of legal options: {vis_finish_options}"
        )

        self._remove_cards(chosen[2], chosen[0], chosen[1])
        self.history.append((chosen[2], (chosen[0], chosen[1], 0)))
        return chosen[2]

    def _pile_reset(self):
        self.last_move = (0, 0, 0)
        self.passed = set()

    def _is_game_over(self):
        active_players = [p for p in range(self.players) if self.hands[p]]
        return len(active_players) <= 1

    def step(self, player_id, move):
        card_val, count, twos = move
        pile_reset = False
        finisher = None

        if move == (0, 0, 0):
            self.passed.add(player_id)
        else:
            self.last_move = move

            rcount = count - twos
            self._remove_cards(player_id, card_val, rcount)
            self._remove_cards(player_id, 15, twos)

        self.history.append((player_id, move))
        self.first_turn = False

        if move != (0, 0, 0):
            finisher = self.handle_finishing(card_val, rcount, twos, player_id)

        if finisher is not None:
            self.curr_turn = finisher
            self._pile_reset()
            self.game_over = self._is_game_over()

            return self._get_state(self.curr_turn), self.game_over

        if len(self.passed) >= len(self.playing) - 1:
            pile_reset = True

        if not self.hands[player_id]:
            print(f"Player {player_id} is out!")
            self.playing.remove(player_id)
            self.game_over = self._is_game_over()

        self.curr_turn = (self.curr_turn + 1) % self.players
        while self.curr_turn in self.passed or self.curr_turn not in self.playing:
            self.curr_turn = (self.curr_turn + 1) % self.players

        if pile_reset:
            self._pile_reset()

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


env = Presidenten()
state = env.full_reset()

print(
    f"Initial Player {env.curr_turn} Starting Hand:", env.visualize_hand(state["hand"])
)
print("-" * 50)

while not env.game_over:
    curr_player = env.curr_turn
    legal_moves = env.get_legal_moves(curr_player)

    # Pick a random legal move
    chosen_move = random.choice(legal_moves)

    vis_options = [env.visualize_move(move) for move in legal_moves]
    print(
        f"Player {curr_player} plays [{env.visualize_move(chosen_move)}] out of legal options: {vis_options}"
    )

    state, game_over = env.step(curr_player, chosen_move)
    if game_over:
        print(f"\nGame Over!")
        break

    print("-" * 50)
    print(
        f"Next up: Player {env.curr_turn}'s Hand: {env.visualize_hand(env.hands[env.curr_turn])}"
    )
