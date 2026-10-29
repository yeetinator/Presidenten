import { playSound } from "./audio";
import type { GameStateUpdate } from "../stores/gameStore";

// Other players' moves sound a little softer than the human's own
const OTHER_PLAYER_GAIN = 0.7;

function playerGain(seat: number | null) {
  return seat === 0 ? 1 : OTHER_PLAYER_GAIN;
}

function sameCards(left: string[], right: string[]) {
  return left.length === right.length && left.every((card, index) => card === right[index]);
}

function handsBySeat(state: GameStateUpdate): Record<number, string[]> {
  return { ...state.opp_suited_hands, 0: state.suited_hand };
}

// The server sends whole snapshots, so what just happened is read from the
// difference between the table before and after an update
export function playStateSounds(prev: GameStateUpdate | null, next: GameStateUpdate) {
  if (!prev) return;

  const prevPile = prev.suit_last_move ?? [];
  const nextPile = next.suit_last_move ?? [];
  if (prevPile.length > 0 && nextPile.length === 0) playSound("pileClear");
  else if (nextPile.length > 0 && !sameCards(prevPile, nextPile)) {
    // A jump-in adds the missing cards to the play on the pile
    const isQuadFinish =
      prevPile.length > 0 &&
      nextPile.length > prevPile.length &&
      prevPile.every((card, index) => card === nextPile[index]);
    playSound(isQuadFinish ? "quad" : "cardPlay", { gain: playerGain(next.pile_leader) });
  }

  // Passes are forgotten once the pile clears
  const newPass = (next.passed ?? []).find((seat) => !(prev.passed ?? []).includes(seat));
  if (newPass !== undefined && nextPile.length > 0) playSound("pass", { gain: playerGain(newPass) });

  const prevHands = handsBySeat(prev);
  const nextHands = handsBySeat(next);
  let someoneOut: number | null = null;
  let swapped = false;
  for (const [seatStr, hand] of Object.entries(nextHands)) {
    const seat = Number(seatStr);
    const prevHand = prevHands[seat] ?? [];
    if (prevHand.length > 0 && hand.length === 0) {
      if (someoneOut !== 0) someoneOut = seat;
    }
    // Only the card exchange changes a hand without changing its size
    else if (prevHand.length === hand.length && !sameCards(prevHand, hand)) swapped = true;
  }
  if (someoneOut !== null) playSound(someoneOut === 0 ? "playerOutOwn" : "playerOut");
  if (swapped) playSound("cardSwap");
}
