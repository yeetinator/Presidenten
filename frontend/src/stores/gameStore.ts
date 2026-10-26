import { writable, get } from "svelte/store";
import { tick } from "svelte";
import { TABLE_THEMES, type ThemeKey, type TableTheme } from "../lib/themes";

export interface GameStateUpdate {
  suited_hand: string[];
  opp_suited_hands: Record<number, string[]>;
  is_finish_prompt: boolean;
  resume_turn: number | null;
  curr_turn: number | null;
  player_roles: Record<number, string | null>;
  player_types: Record<number, string>;
  legal_moves_suits: string[][];
  suit_last_move: string[];
  can_pass: boolean;
  passed: number[];
  pile_leader: number | null;
}

export interface DealCardsMessage {
  type: "DEAL_CARDS";
  state: GameStateUpdate;
}

export interface StateUpdateMessage {
  type: "STATE_UPDATE";
  state: GameStateUpdate;
  clearJump?: boolean;
  prompt_id?: number;
}

export interface RoundOverMessage {
  type: "ROUND_OVER";
  scores: Record<number, [number, number]>;
  roles: Record<string, string>;
  out_order: number[];
}

export type RoundSummary = Omit<RoundOverMessage, "type">;

export interface JumpInPromptMessage {
  type: "JUMP_IN_PROMPT";
  state: GameStateUpdate;
  message: string;
  prompt_id: number;
}

export interface JumpInPrompt {
  message: string;
}

export interface ExchangePrompt {
  requiredCards: number;
  canChoose: boolean;
}

export interface ExchangePromptMessage {
  type: "EXCHANGE_PROMPT";
  state: GameStateUpdate;
  required_cards: number;
  can_choose: boolean;
  prompt_id: number;
}

const initialThemeKey = "emerald";
export const currTheme = writable<TableTheme>(TABLE_THEMES[initialThemeKey] || TABLE_THEMES.emerald);
export const state = writable<GameStateUpdate | null>(null);
export const inGame = writable<boolean>(false);
export const lobbyNotice = writable<string | null>(null);
export const connectionStatus = writable<ConnectionStatus>("disconnected");
export const selectedCards = writable<string[]>([]);
export const roundSummary = writable<RoundSummary | null>(null);
export const jumpInPrompt = writable<JumpInPrompt | null>(null);
export const exchangePrompt = writable<ExchangePrompt | null>(null);
export const isAnimating = writable<boolean>(false);
export const fastForwardMode = writable<boolean>(false);
export const revealedBotSeat = writable<number | null>(null);
export const isDealing = writable<boolean>(false);

type IncomingWebSocketMessage =
  | StateUpdateMessage
  | JumpInPromptMessage
  | ExchangePromptMessage
  | RoundOverMessage
  | DealCardsMessage
  | {
    type: string;
    [key: string]: unknown;
  };

type ConnectionStatus = "disconnected" | "connecting" | "connected" | "error";

const RECONNECT_DELAY_MS = 2000;
const JUMP_IN_MESSAGE = "JUMP IN!";
// No card transition runs this long; Svelte skips the end event of an interrupted
// transition, so an element still marked as animating by then is released
const ANIMATION_TIMEOUT_MS = 1000;

let socket: WebSocket | null = null;
let currUrl: string | null = null;
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
let stateUpdateQueue: (() => void)[] = [];
const animatingElements = new Map<EventTarget, ReturnType<typeof setTimeout>>();
// Id of the prompt on screen; replies echo it so the server can ignore stale ones
let activePromptId: number | null = null;

export function startAnimation(event: Event) {
  const element = event.target;
  if (!element) return;

  clearTimeout(animatingElements.get(element));
  animatingElements.set(element, setTimeout(() => releaseAnimation(element), ANIMATION_TIMEOUT_MS));
  isAnimating.set(true);
}

export function endAnimation(event: Event) {
  if (event.target) releaseAnimation(event.target);
}

function releaseAnimation(element: EventTarget) {
  clearTimeout(animatingElements.get(element));
  animatingElements.delete(element);
  if (animatingElements.size === 0) isAnimating.set(false);
}

export function setTheme(key: ThemeKey) {
  if (TABLE_THEMES[key]) currTheme.set(TABLE_THEMES[key]);
}

function isGameStateUpdate(value: unknown): value is GameStateUpdate {
  if (!value || typeof value !== "object") return false;

  const candidate = value as Partial<GameStateUpdate>;
  return (
    Array.isArray(candidate.suited_hand) &&
    typeof candidate.is_finish_prompt === "boolean" &&
    typeof candidate.can_pass === "boolean" &&
    !!candidate.player_roles &&
    typeof candidate.player_roles === "object" &&
    !Array.isArray(candidate.player_roles) &&
    !!candidate.player_types &&
    typeof candidate.player_types === "object" &&
    !Array.isArray(candidate.player_types) &&
    Array.isArray(candidate.legal_moves_suits) &&
    Array.isArray(candidate.suit_last_move)
  );
}

function isRoundOverMessage(value: unknown): value is RoundOverMessage {
  if (!value || typeof value !== "object") return false;

  const candidate = value as Partial<RoundOverMessage>;
  return (
    candidate.type === "ROUND_OVER" &&
    !!candidate.scores &&
    !!candidate.roles &&
    Array.isArray(candidate.out_order)
  );
}

function clearSelectedCards() {
  selectedCards.set([]);
}

function clearState() {
  state.set(null);
}

function clearAnimations() {
  // Empty the queue first: going idle immediately runs the next queued update
  stateUpdateQueue = [];
  animatingElements.forEach((timer) => clearTimeout(timer));
  animatingElements.clear();
  isAnimating.set(false);
}

function clearFastForwardMode() {
  fastForwardMode.set(false);
}

function clearRoundSummary() {
  roundSummary.set(null);
  revealedBotSeat.set(null);
}

function clearExchangePrompt() {
  exchangePrompt.set(null);
}

function clearJumpInPrompt() {
  jumpInPrompt.set(null);
}

function clearDealing() {
  isDealing.set(false);
}

function clearAll() {
  clearSelectedCards();
  clearState();
  clearAnimations();
  clearFastForwardMode();
  clearRoundSummary();
  clearExchangePrompt();
  clearJumpInPrompt();
  clearDealing();
  activePromptId = null;
  inGame.set(false);
  lobbyNotice.set(null);
}

function buildAutoFinishMoves(suitLastMove: string[]) {
  const groupedMoves: string[][] = [];

  for (const suit of suitLastMove) {
    const cardValue = stripSuitCard(suit);
    const lastGroup = groupedMoves[groupedMoves.length - 1];
    if (lastGroup && lastGroup[0] === cardValue) lastGroup.push(cardValue);
    else groupedMoves.push([cardValue]);
  }

  return groupedMoves
    .map((group) => {
      const remainingCards = 4 - group.length;
      if (remainingCards <= 0) return null;
      return Array.from({ length: remainingCards }, () => group[0]);
    })
    .filter((move): move is string[] => Array.isArray(move));
}

function arraysEqual(left: string[], right: string[]) {
  if (left.length !== right.length) return false;
  return left.every((value, index) => value === right[index]);
}

function getAutoFinishMove(): string[][] | null;
function getAutoFinishMove(clickedCardValue: string): string[] | null;
function getAutoFinishMove(clickedCardValue: string | null = null) {
  const currState = get(state);
  if (
    !currState?.legal_moves_suits?.length ||
    !currState.suit_last_move?.length
  )
    return null;

  const candidateMoves = buildAutoFinishMoves(currState.suit_last_move);
  const legalMoves = candidateMoves.filter((candidateMove) =>
    currState.legal_moves_suits.some((legalMove) =>
      arraysEqual(legalMove, candidateMove),
    ),
  );

  if (legalMoves.length === 0) return null;

  if (clickedCardValue !== null) return legalMoves.find((move) => move.includes(clickedCardValue)) ?? null;

  return legalMoves;
}

function stripSuitCard(suitCard: string) {
  return suitCard.slice(0, -1).toUpperCase();
}

function getHighestSuitCards(suitedHand: string[], count: number) {
  return suitedHand.slice(-count);
}

function enqueue(action: () => void) {
  if (get(isAnimating)) stateUpdateQueue.push(action);
  else action();
}

function handleSocketMessage(event: MessageEvent<string>) {
  let payload: IncomingWebSocketMessage;
  try {
    payload = JSON.parse(event.data) as IncomingWebSocketMessage;
  } catch {
    return;
  }

  switch (payload.type) {
    case "DEAL_CARDS": {
      const msg = payload as DealCardsMessage;
      if (!isGameStateUpdate(msg.state)) return;

      const nextState = msg.state;
      enqueue(() => {
        activePromptId = null;
        isDealing.set(true);
        state.set(nextState);
        clearSelectedCards();
        clearExchangePrompt();
        clearJumpInPrompt();
      });
      return;
    }

    case "STATE_UPDATE": {
      const msg = payload as StateUpdateMessage;
      if (!isGameStateUpdate(msg.state)) return;

      const nextState = msg.state;
      const keepJumpIn = msg.clearJump === false;
      const promptId = typeof msg.prompt_id === "number" ? msg.prompt_id : null;
      enqueue(() => {
        activePromptId = promptId;
        state.set(nextState);
        clearExchangePrompt();
        if (!keepJumpIn) clearJumpInPrompt();
        // The human's own turn offers a pending jump-in next to a normal move
        else if (!get(jumpInPrompt)) jumpInPrompt.set({ message: JUMP_IN_MESSAGE });
      });
      return;
    }

    case "JUMP_IN_PROMPT": {
      const msg = payload as JumpInPromptMessage;
      if (!isGameStateUpdate(msg.state) || typeof msg.prompt_id !== "number") return;

      const nextState = msg.state;
      const message = typeof msg.message === "string" ? msg.message : JUMP_IN_MESSAGE;
      const promptId = msg.prompt_id;
      enqueue(() => {
        activePromptId = promptId;
        state.set(nextState);
        clearJumpInPrompt();
        jumpInPrompt.set({ message });
      });
      return;
    }

    case "EXCHANGE_PROMPT": {
      const msg = payload as ExchangePromptMessage;
      if (
        !isGameStateUpdate(msg.state) ||
        typeof msg.required_cards !== "number" ||
        typeof msg.can_choose !== "boolean" ||
        typeof msg.prompt_id !== "number"
      )
        return;

      const nextState = msg.state;
      const requiredCards = msg.required_cards;
      const canChoose = msg.can_choose;
      const promptId = msg.prompt_id;
      enqueue(() => {
        activePromptId = promptId;
        state.set(nextState);
        if (!canChoose && requiredCards > 0) selectedCards.set(getHighestSuitCards(nextState.suited_hand, requiredCards));

        exchangePrompt.set({ requiredCards, canChoose });
      });
      return;
    }

    case "ROUND_OVER": {
      if (!isRoundOverMessage(payload)) return;

      clearJumpInPrompt();
      clearExchangePrompt();
      clearFastForwardMode();
      roundSummary.set({
        scores: payload.scores,
        roles: payload.roles,
        out_order: payload.out_order,
      });
      return;
    }

    case "START_REJECTED": {
      const { message } = payload as { message?: unknown };
      inGame.set(false);
      lobbyNotice.set(typeof message === "string" ? message : "The server couldn't start that game.");
      return;
    }

    case "REVEAL_BOT": {
      const { seat } = payload as { seat?: unknown };
      if (typeof seat === "number") revealedBotSeat.set(seat);
      return;
    }

    default:
      return;
  }
}

function attachSocketListeners(activeSocket: WebSocket) {
  activeSocket.addEventListener("open", () => connectionStatus.set("connected"));
  activeSocket.addEventListener("error", () => connectionStatus.set("error"));
  activeSocket.addEventListener("close", () => {
    if (socket !== activeSocket) return; // closed on purpose by disconnect()

    connectionStatus.set("disconnected");
    socket = null;
    if (get(inGame)) {
      // The server's game ended with the connection, so there is nothing to resume
      clearAll();
      lobbyNotice.set("Lost the connection to the game server.");
    }
    scheduleReconnect();
  });

  activeSocket.addEventListener("message", handleSocketMessage);
}

function scheduleReconnect() {
  if (reconnectTimer || !currUrl) return;

  const url = currUrl;
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    connect(url);
  }, RECONNECT_DELAY_MS);
}

function cancelReconnect() {
  if (reconnectTimer) clearTimeout(reconnectTimer);
  reconnectTimer = null;
}

function connect(url: string) {
  if (socket && socket.readyState <= WebSocket.OPEN && currUrl === url) return;

  cancelReconnect();
  if (socket) {
    socket.close();
    socket = null;
  }

  currUrl = url;
  connectionStatus.set("connecting");

  const nextSocket = new WebSocket(url);
  socket = nextSocket;
  attachSocketListeners(nextSocket);
}

function disconnect() {
  cancelReconnect();
  if (!socket) return;

  socket.close();
  socket = null;
  connectionStatus.set("disconnected");
}

function waitForSocketOpen() {
  return new Promise<void>((resolve, reject) => {
    if (!socket || socket.readyState > WebSocket.OPEN) return reject(new Error("Not connected to the game server."));
    if (socket.readyState === WebSocket.OPEN) return resolve();

    const handleOpen = () => {
      cleanup();
      resolve();
    };

    const handleError = () => {
      cleanup();
      reject(new Error("Unable to open websocket connection."));
    };

    const cleanup = () => {
      socket?.removeEventListener("open", handleOpen);
      socket?.removeEventListener("error", handleError);
    };

    socket.addEventListener("open", handleOpen);
    socket.addEventListener("error", handleError);
  });
}

// Never reconnects on its own: a new connection is a new server session without
// this game, so anything sent there would be silently ignored
async function send(payload: Record<string, unknown>) {
  await waitForSocketOpen();
  socket?.send(JSON.stringify(payload));
}

async function startGame(payload: {
  type: "START_GAME";
  num_players: number;
  num_rounds: number;
  player_types: number[];
}) {
  lobbyNotice.set(null);
  await send(payload);
  inGame.set(true);
}

async function dealtCards() {
  await send({ type: "DEALT_CARDS" });
  clearDealing();
}

async function playSelectedCards() {
  await send({
    type: "PLAY_MOVE",
    prompt_id: activePromptId,
    suits: get(selectedCards),
  });
  clearSelectedCards();
}

async function playJumpInPrompt(finishMove: string[]) {
  if (!finishMove) throw new Error("No finishing move is available.");

  clearJumpInPrompt();
  clearSelectedCards();
  await send({ type: "PLAY_MOVE", prompt_id: activePromptId, suits: finishMove, jump: true });
}

async function passTurn() {
  await send({ type: "PLAY_MOVE", prompt_id: activePromptId, suits: [] });
  clearSelectedCards();
}

async function nextRound() {
  clearSelectedCards();
  clearExchangePrompt();
  clearRoundSummary();
  clearState();
  // Updates still queued from the finished round are obsolete
  clearAnimations();
  await send({ type: "NEXT_ROUND" });
}

async function sendExchangeCards(cards: string[]) {
  await send({ type: "EXCHANGE_CARDS", prompt_id: activePromptId, suits: cards });
}

async function processQueue() {
  if (get(isAnimating)) return;
  while (stateUpdateQueue.length > 0) {
    const nextAction = stateUpdateQueue.shift();
    if (nextAction) {
      nextAction();
      await tick();

      if (get(isAnimating)) break;
    }
  }
}

async function fastForwardGame() {
  fastForwardMode.set(true);
  await send({ type: "FAST_FORWARD" });
}

isAnimating.subscribe((animating) => {
  if (!animating) processQueue();
});

export const gameStore = {
  selectedCards,
  connect,
  disconnect,
  startGame,
  dealtCards,
  clearSelectedCards,
  clearRoundSummary,
  clearExchangePrompt,
  clearFastForwardMode,
  clearAll,
  playSelectedCards,
  playJumpInPrompt,
  passTurn,
  nextRound,
  sendExchangeCards,
  getAutoFinishMove,
  startAnimation,
  endAnimation,
  setTheme,
  fastForwardGame,
  stripSuitCard,
  arraysEqual,
};
