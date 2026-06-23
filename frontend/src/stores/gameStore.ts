import { writable } from "svelte/store";

export interface GameStateUpdate {
  hand: number[];
  cards_in_pile: number[];
  is_finish_prompt: boolean;
  my_role: string;
  curr_turn: number | null;
  round: number;
  player_roles: Record<number, string | null>;
  player_types: Record<number, string>;
  legal_moves: [number, number, number][];
  last_move: [number, number, number];
  role_pairs: [string, string, number][];
  opp_hand_counts: Record<number, number>;
  first_turn: boolean;
  clubs_3_holder: number;
}

export interface VisualCard {
  id: string;
  value: number;
  suit: "hearts" | "diamonds" | "clubs" | "spades";
}

export interface StateUpdateMessage {
  type: "STATE_UPDATE";
  state: GameStateUpdate;
}

export interface MoveTuple {
  card_value: number;
  count: number;
  twos_used: number;
}

export interface RoundOverMessage {
  type: "ROUND_OVER";
  scores: Record<string, [number, number]>;
  roles: Record<string, string>;
  out_order: number[];
}

export interface RoundSummary {
  scores: Record<string, [number, number]>;
  roles: Record<string, string>;
  out_order: number[];
}

export interface JumpInPromptMessage {
  type: "JUMP_IN_PROMPT";
  state: GameStateUpdate;
  timeout_seconds: number;
  message: string;
}

export interface JumpInPrompt {
  message: string;
  timeoutSeconds: number;
  state: GameStateUpdate;
}

export interface ExchangePrompt {
  state: GameStateUpdate;
  requiredCards: number;
  canChoose: boolean;
}

export const logs = writable<string[]>([]);
export const state = writable<GameStateUpdate | null>(null);
export const connectionStatus = writable<ConnectionStatus>("disconnected");
export const lastMessageType = writable<string | null>(null);
export const selectedCards = writable<number[]>([]);
export const roundSummary = writable<RoundSummary | null>(null);
export const jumpInPrompt = writable<JumpInPrompt | null>(null);
export const exchangePrompt = writable<ExchangePrompt | null>(null);

type IncomingWebSocketMessage =
  | StateUpdateMessage
  | JumpInPromptMessage
  | RoundOverMessage
  | {
      type: string;
      [key: string]: unknown;
    };

type ConnectionStatus = "disconnected" | "connecting" | "connected" | "error";

let socket: WebSocket | null = null;
let currentUrl: string | null = null;
let latestSelectedCards: number[] = [];
let latestState: GameStateUpdate | null = null;
let jumpInPromptTimeout: ReturnType<typeof setTimeout> | null = null;

selectedCards.subscribe((value) => {
  latestSelectedCards = value;
});

state.subscribe((value) => {
  latestState = value;
});

function isGameStateUpdate(value: unknown): value is GameStateUpdate {
  if (!value || typeof value !== "object") return false;

  const candidate = value as Partial<GameStateUpdate>;
  return (
    Array.isArray(candidate.hand) &&
    Array.isArray(candidate.cards_in_pile) &&
    typeof candidate.is_finish_prompt === "boolean" &&
    !!candidate.player_roles &&
    typeof candidate.player_roles === "object" &&
    !Array.isArray(candidate.player_roles) &&
    !!candidate.player_types &&
    typeof candidate.player_types === "object" &&
    !Array.isArray(candidate.player_types) &&
    Array.isArray(candidate.legal_moves) &&
    Array.isArray(candidate.last_move)
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

function normalizeSelectedCards(cards: number[]): MoveTuple | null {
  if (cards.length === 0) return null;

  const twosUsed = cards.filter((card) => card === 15).length;
  const nonTwoCards = cards.filter((card) => card !== 15);

  if (nonTwoCards.length > 0) {
    const [firstCard] = nonTwoCards;
    if (!nonTwoCards.every((card) => card === firstCard)) return null;

    return {
      card_value: firstCard,
      count: cards.length,
      twos_used: twosUsed,
    };
  }

  return {
    card_value: 15,
    count: cards.length,
    twos_used: 0,
  };
}

function clearSelectedCards() {
  selectedCards.set([]);
}

function clearRoundSummary() {
  roundSummary.set(null);
}

function clearExchangePrompt() {
  exchangePrompt.set(null);
}

function clearLogs() {
  logs.set([]);
}

function clearJumpInPrompt() {
  if (jumpInPromptTimeout) {
    clearTimeout(jumpInPromptTimeout);
    jumpInPromptTimeout = null;
  }
  jumpInPrompt.set(null);
}

function selectCard(card: number) {
  selectedCards.update((cards) => [...cards, card]);
}

function getSelectedMoveTuple() {
  return normalizeSelectedCards(latestSelectedCards);
}

function getAutoFinishMove() {
  if (!latestState?.legal_moves) return null;
  return latestState.legal_moves.find((move) => move[0] !== 0) ?? null;
}

function attachSocketListeners(activeSocket: WebSocket) {
  activeSocket.addEventListener("open", () =>
    connectionStatus.set("connected"),
  );
  activeSocket.addEventListener("error", () => connectionStatus.set("error"));
  activeSocket.addEventListener("close", () => {
    if (socket === activeSocket) {
      connectionStatus.set("disconnected");
      socket = null;
    }
  });

  activeSocket.addEventListener("message", (event: MessageEvent<string>) => {
    try {
      const payload = JSON.parse(event.data) as IncomingWebSocketMessage;
      lastMessageType.set(payload.type);

      if (payload.type === "STATE_UPDATE" && isGameStateUpdate(payload.state)) {
        state.set(payload.state);
        clearExchangePrompt();
      }

      if (
        payload.type === "JUMP_IN_PROMPT" &&
        payload.state &&
        isGameStateUpdate(payload.state)
      ) {
        state.set(payload.state);
        clearJumpInPrompt();
        jumpInPrompt.set({
          message:
            typeof payload.message === "string" ? payload.message : "JUMP IN!",
          timeoutSeconds:
            typeof payload.timeout_seconds === "number"
              ? payload.timeout_seconds
              : 1.5,
          state: payload.state,
        });

        const promptTimeout =
          typeof payload.timeout_seconds === "number"
            ? payload.timeout_seconds
            : 1.5;

        jumpInPromptTimeout = setTimeout(() => {
          jumpInPrompt.set(null);
          jumpInPromptTimeout = null;
        }, promptTimeout * 1000);
      }

      if (
        payload.type === "EXCHANGE_PROMPT" &&
        payload.state &&
        isGameStateUpdate(payload.state) &&
        typeof payload.required_cards === "number" &&
        typeof payload.can_choose === "boolean"
      ) {
        state.set(payload.state);
        if (!payload.can_choose && payload.required_cards > 0) {
          const hand = payload.state.hand;
          const count = payload.required_cards;
          const highCards = hand.slice(hand.length - count);
          selectedCards.set(highCards);
        } else {
          selectedCards.set([]);
        }
        exchangePrompt.set({
          state: payload.state,
          requiredCards: payload.required_cards,
          canChoose: payload.can_choose,
        });
      }

      if (isRoundOverMessage(payload)) {
        clearJumpInPrompt();
        clearExchangePrompt();
        roundSummary.set({
          scores: payload.scores,
          roles: payload.roles,
          out_order: payload.out_order,
        });
      }

      if (
        (payload.type === "GAME_LOG" || payload.type === "LOG_ALERT") &&
        typeof payload.message === "string"
      ) {
        logs.update((currLogs) => [...currLogs, payload.message as string]);
      }
    } catch {
      // Ignore malformed payloads; other valid messages keep flowing.
    }
  });
}

function connect(url: string) {
  if (socket && socket.readyState <= WebSocket.OPEN && currentUrl === url)
    return;

  if (socket) {
    socket.close();
    socket = null;
  }

  currentUrl = url;
  connectionStatus.set("connecting");

  const nextSocket = new WebSocket(url);
  socket = nextSocket;
  attachSocketListeners(nextSocket);
}

function disconnect() {
  if (!socket) return;

  socket.close();
  socket = null;
  connectionStatus.set("disconnected");
}

function waitForSocketOpen() {
  return new Promise<void>((resolve, reject) => {
    if (!socket) return reject(new Error("WebSocket is not initialized."));
    if (socket.readyState === WebSocket.OPEN) return resolve();
    if (socket.readyState !== WebSocket.CONNECTING)
      return reject(new Error("WebSocket is not connected."));

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

async function send(payload: Record<string, unknown>) {
  if (!socket || socket.readyState === WebSocket.CLOSED) {
    if (!currentUrl) throw new Error("No websocket URL has been configured.");
    connect(currentUrl);
  }

  await waitForSocketOpen();
  socket?.send(JSON.stringify(payload));
}

async function startGame(payload: {
  type: "START_GAME";
  num_players: number;
  num_rounds: number;
  player_types: number[];
}) {
  await send(payload);
}

async function playSelectedCards() {
  const cards = getSelectedMoveTuple();
  if (!cards) throw new Error("Invalid selection structure.");

  await send({
    type: "PLAY_MOVE",
    move: [cards.card_value, cards.count, cards.twos_used],
  });
  clearSelectedCards();
}

async function playJumpInPrompt() {
  const move = getAutoFinishMove();
  if (!move) throw new Error("No finishing move is available.");

  clearJumpInPrompt();
  clearSelectedCards();
  await send({ type: "PLAY_MOVE", move });
}

async function passTurn() {
  await send({ type: "PLAY_MOVE", move: [0, 0, 0] });
  clearSelectedCards();
}

async function nextRound() {
  clearSelectedCards();
  clearExchangePrompt();
  clearRoundSummary();
  await send({ type: "NEXT_ROUND" });
}

function quitGame() {
  clearSelectedCards();
  clearExchangePrompt();
  disconnect();
}

async function sendExchangeCards(cards: number[]) {
  await send({ type: "EXCHANGE_CARDS", cards });
}

export const gameStore = {
  state,
  connectionStatus,
  lastMessageType,
  selectedCards,
  roundSummary,
  jumpInPrompt,
  exchangePrompt,
  connect,
  disconnect,
  send,
  startGame,
  selectCard,
  clearSelectedCards,
  clearRoundSummary,
  clearExchangePrompt,
  clearLogs,
  clearJumpInPrompt,
  playSelectedCards,
  playJumpInPrompt,
  passTurn,
  nextRound,
  sendExchangeCards,
  quitGame,
  getAutoFinishMove,
};
