import { writable } from "svelte/store";

export interface GameStateUpdate {
  hand: number[];
  cards_in_pile: number[];
  is_finish_prompt: boolean;
  player_roles: Record<number, string | null>;
  legal_moves: [number, number, number][];
  last_move: [number, number, number];
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

export const logs = writable<string[]>([]);
export const state = writable<GameStateUpdate | null>(null);
export const connectionStatus = writable<ConnectionStatus>("disconnected");
export const lastMessageType = writable<string | null>(null);
export const selectedCards = writable<number[]>([]);
export const roundSummary = writable<RoundSummary | null>(null);

type IncomingWebSocketMessage =
  | StateUpdateMessage
  | RoundOverMessage
  | {
      type: string;
      [key: string]: unknown;
    };

type ConnectionStatus = "disconnected" | "connecting" | "connected" | "error";

let socket: WebSocket | null = null;
let currentUrl: string | null = null;
let latestSelectedCards: number[] = [];

selectedCards.subscribe((value) => {
  latestSelectedCards = value;
});

function isGameStateUpdate(value: unknown): value is GameStateUpdate {
  if (!value || typeof value !== "object") {
    return false;
  }

  const candidate = value as Partial<GameStateUpdate>;
  const hasHand = Array.isArray(candidate.hand);
  const hasCardsInPile = Array.isArray(candidate.cards_in_pile);
  const hasFinishPrompt = typeof candidate.is_finish_prompt === "boolean";
  const hasPlayerRoles =
    !!candidate.player_roles &&
    typeof candidate.player_roles === "object" &&
    !Array.isArray(candidate.player_roles);

  const hasLegalMoves = Array.isArray(candidate.legal_moves);
  const hasLastMove = Array.isArray(candidate.last_move);

  return (
    hasHand &&
    hasCardsInPile &&
    hasFinishPrompt &&
    hasPlayerRoles &&
    hasLegalMoves &&
    hasLastMove
  );
}

function isRoundOverMessage(value: unknown): value is RoundOverMessage {
  if (!value || typeof value !== "object") {
    return false;
  }

  const candidate = value as Partial<RoundOverMessage>;
  return (
    candidate.type === "ROUND_OVER" &&
    !!candidate.scores &&
    !!candidate.roles &&
    Array.isArray(candidate.out_order)
  );
}

function normalizeSelectedCards(cards: number[]): MoveTuple | null {
  if (cards.length === 0) {
    return null;
  }

  const twosUsed = cards.filter((card) => card === 15).length;
  const nonTwoCards = cards.filter((card) => card !== 15);

  if (nonTwoCards.length > 0) {
    const [firstCard] = nonTwoCards;
    if (!nonTwoCards.every((card) => card === firstCard)) {
      return null;
    }

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

function clearLogs() {
  logs.set([]);
}

function selectCard(card: number) {
  selectedCards.update((cards) => [...cards, card]);
}

function getSelectedMoveTuple() {
  return normalizeSelectedCards(latestSelectedCards);
}

function attachSocketListeners(activeSocket: WebSocket) {
  activeSocket.addEventListener("open", () => {
    connectionStatus.set("connected");
  });

  activeSocket.addEventListener("error", () => {
    connectionStatus.set("error");
  });

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
      }

      if (
        payload.type === "JUMP_IN_PROMPT" &&
        payload.state &&
        isGameStateUpdate(payload.state)
      ) {
        state.set(payload.state);
      }

      if (isRoundOverMessage(payload)) {
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
  if (socket && socket.readyState <= WebSocket.OPEN && currentUrl === url) {
    return;
  }

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
  if (!socket) {
    return;
  }

  socket.close();
  socket = null;
  connectionStatus.set("disconnected");
}

function waitForSocketOpen() {
  return new Promise<void>((resolve, reject) => {
    if (!socket) {
      reject(new Error("WebSocket is not initialized."));
      return;
    }

    if (socket.readyState === WebSocket.OPEN) {
      resolve();
      return;
    }

    if (socket.readyState !== WebSocket.CONNECTING) {
      reject(new Error("WebSocket is not connected."));
      return;
    }

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
    if (!currentUrl) {
      throw new Error(
        "No websocket URL has been configured. Call connect(url) first.",
      );
    }
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

  if (!cards) {
    throw new Error(
      "Selected cards must all be the same rank, optionally with 2s.",
    );
  }

  await send({
    type: "PLAY_MOVE",
    move: [cards.card_value, cards.count, cards.twos_used],
  });
  clearSelectedCards();
}

async function passTurn() {
  await send({
    type: "PLAY_MOVE",
    move: [0, 0, 0],
  });
  clearSelectedCards();
}

async function nextRound() {
  clearSelectedCards();
  clearRoundSummary();
  await send({
    type: "NEXT_ROUND",
  });
}

function quitGame() {
  clearSelectedCards();
  disconnect();
}

export const gameStore = {
  state,
  connectionStatus,
  lastMessageType,
  selectedCards,
  roundSummary,
  connect,
  disconnect,
  send,
  startGame,
  selectCard,
  clearSelectedCards,
  clearRoundSummary,
  clearLogs,
  playSelectedCards,
  passTurn,
  nextRound,
  quitGame,
};
