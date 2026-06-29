import { writable, get } from "svelte/store";
import { tick } from "svelte";

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

export interface StateUpdateMessage {
  type: "STATE_UPDATE";
  state: GameStateUpdate;
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

export interface ExchangePromptMessage {
  type: "EXCHANGE_PROMPT";
  state: GameStateUpdate;
  required_cards: number;
  can_choose: boolean;
}

export const logs = writable<string[]>([]);
export const state = writable<GameStateUpdate | null>(null);
export const connectionStatus = writable<ConnectionStatus>("disconnected");
export const lastMessageType = writable<string | null>(null);
export const selectedCards = writable<string[]>([]);
export const roundSummary = writable<RoundSummary | null>(null);
export const jumpInPrompt = writable<JumpInPrompt | null>(null);
export const exchangePrompt = writable<ExchangePrompt | null>(null);
export const isAnimating = writable<boolean>(false);
export const revealedBotSeat = writable<number | null>(null);

type IncomingWebSocketMessage =
  | StateUpdateMessage
  | JumpInPromptMessage
  | ExchangePromptMessage
  | RoundOverMessage
  | {
      type: string;
      [key: string]: unknown;
    };

type ConnectionStatus = "disconnected" | "connecting" | "connected" | "error";

let socket: WebSocket | null = null;
let currentUrl: string | null = null;
let jumpInPromptTimeout: ReturnType<typeof setTimeout> | null = null;
let stateUpdateQueue: (() => void)[] = [];
let activeAnimationsCount = 0;

export function startAnimation() {
  activeAnimationsCount++;
  isAnimating.set(true);
}

export function endAnimation() {
  activeAnimationsCount = Math.max(0, activeAnimationsCount - 1);
  if (activeAnimationsCount === 0) isAnimating.set(false);
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
  activeAnimationsCount = 0;
  isAnimating.set(false);
  stateUpdateQueue = [];
}

function clearLastMessageType() {
  lastMessageType.set(null);
}

function clearRoundSummary() {
  roundSummary.set(null);
  revealedBotSeat.set(null);
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

function clearAll() {
  clearSelectedCards();
  clearState();
  clearAnimations();
  clearLastMessageType();
  clearRoundSummary();
  clearExchangePrompt();
  clearLogs();
  clearJumpInPrompt();
}

function getAutoFinishMove() {
  const currState = get(state);
  if (!currState?.legal_moves_suits) return null;
  return currState.legal_moves_suits[0] ?? null;
}

function getHighestSuitCards(suitedHand: string[], count: number) {
  return suitedHand.slice(-count);
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

      if (payload.type === "STATE_UPDATE") {
        const msg = payload as StateUpdateMessage;
        if (isGameStateUpdate(msg.state)) {
          const action = () => {
            state.set(msg.state);
            clearExchangePrompt();
            clearJumpInPrompt();
          };
          if (get(isAnimating)) stateUpdateQueue.push(action);
          else action();
        }
      }

      if (payload.type === "JUMP_IN_PROMPT") {
        const msg = payload as JumpInPromptMessage;
        if (isGameStateUpdate(msg.state)) {
          const action = () => {
            state.set(msg.state);
            clearJumpInPrompt();
            jumpInPrompt.set({
              message:
                typeof msg.message === "string" ? msg.message : "JUMP IN!",
              timeoutSeconds:
                typeof msg.timeout_seconds === "number"
                  ? msg.timeout_seconds
                  : 1.5,
              state: msg.state,
            });

            const promptTimeout =
              typeof msg.timeout_seconds === "number"
                ? msg.timeout_seconds
                : 1.5;
            jumpInPromptTimeout = setTimeout(() => {
              jumpInPrompt.set(null);
              jumpInPromptTimeout = null;
            }, promptTimeout * 1000);
          };
          if (get(isAnimating)) stateUpdateQueue.push(action);
          else action();
        }
      }

      if (payload.type === "EXCHANGE_PROMPT") {
        const msg = payload as ExchangePromptMessage;
        if (
          isGameStateUpdate(msg.state) &&
          typeof msg.required_cards === "number" &&
          typeof msg.can_choose === "boolean"
        ) {
          const action = () => {
            state.set(msg.state);
            if (!msg.can_choose && msg.required_cards > 0) {
              selectedCards.set(
                getHighestSuitCards(msg.state.suited_hand, msg.required_cards),
              );
            }
            exchangePrompt.set({
              state: msg.state,
              requiredCards: msg.required_cards,
              canChoose: msg.can_choose,
            });
          };
          if (get(isAnimating)) stateUpdateQueue.push(action);
          else action();
        }
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
      )
        logs.update((currLogs) => [...currLogs, payload.message as string]);

      if (payload.type === "REVEAL_BOT" && typeof payload.seat === "number")
        revealedBotSeat.set(payload.seat);
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
  await send({
    type: "PLAY_MOVE",
    suits: get(selectedCards),
  });
  clearSelectedCards();
}

async function playJumpInPrompt() {
  const move = getAutoFinishMove();
  if (!move) throw new Error("No finishing move is available.");

  clearJumpInPrompt();
  clearSelectedCards();
  await send({ type: "PLAY_MOVE", suits: move });
}

async function passTurn() {
  await send({ type: "PLAY_MOVE", suits: [] });
  clearSelectedCards();
}

async function nextRound() {
  clearSelectedCards();
  clearExchangePrompt();
  clearRoundSummary();
  clearState();
  await send({ type: "NEXT_ROUND" });
}

async function sendExchangeCards(cards: string[]) {
  await send({
    type: "EXCHANGE_CARDS",
    suits: cards,
  });
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

isAnimating.subscribe((animating) => {
  if (!animating) processQueue();
});

export const gameStore = {
  state,
  connectionStatus,
  lastMessageType,
  selectedCards,
  roundSummary,
  revealedBotSeat,
  jumpInPrompt,
  exchangePrompt,
  connect,
  disconnect,
  send,
  startGame,
  clearSelectedCards,
  clearRoundSummary,
  clearExchangePrompt,
  clearLogs,
  clearJumpInPrompt,
  clearAll,
  playSelectedCards,
  playJumpInPrompt,
  passTurn,
  nextRound,
  sendExchangeCards,
  getAutoFinishMove,
  startAnimation,
  endAnimation,
};
