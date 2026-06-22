import { writable } from "svelte/store";

export interface GameStateUpdate {
  hand: number[];
  cards_in_pile: number[];
  is_finish_prompt: boolean;
  player_roles: Record<number, string | null>;
}

export interface StateUpdateMessage {
  type: "STATE_UPDATE";
  state: GameStateUpdate;
}

export const logs = writable<string[]>([]);

type IncomingWebSocketMessage =
  | StateUpdateMessage
  | {
      type: string;
      [key: string]: unknown;
    };

type ConnectionStatus = "disconnected" | "connecting" | "connected" | "error";

const state = writable<GameStateUpdate | null>(null);
const connectionStatus = writable<ConnectionStatus>("disconnected");
const lastMessageType = writable<string | null>(null);

let socket: WebSocket | null = null;
let currentUrl: string | null = null;

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

  return hasHand && hasCardsInPile && hasFinishPrompt && hasPlayerRoles;
}

function isStateUpdateMessage(value: unknown): value is StateUpdateMessage {
  if (!value || typeof value !== "object") {
    return false;
  }

  const candidate = value as Partial<StateUpdateMessage>;
  return (
    candidate.type === "STATE_UPDATE" && isGameStateUpdate(candidate.state)
  );
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

      if (
        payload.type === "STATUS_UPDATE" ||
        payload.type === "JUMP_IN_PROMPT"
      ) {
        if (payload.state && isGameStateUpdate(payload.state)) {
          state.set(payload.state);
        }
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

export const gameStore = {
  state,
  connectionStatus,
  lastMessageType,
  connect,
  disconnect,
  send,
  startGame,
};
