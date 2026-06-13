const socketUrl = `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}/ws`;
let socket = null;

const CARD_NAMES = {
    11: "J",
    12: "Q",
    13: "K",
    14: "A",
    15: "2",
};

const BOT_KIND_BY_SELECT_VALUE = {
    0: "random",
    1: "baseline",
    2: "ismcts",
};

const state = {
    config: {
        human_player: 0,
        player_kinds: ["human", "baseline", "baseline", "baseline"],
        ismcts_iterations: 140,
        max_rounds: 10,
    },
    current: null,
};

const el = (id) => document.getElementById(id);

function cardName(value) {
    return CARD_NAMES[value] ?? String(value);
}

function moveLabel(move) {
    if (!move || move.length !== 3) {
        return "-";
    }
    const [card, count, twos] = move;
    if (card === 0 && count === 0 && twos === 0) {
        return "Pass";
    }
    if (twos > 0) {
        return `${count}x ${cardName(card)} (${count - twos}x ${cardName(card)} + ${twos}x 2)`;
    }
    return `${count}x ${cardName(card)}`;
}

function pushLog(message, kind = "info") {
    const container = el("logArea");
    if (!container) {
        return;
    }

    const row = document.createElement("div");
    row.className = `log-entry ${kind === "warn" ? "log-warn" : kind === "error" ? "log-error" : ""}`;
    row.textContent = message;
    container.prepend(row);
}

function setText(id, value) {
    const target = el(id);
    if (target) {
        target.textContent = value;
    }
}

function setHidden(id, hidden) {
    const target = el(id);
    if (target) {
        target.classList.toggle("hidden", hidden);
    }
}

function readConfig() {
    const kinds = ["human", "baseline", "baseline", "baseline"];

    [el("bot1"), el("bot2"), el("bot3")].forEach((select, index) => {
        if (!select) {
            return;
        }
        kinds[index + 1] = BOT_KIND_BY_SELECT_VALUE[select.value] ?? "baseline";
    });

    const roundsInput = el("rounds");
    if (roundsInput) {
        state.config.max_rounds = Math.max(1, Math.min(Number(roundsInput.value) || 10, 100));
    }

    state.config.human_player = 0;
    state.config.player_kinds = kinds;
}

function renderPlayers(players) {
    const container = el("players");
    if (!container) {
        return;
    }

    container.innerHTML = "";
    players.forEach((player) => {
        const row = document.createElement("div");
        row.className = `player-row flex items-center justify-between gap-3 rounded-2xl border px-4 py-3 ${player.is_human ? "border-cyan-300/30 bg-cyan-400/10" : "border-white/10 bg-white/5"}`;

        const left = document.createElement("div");
        left.innerHTML = `<div class="text-sm font-bold text-white">${player.label}</div><div class="text-xs uppercase tracking-[0.18em] text-slate-400">${player.role}</div>`;

        const right = document.createElement("div");
        right.className = "text-right text-xs text-slate-300";
        right.innerHTML = `<div>${player.hand_count} cards left</div><div>${player.round_wins} rounds won</div>`;

        row.append(left, right);
        container.appendChild(row);
    });
}

function renderScores(scores) {
    const container = el("scoreArea");
    if (!container) {
        return;
    }

    container.innerHTML = "";
    scores.forEach((score) => {
        const row = document.createElement("div");
        row.className = "flex items-center justify-between rounded-2xl bg-white/5 px-4 py-3";
        row.innerHTML = `<span class="text-slate-300">Seat ${score.player_id}</span><span class="font-semibold text-white">${score.score} pts · ${score.round_wins} rounds</span>`;
        container.appendChild(row);
    });
}

function renderHistory(history) {
    const container = el("historyArea");
    if (!container) {
        return;
    }

    container.innerHTML = "";
    if (!history.length) {
        const empty = document.createElement("div");
        empty.className = "rounded-2xl border border-dashed border-white/10 px-4 py-3 text-sm text-slate-400";
        empty.textContent = "No moves yet.";
        container.appendChild(empty);
        return;
    }

    history.slice().reverse().forEach((entry) => {
        const row = document.createElement("div");
        row.className = "rounded-2xl border border-white/10 bg-white/5 px-4 py-3";
        row.innerHTML = `<div class="text-xs uppercase tracking-[0.18em] text-slate-400">${entry.label}</div><div class="mt-1 text-sm text-white">${entry.move_label}</div>`;
        container.appendChild(row);
    });
}

function renderHand(hand) {
    const container = el("handArea");
    if (!container) {
        return;
    }

    container.innerHTML = "";
    setText("handHint", `${hand.length} cards`);

    hand.forEach((card) => {
        const chip = document.createElement("div");
        chip.className = "chip rounded-2xl px-4 py-3 text-sm font-bold text-white";
        chip.textContent = cardName(card);
        container.appendChild(chip);
    });
}

function renderMoves(moves, moveLabels, yourTurn, currentTurnLabel) {
    const container = el("moveArea");
    if (!container) {
        return;
    }

    container.innerHTML = "";
    setText("moveHint", yourTurn ? "Choose a legal move" : `Waiting for ${currentTurnLabel}`);

    if (!moves.length) {
        const empty = document.createElement("div");
        empty.className = "rounded-2xl border border-dashed border-white/10 px-4 py-3 text-sm text-slate-400";
        empty.textContent = yourTurn ? "No legal moves available." : `Waiting for ${currentTurnLabel}.`;
        container.appendChild(empty);
        return;
    }

    moves.forEach((move, index) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = `chip rounded-2xl px-4 py-3 text-sm font-bold text-white disabled:opacity-40 ${yourTurn ? "selected" : ""}`;
        button.textContent = moveLabels[index] ?? moveLabel(move);
        button.disabled = !yourTurn;
        button.addEventListener("click", () => {
            if (socket && socket.readyState === WebSocket.OPEN) {
                socket.send(JSON.stringify({ type: "move", move }));
            }
        });
        container.appendChild(button);
    });
}

function renderCurrent() {
    const current = state.current;
    if (!current) {
        return;
    }

    setText("phaseTitle", current.finished ? "Game complete" : current.your_turn ? "Your turn" : "Live table");
    setText("sessionId", `Round ${current.round_number ?? "-"}`);
    setText("roundInfo", `${current.round_number ?? "-"}${current.max_rounds != null ? ` / ${current.max_rounds}` : ""}`);
    setText("turnInfo", current.current_turn_label ?? "-");
    setText("lastMoveInfo", current.last_move_label ?? "-");
    setText("roleInfo", current.your_role ?? "-");
    setText("statusLine", current.message || (current.your_turn ? "Your move." : "Bots are playing."));
    setText("activePlayersInfo", (current.active_players || []).join(", ") || "-");
    setText("passedInfo", (current.passed || []).join(", ") || "-");
    setText("leaderInfo", current.current_turn_label ?? "-");
    setText("exchangeInfo", current.exchange_requirement ? `${current.exchange_requirement.role} -> ${current.exchange_requirement.partner_role} (${current.exchange_requirement.count})` : "None");
    setText("exchangeText", current.exchange_requirement ? `Select exactly ${current.exchange_requirement.count} card(s) to pass to ${current.exchange_requirement.partner_role}.` : "");

    renderHand(current.hand || []);
    renderMoves(current.legal_moves || [], current.legal_move_labels || [], Boolean(current.your_turn), current.current_turn_label ?? "the bots");
    renderPlayers(current.player_states || []);
    renderScores(current.scores || []);
    renderHistory(current.history || []);

    setHidden("exchangePanel", !current.exchange_requirement);
    const confirmBtn = el("confirmExchangeBtn");
    if (confirmBtn) {
        confirmBtn.disabled = true;
    }
}

function handlePayload(payload) {
    if (payload.message) {
        pushLog(payload.message, payload.type === "game_over" ? "warn" : payload.type === "error" ? "error" : "info");
    }

    if (payload.state) {
        state.current = payload.state;
        renderCurrent();
    }
}

function connect() {
    if (socket) {
        socket.close();
    }

    socket = new WebSocket(socketUrl);

    socket.addEventListener("open", () => {
        pushLog("Socket connected.");
    });

    socket.addEventListener("close", () => {
        pushLog("Socket disconnected.", "warn");
    });

    socket.addEventListener("message", (event) => {
        handlePayload(JSON.parse(event.data));
    });
}

const startButton = el("startBtn");
if (startButton) {
    startButton.addEventListener("click", () => {
        readConfig();
        if (socket && socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify({ type: "start", config: state.config }));
            pushLog("Starting a new match...");
        }
    });
}

const confirmExchangeBtn = el("confirmExchangeBtn");
if (confirmExchangeBtn) {
    confirmExchangeBtn.addEventListener("click", () => {
        if (socket && socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify({ type: "exchange", selected_indices: [] }));
        }
    });
}

const clearLogBtn = el("clearLogBtn");
if (clearLogBtn) {
    clearLogBtn.addEventListener("click", () => {
        const container = el("logArea");
        if (container) {
            container.innerHTML = "";
        }
    });
}

readConfig();
connect();
