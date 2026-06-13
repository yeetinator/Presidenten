const log = document.getElementById('log');
const statusEl = document.getElementById('status');
const turnCountEl = document.getElementById('turn-count');
const activeBotEl = document.getElementById('active-bot');
const botSelect = document.getElementById('bot-select');
const moveForm = document.getElementById('move-form');
const moveInput = document.getElementById('move-input');
const resetBtn = document.getElementById('reset-btn');

let socket;
let currentBot = '';

function addEntry(role, text) {
    const article = document.createElement('article');
    article.className = `entry ${role}`;

    const roleEl = document.createElement('div');
    roleEl.className = 'role';
    roleEl.textContent = role;

    const textEl = document.createElement('div');
    textEl.className = 'text';
    textEl.textContent = text;

    article.append(roleEl, textEl);
    log.append(article);
    log.scrollTop = log.scrollHeight;
}

function setStatus(message) {
    statusEl.textContent = message;
}

function connect() {
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    socket = new WebSocket(`${protocol}://${window.location.host}/ws/play`);

    socket.addEventListener('open', () => {
        setStatus('Connected.');
    });

    socket.addEventListener('message', (event) => {
        const payload = JSON.parse(event.data);

        if (payload.type === 'ready') {
            populateBots(payload.bots);
            currentBot = payload.bot_name;
            activeBotEl.textContent = currentBot;
            setStatus(payload.message);
            return;
        }

        if (payload.type === 'state') {
            currentBot = payload.bot_name;
            activeBotEl.textContent = currentBot;
            setStatus(payload.message);
            log.innerHTML = '';
            turnCountEl.textContent = '0';
            return;
        }

        if (payload.type === 'turn') {
            addEntry('You', payload.user);
            addEntry(payload.bot_name, payload.bot);
            turnCountEl.textContent = String(payload.turn);
            setStatus(`Turn ${payload.turn} complete.`);
            moveInput.value = '';
            moveInput.focus();
            return;
        }

        if (payload.type === 'error') {
            setStatus(payload.message);
        }
    });

    socket.addEventListener('close', () => {
        setStatus('Disconnected. Reconnecting...');
        window.setTimeout(connect, 1200);
    });

    socket.addEventListener('error', () => {
        setStatus('WebSocket error.');
    });
}

function populateBots(bots) {
    botSelect.innerHTML = '';
    for (const bot of bots) {
        const option = document.createElement('option');
        option.value = bot;
        option.textContent = bot;
        botSelect.append(option);
    }
    botSelect.value = currentBot || bots[0];
}

botSelect.addEventListener('change', () => {
    if (socket?.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: 'select_bot', bot_name: botSelect.value }));
    }
});

moveForm.addEventListener('submit', (event) => {
    event.preventDefault();
    const text = moveInput.value.trim();
    if (!text || socket?.readyState !== WebSocket.OPEN) {
        return;
    }

    socket.send(JSON.stringify({ type: 'move', text }));
});

resetBtn.addEventListener('click', () => {
    if (socket?.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: 'select_bot', bot_name: botSelect.value }));
        setStatus('Match reset.');
    }
});

connect();
