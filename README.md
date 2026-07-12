# President RL

President is a card game project with two parts:

- A Python backend for the game engine, CLI tools, RL training, and the FastAPI websocket server.
- A Svelte frontend for the browser UI.

<video src="./public/jump_in.mp4" autoplay loop muted playsinline width="600"></video>

## Architecture

<img src="./public/architecture.png" width="600"></img>

I kept the game engine to be synchronous to allow multiprocessing across CPU cores for ISMCTS parallelism and RL training without event-loop bottlenecks.

## Bot Hierarchy

- **Baseline**: Hardcoded rule-based bot modeled after human playstyles.
- **ISMCTS**: (Information Set Monte Carlo Tree Search), creates possible game states from the current state, plays them out, results are backpropagated up the tree to determine which move is statistically the most successful.
- **DMC**: RL model updated via round placements rewards.
- **Master-Student DMC**: Master is trained with knowing opponent hands, guiding a student model.
- **Asymmetric PPO Actor-Critic**: (Proximal Policy Optimization) with action masking, Critic sees opponent hands to compute GAE (Generalized Advantage Estimation) advantages.

## Custom Rules

- Must play card value higher and card count equal or higher than currently on the pile.
- 2s act as wildcards when played with other cards, or as the highest card value.
- A player can jump-in out-of-turn if he can make a quad out of the last played move.
- After a round ends, 'winning' players ((Vice-)President, Secretary) can choose which cards to pass. 'Losing' players ((High-)Scum, Clerk) must pass their highest cards. Citizens are exempt from card passing.

## Repo Layout

```text
Presidenten/
├── backend/                    # Python codebase
│   ├── dmc_loop/               # DMC training loops (orchestrates training & eval cycles)
│   ├── game/                   # Core synchronous game logic
│   ├── playerTypes/            # Playable bots & saved RL model checkpoints
│   ├── ppo_loop/               # PPO training loops (orchestrates training & eval cycles)
│   ├── benchmark.py            # Pit bots head-to-head
│   ├── main.py                 # FastAPI WebSocket Server
│   └── utils.py                # Shared utilities & helper functions
│
├── frontend/                   # Svelte/TypeScript codebase
│   ├── public/                 # Static assets, card SVGs
│   └── src/
│       ├── assets/             # Reusable UI assets
│       ├── lib/
│       │   ├── components/     # Svelte game components
│       │   ├── themes.ts       # Color themes
│       │   └── transitions.ts  # Svelte card & game transitions
│       ├── stores/             # Svelte state management
│       └── App.svelte          # Main Application Entry
```

## Quick Start

### Launch Backend

```Bash
cd backend
python -m venv venv
(linux) source venv/bin/activate || (windows) .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload
```

### Launch Frontend

```Bash
cd frontend
npm i
npm run dev
```
