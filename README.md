# President RL

President is a card game project with two parts:

- A Python backend for the game engine, bots, CLI tournaments, RL training, and the FastAPI websocket server.
- A Svelte frontend for the browser UI, where you play against 3 to 6 bots.

![Jump In Demo](./public/jump_in.gif)

## Architecture

<img src="./public/architecture.png" width="600"></img>

I kept the game engine synchronous to allow multiprocessing across CPU cores for ISMCTS parallelism and RL training without event-loop bottlenecks. The server runs bot decisions in a thread (`asyncio.to_thread`) so the websocket stays responsive.

## Bots

- **Random**: Picks a random legal move. Available in the CLI only.
- **Baseline**: Hardcoded rule-based bot modeled after human playstyles.
- **ISMCTS**: (Information Set Monte Carlo Tree Search), creates possible game states from the current state, plays them out, results are backpropagated up the tree to determine which move is statistically the most successful. The possible states favour hands that explain the opponents' moves so far this round. Thinks for 1 second per move.
- **DMC**: (Deep Monte Carlo) Q-value network over 125 move actions, trained through self-play with round placement rewards. It also chooses which cards to pass. The web game averages two independently trained models (`bots/DMC.pt` and `bots/DMCW.pt`) into an ensemble.

### Planned

- **Master-Student DMC**: Master is trained with knowing opponent hands, guiding a student model.

### Tried

- **Asymmetric PPO Actor-Critic**: (Proximal Policy Optimization) with action masking, Critic sees opponent hands to compute GAE (Generalized Advantage Estimation) advantages. It plays at the DMC ensemble's level but doesn't beat it, see [PPO Training](#ppo-training).

## Custom Rules

- 4 to 7 players. The holder of the 3 of Clubs starts the first round by playing 3s, the Scum starts every round after that.
- Must play card value higher and card count equal or higher than currently on the pile, up to 4 cards.
- 2s act as wildcards when played with other cards, or as the highest card value. Playing 2s on their own clears the pile.
- A player can jump in out of turn if they can make a quad out of the last played move, even after passing. Jumping in clears the pile and the jumper leads the next one.
- Finishing your hand with a 2 ranks you last for that round.
- After a round ends, 'winning' players ((Vice-)President, Secretary) can choose which cards to pass. 'Losing' players ((High-)Scum, Clerk) must pass their highest cards. Citizens are exempt from card passing.

| Players | Roles (best to worst)                                         | Cards exchanged         |
| ------- | ------------------------------------------------------------- | ----------------------- |
| 4       | President, Vice-President, High-Scum, Scum                    | 2 / 1                   |
| 5       | President, Vice-President, Citizen, High-Scum, Scum           | 2 / 1                   |
| 6       | President, Vice-President, Secretary, Clerk, High-Scum, Scum  | 3 / 2 / 1               |
| 7       | Same as 6, with a Citizen in the middle                       | 3 / 2 / 1               |

Cards exchanged are President ↔ Scum / Vice-President ↔ High-Scum / Secretary ↔ Clerk.

## Repo Layout

```text
Presidenten/
├── backend/                    # Python codebase (run commands from here)
│   ├── bots/                   # Trained DMC checkpoints used by the server
│   ├── dmc_loop/               # DMC training loop (train, evaluate, orchestrate)
│   ├── game/                   # Synchronous game engine, CLI prompts & tournaments
│   ├── playerTypes/            # Human player & bots (Random, Baseline, ISMCTS, DMC, PPO)
│   ├── ppo_loop/               # PPO training loop (distill, train, evaluate, orchestrate, gates)
│   ├── tests/                  # Pytest suite (engine, jump-ins, server, full games)
│   ├── utils/                  # Self-play, training, Elo league & head-to-head helpers
│   ├── benchmark.py            # Pit two DMC checkpoints head-to-head
│   ├── main.py                 # FastAPI WebSocket server
│   └── run_cli.py              # Terminal tournaments between any player types
│
├── frontend/                   # Svelte/TypeScript codebase
│   ├── public/                 # Static assets, card SVGs
│   └── src/
│       ├── assets/             # Card & rules components
│       ├── lib/
│       │   ├── components/     # Lobby, opponent seats & round summary
│       │   ├── themes.ts       # Table color themes
│       │   ├── transitions.ts  # Svelte card & game transitions
│       │   └── types.ts        # Shared types
│       ├── stores/             # Svelte state management
│       └── App.svelte          # Main game table
│
├── public/                     # README images
├── requirements.txt            # Backend dependencies
└── requirements-dev.txt        # Backend dependencies + pytest
```

## Quick Start

Requires Python 3.13 and Node.js 20.19+ (or 22.12+).

### Launch Backend

```Bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
cd backend
uvicorn main:app --reload
```

The server listens on `ws://localhost:8000/ws/game`. It uses the GPU when one is available and falls back to the CPU otherwise.

On Windows, `pip` installs the CPU build of PyTorch. For CUDA, install PyTorch from the [PyTorch index](https://pytorch.org/get-started/locally/) first, e.g. `pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu132`. On Linux, the default build already includes CUDA.

### Launch Frontend

```Bash
cd frontend
npm i
npm run dev
```

To point the frontend at a different server, set `VITE_WS_URL` (e.g. in `frontend/.env.local`).

## Backend Tools

All commands are run from `backend/` with the virtual environment active.

### CLI Tournaments

```Bash
python run_cli.py
```

Prompts for the number of players, rounds, and a player type per seat (Human, Random, Baseline, ISMCTS, DMC), then plays 1000 games and prints the scores. Without a human you can pick game parallelism (whole games across processes) or search parallelism (ISMCTS searches across processes).

### DMC Training

```Bash
python -m dmc_loop.orchestrate_dmc
```

Runs league cycles until training plateaus, resuming from `snapshots/latest_model.pt` if it exists:

1. **Train** (`dmc_loop.train_dmc`): 2000 batches of self-play against the live model, elite snapshots and the Baseline bot.
2. **Evaluate** (`dmc_loop.evaluate_dmc`): Elo tournament of the new snapshots, elites, milestones and the Baseline bot. The top 8 become the new elites.
3. **Stop check** (every 3 cycles): Head-to-head of the best elite against an anchor. If it stops improving, the learning rate is cut, and training stops once it reaches its floor.

Snapshots, elites and evaluation results are written to `backend/snapshots/`.

### PPO Training

PPO trains a policy that plays from its own information, with a critic that also sees every opponent's hand during training, for lower-variance advantages than DMC's Monte Carlo targets. It starts from the DMC ensemble and passes cards with the DMC passer.

1. **Distill** (`python -m ppo_loop.distill_ppo`): clones the DMC ensemble's policy into the actor and pretrains the critics on its games. `--dagger` continues with the student playing, `--critics` trains only the critics, and `--arch flat|seat|pairwise` picks the critic architecture.
2. **Gate 1** (`python -m ppo_loop.gates_ppo gate1`): the distilled actor has to play at parity with the ensemble over 2400 head-to-head games.
3. **Train** (`python -m ppo_loop.orchestrate_ppo`): league cycles as for DMC. Every 3 cycles, gate 2 plays the best elite against the ensemble and the distilled start, and gives up if training stops improving.

Snapshots, the frozen DMC elites and gate results are written to `backend/snapshots_ppo/`.

**Outcome.** The distilled actor passed gate 1 (−0.005, 95% CI [−0.011, +0.001] against the ensemble). Fifteen training cycles (about 1.8M games) then gained about +0.005 over the distilled start, which brought PPO level with the ensemble (final model +0.002, [−0.004, +0.009]) but not past it, so gate 2 gave up.

The premise didn't hold: seeing every hand barely helps to predict a round's outcome. The critic with every opponent's hand explains 0.77 of the outcome's variance, against 0.74 for one with only the actor's information. That makes the advantages only about 1.2× less noisy than without the hidden hands. Flat, per-seat and pairwise critics all stall there. Replaying rounds from the true state shows that about 0.88 is predictable, so the gap lies in judging how the rest of a round plays out, not in hidden information. Giving the critic each opponent's played cards made no difference either.

### Benchmark

```Bash
python benchmark.py
```

Plays two checkpoints (`MASTER_PATH` vs `OLD_PATH`) against each other in paired, seeded games and reports the score difference with a 95% confidence interval, stopping early once the result is clear. A tuple of checkpoints, such as `("bots/DMC.pt", "bots/DMCW.pt")`, plays as their averaged ensemble.

### Tests

```Bash
pip install -r requirements-dev.txt
pytest
```
