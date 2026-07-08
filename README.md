# President

President is a card game project with two parts:

- A Python backend for the game engine, CLI tools, Deep Monte Carlo training, and the FastAPI websocket server.
- A Svelte frontend for the browser UI.

## Prerequisites

- Python 3.10 or newer
- Node.js 18 or newer
- `pip` and `npm`

## Setup

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install the Python dependencies.
4. Install the frontend dependencies.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1 | source .venv/bin/activate
pip install -r requirements.txt
cd frontend
npm install
```

## Run the backend

The backend code lives in `backend/`, so run the Python commands from that folder.

```powershell
cd backend
uvicorn main:app --reload
```

## Run the frontend

Start the frontend development server from the `frontend/` folder:

```powershell
cd frontend
npm run dev
```