from __future__ import annotations

import os

import uvicorn

if __name__ == "__main__":
    port = int(os.getenv("PRESIDENTEN_WEB_PORT", "8010"))
    uvicorn.run("app:app", host="127.0.0.1", port=port, reload=True)
