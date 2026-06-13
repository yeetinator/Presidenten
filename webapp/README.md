# Bot Arena Web UI

Run it from the repository root with:

```bash
pip install -r webapp/requirements.txt
uvicorn webapp.main:app --reload
```

Open http://127.0.0.1:8000 in your browser.

If Windows refuses to bind port 8000, use:

```bash
python -m webapp.run
```

That launcher listens on port 8010 by default and can be overridden with PRESIDENTEN_WEB_PORT.

To wire in your own bots, replace the default entries in `webapp/bots.py` with adapters that implement `respond(history, user_message)`.
