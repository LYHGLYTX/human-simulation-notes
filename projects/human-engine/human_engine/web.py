"""Game-mode web UI (spec §5.2). Zero-dependency: http.server + static page.

Run:  python -m human_engine.web   (then open http://localhost:8000)
API:
  GET  /api/state            -> full state snapshot + persona summary
  POST /api/event {text}     -> run event pipeline, return result + options
  POST /api/act   {kind}     -> execute chosen behavior
  POST /api/reset            -> new engine
"""
from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from .engine import Engine
from .persona import default_persona

_lock = threading.Lock()
engine = Engine(seed=11)


def _snapshot() -> dict:
    s = engine.state
    return {
        "t": s.t,
        "emotion_label": s.emotion_label,
        "emotion_strength": s.emotion_strength,
        "pad": s.pad,
        "mood_pad": s.mood_pad,
        "stress": s.stress,
        "resources": s.resources,
        "threshold": s.threshold,
        "wear": s.wear,
        "gas_phase": s.gas_phase,
        "crashed": s.crashed,
        "crash_type": s.crash_type,
        "self_control": s.self_control,
        "impulse": s.impulse,
        "guilt": s.guilt,
        "shame": s.shame,
        "vigilance": s.vigilance,
        "depression_tendency": s.depression_tendency,
        "energy": s.energy,
        "sleep_debt": s.sleep_debt,
        "disengagement": max(s.moral_disengagement.values()),
        "relations": engine.relations.snapshot(),
        "support": engine.relations.support_score(),
        "persona": engine.persona.summary_text(),
        "memory": engine.memory.summarize(),
        "history": [{"text": m.text, "valence": m.valence,
                     "importance": m.importance}
                    for m in engine.memory.episodic[-6:]],
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode())

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/state":
            with _lock:
                self._send(_snapshot())
        elif path in ("/", "/index.html"):
            html = _load_page()
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        global engine
        path = urlparse(self.path).path
        data = self._read_json()
        with _lock:
            if path == "/api/event":
                raw = data.get("text", "")
                engine.tick(60)
                out = engine.handle_event(raw)
                out["options"] = engine.options(out)
                out["state"] = _snapshot()
                self._send(out)
            elif path == "/api/act":
                kind = data.get("kind", "neutral_act")
                engine.tick(30)
                out = engine.act(kind)
                out["options"] = engine.options()
                out["state"] = _snapshot()
                self._send(out)
            elif path == "/api/reset":
                engine = Engine(seed=11)
                self._send({"ok": True})
            elif path == "/api/sleep":
                r = engine.sleep(8.0)
                self._send({"ok": True, "sleep": r, "state": _snapshot()})
            else:
                self._send({"error": "not found"}, 404)


def _load_page() -> str:
    path = os.path.join(os.path.dirname(__file__), "web_ui.html")
    with open(path, encoding="utf-8") as f:
        return f.read()


def main(port: int = 8000):
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"human-engine 游戏模式: http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
