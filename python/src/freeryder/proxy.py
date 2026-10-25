"""Local OpenAI-compatible proxy: point any OpenAI client at http://localhost:PORT/v1."""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .picker import FreeRyder, NoModelAvailable

log = logging.getLogger("freeryder")

MODEL_ALIAS = "freeryder"


def make_handler(ryder: FreeRyder) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # route through logging instead of stderr
            log.debug("%s - " + fmt, self.address_string(), *args)

        def _send_json(self, status: int, obj: dict) -> None:
            data = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.rstrip("/") in ("/v1/models", "/models"):
                self._send_json(200, {"object": "list", "data": [
                    {"id": MODEL_ALIAS, "object": "model", "owned_by": "freeryder",
                     "current": ryder.current}]})
            elif self.path.rstrip("/") in ("/status", "/v1/status"):
                self._send_json(200, {
                    "current": ryder.current,
                    "scanned_at": ryder.scanned_at,
                    "ranking": [{"model": c.model, "score": round(c.score, 3),
                                 "bench_latency": round(c.bench_latency, 2),
                                 "cooling_until": c.cooldown_until or None}
                                for c in ryder.ranking],
                })
            else:
                self._send_json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
                self._send_json(404, {"error": {"message": "not found"}})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                self._send_json(400, {"error": {"message": "invalid JSON"}})
                return
            stream = bool(payload.pop("stream", False))
            payload.pop("stream_options", None)
            try:
                res = ryder.chat(payload)
            except NoModelAvailable as e:
                self._send_json(503, {"error": {"message": str(e)}})
                return
            if not res.ok:
                self._send_json(res.status or 502, {"error": {
                    "message": "all free models failed", "upstream": res.error}})
                return
            if stream:
                self._send_as_stream(res.body)
            else:
                self._send_json(200, res.body)

        def _send_as_stream(self, body: dict) -> None:
            """Replay a finished completion as SSE so streaming clients keep working.

            Failover needs the whole answer first, so this is not token-by-token streaming.
            """
            choice = (body.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            delta = {"role": "assistant", "content": msg.get("content") or ""}
            if msg.get("tool_calls"):
                delta["tool_calls"] = [{**tc, "index": i} for i, tc in enumerate(msg["tool_calls"])]
            base = {k: body.get(k) for k in ("id", "created", "model")}
            chunks = [
                {**base, "object": "chat.completion.chunk",
                 "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                {**base, "object": "chat.completion.chunk", "usage": body.get("usage"),
                 "choices": [{"index": 0, "delta": {},
                              "finish_reason": choice.get("finish_reason", "stop")}]},
            ]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            for c in chunks:
                self.wfile.write(b"data: " + json.dumps(c).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
            self.close_connection = True

    return Handler


def serve(ryder: FreeRyder, host: str = "127.0.0.1", port: int = 8765) -> None:
    server = ThreadingHTTPServer((host, port), make_handler(ryder))
    log.info("FreeRyder proxy on http://%s:%d/v1 (model name: %s)", host, port, MODEL_ALIAS)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
