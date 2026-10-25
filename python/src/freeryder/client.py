"""Minimal OpenRouter HTTP client (stdlib only)."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

BASE_URL = "https://openrouter.ai/api/v1"


@dataclass
class CallResult:
    """Outcome of one chat call. `status` is the HTTP status, 0 for network errors/timeouts."""

    ok: bool
    status: int
    latency: float
    body: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    @property
    def text(self) -> str:
        try:
            return self.body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            return ""

    @property
    def rate_limited(self) -> bool:
        return self.status == 429

    @property
    def gone(self) -> bool:
        """Model no longer exists, no provider serves it for this account, or it is
        restricted (e.g. "only available on agentic harnesses")."""
        if self.status in (404, 410):
            return True
        return self.status == 403 and "only available" in self.error


class OpenRouterClient:
    def __init__(self, api_key: str | None, base_url: str = BASE_URL,
                 app_name: str = "freeryder"):
        """`api_key` is your OpenRouter key. Only `list_models` works without one."""
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.app_name = app_name

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json", "X-Title": self.app_name}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def get_json(self, path: str, timeout: float = 30) -> dict[str, Any]:
        req = urllib.request.Request(self.base_url + path, headers=self._headers())
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())

    def list_models(self) -> list[dict[str, Any]]:
        return self.get_json("/models")["data"]

    def chat(self, payload: dict[str, Any], timeout: float = 60) -> CallResult:
        """POST /chat/completions (non-streaming). Never raises for HTTP/network errors."""
        data = json.dumps(payload).encode()
        req = urllib.request.Request(self.base_url + "/chat/completions", data=data,
                                     headers=self._headers(), method="POST")
        start = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read())
                latency = time.monotonic() - start
        except urllib.error.HTTPError as e:
            raw = e.read().decode(errors="replace")[:500]
            return CallResult(False, e.code, time.monotonic() - start, error=raw)
        except Exception as e:  # timeouts, connection resets, bad JSON
            return CallResult(False, 0, time.monotonic() - start, error=repr(e))
        # OpenRouter sometimes returns 200 with an error object (e.g. upstream 429).
        if "error" in body:
            err = body["error"] or {}
            code = err.get("code") if isinstance(err.get("code"), int) else 502
            return CallResult(False, code, latency, body, json.dumps(err)[:500])
        return CallResult(True, 200, latency, body)
