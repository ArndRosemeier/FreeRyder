"""Use the best free model until it congests or disappears, then fail over or rescan."""

from __future__ import annotations

import json
import logging
import os
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .bench import Benchmark, ModelReport, sample
from .catalog import free_text_models
from .client import CallResult, OpenRouterClient

log = logging.getLogger("freeryder")

DEFAULT_STATE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "freeryder" / "state.json"


@dataclass
class Config:
    sample_size: int = 10          # models benchmarked per scan
    parallel: int = 5              # models benchmarked at once
    min_score: float = 0.5         # models below this never get used
    max_failures: int = 2          # consecutive failures before a model counts as congested
    cooldown: float = 600.0        # seconds a congested model is skipped
    slow_factor: float = 3.0       # rolling median latency > factor * benchmark median => congested
    slow_floor: float = 20.0       # ... but only once it is also above this many seconds
    rescan_after: float = 6 * 3600 # rescan even when healthy, to pick up new models
    min_rescan_gap: float = 120.0  # never rescan more often than this
    max_attempts: int = 4          # models tried per request before giving up
    timeout: float = 90.0
    # Last resort when no benchmarked model is healthy: OpenRouter's own free router.
    fallback_model: str | None = "openrouter/free"
    state_file: Path | None = DEFAULT_STATE


@dataclass
class Candidate:
    model: str
    score: float
    bench_latency: float
    failures: int = 0
    cooldown_until: float = 0.0
    recent: deque = field(default_factory=lambda: deque(maxlen=5))

    def available(self, now: float) -> bool:
        return now >= self.cooldown_until


class NoModelAvailable(RuntimeError):
    pass


class FreeRyder:
    def __init__(self, api_key: str | None = None, config: Config | None = None, *,
                 client: OpenRouterClient | None = None, benchmark: Benchmark | None = None):
        """Pass your OpenRouter `api_key` (or a ready `client`)."""
        if client is None:
            if not api_key:
                raise ValueError("FreeRyder needs an OpenRouter api_key")
            client = OpenRouterClient(api_key)
        self.client = client
        self.cfg = config or Config()
        self.bench = benchmark or Benchmark(self.client, timeout=self.cfg.timeout)
        self.ranking: list[Candidate] = []
        self.scanned_at = 0.0
        self.last_reports: list[ModelReport] = []
        self.last_error: Exception | None = None
        self._lock = threading.RLock()
        self._scan_lock = threading.Lock()
        self._scan_thread: threading.Thread | None = None
        self._last_scan_start = 0.0
        self._load_state()

    # ---- warm-up --------------------------------------------------------------------------

    def start(self) -> "FreeRyder":
        """Warm up: start discovery in a background thread and return immediately.

        Skipped if a fresh ranking is already cached. You can call `chat` right away: until
        the first scan finishes, requests go to cached models or the fallback router.
        """
        self._kick_scan()
        return self

    def wait_ready(self, timeout: float | None = None) -> bool:
        """Block until a running background scan has finished. True if a model is ready."""
        t = self._scan_thread
        if t is not None:
            t.join(timeout)
        return self.current is not None

    @property
    def scanning(self) -> bool:
        t = self._scan_thread
        return t is not None and t.is_alive()

    def _kick_scan(self, force: bool = False) -> None:
        """Start a background scan if needed (nothing usable, stale, or forced)."""
        with self._lock:
            if self.scanning:
                return
            now = time.time()
            stale = now - self.scanned_at > self.cfg.rescan_after
            usable = any(c.available(now) for c in self.ranking)
            if not (force or stale or not usable):
                return
            if now - self._last_scan_start < self.cfg.min_rescan_gap:
                return
            self._last_scan_start = now
            self._scan_thread = threading.Thread(target=self._scan_safely, name="freeryder-scan",
                                                 daemon=True)
            self._scan_thread.start()

    def _scan_safely(self) -> None:
        try:
            self.scan()
            self.last_error = None
        except Exception as e:  # network down, bad key...: keep serving what we have
            self.last_error = e
            log.warning("background scan failed: %r", e)

    # ---- scanning -------------------------------------------------------------------------

    def scan(self) -> list[ModelReport]:
        """List free models, benchmark a random sample, and rebuild the ranking (blocking).

        Most callers want `start()` instead, which does this in the background.
        """
        with self._scan_lock:
            free = [m["id"] for m in free_text_models(self.client.list_models())]
            reachable = self.bench.probe(free, parallel=self.cfg.parallel)
            with self._lock:
                previous = [c.model for c in self.ranking]
            chosen = sample(reachable, self.cfg.sample_size, prefer=previous[:3])
            log.info("benchmarking %d of %d reachable free models (%d listed)",
                     len(chosen), len(reachable), len(free))
            reports = self.bench.run(chosen, parallel=self.cfg.parallel)
            with self._lock:
                self.last_reports = reports
                self.ranking = [Candidate(r.model, r.score, r.median_latency or 0.0)
                                for r in reports if r.score >= self.cfg.min_score]
                self.scanned_at = time.time()
                self._save_state()
            for r in reports:
                log.info("%s", r.summary())
            return reports

    @property
    def current(self) -> str | None:
        now = time.time()
        with self._lock:
            return next((c.model for c in self.ranking if c.available(now)), None)

    # ---- health bookkeeping ---------------------------------------------------------------

    def _record(self, cand: Candidate, res: CallResult) -> None:
        with self._lock:
            if res.gone:
                log.warning("%s disappeared (%s); dropping it", cand.model, res.status)
                self.ranking = [c for c in self.ranking if c is not cand]
                return
            if res.ok:
                cand.failures = 0
                cand.recent.append(res.latency)
                med = statistics.median(cand.recent)
                if (len(cand.recent) >= 3 and med > self.cfg.slow_floor
                        and med > self.cfg.slow_factor * max(cand.bench_latency, 1.0)):
                    log.warning("%s is slowing down (median %.1fs); cooling it down", cand.model, med)
                    self._cool(cand)
                return
            cand.failures += 1
            log.info("%s failed (%s %s)", cand.model, res.status, res.error[:120])
            if cand.failures >= self.cfg.max_failures or res.rate_limited:
                self._cool(cand)

    def _cool(self, cand: Candidate) -> None:
        cand.cooldown_until = time.time() + self.cfg.cooldown
        cand.failures = 0
        cand.recent.clear()

    def _available(self, tried: set[str]) -> list[Candidate]:
        now = time.time()
        with self._lock:
            return [c for c in self.ranking if c.available(now) and c.model not in tried]

    # ---- public API -----------------------------------------------------------------------

    def chat(self, payload: dict[str, Any]) -> CallResult:
        """Send an OpenAI-style chat payload to the best healthy free model.

        `payload["model"]` is replaced. Never waits for a scan when it can avoid it: it fails
        over to the next-ranked model, starts a background rescan when the ranking runs dry
        or gets stale, and meanwhile uses `fallback_model`. Only without a fallback does it
        wait for a running scan. Returns the last failure if every attempt fails.
        """
        self._kick_scan()
        last: CallResult | None = None
        tried: set[str] = set()
        waited = False
        attempts = 0
        while attempts < self.cfg.max_attempts:
            cands = self._available(tried)
            if not cands:
                self._kick_scan()
                if self.cfg.fallback_model or waited or not self.scanning:
                    break
                self.wait_ready()  # no fallback: the scan is our only hope
                waited = True
                continue
            cand = cands[0]
            tried.add(cand.model)
            attempts += 1
            res = self.client.chat({**payload, "model": cand.model}, timeout=self.cfg.timeout)
            self._record(cand, res)
            if res.ok:
                return res
            last = res
            # A client error that is not about the model (bad request) won't improve elsewhere.
            if res.status in (400, 401, 402, 413):
                return res
        fb = self.cfg.fallback_model
        if fb and fb not in tried:
            log.info("no benchmarked model available; using %s", fb)
            res = self.client.chat({**payload, "model": fb}, timeout=self.cfg.timeout)
            if res.ok or last is None:
                return res
        if last is None:
            raise NoModelAvailable("no free model passed the benchmark; try again later")
        return last

    def complete(self, prompt: str, **params: Any) -> str:
        """Convenience: one user prompt in, answer text out."""
        res = self.chat({"messages": [{"role": "user", "content": prompt}], **params})
        if not res.ok:
            raise RuntimeError(f"all free models failed: {res.status} {res.error}")
        return res.text

    # ---- persistence ----------------------------------------------------------------------

    def _save_state(self) -> None:
        path = self.cfg.state_file
        if not path:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "scanned_at": self.scanned_at,
                "ranking": [{"model": c.model, "score": c.score, "bench_latency": c.bench_latency}
                            for c in self.ranking],
            }, indent=1))
        except OSError as e:
            log.debug("could not save state: %s", e)

    def _load_state(self) -> None:
        path = self.cfg.state_file
        if not path or not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            self.ranking = [Candidate(r["model"], r["score"], r["bench_latency"])
                            for r in data["ranking"]]
            self.scanned_at = float(data["scanned_at"])
        except (OSError, ValueError, KeyError) as e:
            log.debug("ignoring bad state file: %s", e)
