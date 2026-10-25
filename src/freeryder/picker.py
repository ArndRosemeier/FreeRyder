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
    def __init__(self, client: OpenRouterClient | None = None, config: Config | None = None,
                 benchmark: Benchmark | None = None):
        self.client = client or OpenRouterClient()
        self.cfg = config or Config()
        self.bench = benchmark or Benchmark(self.client, timeout=self.cfg.timeout)
        self.ranking: list[Candidate] = []
        self.scanned_at = 0.0
        self.last_reports: list[ModelReport] = []
        self._lock = threading.RLock()
        self._scan_lock = threading.Lock()
        self._load_state()

    # ---- scanning -------------------------------------------------------------------------

    def scan(self) -> list[ModelReport]:
        """List free models, benchmark a random sample, and rebuild the ranking."""
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

    def _maybe_rescan(self, force: bool = False) -> None:
        now = time.time()
        stale = now - self.scanned_at > self.cfg.rescan_after
        usable = any(c.available(now) for c in self.ranking)
        if (force or stale or not usable) and now - self.scanned_at >= self.cfg.min_rescan_gap:
            self.scan()

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

    def _candidates(self) -> list[Candidate]:
        with self._lock:
            self._maybe_rescan()
            now = time.time()
            return [c for c in self.ranking if c.available(now)]

    # ---- public API -----------------------------------------------------------------------

    def chat(self, payload: dict[str, Any]) -> CallResult:
        """Send an OpenAI-style chat payload to the best healthy free model.

        `payload["model"]` is replaced. Fails over to the next-ranked model on errors and
        rescans when no ranked model is usable. Returns the last failure if all attempts fail.
        """
        last: CallResult | None = None
        tried: set[str] = set()
        rescanned = False
        for _ in range(self.cfg.max_attempts):
            cands = [c for c in self._candidates() if c.model not in tried]
            if not cands and not rescanned:
                with self._lock:
                    self._maybe_rescan(force=True)
                rescanned = True
                cands = [c for c in self._candidates() if c.model not in tried]
            if not cands:
                break
            cand = cands[0]
            tried.add(cand.model)
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
            log.info("no benchmarked model worked; falling back to %s", fb)
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
