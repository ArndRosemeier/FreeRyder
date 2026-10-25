"""Benchmark free models for responsiveness and capability."""

from __future__ import annotations

import random
import statistics
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .client import CallResult, OpenRouterClient
from .tasks import TASKS, Task, clean, final_answer  # noqa: F401 (re-exported)

@dataclass
class Weights:
    quality: float = 0.6
    reliability: float = 0.25
    speed: float = 0.15
    # Median latency (seconds) at which the speed score drops to 0.5.
    latency_half: float = 10.0


@dataclass
class ModelReport:
    model: str
    # (task name, call result, task score 0..1)
    calls: list[tuple[str, CallResult, float]] = field(default_factory=list)
    score: float = 0.0

    @property
    def points(self) -> float:
        return sum(s for _, _, s in self.calls)

    @property
    def successes(self) -> list[CallResult]:
        return [r for _, r, _ in self.calls if r.ok]

    @property
    def quality(self) -> float:
        return self.points / len(self.calls) if self.calls else 0.0

    @property
    def reliability(self) -> float:
        return len(self.successes) / len(self.calls) if self.calls else 0.0

    @property
    def rate_limited(self) -> int:
        return sum(1 for _, r, _ in self.calls if r.rate_limited)

    @property
    def gone(self) -> bool:
        return any(r.gone for _, r, _ in self.calls)

    @property
    def median_latency(self) -> float | None:
        lat = [r.latency for r in self.successes]
        return statistics.median(lat) if lat else None

    def compute_score(self, w: Weights) -> float:
        if not self.successes:
            self.score = 0.0
            return 0.0
        speed = 1.0 / (1.0 + self.median_latency / w.latency_half)
        self.score = w.quality * self.quality + w.reliability * self.reliability + w.speed * speed
        return self.score

    def summary(self) -> str:
        lat = f"{self.median_latency:.1f}s" if self.median_latency is not None else "-"
        tasks = " ".join(f"{n}={s:.1f}".replace(".0", "") for n, _, s in self.calls)
        return (f"{self.score:.3f}  {self.model:<50} quality {self.points:.1f}/{len(self.calls)}  "
                f"ok {len(self.successes)}/{len(self.calls)}  429s {self.rate_limited}  "
                f"median {lat}  [{tasks}]")


class Benchmark:
    def __init__(self, client: OpenRouterClient, tasks: list[Task] | None = None,
                 weights: Weights | None = None, max_tokens: int = 8192,
                 timeout: float = 60, abort_after_failures: int = 3):
        self.client = client
        self.tasks = tasks or TASKS
        self.weights = weights or Weights()
        self.max_tokens = max_tokens
        self.timeout = timeout
        # Stop testing a model after this many consecutive failed calls (saves time on dead models).
        self.abort_after_failures = abort_after_failures

    def run_model(self, model: str) -> ModelReport:
        report = ModelReport(model)
        consecutive_failures = 0
        for task in self.tasks:
            if consecutive_failures >= self.abort_after_failures:
                report.calls.append((task.name, CallResult(False, 0, 0.0, error="skipped"), 0.0))
                continue
            res = self.client.chat({
                "model": model,
                "messages": [{"role": "user", "content": task.prompt}],
                "temperature": 0,
                "max_tokens": self.max_tokens,
            }, timeout=self.timeout)
            if res.gone:
                consecutive_failures = self.abort_after_failures
            else:
                consecutive_failures = 0 if res.ok else consecutive_failures + 1
            report.calls.append((task.name, res, float(task.check(res.text)) if res.ok else 0.0))
        report.compute_score(self.weights)
        return report

    def probe(self, models: list[str], parallel: int = 5) -> list[str]:
        """Drop models this account can't use at all (404/403: removed, blocked by privacy
        settings, agent-only). Rate-limited models are kept: they may recover."""
        def usable(model: str) -> bool:
            res = self.client.chat({"model": model, "max_tokens": 16, "temperature": 0,
                                    "messages": [{"role": "user", "content": "Say OK."}]},
                                   timeout=self.timeout)
            return not res.gone
        with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
            flags = list(pool.map(usable, models))
        return [m for m, keep in zip(models, flags) if keep]

    def run(self, models: list[str], parallel: int = 5) -> list[ModelReport]:
        """Benchmark models concurrently (tasks within a model run in sequence). Best first."""
        with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
            reports = list(pool.map(self.run_model, models))
        return sorted(reports, key=lambda r: r.score, reverse=True)


def sample(models: list[str], k: int = 10, rng: random.Random | None = None,
           prefer: list[str] | None = None) -> list[str]:
    """Pick up to k models at random; ids in `prefer` (e.g. last winners) are kept first."""
    rng = rng or random.Random()
    keep = [m for m in (prefer or []) if m in models][:k]
    rest = [m for m in models if m not in keep]
    rng.shuffle(rest)
    return keep + rest[: k - len(keep)]
