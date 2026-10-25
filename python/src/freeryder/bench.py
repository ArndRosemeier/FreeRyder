"""Benchmark free models for responsiveness and capability."""

from __future__ import annotations

import json
import random
import re
import statistics
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from .client import CallResult, OpenRouterClient

ANSWER_HINT = "\nEnd your reply with a final line of the form `ANSWER: <answer>`."
_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)


def clean(text: str) -> str:
    return _THINK.sub("", text or "").strip()


def final_answer(text: str) -> str:
    matches = re.findall(r"ANSWER:\s*(.+)", clean(text), re.I)
    return matches[-1].strip().strip("`*. ").lower() if matches else ""


def _answer_is(expected: str) -> Callable[[str], bool]:
    return lambda text: final_answer(text) == expected


def _exact_banana(text: str) -> bool:
    return clean(text).strip().strip(".").strip() == "banana"


def _json_ada(text: str) -> bool:
    t = clean(text)
    t = re.sub(r"^```(?:json)?|```$", "", t.strip(), flags=re.M).strip()
    try:
        return json.loads(t) == {"name": "Ada", "year": 1815}
    except ValueError:
        return False


@dataclass(frozen=True)
class Task:
    name: str
    prompt: str
    check: Callable[[str], bool]


TASKS: list[Task] = [
    Task("arithmetic", "What is 17 * 23 + 19?" + ANSWER_HINT, _answer_is("410")),
    Task("time", "A train leaves at 14:35 and arrives at 17:10 the same day. "
         "How many minutes is the trip? Give just the number." + ANSWER_HINT, _answer_is("155")),
    Task("letters", "How many times does the letter 'r' appear in the text "
         "\"strawberry raspberry\"? Give just the number." + ANSWER_HINT, _answer_is("6")),
    Task("logic", "All bloops are razzies. All razzies are lazzies. Are all bloops "
         "definitely lazzies? Answer yes or no." + ANSWER_HINT, _answer_is("yes")),
    Task("reverse", "Reverse the string \"freeryder\" character by character." + ANSWER_HINT,
         _answer_is("redyreerf")),
    Task("code", "What does this Python expression evaluate to?\n"
         "sum(i * i for i in range(5) if i % 2 == 0)" + ANSWER_HINT, _answer_is("20")),
    Task("instruction", "Reply with exactly the word banana in lowercase, with no "
         "punctuation, explanation or anything else.", _exact_banana),
    Task("json", "Output only a JSON object with exactly two keys: \"name\" set to the "
         "string \"Ada\" and \"year\" set to the number 1815. No prose, no code fences.",
         _json_ada),
]


@dataclass
class Weights:
    quality: float = 0.6
    reliability: float = 0.25
    speed: float = 0.15
    # Median latency (seconds) at which the speed score drops to 0.5.
    latency_half: float = 6.0


@dataclass
class ModelReport:
    model: str
    calls: list[tuple[str, CallResult, bool]] = field(default_factory=list)
    score: float = 0.0

    @property
    def passed(self) -> int:
        return sum(1 for _, _, ok in self.calls if ok)

    @property
    def successes(self) -> list[CallResult]:
        return [r for _, r, _ in self.calls if r.ok]

    @property
    def quality(self) -> float:
        return self.passed / len(self.calls) if self.calls else 0.0

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
        return (f"{self.score:.3f}  {self.model:<55} quality {self.passed}/{len(self.calls)}  "
                f"ok {len(self.successes)}/{len(self.calls)}  429s {self.rate_limited}  "
                f"median {lat}")


class Benchmark:
    def __init__(self, client: OpenRouterClient, tasks: list[Task] | None = None,
                 weights: Weights | None = None, max_tokens: int = 2048,
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
                report.calls.append((task.name, CallResult(False, 0, 0.0, error="skipped"), False))
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
            report.calls.append((task.name, res, res.ok and task.check(res.text)))
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
