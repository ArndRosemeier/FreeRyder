"""Capability tasks. Each check returns a score from 0.0 to 1.0.

The riddles are original (not well-known puzzles a model may have memorised); their
answers were verified by brute force over every possible assignment.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable

ANSWER_HINT = "\nThink it through, then end your reply with a final line of the form `ANSWER: <answer>`."
_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*\n?(.*?)\n?```\s*$", re.S)


def clean(text: str) -> str:
    """Drop <think> blocks some models put in the content."""
    return _THINK.sub("", text or "").strip()


def final_answer(text: str) -> str:
    matches = re.findall(r"ANSWER:\s*(.+)", clean(text), re.I)
    return matches[-1].strip().strip("`*. ").lower() if matches else ""


def _names(answer: str) -> list[str]:
    return [w for w in re.split(r"[^a-z]+", answer.lower()) if w and w != "and"]


@dataclass(frozen=True)
class Task:
    name: str
    prompt: str
    check: Callable[[str], float]


# ---- strict JSON ---------------------------------------------------------------------------

JSON_PROMPT = """Extract the order below into JSON with exactly this shape:
{"order_id": <integer>, "customer": <string>, "date": "<YYYY-MM-DD>",
 "items": [{"name": <singular noun, lowercase>, "qty": <integer>, "unit_price": <number>}],
 "total": <number>}

Order text: "Order 4471 came in from Mirela Petrov on 3 March 2025. She bought two lamps
at 24.50 each and one rug for 89.00. Shipping was free."

Output only the JSON object: no prose, no code fences."""

_JSON_EXPECTED = {
    "order_id": 4471, "customer": "Mirela Petrov", "date": "2025-03-03",
    "items": [{"name": "lamp", "qty": 2, "unit_price": 24.5},
              {"name": "rug", "qty": 1, "unit_price": 89.0}],
    "total": 138.0,
}


def check_json(text: str) -> float:
    """1.0 for the exact object as raw JSON, 0.5 if it was wrapped in a code fence."""
    raw = clean(text)
    factor = 1.0
    m = _FENCE.match(raw)
    if m:
        raw, factor = m.group(1).strip(), 0.5
    try:
        data = json.loads(raw)
    except ValueError:
        return 0.0
    if not isinstance(data, dict) or set(data) != set(_JSON_EXPECTED):
        return 0.0
    try:
        items = data["items"]
        ok = (
            data["order_id"] == 4471 and not isinstance(data["order_id"], bool)
            and str(data["customer"]).strip() == "Mirela Petrov"
            and data["date"] == "2025-03-03"
            and abs(float(data["total"]) - 138.0) < 1e-6
            and isinstance(items, list) and len(items) == 2
            and all(set(i) == {"name", "qty", "unit_price"} for i in items)
            and [(i["name"], i["qty"]) for i in items] == [("lamp", 2), ("rug", 1)]
            and abs(float(items[0]["unit_price"]) - 24.5) < 1e-6
            and abs(float(items[1]["unit_price"]) - 89.0) < 1e-6
        )
    except (TypeError, ValueError, KeyError):
        return 0.0
    return factor if ok else 0.0


# ---- logic riddles -------------------------------------------------------------------------

EASY_PROMPT = """Three boxes stand in a row, from left to right: red, green, blue.
A coin is in exactly one of them. It is not in the red box, and it is not in the box
directly to the right of the red box. Which box holds the coin?""" + ANSWER_HINT


def check_easy(text: str) -> float:
    return float(_names(final_answer(text)) in (["blue"], ["blue", "box"], ["the", "blue", "box"]))


MEDIUM_PROMPT = """Ada, Ben, Cleo, Dev and Eli stand in a row in positions 1 to 5, left to right.
- Cleo is at neither end of the row.
- Ben stands directly to the left of Eli.
- Ada stands somewhere to the right of Dev.
- Cleo does not stand next to Dev.
- Ada is not in position 5.
Give the full order from left to right as names separated by commas.""" + ANSWER_HINT


def check_medium(text: str) -> float:
    return float(_names(final_answer(text)) == ["dev", "ada", "cleo", "ben", "eli"])


HARD_PROMPT = """Pell, Quin, Rook, Sage and Tess each either always tell the truth or always lie.
- Pell says: "Quin is a liar."
- Quin says: "Rook and Sage are the same type (both truthful or both liars)."
- Rook says: "Exactly two of us five are liars."
- Sage says: "At least one of Pell and Rook is a liar."
- Tess says: "Pell and I are different types, or the number of liars among us five is odd."
Which of them are liars? Give their names in alphabetical order, separated by commas.""" + ANSWER_HINT


def check_hard(text: str) -> float:
    return float(_names(final_answer(text)) == ["pell", "tess"])


# ---- human-sounding story ------------------------------------------------------------------

STORY_PROMPT = """Write a short story of about 150 words about someone who finds an old key in
the pocket of a second-hand coat. Write it the way a good human author would. Output only
the story."""

# Words and phrases that show up far more often in LLM prose than in human writing.
AI_TELLS = [
    "delve", "tapestry", "testament", "symphony", "palpable", "intricate", "unwavering",
    "resonat", "beacon", "embark", "realm", "ethereal", "kaleidoscope", "labyrinth",
    "enigmatic", "solace", "bustling", "nestled", "vibrant", "whisper of", "whispers of",
    "a sense of", "the weight of", "in that moment", "little did", "couldn't help but",
    "a breath she didn't know", "a breath he didn't know", "a breath they didn't know",
    "it wasn't just", "not just a", "something shifted", "the air was thick", "elara",
    "seraphina", "lyra", "eldoria", "willowbrook", "once upon a time",
]
_DASHES = re.compile(r"—| – | -- ")


def story_tells(text: str) -> dict[str, int]:
    """Count AI-writing tells: em dashes (and dash look-alikes) plus stock phrases."""
    t = clean(text)
    low = t.lower().replace("’", "'")
    found = {"dash": len(_DASHES.findall(t))}
    for phrase in AI_TELLS:
        n = low.count(phrase)
        if n:
            found[phrase] = n
    found["title/markdown"] = int(bool(re.search(r"^\s*(#|\*\*)", t)))
    return {k: v for k, v in found.items() if v}


def check_story(text: str) -> float:
    """1.0 with no tells, minus 0.2 per tell. 0 if it isn't roughly a 150-word story."""
    words = len(clean(text).split())
    if not 80 <= words <= 300:
        return 0.0
    return max(0.0, 1.0 - 0.2 * sum(story_tells(text).values()))


TASKS: list[Task] = [
    Task("json", JSON_PROMPT, check_json),
    Task("riddle-easy", EASY_PROMPT, check_easy),
    Task("riddle-medium", MEDIUM_PROMPT, check_medium),
    Task("riddle-hard", HARD_PROMPT, check_hard),
    Task("story", STORY_PROMPT, check_story),
]
