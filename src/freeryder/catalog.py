"""Find OpenRouter's currently free text models."""

from __future__ import annotations

from typing import Any

# Routers and safety classifiers are free but not general chat models.
EXCLUDE_PREFIXES = ("openrouter/",)
EXCLUDE_SUBSTRINGS = ("content-safety", "guard", "safeguard")


def is_free(model: dict[str, Any]) -> bool:
    pricing = model.get("pricing") or {}
    zero = all(str(pricing.get(k, "0")) in ("0", "0.0") for k in ("prompt", "completion", "request"))
    return model["id"].endswith(":free") or (zero and "prompt" in pricing)


def is_text_chat(model: dict[str, Any]) -> bool:
    arch = model.get("architecture") or {}
    return arch.get("output_modalities", ["text"]) == ["text"]


def free_text_models(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for m in models:
        mid = m["id"]
        if mid.startswith(EXCLUDE_PREFIXES) or any(s in mid for s in EXCLUDE_SUBSTRINGS):
            continue
        if is_free(m) and is_text_chat(m):
            out.append(m)
    return out
