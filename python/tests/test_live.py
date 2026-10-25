"""Live test against OpenRouter. Runs only when OPEN_ROUTER_KEY_JUST_FOR_FREE_MODELS is set."""

import os

import pytest

from freeryder import Config, FreeRyder

KEY = os.environ.get("OPEN_ROUTER_KEY_JUST_FOR_FREE_MODELS")

pytestmark = pytest.mark.skipif(not KEY, reason="OPEN_ROUTER_KEY_JUST_FOR_FREE_MODELS not set")


def test_live_scan_and_chat(tmp_path):
    ryder = FreeRyder(KEY, Config(sample_size=3, state_file=tmp_path / "state.json"))
    reports = ryder.scan()
    assert reports, "no reachable free models"
    for r in reports:
        print(r.summary())
    res = ryder.chat({"messages": [{"role": "user", "content": "Reply with the word pong."}]})
    assert res.ok, res.error
    assert "pong" in res.text.lower()
