import json
import threading
import urllib.request

from freeryder.bench import TASKS, Benchmark, final_answer, sample
from freeryder.catalog import free_text_models
from freeryder.client import CallResult
from freeryder.picker import Config, FreeRyder
from freeryder.proxy import make_handler

GOOD_ANSWERS = {
    "arithmetic": "ANSWER: 410", "time": "ANSWER: 155", "letters": "ANSWER: 6",
    "logic": "ANSWER: yes", "reverse": "ANSWER: redyreerf", "code": "ANSWER: 20",
    "instruction": "banana", "json": '{"name": "Ada", "year": 1815}',
}


def ok(text, latency=1.0, model="m"):
    return CallResult(True, 200, latency, {"model": model, "choices": [
        {"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]})


class FakeClient:
    """behaviour[model] = 'good' | 'dumb' | 'busy' | 'gone' | callable(n)->CallResult"""

    def __init__(self, behaviour, catalog=None):
        self.behaviour = behaviour
        self.catalog = catalog or [
            {"id": m, "pricing": {"prompt": "0", "completion": "0"},
             "architecture": {"output_modalities": ["text"]}} for m in behaviour]
        self.calls = []
        self.lock = threading.Lock()

    def list_models(self):
        return self.catalog

    def chat(self, payload, timeout=60):
        model = payload["model"]
        if payload["messages"][-1]["content"] == "Say OK.":  # reachability probe
            return CallResult(False, 404, 0.1) if self.behaviour[model] == "gone" else ok("OK")
        with self.lock:
            self.calls.append(model)
            n = self.calls.count(model)
        b = self.behaviour[model]
        if callable(b):
            return b(n)
        if b == "busy":
            return CallResult(False, 429, 0.1, error="rate limited")
        if b == "gone":
            return CallResult(False, 404, 0.1, error="no endpoints")
        prompt = payload["messages"][-1]["content"]
        task = next((t for t in TASKS if t.prompt == prompt), None)
        if b == "good":
            return ok(GOOD_ANSWERS[task.name] if task else "hello", model=model)
        return ok("ANSWER: no idea", latency=0.5, model=model)


def ryder(behaviour, **cfg):
    cfg.setdefault("fallback_model", None)
    client = FakeClient(behaviour)
    return client, FreeRyder(client, Config(state_file=None, min_rescan_gap=0, **cfg))


def test_good_answers_pass_every_task():
    for t in TASKS:
        assert t.check(GOOD_ANSWERS[t.name]), t.name


def test_final_answer_ignores_think_blocks():
    assert final_answer("<think>ANSWER: 1</think>so... ANSWER: **42**.") == "42"


def test_catalog_filters_non_text_and_routers():
    models = [
        {"id": "a/x:free", "pricing": {"prompt": "0", "completion": "0"},
         "architecture": {"output_modalities": ["text"]}},
        {"id": "openrouter/free", "pricing": {"prompt": "0", "completion": "0"},
         "architecture": {"output_modalities": ["text"]}},
        {"id": "g/music", "pricing": {"prompt": "0", "completion": "0"},
         "architecture": {"output_modalities": ["text", "audio"]}},
        {"id": "n/content-safety:free", "pricing": {"prompt": "0", "completion": "0"},
         "architecture": {"output_modalities": ["text"]}},
        {"id": "paid/model", "pricing": {"prompt": "0.000001", "completion": "0.000002"},
         "architecture": {"output_modalities": ["text"]}},
    ]
    assert [m["id"] for m in free_text_models(models)] == ["a/x:free"]


def test_sample_caps_and_keeps_preferred():
    models = [f"m{i}" for i in range(30)]
    s = sample(models, 10, prefer=["m7", "gone"])
    assert len(s) == 10 and s[0] == "m7" and len(set(s)) == 10


def test_scan_ranks_good_over_dumb_and_busy():
    _, r = ryder({"good": "good", "dumb": "dumb", "busy": "busy"})
    reports = r.scan()
    assert reports[0].model == "good"
    assert [c.model for c in r.ranking] == ["good"]  # others below min_score
    assert r.current == "good"


def test_failover_when_best_congests_then_rescan():
    state = {"congested": False}

    def flaky(n):
        if state["congested"]:
            return CallResult(False, 429, 0.1, error="busy")
        return FakeClient({"x": "good"}).chat(
            {"model": "x", "messages": [{"role": "user", "content": TASKS[n - 1].prompt}]})

    def second(n):
        r = FakeClient({"y": "good"}).chat(
            {"model": "y", "messages": [{"role": "user",
                                         "content": TASKS[(n - 1) % len(TASKS)].prompt}]})
        r.latency = 5.0
        return r

    client, r = ryder({"best": flaky, "second": second})
    r.scan()
    assert r.current == "best"
    state["congested"] = True
    res = r.chat({"messages": [{"role": "user", "content": "hi"}]})
    assert res.ok and r.current == "second"


def test_gone_model_is_dropped():

    def vanishing(n):
        if n > len(TASKS):
            return CallResult(False, 404, 0.1, error="gone")
        return ok(GOOD_ANSWERS[TASKS[n - 1].name], latency=0.5)

    client, r = ryder({"a": vanishing, "b": "good"})
    r.scan()
    assert r.current == "a"
    assert r.chat({"messages": [{"role": "user", "content": "hi"}]}).ok
    assert [c.model for c in r.ranking] == ["b"]


def test_rescan_when_everything_is_cooling():
    client, r = ryder({"a": "good"})
    r.scan()
    for c in r.ranking:
        c.cooldown_until = 1e18
    client.behaviour["a"] = "good"
    assert r.chat({"messages": [{"role": "user", "content": "hi"}]}).ok


def test_proxy_round_trip_and_stream():
    from http.server import ThreadingHTTPServer
    _, r = ryder({"good": "good"})
    r.scan()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(r))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}/v1"
    try:
        body = json.dumps({"model": "freeryder",
                           "messages": [{"role": "user", "content": "hi"}]}).encode()
        req = urllib.request.Request(base + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        out = json.loads(urllib.request.urlopen(req).read())
        assert out["choices"][0]["message"]["content"] == "hello"
        body = json.dumps({"messages": [{"role": "user", "content": "hi"}],
                           "stream": True}).encode()
        req = urllib.request.Request(base + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        sse = urllib.request.urlopen(req).read().decode()
        assert '"content": "hello"' in sse and sse.rstrip().endswith("data: [DONE]")
        status = json.loads(urllib.request.urlopen(base + "/status").read())
        assert status["current"] == "good"
    finally:
        srv.shutdown()


def test_probe_skips_unusable_models():
    client, r = ryder({"dead": "gone", "good": "good"})
    reports = r.scan()
    assert [x.model for x in reports] == ["good"]


def test_fallback_router_when_nothing_works():
    client, r = ryder({"a": "good", "openrouter/free": "good"},
                      fallback_model="openrouter/free")
    client.catalog = client.catalog[:1]
    r.scan()
    client.behaviour["a"] = "busy"
    res = r.chat({"messages": [{"role": "user", "content": "hi"}]})
    assert res.ok and client.calls[-1] == "openrouter/free"
