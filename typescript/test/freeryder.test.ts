import assert from "node:assert/strict";
import type { AddressInfo } from "node:net";
import { test } from "node:test";

import { finalAnswer, sample, TASKS } from "../src/bench.js";
import { freeTextModels } from "../src/catalog.js";
import { CallResult, type ChatClient, type Json, type Model } from "../src/client.js";
import { type Config, FreeRyder } from "../src/picker.js";
import { createProxy } from "../src/proxy.js";

const GOOD_ANSWERS: Record<string, string> = {
  arithmetic: "ANSWER: 410",
  time: "ANSWER: 155",
  letters: "ANSWER: 6",
  logic: "ANSWER: yes",
  reverse: "ANSWER: redyreerf",
  code: "ANSWER: 20",
  instruction: "banana",
  json: '{"name": "Ada", "year": 1815}',
};

const ok = (text: string, latency = 1, model = "m") =>
  new CallResult(true, 200, latency, {
    model,
    choices: [{ message: { role: "assistant", content: text }, finish_reason: "stop" }],
  });

type Behaviour = "good" | "dumb" | "busy" | "gone" | ((n: number) => CallResult);

/** behaviour[model] decides how each fake model answers. */
class FakeClient implements ChatClient {
  calls: string[] = [];
  catalog: Model[];

  constructor(public behaviour: Record<string, Behaviour>) {
    this.catalog = Object.keys(behaviour).map((id) => ({
      id,
      pricing: { prompt: "0", completion: "0" },
      architecture: { output_modalities: ["text"] },
    }));
  }

  async listModels(): Promise<Model[]> {
    return this.catalog;
  }

  async chat(payload: Json): Promise<CallResult> {
    const model: string = payload.model;
    const prompt: string = payload.messages.at(-1).content;
    const b = this.behaviour[model];
    if (prompt === "Say OK.") return b === "gone" ? new CallResult(false, 404, 0.1) : ok("OK");
    this.calls.push(model);
    const n = this.calls.filter((m) => m === model).length;
    if (typeof b === "function") return b(n);
    if (b === "busy") return new CallResult(false, 429, 0.1, {}, "rate limited");
    if (b === "gone") return new CallResult(false, 404, 0.1, {}, "no endpoints");
    const task = TASKS.find((t) => t.prompt === prompt);
    if (b === "good") return ok(task ? GOOD_ANSWERS[task.name] : "hello", 1, model);
    return ok("ANSWER: no idea", 0.5, model);
  }
}

function ryder(behaviour: Record<string, Behaviour>, cfg: Partial<Config> = {}) {
  const client = new FakeClient(behaviour);
  const r = new FreeRyder({ client, config: { stateFile: null, minRescanGap: 0, fallbackModel: null, ...cfg } });
  return { client, r };
}

const hi = { messages: [{ role: "user", content: "hi" }] };

test("good answers pass every task", () => {
  for (const t of TASKS) assert.ok(t.check(GOOD_ANSWERS[t.name]), t.name);
});

test("finalAnswer ignores think blocks", () => {
  assert.equal(finalAnswer("<think>ANSWER: 1</think>so... ANSWER: **42**."), "42");
});

test("catalog keeps only free text chat models", () => {
  const free = { prompt: "0", completion: "0" };
  const text = { output_modalities: ["text"] };
  const models: Model[] = [
    { id: "a/x:free", pricing: free, architecture: text },
    { id: "openrouter/free", pricing: free, architecture: text },
    { id: "g/music", pricing: free, architecture: { output_modalities: ["text", "audio"] } },
    { id: "n/content-safety:free", pricing: free, architecture: text },
    { id: "paid/model", pricing: { prompt: "0.000001", completion: "0.000002" }, architecture: text },
  ];
  assert.deepEqual(freeTextModels(models).map((m) => m.id), ["a/x:free"]);
});

test("sample caps and keeps preferred", () => {
  const models = Array.from({ length: 30 }, (_, i) => `m${i}`);
  const s = sample(models, 10, ["m7", "gone"]);
  assert.equal(s.length, 10);
  assert.equal(s[0], "m7");
  assert.equal(new Set(s).size, 10);
});

test("key is required", () => {
  assert.throws(() => new FreeRyder({ config: { stateFile: null } }));
});

test("scan ranks good over dumb and busy", async () => {
  const { r } = ryder({ good: "good", dumb: "dumb", busy: "busy" });
  const reports = await r.scan();
  assert.equal(reports[0].model, "good");
  assert.deepEqual(r.ranking.map((c) => c.model), ["good"]);
  assert.equal(r.current, "good");
});

test("probe skips unusable models", async () => {
  const { r } = ryder({ dead: "gone", good: "good" });
  assert.deepEqual((await r.scan()).map((x) => x.model), ["good"]);
});

test("fails over when the best congests", async () => {
  let congested = false;
  const answer = (n: number, latency: number) => ok(GOOD_ANSWERS[TASKS[(n - 1) % TASKS.length].name], latency);
  const { r } = ryder({
    best: (n) => (congested ? new CallResult(false, 429, 0.1, {}, "busy") : answer(n, 1)),
    second: (n) => answer(n, 5),
  });
  await r.scan();
  assert.equal(r.current, "best");
  congested = true;
  const res = await r.chat(hi);
  assert.ok(res.ok);
  assert.equal(r.current, "second");
});

test("a vanished model is dropped", async () => {
  const { r } = ryder({
    a: (n) => (n > TASKS.length ? new CallResult(false, 404, 0.1, {}, "gone") : ok(GOOD_ANSWERS[TASKS[n - 1].name], 0.5)),
    b: "good",
  });
  await r.scan();
  assert.equal(r.current, "a");
  assert.ok((await r.chat(hi)).ok);
  assert.deepEqual(r.ranking.map((c) => c.model), ["b"]);
});

test("rescans when everything is cooling", async () => {
  const { r } = ryder({ a: "good" });
  await r.scan();
  for (const c of r.ranking) c.cooldownUntil = 1e18;
  assert.ok((await r.chat(hi)).ok);
});

test("falls back to the router when nothing works", async () => {
  const { client, r } = ryder({ a: "good", "openrouter/free": "good" }, { fallbackModel: "openrouter/free" });
  client.catalog = client.catalog.slice(0, 1);
  await r.scan();
  client.behaviour.a = "busy";
  const res = await r.chat(hi);
  assert.ok(res.ok);
  assert.equal(client.calls.at(-1), "openrouter/free");
});

test("proxy round trip, stream and status", async () => {
  const { r } = ryder({ good: "good" });
  await r.scan();
  const server = createProxy(r).listen(0, "127.0.0.1");
  await new Promise((res) => server.once("listening", res));
  const base = `http://127.0.0.1:${(server.address() as AddressInfo).port}/v1`;
  try {
    const post = (body: Json) =>
      fetch(base + "/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
    const out = await (await post({ model: "freeryder", ...hi })).json();
    assert.equal(out.choices[0].message.content, "hello");
    const sse = await (await post({ ...hi, stream: true })).text();
    assert.ok(sse.includes('"content":"hello"'));
    assert.ok(sse.trimEnd().endsWith("data: [DONE]"));
    const status = await (await fetch(base + "/status")).json();
    assert.equal(status.current, "good");
  } finally {
    server.close();
  }
});
