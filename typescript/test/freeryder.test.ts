import assert from "node:assert/strict";
import type { AddressInfo } from "node:net";
import { test } from "node:test";

import { sample } from "../src/bench.js";
import { checkJson, checkStory, finalAnswer, storyTells, TASKS } from "../src/tasks.js";
import { freeTextModels } from "../src/catalog.js";
import { CallResult, type ChatClient, type Json, type Model } from "../src/client.js";
import { type Config, FreeRyder } from "../src/picker.js";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createProxy } from "../src/proxy.js";

const STORY =
  "The coat cost four euros and smelled of somebody else's cigarettes. Marta bought it " +
  "anyway, because November had arrived early and her own coat had a broken zip. " +
  "On the tram home she put her hands in the pockets and found a key. It was brass, " +
  "heavy, with a paper tag tied on by a bit of red string. The tag said 14B in pencil. " +
  "She turned it over. Nothing on the back. At her stop she almost left it on the seat, " +
  "then didn't. That evening she walked past the old flats on Kettengasse, where the " +
  "buildings still had numbers painted above the doors, and counted up to fourteen. " +
  "There was no B. There was a bakery, closed, and a woman smoking in the doorway who " +
  "looked at the coat and then at Marta's face for a long time. You found it, the woman " +
  "said. He always said someone would.";

const GOOD_ANSWERS: Record<string, string> = {
  json:
    '{"order_id": 4471, "customer": "Mirela Petrov", "date": "2025-03-03", ' +
    '"items": [{"name": "lamp", "qty": 2, "unit_price": 24.50}, ' +
    '{"name": "rug", "qty": 1, "unit_price": 89}], "total": 138}',
  "riddle-easy": "Not red, not green.\nANSWER: blue",
  "riddle-medium": "ANSWER: Dev, Ada, Cleo, Ben, Eli",
  "riddle-hard": "<think>ANSWER: Quin, Sage</think>Case analysis...\nANSWER: **Pell, Tess**",
  story: STORY,
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
  for (const t of TASKS) {
    assert.equal(t.check(GOOD_ANSWERS[t.name]), 1, t.name);
    assert.equal(t.check("ANSWER: no idea"), 0, t.name);
  }
});

test("JSON in a code fence gets half", () => {
  assert.equal(checkJson("```json\n" + GOOD_ANSWERS.json + "\n```"), 0.5);
  assert.equal(checkJson(GOOD_ANSWERS.json.replace("4471", '"4471"')), 0);
});

test("story tells are counted", () => {
  const ai = STORY.replace("She turned it over.", "She turned it over\u2014a testament to a sense of wonder.");
  assert.deepEqual(storyTells(ai), { dash: 1, testament: 1, "a sense of": 1 });
  assert.ok(Math.abs(checkStory(ai) - 0.4) < 1e-9);
  assert.equal(checkStory("Too short."), 0);
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
  assert.equal(res.body.model, "openrouter/free");
  await r.waitReady();
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

test("start returns immediately and uses the fallback while warming up", async () => {
  let release!: () => void;
  const gate = new Promise<void>((res) => (release = res));
  const { client, r } = ryder(
    {
      best: "good",
      "openrouter/free": "good",
    },
    { fallbackModel: "openrouter/free" },
  );
  client.catalog = client.catalog.slice(0, 1);
  // Slow the benchmark down until we release it.
  const chat = client.chat.bind(client);
  client.chat = async (payload: Json) => {
    if (payload.model === "best" && payload.messages.at(-1).content !== "Say OK.") await gate;
    return chat(payload);
  };
  assert.equal(r.start(), r);
  assert.ok(r.scanning);
  const early = await r.chat(hi);
  assert.equal(early.body.model, "openrouter/free");
  release();
  assert.ok(await r.waitReady());
  assert.equal(r.current, "best");
  assert.equal((await r.chat(hi)).body.model, "best");
});

test("chat without a fallback waits for warm-up", async () => {
  const { r } = ryder({ a: "good" });
  const res = await r.start().chat(hi);
  assert.equal(res.body.model, "a");
});

test("start skips the scan when the cache is fresh", async () => {
  const stateFile = join(mkdtempSync(join(tmpdir(), "freeryder-")), "state.json");
  const client = new FakeClient({ a: "good" });
  await new FreeRyder({ client, config: { stateFile } }).scan();
  const n = client.calls.length;
  const r = new FreeRyder({ client, config: { stateFile } }).start();
  assert.ok(!r.scanning);
  assert.equal(r.current, "a");
  assert.equal(client.calls.length, n);
});

test("a background scan error is kept, not thrown", async () => {
  const client = new FakeClient({});
  client.listModels = async () => {
    throw new Error("network down");
  };
  const r = new FreeRyder({ client, config: { stateFile: null, fallbackModel: null } });
  await r.start().waitReady();
  assert.match(String(r.lastError), /network down/);
});
