/** Live test against OpenRouter. Runs only when OPEN_ROUTER_KEY_JUST_FOR_FREE_MODELS is set. */

import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import { FreeRyder } from "../src/picker.js";

const KEY = process.env.OPEN_ROUTER_KEY_JUST_FOR_FREE_MODELS;

test("live scan and chat", { skip: !KEY && "OPEN_ROUTER_KEY_JUST_FOR_FREE_MODELS not set", timeout: 600_000 }, async () => {
  const stateFile = join(mkdtempSync(join(tmpdir(), "freeryder-")), "state.json");
  const ryder = new FreeRyder({ apiKey: KEY, config: { sampleSize: 3, stateFile } });
  const reports = await ryder.scan();
  assert.ok(reports.length, "no reachable free models");
  for (const r of reports) console.log(r.summary());
  const res = await ryder.chat({ messages: [{ role: "user", content: "Reply with the word pong." }] });
  assert.ok(res.ok, res.error);
  assert.match(res.text.toLowerCase(), /pong/);
});
