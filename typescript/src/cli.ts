#!/usr/bin/env node
/** Command line: freeryder list | scan | ask | serve. */

import { parseArgs } from "node:util";

import { freeTextModels } from "./catalog.js";
import { OpenRouterClient } from "./client.js";
import { FreeRyder } from "./picker.js";
import { serve } from "./proxy.js";

const USAGE = `usage: freeryder [options] <list | scan | ask "<prompt>" | serve>

options:
  --api-key KEY     OpenRouter API key (needed for everything but list)
  --sample N        models benchmarked per scan (default 10)
  --min-score X     minimum benchmark score to use a model (default 0.5)
  --host HOST       proxy host for serve (default 127.0.0.1)
  --port PORT       proxy port for serve (default 8765)
  -v, --verbose     log progress to stderr`;

export async function main(argv = process.argv.slice(2)): Promise<number> {
  const { values, positionals } = parseArgs({
    args: argv,
    allowPositionals: true,
    options: {
      "api-key": { type: "string" },
      sample: { type: "string", default: "10" },
      "min-score": { type: "string", default: "0.5" },
      host: { type: "string", default: "127.0.0.1" },
      port: { type: "string", default: "8765" },
      verbose: { type: "boolean", short: "v", default: false },
      help: { type: "boolean", short: "h", default: false },
    },
  });
  const [cmd, ...rest] = positionals;
  if (values.help || !cmd || !["list", "scan", "ask", "serve"].includes(cmd)) {
    console.error(USAGE);
    return values.help ? 0 : 2;
  }
  const apiKey = values["api-key"];
  const client = new OpenRouterClient(apiKey);
  if (cmd === "list") {
    for (const m of freeTextModels(await client.listModels())) {
      console.log(`${m.id.padEnd(55)} ctx ${m.context_length}`);
    }
    return 0;
  }
  if (!apiKey) {
    console.error("--api-key is required for this command.");
    return 2;
  }
  const log = (msg: string) => console.error(`${new Date().toISOString()} ${msg}`);
  const ryder = new FreeRyder({
    client,
    config: {
      sampleSize: Number(values.sample),
      minScore: Number(values["min-score"]),
      log: values.verbose || cmd === "serve" ? log : () => {},
    },
  });

  if (cmd === "scan") {
    const reports = await ryder.scan();
    console.log("score  model" + " ".repeat(52) + "details");
    for (const r of reports) console.log(r.summary());
    console.log(`\nusing: ${ryder.current ?? "none (no model passed min score)"}`);
  } else if (cmd === "ask") {
    const res = await ryder.chat({ messages: [{ role: "user", content: rest.join(" ") }] });
    if (!res.ok) {
      console.error(`failed: ${res.status} ${res.error}`);
      return 1;
    }
    console.error(`[${res.body.model}]`);
    console.log(res.text);
  } else if (cmd === "serve") {
    if (ryder.current === null) await ryder.scan();
    serve(ryder, values.host, Number(values.port));
    return -1; // keep running
  }
  return 0;
}

main().then(
  (code) => {
    if (code >= 0) process.exit(code);
  },
  (err) => {
    console.error(err);
    process.exit(1);
  },
);
