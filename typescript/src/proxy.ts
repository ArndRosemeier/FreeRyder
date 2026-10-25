/** Local OpenAI-compatible proxy: point any OpenAI client at http://localhost:PORT/v1. */

import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";

import type { Json } from "./client.js";
import { type FreeRyder, NoModelAvailable } from "./picker.js";

export const MODEL_ALIAS = "freeryder";

function sendJson(res: ServerResponse, status: number, obj: Json): void {
  const data = JSON.stringify(obj);
  res.writeHead(status, { "Content-Type": "application/json", "Content-Length": Buffer.byteLength(data) });
  res.end(data);
}

async function readBody(req: IncomingMessage): Promise<string> {
  const chunks: Buffer[] = [];
  for await (const c of req) chunks.push(c as Buffer);
  return Buffer.concat(chunks).toString("utf8");
}

/**
 * Replay a finished completion as SSE so streaming clients keep working.
 * Failover needs the whole answer first, so this is not token-by-token streaming.
 */
function sendAsStream(res: ServerResponse, body: Json): void {
  const choice = body.choices?.[0] ?? {};
  const msg = choice.message ?? {};
  const delta: Json = { role: "assistant", content: msg.content ?? "" };
  if (msg.tool_calls) delta.tool_calls = msg.tool_calls.map((tc: Json, index: number) => ({ ...tc, index }));
  const base = { id: body.id, created: body.created, model: body.model };
  const chunks = [
    { ...base, object: "chat.completion.chunk", choices: [{ index: 0, delta, finish_reason: null }] },
    {
      ...base,
      object: "chat.completion.chunk",
      usage: body.usage,
      choices: [{ index: 0, delta: {}, finish_reason: choice.finish_reason ?? "stop" }],
    },
  ];
  res.writeHead(200, { "Content-Type": "text/event-stream", "Cache-Control": "no-cache", Connection: "close" });
  for (const c of chunks) res.write(`data: ${JSON.stringify(c)}\n\n`);
  res.end("data: [DONE]\n\n");
}

export function createProxy(ryder: FreeRyder): Server {
  return createServer(async (req, res) => {
    const path = (req.url ?? "/").split("?")[0].replace(/\/+$/, "");
    try {
      if (req.method === "GET" && (path === "/v1/models" || path === "/models")) {
        return sendJson(res, 200, {
          object: "list",
          data: [{ id: MODEL_ALIAS, object: "model", owned_by: "freeryder", current: ryder.current }],
        });
      }
      if (req.method === "GET" && (path === "/status" || path === "/v1/status")) {
        return sendJson(res, 200, {
          current: ryder.current,
          scanned_at: ryder.scannedAt,
          ranking: ryder.ranking.map((c) => ({
            model: c.model,
            score: Math.round(c.score * 1000) / 1000,
            bench_latency: Math.round(c.benchLatency * 100) / 100,
            cooling_until: c.cooldownUntil || null,
          })),
        });
      }
      if (req.method === "POST" && (path === "/v1/chat/completions" || path === "/chat/completions")) {
        let payload: Json;
        try {
          payload = JSON.parse((await readBody(req)) || "{}");
        } catch {
          return sendJson(res, 400, { error: { message: "invalid JSON" } });
        }
        const stream = Boolean(payload.stream);
        delete payload.stream;
        delete payload.stream_options;
        let result;
        try {
          result = await ryder.chat(payload);
        } catch (e) {
          if (e instanceof NoModelAvailable) return sendJson(res, 503, { error: { message: e.message } });
          throw e;
        }
        if (!result.ok) {
          return sendJson(res, result.status || 502, {
            error: { message: "all free models failed", upstream: result.error },
          });
        }
        return stream ? sendAsStream(res, result.body) : sendJson(res, 200, result.body);
      }
      sendJson(res, 404, { error: { message: "not found" } });
    } catch (e) {
      sendJson(res, 500, { error: { message: String(e) } });
    }
  });
}

export function serve(ryder: FreeRyder, host = "127.0.0.1", port = 8765): Server {
  const server = createProxy(ryder);
  server.listen(port, host, () =>
    ryder.cfg.log(`FreeRyder proxy on http://${host}:${port}/v1 (model name: ${MODEL_ALIAS})`),
  );
  return server;
}
