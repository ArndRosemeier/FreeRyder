/** Minimal OpenRouter HTTP client (uses the built-in fetch). */

export const BASE_URL = "https://openrouter.ai/api/v1";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type Json = any;

/** Outcome of one chat call. `status` is the HTTP status, 0 for network errors/timeouts. */
export class CallResult {
  constructor(
    public ok: boolean,
    public status: number,
    public latency: number, // seconds
    public body: Json = {},
    public error = "",
  ) {}

  get text(): string {
    return this.body?.choices?.[0]?.message?.content ?? "";
  }

  get rateLimited(): boolean {
    return this.status === 429;
  }

  /** Model no longer exists, no provider serves it for this account, or it is restricted. */
  get gone(): boolean {
    if (this.status === 404 || this.status === 410) return true;
    return this.status === 403 && this.error.includes("only available");
  }
}

export interface Model {
  id: string;
  pricing?: Record<string, string>;
  architecture?: { output_modalities?: string[] };
  context_length?: number;
  [k: string]: Json;
}

export interface ChatClient {
  listModels(): Promise<Model[]>;
  chat(payload: Json, timeoutSeconds?: number): Promise<CallResult>;
}

export class OpenRouterClient implements ChatClient {
  readonly baseUrl: string;

  /** `apiKey` is your OpenRouter key. Only `listModels` works without one. */
  constructor(
    readonly apiKey: string | undefined,
    baseUrl = BASE_URL,
    readonly appName = "freeryder",
  ) {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
  }

  private headers(): Record<string, string> {
    const h: Record<string, string> = { "Content-Type": "application/json", "X-Title": this.appName };
    if (this.apiKey) h.Authorization = `Bearer ${this.apiKey}`;
    return h;
  }

  async listModels(): Promise<Model[]> {
    const res = await fetch(this.baseUrl + "/models", {
      headers: this.headers(),
      signal: AbortSignal.timeout(30_000),
    });
    if (!res.ok) throw new Error(`GET /models failed: ${res.status}`);
    return ((await res.json()) as { data: Model[] }).data;
  }

  /** POST /chat/completions (non-streaming). Never throws for HTTP/network errors. */
  async chat(payload: Json, timeoutSeconds = 60): Promise<CallResult> {
    const start = performance.now();
    const elapsed = () => (performance.now() - start) / 1000;
    let res: Response;
    let raw: string;
    try {
      res = await fetch(this.baseUrl + "/chat/completions", {
        method: "POST",
        headers: this.headers(),
        body: JSON.stringify(payload),
        signal: AbortSignal.timeout(timeoutSeconds * 1000),
      });
      raw = await res.text();
    } catch (e) {
      return new CallResult(false, 0, elapsed(), {}, String(e));
    }
    const latency = elapsed();
    if (!res.ok) return new CallResult(false, res.status, latency, {}, raw.slice(0, 500));
    let body: Json;
    try {
      body = JSON.parse(raw);
    } catch (e) {
      return new CallResult(false, 0, latency, {}, `bad JSON: ${String(e)}`);
    }
    // OpenRouter sometimes returns 200 with an error object (e.g. upstream 429).
    if (body && "error" in body) {
      const err = body.error ?? {};
      const code = Number.isInteger(err.code) ? err.code : 502;
      return new CallResult(false, code, latency, body, JSON.stringify(err).slice(0, 500));
    }
    return new CallResult(true, 200, latency, body);
  }
}
