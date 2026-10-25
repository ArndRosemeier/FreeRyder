/** Use the best free model until it congests or disappears, then fail over or rescan. */

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

import { Benchmark, median, type ModelReport, sample } from "./bench.js";
import { freeTextModels } from "./catalog.js";
import { CallResult, type ChatClient, type Json, OpenRouterClient } from "./client.js";

export interface Config {
  sampleSize: number; // models benchmarked per scan
  parallel: number; // models benchmarked at once
  minScore: number; // models below this never get used
  maxFailures: number; // consecutive failures before a model counts as congested
  cooldown: number; // seconds a congested model is skipped
  slowFactor: number; // rolling median latency > factor * benchmark median => congested
  slowFloor: number; // ... but only once it is also above this many seconds
  rescanAfter: number; // seconds; rescan even when healthy, to pick up new models
  minRescanGap: number; // seconds; never rescan more often than this
  maxAttempts: number; // models tried per request before giving up
  timeout: number; // seconds per call
  /** Last resort when no benchmarked model is healthy: OpenRouter's own free router. */
  fallbackModel: string | null;
  /** Where the ranking is cached between runs; null disables it. */
  stateFile: string | null;
  log: (msg: string) => void;
}

export const DEFAULT_CONFIG: Config = {
  sampleSize: 10,
  parallel: 5,
  minScore: 0.5,
  maxFailures: 2,
  cooldown: 600,
  slowFactor: 3,
  slowFloor: 20,
  rescanAfter: 6 * 3600,
  minRescanGap: 120,
  maxAttempts: 4,
  timeout: 90,
  fallbackModel: "openrouter/free",
  stateFile: join(process.env.XDG_CACHE_HOME ?? join(homedir(), ".cache"), "freeryder", "state.json"),
  log: () => {},
};

export interface Candidate {
  model: string;
  score: number;
  benchLatency: number;
  failures: number;
  cooldownUntil: number; // epoch seconds
  recent: number[];
}

const now = () => Date.now() / 1000;

function candidate(model: string, score: number, benchLatency: number): Candidate {
  return { model, score, benchLatency, failures: 0, cooldownUntil: 0, recent: [] };
}

export class NoModelAvailable extends Error {}

export interface FreeRyderOptions {
  /** Your OpenRouter API key. Required unless `client` is given. */
  apiKey?: string;
  config?: Partial<Config>;
  client?: ChatClient;
  benchmark?: Benchmark;
}

export class FreeRyder {
  readonly client: ChatClient;
  readonly cfg: Config;
  readonly bench: Benchmark;
  ranking: Candidate[] = [];
  scannedAt = 0;
  lastReports: ModelReport[] = [];
  private scanning: Promise<ModelReport[]> | null = null;

  constructor(opts: FreeRyderOptions) {
    if (!opts.client && !opts.apiKey) throw new Error("FreeRyder needs an OpenRouter apiKey");
    this.client = opts.client ?? new OpenRouterClient(opts.apiKey);
    this.cfg = { ...DEFAULT_CONFIG, ...opts.config };
    this.bench = opts.benchmark ?? new Benchmark(this.client, { timeout: this.cfg.timeout });
    this.loadState();
  }

  // ---- scanning ------------------------------------------------------------------------

  /** List free models, benchmark a random sample, and rebuild the ranking. */
  scan(): Promise<ModelReport[]> {
    // Concurrent callers share one scan.
    this.scanning ??= this.doScan().finally(() => (this.scanning = null));
    return this.scanning;
  }

  private async doScan(): Promise<ModelReport[]> {
    const free = freeTextModels(await this.client.listModels()).map((m) => m.id);
    const reachable = await this.bench.probe(free, this.cfg.parallel);
    const previous = this.ranking.map((c) => c.model).slice(0, 3);
    const chosen = sample(reachable, this.cfg.sampleSize, previous);
    this.cfg.log(`benchmarking ${chosen.length} of ${reachable.length} reachable free models (${free.length} listed)`);
    const reports = await this.bench.run(chosen, this.cfg.parallel);
    this.lastReports = reports;
    this.ranking = reports
      .filter((r) => r.score >= this.cfg.minScore)
      .map((r) => candidate(r.model, r.score, r.medianLatency ?? 0));
    this.scannedAt = now();
    this.saveState();
    for (const r of reports) this.cfg.log(r.summary());
    return reports;
  }

  private async maybeRescan(force = false): Promise<void> {
    const t = now();
    const stale = t - this.scannedAt > this.cfg.rescanAfter;
    const usable = this.ranking.some((c) => t >= c.cooldownUntil);
    if ((force || stale || !usable) && (this.scanning || t - this.scannedAt >= this.cfg.minRescanGap)) {
      await this.scan();
    }
  }

  /** The model the next request goes to, or null if none is healthy. */
  get current(): string | null {
    const t = now();
    return this.ranking.find((c) => t >= c.cooldownUntil)?.model ?? null;
  }

  // ---- health bookkeeping --------------------------------------------------------------

  private record(cand: Candidate, res: CallResult): void {
    if (res.gone) {
      this.cfg.log(`${cand.model} disappeared (${res.status}); dropping it`);
      this.ranking = this.ranking.filter((c) => c !== cand);
      return;
    }
    if (res.ok) {
      cand.failures = 0;
      cand.recent = [...cand.recent, res.latency].slice(-5);
      const med = median(cand.recent);
      if (
        cand.recent.length >= 3 &&
        med > this.cfg.slowFloor &&
        med > this.cfg.slowFactor * Math.max(cand.benchLatency, 1)
      ) {
        this.cfg.log(`${cand.model} is slowing down (median ${med.toFixed(1)}s); cooling it down`);
        this.cool(cand);
      }
      return;
    }
    cand.failures += 1;
    this.cfg.log(`${cand.model} failed (${res.status} ${res.error.slice(0, 120)})`);
    if (cand.failures >= this.cfg.maxFailures || res.rateLimited) this.cool(cand);
  }

  private cool(cand: Candidate): void {
    cand.cooldownUntil = now() + this.cfg.cooldown;
    cand.failures = 0;
    cand.recent = [];
  }

  private async candidates(tried: Set<string>): Promise<Candidate[]> {
    await this.maybeRescan();
    const t = now();
    return this.ranking.filter((c) => t >= c.cooldownUntil && !tried.has(c.model));
  }

  // ---- public API ----------------------------------------------------------------------

  /**
   * Send an OpenAI-style chat payload to the best healthy free model.
   * `payload.model` is replaced. Fails over to the next-ranked model on errors, rescans when
   * no ranked model is usable, and finally tries `fallbackModel`.
   * Resolves with the last failure if everything fails.
   */
  async chat(payload: Json): Promise<CallResult> {
    let last: CallResult | null = null;
    const tried = new Set<string>();
    let rescanned = false;
    for (let i = 0; i < this.cfg.maxAttempts; i++) {
      let cands = await this.candidates(tried);
      if (!cands.length && !rescanned) {
        await this.maybeRescan(true);
        rescanned = true;
        cands = await this.candidates(tried);
      }
      if (!cands.length) break;
      const cand = cands[0];
      tried.add(cand.model);
      const res = await this.client.chat({ ...payload, model: cand.model }, this.cfg.timeout);
      this.record(cand, res);
      if (res.ok) return res;
      last = res;
      // A client error that is not about the model won't improve elsewhere.
      if ([400, 401, 402, 413].includes(res.status)) return res;
    }
    const fb = this.cfg.fallbackModel;
    if (fb && !tried.has(fb)) {
      this.cfg.log(`no benchmarked model worked; falling back to ${fb}`);
      const res = await this.client.chat({ ...payload, model: fb }, this.cfg.timeout);
      if (res.ok || last === null) return res;
    }
    if (last === null) throw new NoModelAvailable("no free model passed the benchmark; try again later");
    return last;
  }

  /** Convenience: one user prompt in, answer text out. */
  async complete(prompt: string, params: Json = {}): Promise<string> {
    const res = await this.chat({ messages: [{ role: "user", content: prompt }], ...params });
    if (!res.ok) throw new Error(`all free models failed: ${res.status} ${res.error}`);
    return res.text;
  }

  // ---- persistence ---------------------------------------------------------------------

  private saveState(): void {
    const path = this.cfg.stateFile;
    if (!path) return;
    try {
      mkdirSync(dirname(path), { recursive: true });
      const ranking = this.ranking.map(({ model, score, benchLatency }) => ({
        model,
        score,
        bench_latency: benchLatency,
      }));
      writeFileSync(path, JSON.stringify({ scanned_at: this.scannedAt, ranking }, null, 1));
    } catch {
      /* cache only */
    }
  }

  private loadState(): void {
    const path = this.cfg.stateFile;
    if (!path) return;
    try {
      const data = JSON.parse(readFileSync(path, "utf8"));
      this.ranking = data.ranking.map((r: Json) => candidate(r.model, r.score, r.bench_latency));
      this.scannedAt = Number(data.scanned_at);
    } catch {
      /* missing or bad cache */
    }
  }
}
