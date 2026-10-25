/** Benchmark free models for responsiveness and capability. */

import { CallResult, type ChatClient } from "./client.js";
import { TASKS, type Task } from "./tasks.js";

export { clean, finalAnswer, TASKS, type Task } from "./tasks.js";

export interface Weights {
  quality: number;
  reliability: number;
  speed: number;
  /** Median latency (seconds) at which the speed score drops to 0.5. */
  latencyHalf: number;
}

export const DEFAULT_WEIGHTS: Weights = { quality: 0.6, reliability: 0.25, speed: 0.15, latencyHalf: 10 };

export interface TaskCall {
  task: string;
  result: CallResult;
  /** Task score, 0 to 1. */
  score: number;
}

export class ModelReport {
  calls: TaskCall[] = [];
  score = 0;

  constructor(readonly model: string) {}

  get points(): number {
    return this.calls.reduce((a, c) => a + c.score, 0);
  }
  get successes(): CallResult[] {
    return this.calls.filter((c) => c.result.ok).map((c) => c.result);
  }
  get quality(): number {
    return this.calls.length ? this.points / this.calls.length : 0;
  }
  get reliability(): number {
    return this.calls.length ? this.successes.length / this.calls.length : 0;
  }
  get rateLimited(): number {
    return this.calls.filter((c) => c.result.rateLimited).length;
  }
  get gone(): boolean {
    return this.calls.some((c) => c.result.gone);
  }
  get medianLatency(): number | null {
    const lat = this.successes.map((r) => r.latency);
    return lat.length ? median(lat) : null;
  }

  computeScore(w: Weights): number {
    const med = this.medianLatency;
    if (med === null) return (this.score = 0);
    const speed = 1 / (1 + med / w.latencyHalf);
    return (this.score = w.quality * this.quality + w.reliability * this.reliability + w.speed * speed);
  }

  summary(): string {
    const lat = this.medianLatency === null ? "-" : `${this.medianLatency.toFixed(1)}s`;
    const tasks = this.calls.map((c) => `${c.task}=${Number(c.score.toFixed(1))}`).join(" ");
    return (
      `${this.score.toFixed(3)}  ${this.model.padEnd(50)} quality ${this.points.toFixed(1)}/${this.calls.length}  ` +
      `ok ${this.successes.length}/${this.calls.length}  429s ${this.rateLimited}  median ${lat}  [${tasks}]`
    );
  }
}

export function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

/** Run `fn` over `items` with at most `limit` in flight, keeping order. */
export async function mapLimit<T, R>(items: T[], limit: number, fn: (x: T) => Promise<R>): Promise<R[]> {
  const out: R[] = new Array(items.length);
  let next = 0;
  const worker = async () => {
    while (next < items.length) {
      const i = next++;
      out[i] = await fn(items[i]);
    }
  };
  await Promise.all(Array.from({ length: Math.max(1, Math.min(limit, items.length)) }, worker));
  return out;
}

export interface BenchmarkOptions {
  tasks?: Task[];
  weights?: Weights;
  maxTokens?: number;
  timeout?: number; // seconds
  /** Stop testing a model after this many consecutive failed calls. */
  abortAfterFailures?: number;
}

export class Benchmark {
  readonly tasks: Task[];
  readonly weights: Weights;
  readonly maxTokens: number;
  readonly timeout: number;
  readonly abortAfterFailures: number;

  constructor(readonly client: ChatClient, opts: BenchmarkOptions = {}) {
    this.tasks = opts.tasks ?? TASKS;
    this.weights = opts.weights ?? DEFAULT_WEIGHTS;
    this.maxTokens = opts.maxTokens ?? 8192;
    this.timeout = opts.timeout ?? 60;
    this.abortAfterFailures = opts.abortAfterFailures ?? 3;
  }

  async runModel(model: string): Promise<ModelReport> {
    const report = new ModelReport(model);
    let failures = 0;
    for (const task of this.tasks) {
      if (failures >= this.abortAfterFailures) {
        report.calls.push({ task: task.name, result: new CallResult(false, 0, 0, {}, "skipped"), score: 0 });
        continue;
      }
      const res = await this.client.chat(
        { model, messages: [{ role: "user", content: task.prompt }], temperature: 0, max_tokens: this.maxTokens },
        this.timeout,
      );
      failures = res.gone ? this.abortAfterFailures : res.ok ? 0 : failures + 1;
      report.calls.push({ task: task.name, result: res, score: res.ok ? Number(task.check(res.text)) : 0 });
    }
    report.computeScore(this.weights);
    return report;
  }

  /** Drop models this account can't use at all; rate-limited ones are kept. */
  async probe(models: string[], parallel = 5): Promise<string[]> {
    const flags = await mapLimit(models, parallel, async (model) => {
      const res = await this.client.chat(
        { model, max_tokens: 16, temperature: 0, messages: [{ role: "user", content: "Say OK." }] },
        this.timeout,
      );
      return !res.gone;
    });
    return models.filter((_, i) => flags[i]);
  }

  /** Benchmark models concurrently (tasks within a model run in sequence). Best first. */
  async run(models: string[], parallel = 5): Promise<ModelReport[]> {
    const reports = await mapLimit(models, parallel, (m) => this.runModel(m));
    return reports.sort((a, b) => b.score - a.score);
  }
}

/** Pick up to k models at random; ids in `prefer` (e.g. last winners) are kept first. */
export function sample(models: string[], k = 10, prefer: string[] = [], random = Math.random): string[] {
  const keep = prefer.filter((m) => models.includes(m)).slice(0, k);
  const rest = models.filter((m) => !keep.includes(m));
  for (let i = rest.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [rest[i], rest[j]] = [rest[j], rest[i]];
  }
  return [...keep, ...rest.slice(0, k - keep.length)];
}
