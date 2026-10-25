/** Benchmark free models for responsiveness and capability. */

import { CallResult, type ChatClient } from "./client.js";

const ANSWER_HINT = "\nEnd your reply with a final line of the form `ANSWER: <answer>`.";

export function clean(text: string): string {
  return (text ?? "").replace(/<think>[\s\S]*?<\/think>/gi, "").trim();
}

function stripChars(s: string, chars: string): string {
  let a = 0;
  let b = s.length;
  while (a < b && chars.includes(s[a])) a++;
  while (b > a && chars.includes(s[b - 1])) b--;
  return s.slice(a, b);
}

export function finalAnswer(text: string): string {
  const matches = [...clean(text).matchAll(/ANSWER:\s*(.+)/gi)];
  if (!matches.length) return "";
  return stripChars(matches[matches.length - 1][1].trim(), "`*. ").toLowerCase();
}

const answerIs = (expected: string) => (text: string) => finalAnswer(text) === expected;

function exactBanana(text: string): boolean {
  return stripChars(clean(text), ". \n") === "banana";
}

function jsonAda(text: string): boolean {
  const t = clean(text).replace(/^```(?:json)?|```$/gm, "").trim();
  try {
    const v = JSON.parse(t);
    return v && typeof v === "object" && Object.keys(v).length === 2 && v.name === "Ada" && v.year === 1815;
  } catch {
    return false;
  }
}

export interface Task {
  name: string;
  prompt: string;
  check: (text: string) => boolean;
}

export const TASKS: Task[] = [
  { name: "arithmetic", prompt: "What is 17 * 23 + 19?" + ANSWER_HINT, check: answerIs("410") },
  {
    name: "time",
    prompt:
      "A train leaves at 14:35 and arrives at 17:10 the same day. " +
      "How many minutes is the trip? Give just the number." + ANSWER_HINT,
    check: answerIs("155"),
  },
  {
    name: "letters",
    prompt:
      "How many times does the letter 'r' appear in the text " +
      '"strawberry raspberry"? Give just the number.' + ANSWER_HINT,
    check: answerIs("6"),
  },
  {
    name: "logic",
    prompt:
      "All bloops are razzies. All razzies are lazzies. Are all bloops " +
      "definitely lazzies? Answer yes or no." + ANSWER_HINT,
    check: answerIs("yes"),
  },
  {
    name: "reverse",
    prompt: 'Reverse the string "freeryder" character by character.' + ANSWER_HINT,
    check: answerIs("redyreerf"),
  },
  {
    name: "code",
    prompt:
      "What does this Python expression evaluate to?\n" +
      "sum(i * i for i in range(5) if i % 2 == 0)" + ANSWER_HINT,
    check: answerIs("20"),
  },
  {
    name: "instruction",
    prompt:
      "Reply with exactly the word banana in lowercase, with no " +
      "punctuation, explanation or anything else.",
    check: exactBanana,
  },
  {
    name: "json",
    prompt:
      'Output only a JSON object with exactly two keys: "name" set to the ' +
      'string "Ada" and "year" set to the number 1815. No prose, no code fences.',
    check: jsonAda,
  },
];

export interface Weights {
  quality: number;
  reliability: number;
  speed: number;
  /** Median latency (seconds) at which the speed score drops to 0.5. */
  latencyHalf: number;
}

export const DEFAULT_WEIGHTS: Weights = { quality: 0.6, reliability: 0.25, speed: 0.15, latencyHalf: 6 };

export interface TaskCall {
  task: string;
  result: CallResult;
  passed: boolean;
}

export class ModelReport {
  calls: TaskCall[] = [];
  score = 0;

  constructor(readonly model: string) {}

  get passed(): number {
    return this.calls.filter((c) => c.passed).length;
  }
  get successes(): CallResult[] {
    return this.calls.filter((c) => c.result.ok).map((c) => c.result);
  }
  get quality(): number {
    return this.calls.length ? this.passed / this.calls.length : 0;
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
    return (
      `${this.score.toFixed(3)}  ${this.model.padEnd(55)} quality ${this.passed}/${this.calls.length}  ` +
      `ok ${this.successes.length}/${this.calls.length}  429s ${this.rateLimited}  median ${lat}`
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
    this.maxTokens = opts.maxTokens ?? 2048;
    this.timeout = opts.timeout ?? 60;
    this.abortAfterFailures = opts.abortAfterFailures ?? 3;
  }

  async runModel(model: string): Promise<ModelReport> {
    const report = new ModelReport(model);
    let failures = 0;
    for (const task of this.tasks) {
      if (failures >= this.abortAfterFailures) {
        report.calls.push({ task: task.name, result: new CallResult(false, 0, 0, {}, "skipped"), passed: false });
        continue;
      }
      const res = await this.client.chat(
        { model, messages: [{ role: "user", content: task.prompt }], temperature: 0, max_tokens: this.maxTokens },
        this.timeout,
      );
      failures = res.gone ? this.abortAfterFailures : res.ok ? 0 : failures + 1;
      report.calls.push({ task: task.name, result: res, passed: res.ok && task.check(res.text) });
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
