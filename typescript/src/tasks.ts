/**
 * Capability tasks. Each check returns a score from 0 to 1.
 *
 * The riddles are original (not well-known puzzles a model may have memorised); their
 * answers were verified by brute force over every possible assignment.
 */

export const ANSWER_HINT =
  "\nThink it through, then end your reply with a final line of the form `ANSWER: <answer>`.";

export interface Task {
  name: string;
  prompt: string;
  check: (text: string) => number;
}

/** Drop <think> blocks some models put in the content. */
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

function names(answer: string): string[] {
  return answer
    .toLowerCase()
    .split(/[^a-z]+/)
    .filter((w) => w && w !== "and");
}

const sameList = (a: string[], b: string[]) => a.length === b.length && a.every((x, i) => x === b[i]);
const close = (x: unknown, y: number) => typeof x === "number" && Math.abs(x - y) < 1e-6;

// ---- strict JSON -----------------------------------------------------------------------------

export const JSON_PROMPT = `Extract the order below into JSON with exactly this shape:
{"order_id": <integer>, "customer": <string>, "date": "<YYYY-MM-DD>",
 "items": [{"name": <singular noun, lowercase>, "qty": <integer>, "unit_price": <number>}],
 "total": <number>}

Order text: "Order 4471 came in from Mirela Petrov on 3 March 2025. She bought two lamps
at 24.50 each and one rug for 89.00. Shipping was free."

Output only the JSON object: no prose, no code fences.`;

const sameKeys = (o: object, keys: string[]) =>
  sameList(Object.keys(o).sort(), [...keys].sort());

/** 1 for the exact object as raw JSON, 0.5 if it was wrapped in a code fence. */
export function checkJson(text: string): number {
  let raw = clean(text);
  let factor = 1;
  const m = raw.match(/^\s*```[a-zA-Z]*\s*\n?([\s\S]*?)\n?```\s*$/);
  if (m) {
    raw = m[1].trim();
    factor = 0.5;
  }
  let d: any; // eslint-disable-line @typescript-eslint/no-explicit-any
  try {
    d = JSON.parse(raw);
  } catch {
    return 0;
  }
  if (!d || typeof d !== "object" || Array.isArray(d)) return 0;
  if (!sameKeys(d, ["order_id", "customer", "date", "items", "total"])) return 0;
  const items = d.items;
  const ok =
    d.order_id === 4471 &&
    typeof d.customer === "string" &&
    d.customer.trim() === "Mirela Petrov" &&
    d.date === "2025-03-03" &&
    close(d.total, 138) &&
    Array.isArray(items) &&
    items.length === 2 &&
    items.every((i) => i && typeof i === "object" && sameKeys(i, ["name", "qty", "unit_price"])) &&
    items[0].name === "lamp" &&
    items[0].qty === 2 &&
    close(items[0].unit_price, 24.5) &&
    items[1].name === "rug" &&
    items[1].qty === 1 &&
    close(items[1].unit_price, 89);
  return ok ? factor : 0;
}

// ---- logic riddles ---------------------------------------------------------------------------

export const EASY_PROMPT =
  `Three boxes stand in a row, from left to right: red, green, blue.
A coin is in exactly one of them. It is not in the red box, and it is not in the box
directly to the right of the red box. Which box holds the coin?` + ANSWER_HINT;

export function checkEasy(text: string): number {
  const n = names(finalAnswer(text));
  return [["blue"], ["blue", "box"], ["the", "blue", "box"]].some((x) => sameList(n, x)) ? 1 : 0;
}

export const MEDIUM_PROMPT =
  `Ada, Ben, Cleo, Dev and Eli stand in a row in positions 1 to 5, left to right.
- Cleo is at neither end of the row.
- Ben stands directly to the left of Eli.
- Ada stands somewhere to the right of Dev.
- Cleo does not stand next to Dev.
- Ada is not in position 5.
Give the full order from left to right as names separated by commas.` + ANSWER_HINT;

export function checkMedium(text: string): number {
  return sameList(names(finalAnswer(text)), ["dev", "ada", "cleo", "ben", "eli"]) ? 1 : 0;
}

export const HARD_PROMPT =
  `Pell, Quin, Rook, Sage and Tess each either always tell the truth or always lie.
- Pell says: "Quin is a liar."
- Quin says: "Rook and Sage are the same type (both truthful or both liars)."
- Rook says: "Exactly two of us five are liars."
- Sage says: "At least one of Pell and Rook is a liar."
- Tess says: "Pell and I are different types, or the number of liars among us five is odd."
Which of them are liars? Give their names in alphabetical order, separated by commas.` + ANSWER_HINT;

export function checkHard(text: string): number {
  return sameList(names(finalAnswer(text)), ["pell", "tess"]) ? 1 : 0;
}

// ---- human-sounding story --------------------------------------------------------------------

export const STORY_PROMPT = `Write a short story of about 150 words about someone who finds an old key in
the pocket of a second-hand coat. Write it the way a good human author would. Output only
the story.`;

/** Words and phrases that show up far more often in LLM prose than in human writing. */
export const AI_TELLS = [
  "delve", "tapestry", "testament", "symphony", "palpable", "intricate", "unwavering",
  "resonat", "beacon", "embark", "realm", "ethereal", "kaleidoscope", "labyrinth",
  "enigmatic", "solace", "bustling", "nestled", "vibrant", "whisper of", "whispers of",
  "a sense of", "the weight of", "in that moment", "little did", "couldn't help but",
  "a breath she didn't know", "a breath he didn't know", "a breath they didn't know",
  "it wasn't just", "not just a", "something shifted", "the air was thick", "elara",
  "seraphina", "lyra", "eldoria", "willowbrook", "once upon a time",
];

function countOf(hay: string, needle: string): number {
  let n = 0;
  for (let i = hay.indexOf(needle); i !== -1; i = hay.indexOf(needle, i + needle.length)) n++;
  return n;
}

/** Count AI-writing tells: em dashes (and dash look-alikes) plus stock phrases. */
export function storyTells(text: string): Record<string, number> {
  const t = clean(text);
  const low = t.toLowerCase().replaceAll("’", "'");
  const found: Record<string, number> = { dash: (t.match(/—| – | -- /g) ?? []).length };
  for (const phrase of AI_TELLS) found[phrase] = countOf(low, phrase);
  found["title/markdown"] = /^\s*(#|\*\*)/.test(t) ? 1 : 0;
  return Object.fromEntries(Object.entries(found).filter(([, v]) => v > 0));
}

/** 1 with no tells, minus 0.2 per tell. 0 if it isn't roughly a 150-word story. */
export function checkStory(text: string): number {
  const words = clean(text).split(/\s+/).filter(Boolean).length;
  if (words < 80 || words > 300) return 0;
  const tells = Object.values(storyTells(text)).reduce((a, b) => a + b, 0);
  return Math.max(0, 1 - 0.2 * tells);
}

export const TASKS: Task[] = [
  { name: "json", prompt: JSON_PROMPT, check: checkJson },
  { name: "riddle-easy", prompt: EASY_PROMPT, check: checkEasy },
  { name: "riddle-medium", prompt: MEDIUM_PROMPT, check: checkMedium },
  { name: "riddle-hard", prompt: HARD_PROMPT, check: checkHard },
  { name: "story", prompt: STORY_PROMPT, check: checkStory },
];
