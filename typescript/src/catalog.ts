/** Find OpenRouter's currently free text models. */

import type { Model } from "./client.js";

// Routers and safety classifiers are free but not general chat models.
const EXCLUDE_PREFIXES = ["openrouter/"];
const EXCLUDE_SUBSTRINGS = ["content-safety", "guard", "safeguard"];

export function isFree(m: Model): boolean {
  const p = m.pricing ?? {};
  const zero = ["prompt", "completion", "request"].every((k) => ["0", "0.0"].includes(String(p[k] ?? "0")));
  return m.id.endsWith(":free") || (zero && "prompt" in p);
}

export function isTextChat(m: Model): boolean {
  const out = m.architecture?.output_modalities ?? ["text"];
  return out.length === 1 && out[0] === "text";
}

export function freeTextModels(models: Model[]): Model[] {
  return models.filter(
    (m) =>
      !EXCLUDE_PREFIXES.some((p) => m.id.startsWith(p)) &&
      !EXCLUDE_SUBSTRINGS.some((s) => m.id.includes(s)) &&
      isFree(m) &&
      isTextChat(m),
  );
}
