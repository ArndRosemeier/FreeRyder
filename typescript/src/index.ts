/** FreeRyder: find, benchmark and ride the best free OpenRouter model. */

export { Benchmark, DEFAULT_WEIGHTS, ModelReport, TASKS, type Task, type Weights, finalAnswer } from "./bench.js";
export { freeTextModels } from "./catalog.js";
export { CallResult, OpenRouterClient, type ChatClient, type Model } from "./client.js";
export { DEFAULT_CONFIG, FreeRyder, NoModelAvailable, type Config, type FreeRyderOptions } from "./picker.js";
export { createProxy, serve } from "./proxy.js";
