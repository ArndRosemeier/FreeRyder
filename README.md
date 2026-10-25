# FreeRyder

Ride OpenRouter's free models without babysitting them.

OpenRouter's list of free models changes often; some are congested, some are weak.
FreeRyder lists the current free text models, benchmarks up to 10 random ones for
responsiveness and capability, uses the best one until it congests or disappears,
fails over to the next one, and rescans when it runs out. If nothing benchmarked is
healthy, it falls back to OpenRouter's own `openrouter/free` router.

Two implementations with the same behaviour and a shared ranking cache:

- [`python/`](python): Python 3.10+, no dependencies.
- [`typescript/`](typescript): Node 18+, no runtime dependencies.

Each is a library, a CLI and a local OpenAI-compatible proxy. The OpenRouter API key is
always passed in explicitly; FreeRyder never reads it from the environment. Using a
separate key with a small credit limit is a good idea: FreeRyder only calls free models,
so it should never spend anything.

## Python

```bash
cd python && pip install -e .
freeryder list                                   # currently free text models (no key needed)
freeryder --api-key "$KEY" scan                  # benchmark a random sample, save the ranking
freeryder --api-key "$KEY" ask "Hello there"     # one prompt to the current best model
freeryder --api-key "$KEY" serve --port 8765     # local OpenAI-compatible proxy
```

```python
from freeryder import FreeRyder
ryder = FreeRyder(api_key).start()   # returns at once; discovery runs in a background thread
print(ryder.complete("Summarise the plot of Hamlet in one line."))
res = ryder.chat({"messages": [...]})  # OpenAI-style payload, returns a CallResult
```

## TypeScript

```bash
cd typescript && npm install && npm run build
npx freeryder --api-key "$KEY" scan              # same commands as above
```

```ts
import { FreeRyder } from "freeryder";
const ryder = new FreeRyder({ apiKey }).start(); // returns at once; discovery runs in the background
console.log(await ryder.complete("Summarise the plot of Hamlet in one line."));
```

`start()` is the only setup call. Inference works right away: until the first discovery
finishes, requests go to the cached ranking from the last run or to `openrouter/free`.
Later rescans also run in the background, so inference never waits on discovery (unless
you set the fallback to none). `wait_ready()` / `waitReady()` blocks until discovery is
done if you prefer to wait.

## Proxy

With `serve` running, point any OpenAI client at it and use any model name
(e.g. `freeryder`); FreeRyder replaces it:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8765/v1", api_key="unused")
client.chat.completions.create(model="freeryder", messages=[{"role": "user", "content": "Hi"}])
```

`GET /status` shows the ranking and which models are cooling down.

## How it picks

1. **List**: `/models`, keeping free models whose only output is text (routers and
   safety classifiers excluded).
2. **Probe**: one tiny call per model drops the ones this account can't use at all
   (removed, agent-only, or blocked by your OpenRouter privacy settings).
3. **Benchmark** up to 10 random reachable models (previous top 3 always included)
   on 5 tasks, each scored 0 to 1:
   - **Strict JSON**: extract an order into an exact schema. Full marks only for raw JSON,
     half if it comes wrapped in a code fence.
   - **Three logic riddles** (easy, medium, hard). They are original, so a model can't
     have memorised them, and each answer was checked by brute force.
   - **A short story "written like a human"**, scored by counting AI-writing tells: em
     dashes, stock words and phrases ("tapestry", "testament", "a sense of", "Elara"...)
     and a markdown title. Each tell costs 0.2.

   Score = 0.6 × quality + 0.25 × reliability + 0.15 × speed. Models below 0.5 are
   never used.
4. **Ride** the top model. A model is put on a 10-minute cooldown after a 429, two
   failures in a row, or when its recent latency is over 3× its benchmark latency
   (and over 20 s). A model that returns 404/410 is dropped. Each request tries up to
   4 models before falling back to `openrouter/free`.
5. **Rescan** in the background when no ranked model is usable, or every 6 hours to pick
   up new models; `openrouter/free` covers requests until it finishes.
   The ranking is cached in `~/.cache/freeryder/state.json`.

All thresholds are in `Config` (Python) / `config` (TypeScript).

## Notes

- **Privacy setting**: many free endpoints log prompts for training. If your OpenRouter
  account disallows that (Settings → Privacy, "free model training"), those models
  return 404 and FreeRyder skips them.
- **Streaming**: the proxy accepts `stream: true` but sends the whole answer as one
  chunk, because failover needs the complete reply first.
- Free models are rate limited per account (OpenRouter's documented limits apply);
  a scan costs about 8 calls per benchmarked model plus one probe per free model.

## Tests

```bash
cd python && python -m pytest
cd typescript && npm test
```

The live tests call OpenRouter and run only when `OPEN_ROUTER_KEY_JUST_FOR_FREE_MODELS`
is set; the tests are the only place that environment variable is read.
