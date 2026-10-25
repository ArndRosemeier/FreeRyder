# FreeRyder

Ride OpenRouter's free models without babysitting them.

OpenRouter's list of free models changes often; some are congested, some are weak.
FreeRyder lists the current free text models, benchmarks up to 10 random ones for
responsiveness and capability, uses the best one until it congests or disappears,
fails over to the next one, and rescans when it runs out. If nothing benchmarked is
healthy, it falls back to OpenRouter's own `openrouter/free` router.

Pure Python 3.10+, no dependencies.

## Setup

```bash
pip install -e .
export OPEN_ROUTER_KEY_JUST_FOR_FREE_MODELS=...
```

The key is taken from the first of `OPEN_ROUTER_KEY_JUST_FOR_FREE_MODELS`,
`OPENROUTER_API_KEY` and `OPEN_ROUTER_KEY` that is set. Use a separate key with a small
credit limit: FreeRyder only calls free models, so it should never spend anything.

## Use

```bash
freeryder list                 # currently free text models
freeryder scan                 # benchmark a random sample, save the ranking
freeryder ask "Hello there"    # one prompt to the current best model
freeryder serve --port 8765    # local OpenAI-compatible proxy
```

With the proxy running, point any OpenAI client at it and use any model name
(e.g. `freeryder`); FreeRyder replaces it:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8765/v1", api_key="unused")
client.chat.completions.create(model="freeryder", messages=[{"role": "user", "content": "Hi"}])
```

`GET /status` shows the ranking and which models are cooling down.

As a library:

```python
from freeryder import FreeRyder
ryder = FreeRyder()
print(ryder.complete("Summarise the plot of Hamlet in one line."))
```

## How it picks

1. **List**: `/models`, keeping free models whose only output is text (routers and
   safety classifiers excluded).
2. **Probe**: one tiny call per model drops the ones this account can't use at all
   (removed, agent-only, or blocked by your OpenRouter privacy settings).
3. **Benchmark** up to 10 random reachable models (previous top 3 always included)
   on 8 short tasks with checkable answers: arithmetic, time, letter counting, logic,
   string reversal, code tracing, exact instruction following, and strict JSON.
   Score = 0.6 × quality + 0.25 × reliability + 0.15 × speed. Models below 0.5 are
   never used.
4. **Ride** the top model. A model is put on a 10-minute cooldown after a 429, two
   failures in a row, or when its recent latency is over 3× its benchmark latency
   (and over 20 s). A model that returns 404/410 is dropped. Each request tries up to
   4 models before falling back to `openrouter/free`.
5. **Rescan** when no ranked model is usable, or every 6 hours to pick up new models.
   The ranking is cached in `~/.cache/freeryder/state.json`.

All thresholds are in `freeryder.Config`.

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
python -m pytest
```
