"""Command line: freeryder list | scan | ask | serve."""

from __future__ import annotations

import argparse
import logging
import sys

from .catalog import free_text_models
from .client import OpenRouterClient
from .picker import Config, FreeRyder


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="freeryder",
                                description="Benchmark and use the best free OpenRouter model.")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--sample", type=int, default=10, help="models benchmarked per scan (default 10)")
    p.add_argument("--min-score", type=float, default=0.5)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list currently free text models")
    sub.add_parser("scan", help="benchmark a random sample and save the ranking")
    ask = sub.add_parser("ask", help="send one prompt to the current best model")
    ask.add_argument("prompt")
    srv = sub.add_parser("serve", help="run the local OpenAI-compatible proxy")
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8765)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    client = OpenRouterClient()
    if not client.api_key and args.cmd != "list":
        print("Set OPEN_ROUTER_KEY_JUST_FOR_FREE_MODELS (or OPENROUTER_API_KEY).", file=sys.stderr)
        return 2

    if args.cmd == "list":
        for m in free_text_models(client.list_models()):
            print(f"{m['id']:<55} ctx {m.get('context_length')}")
        return 0

    ryder = FreeRyder(client, Config(sample_size=args.sample, min_score=args.min_score))
    if args.cmd == "scan":
        reports = ryder.scan()
        print("score  model" + " " * 52 + "details")
        for r in reports:
            print(r.summary())
        print(f"\nusing: {ryder.current or 'none (no model passed min score)'}")
    elif args.cmd == "ask":
        res = ryder.chat({"messages": [{"role": "user", "content": args.prompt}]})
        if not res.ok:
            print(f"failed: {res.status} {res.error}", file=sys.stderr)
            return 1
        print(f"[{res.body.get('model')}]", file=sys.stderr)
        print(res.text)
    elif args.cmd == "serve":
        from .proxy import serve
        if ryder.current is None:
            ryder.scan()
        serve(ryder, args.host, args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
