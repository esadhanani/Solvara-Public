"""Run a small demonstration or reproducible offline benchmark."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import platform
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .providers import OllamaProvider, synthetic_providers
from .router import Policy, Router, RoutingError


def percentile(values: list[float], fraction: float) -> float | None:
    """Nearest-rank percentile, including all route outcomes."""
    return sorted(values)[max(0, math.ceil(fraction * len(values)) - 1)] if values else None


async def benchmark(repeats: int) -> dict:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    rows = []
    scenarios = {
        "healthy": {},
        "fast_provider_error": {"fail_fast": True},
        "fast_provider_timeout": {"slow_fast": True},
    }
    for scenario, kwargs in scenarios.items():
        for goal in ("latency", "cost", "quality"):
            for index in range(repeats):
                router = Router(synthetic_providers(**kwargs))
                policy = Policy(goal=goal, attempt_timeout_s=0.06, request_timeout_s=0.2)
                base = {"scenario": scenario, "goal": goal, "iteration": index}
                try:
                    result = await router.route(f"synthetic question {index}", "benchmark", policy)
                except RoutingError as exc:
                    rows.append({**base, "status": "route_failure", "request_wall_ms": exc.request_wall_ms,
                                 "candidates": [asdict(c) for c in exc.candidates]})
                else:
                    rows.append({**base, "status": "success", **result.to_dict()})
    wall = [row["request_wall_ms"] for row in rows]
    selected_wall = [row["selected"]["wall_ms"] for row in rows if row["status"] == "success"]
    return {
        "schema_version": 1,
        "mode": "synthetic_offline",
        "notes": [
            "No model inference, GPU measurement, or real provider pricing.",
            "Delay, cost, and quality values are hand-written fixtures.",
            "Selection inputs are deterministic; measured wall times vary by host load.",
            "Request wall time includes all candidate waits, timeouts, retries, and selection.",
            "Observed cost includes successful outputs only, not failed or cancelled remote work.",
        ],
        "environment": {"python": platform.python_version(), "system": platform.system()},
        "summary": {
            "requests": len(rows),
            "route_failures": sum(row["status"] != "success" for row in rows),
            "provider_failures": sum(c["status"] != "success" for row in rows for c in row["candidates"]),
            "selected_providers": dict(Counter(row["selected"]["provider"] for row in rows
                                              if row["status"] == "success")),
            "request_wall_ms": {"p50": percentile(wall, 0.5), "p95": percentile(wall, 0.95)},
            "selected_provider_wall_ms": {"p50": percentile(selected_wall, 0.5),
                                          "p95": percentile(selected_wall, 0.95)},
        },
        "requests": rows,
    }


async def run(args: argparse.Namespace) -> dict:
    if args.command == "benchmark":
        return await benchmark(args.repeats)
    if args.command == "ollama":
        providers = [OllamaProvider(model, model, args.endpoint, args.timeout) for model in args.models]
        policy = Policy(goal="latency", attempt_timeout_s=args.timeout,
                        request_timeout_s=args.timeout)
    else:
        providers = synthetic_providers()
        policy = Policy(goal=args.goal)
    return (await Router(providers).route(args.prompt, "cli", policy)).to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Solvara routing evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="offline synthetic example")
    demo.add_argument("--goal", choices=("latency", "cost", "quality"), default="latency")
    demo.add_argument("--prompt", default="Summarise the supplied evidence")
    bench = sub.add_parser("benchmark", help="offline synthetic benchmark")
    bench.add_argument("--repeats", type=int, default=5)
    ollama = sub.add_parser("ollama", help="call explicitly selected Ollama models")
    ollama.add_argument("--models", nargs="+", required=True)
    ollama.add_argument("--endpoint", default="http://127.0.0.1:11434")
    ollama.add_argument("--timeout", type=float, default=30.0)
    ollama.add_argument("--prompt", required=True)
    for child in (demo, bench, ollama):
        child.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = asyncio.run(run(args))
    except (ValueError, RoutingError) as exc:
        parser.exit(1, f"{type(exc).__name__}: {exc}\n")
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        print(f"Wrote {args.output}")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
