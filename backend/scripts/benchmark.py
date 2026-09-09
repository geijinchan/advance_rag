#!/usr/bin/env python3
"""Serving benchmark CLI (Part A): tokens/sec, TTFT and P95 latency.

Usage:
    python scripts/benchmark.py                          # defaults: 8x24
    python scripts/benchmark.py --concurrency 16 --requests 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.benchmark import BenchmarkRunner  # noqa: E402
from app.llm.service import ModelService  # noqa: E402


async def main_async(concurrency: int, requests: int) -> dict:
    models = ModelService()
    await models.refresh()
    runner = BenchmarkRunner(models)
    result = await runner.run(concurrency=concurrency, total_requests=requests)
    await models.aclose()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="vLLM/serving benchmark")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--requests", type=int, default=24)
    args = parser.parse_args()
    result = asyncio.run(main_async(args.concurrency, args.requests))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
