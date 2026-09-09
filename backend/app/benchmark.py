"""Serving benchmark (Part A): concurrent load against the active backend.

Measures per-request TTFT (time-to-first-token), end-to-end latency, and
aggregate tokens/sec. In vLLM mode this exercises the real streaming API
of the served model — the numbers in the README come from this harness.
In simulation mode it reports the deterministic engine's timings, which
are useful only for plumbing verification (README states this plainly).
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field

from app.llm.base import ChatMessage

logger = logging.getLogger(__name__)

_SAMPLE_QUESTIONS = [
    "What is the warranty period of the X200 controller?",
    "Which firmware version fixed the MODBUS-TCP timeout bug?",
    "What is the operating temperature range of the S200 sensor?",
    "How long does an RMA evaluation take?",
    "What safety standard does the X300 comply with?",
    "Which TCP port does MODBUS-TCP use?",
    "How often should relay contacts be inspected?",
    "What is the accuracy class of the S350 pressure sensor?",
]


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(p / 100 * len(ordered)) - 1))
    return round(ordered[idx], 1)


@dataclass
class RequestStats:
    ttft_ms: float = 0.0
    latency_ms: float = 0.0
    tokens: int = 0
    ok: bool = True
    error: str = ""


@dataclass
class BenchmarkResult:
    mode: str
    concurrency: int
    total_requests: int
    successful: int = 0
    failed: int = 0
    ttft: list[float] = field(default_factory=list)
    latency: list[float] = field(default_factory=list)
    tokens: list[int] = field(default_factory=list)
    duration_s: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_response(self) -> dict:
        total_tokens = sum(self.tokens)
        return {
            "mode": self.mode,
            "concurrency": self.concurrency,
            "total_requests": self.total_requests,
            "successful_requests": self.successful,
            "failed_requests": self.failed,
            "ttft_p50_ms": _percentile(self.ttft, 50),
            "ttft_p95_ms": _percentile(self.ttft, 95),
            "latency_p50_ms": _percentile(self.latency, 50),
            "latency_p95_ms": _percentile(self.latency, 95),
            "tokens_per_sec": round(total_tokens / self.duration_s, 1) if self.duration_s else 0.0,
            "total_tokens": total_tokens,
            "duration_s": round(self.duration_s, 2),
            "notes": self.notes,
        }


class BenchmarkRunner:
    def __init__(self, models) -> None:
        self.models = models
        self.last_result: dict | None = None

    async def run(self, concurrency: int, total_requests: int, question: str | None = None) -> dict:
        result = BenchmarkResult(mode=self.models.mode, concurrency=concurrency,
                                  total_requests=total_requests)
        semaphore = asyncio.Semaphore(concurrency)
        questions = [question] * total_requests if question else None
        t0 = time.perf_counter()

        async def one(i: int) -> None:
            q = questions[i] if questions else _SAMPLE_QUESTIONS[i % len(_SAMPLE_QUESTIONS)]
            stats = RequestStats()
            start = time.perf_counter()
            try:
                async with semaphore:
                    messages = [
                        ChatMessage.system(
                            "Answer the question in one concise sentence using only "
                            "these context passages."
                        ),
                        ChatMessage.user(f"Question: {q}"),
                    ]
                    token_count = 0
                    async for ev in self.models.backend.chat_stream(messages, max_tokens=256):
                        if "ttft_ms" in ev:
                            stats.ttft_ms = float(ev["ttft_ms"])
                        token = ev.get("token", "")
                        if token:
                            token_count += len(token.split())
                    stats.tokens = max(token_count, 1)
            except Exception as exc:  # noqa: BLE001 — benchmark reports failures, never crashes
                stats.ok = False
                stats.error = f"{type(exc).__name__}: {exc}"
            stats.latency_ms = round((time.perf_counter() - start) * 1000, 1)
            if stats.ok:
                result.successful += 1
                result.ttft.append(stats.ttft_ms)
                result.latency.append(stats.latency_ms)
                result.tokens.append(stats.tokens)
            else:
                result.failed += 1
                logger.warning("benchmark request failed: %s", stats.error)

        await asyncio.gather(*(one(i) for i in range(total_requests)))
        result.duration_s = time.perf_counter() - t0
        if result.mode == "simulation":
            result.notes.append(
                "Simulation mode: numbers reflect the deterministic offline engine, "
                "not a GPU-served vLLM instance. Run with `docker compose up vllm` "
                "for real serving benchmarks."
            )
        self.last_result = result.to_response()
        return self.last_result
