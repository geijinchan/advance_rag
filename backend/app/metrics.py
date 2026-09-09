"""Service metrics — lightweight in-process telemetry surfaced at GET /stats.

Counts every /ask (and /ask/stream) call by route, tracks end-to-end latency
distribution (p50/p95), fallback/retry rates, follow-up resolutions, feedback
approval, plus corpus/session/eval snapshots. Deliberately in-memory with a
restart-safe cumulative counter persisted alongside the session store —
production would use Prometheus.

Design notes:
* ``observe_ask`` is called once per completed question (from ask_with_session
  and the eval harness separately tags eval traffic so pinned runs do not
  pollute "live" usage numbers);
* ring-buffer of the last N latencies for p50/p95 without unbounded growth;
* ``snapshot`` merges everything with serving/corpus context for the UI.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from datetime import datetime, timezone
from typing import Any

from app.persist import load_json, save_json

MAX_LATENCIES = 500
RECENT_QUESTIONS = 12


class MetricsCollector:
    def __init__(self, persist_path: str | None = None) -> None:
        self._persist_path = persist_path
        self._started_at = time.time()
        self._route_counts: Counter[str] = Counter()
        self._latencies: deque[float] = deque(maxlen=MAX_LATENCIES)
        self._fallbacks = 0
        self._retries = 0
        self._followups = 0
        self._streamed = 0
        self._recent: deque[dict[str, Any]] = deque(maxlen=RECENT_QUESTIONS)
        self._http_requests = 0
        self._cache_hits = 0
        self._cache_served = 0

        saved = load_json(persist_path) if persist_path else None
        if isinstance(saved, dict):
            # cumulative counters survive restarts (sessions do too, so this
            # keeps /stats consistent with the visible history)
            self._route_counts.update({k: int(v) for k, v in saved.get("routes", {}).items()})
            self._fallbacks = int(saved.get("fallbacks", 0))
            self._retries = int(saved.get("retries", 0))
            self._followups = int(saved.get("followups", 0))
            self._streamed = int(saved.get("streamed", 0))

    # ------------------------------------------------------------- recording
    def observe_ask(
        self,
        *,
        question: str,
        route: str,
        latency_ms: float,
        fallback: bool = False,
        retries: int = 0,
        follow_up: bool = False,
        streamed: bool = False,
        session_id: str | None = None,
        eval_run: bool = False,
        cache_hit: bool = False,
    ) -> None:
        if eval_run:
            return  # pinned eval runs are not user traffic
        self._route_counts[route] += 1
        self._latencies.append(float(latency_ms))
        if fallback:
            self._fallbacks += 1
        self._retries += max(0, int(retries))
        if follow_up:
            self._followups += 1
        if streamed:
            self._streamed += 1
        if cache_hit:
            self._cache_hits += 1
        self._cache_served += 1
        self._recent.append(
            {
                "question": question[:120],
                "route": route,
                "latency_ms": round(latency_ms, 1),
                "fallback": fallback,
                "cache_hit": cache_hit,
                "ts": round(time.time(), 1),
                "session_id": session_id,
            }
        )
        self._persist()

    def observe_http(self) -> None:
        self._http_requests += 1

    def observe_eval(self, report: dict[str, Any]) -> None:
        # kept in-memory only; eval reports are fetched via /eval/results
        self._last_eval = {
            "accuracy": report.get("accuracy"),
            "faithfulness_avg": report.get("faithfulness_avg"),
            "ran_at": report.get("ran_at"),
            "passed": report.get("passed"),
            "total": report.get("total_cases"),
        }

    _last_eval: dict[str, Any] | None = None
    _last_benchmark: dict[str, Any] | None = None

    def observe_benchmark(self, result: dict[str, Any]) -> None:
        self._last_benchmark = {
            "requests": result.get("total_requests"),
            "ok": result.get("successful_requests"),
            "tokens_per_sec": result.get("tokens_per_sec"),
            "p95_ms": result.get("latency_p95_ms"),
        }

    # ---------------------------------------------------------------- readout
    def prometheus_snapshot(self) -> dict[str, Any]:
        """Raw internals for the Prometheus exporter (/metrics).

        Kept separate from `snapshot` (the UI-facing dict) so the exposition
        format can never drift from what /stats renders — same counters, same
        latency ring buffer, just unwrapped.
        """
        total_q = sum(self._route_counts.values())
        return {
            "routes": dict(self._route_counts),
            "fallbacks": self._fallbacks,
            "retries": self._retries,
            "followups": self._followups,
            "streamed": self._streamed,
            "cache_hits": self._cache_hits,
            "cache_served": self._cache_served,
            "cache_hit_ratio": round(self._cache_hits / self._cache_served, 4)
            if self._cache_served
            else 0.0,
            "latencies_seconds": [ms / 1000.0 for ms in self._latencies],
            "uptime_seconds": time.time() - self._started_at,
            "total_questions": total_q,
            "last_eval": dict(self._last_eval) if self._last_eval else None,
            "last_benchmark": dict(self._last_benchmark) if self._last_benchmark else None,
        }

    def _p(self, p: float) -> float:
        if not self._latencies:
            return 0.0
        s = sorted(self._latencies)
        k = (len(s) - 1) * (p / 100.0)
        f = int(k)
        c = min(f + 1, len(s) - 1)
        return round(s[f] + (s[c] - s[f]) * (k - f), 1)

    def snapshot(self, *, sessions: dict | None = None, feedback: dict | None = None,
                 corpus: dict | None = None, serving: dict | None = None,
                 cache: dict | None = None) -> dict[str, Any]:
        total_q = sum(self._route_counts.values())
        uptime_s = time.time() - self._started_at
        return {
            "service": {
                "uptime_seconds": round(uptime_s, 1),
                "uptime_human": _human_duration(uptime_s),
                "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            "questions": {
                "total": total_q,
                "by_route": dict(self._route_counts),
                "fallbacks": self._fallbacks,
                "fallback_rate": round(self._fallbacks / total_q, 4) if total_q else None,
                "query_rewrites": self._retries,
                "follow_ups_resolved": self._followups,
                "streamed_answers": self._streamed,
                "cache_hits": self._cache_hits,
                "cache_hit_rate": round(self._cache_hits / self._cache_served, 4)
                if self._cache_served
                else None,
                "latency_avg_ms": round(sum(self._latencies) / len(self._latencies), 1)
                if self._latencies
                else None,
                "latency_p50_ms": self._p(50),
                "latency_p95_ms": self._p(95),
                "recent": list(self._recent),
            },
            "sessions": sessions or {},
            "feedback": feedback or {},
            "corpus": corpus or {},
            "serving": serving or {},
            "cache": cache or {},
            "last_eval": self._last_eval,
            "last_benchmark": self._last_benchmark,
        }

    # ------------------------------------------------------------ persistence
    def _persist(self) -> None:
        if not self._persist_path:
            return
        save_json(
            self._persist_path,
            {
                "routes": dict(self._route_counts),
                "fallbacks": self._fallbacks,
                "retries": self._retries,
                "followups": self._followups,
                "streamed": self._streamed,
            },
        )


def _human_duration(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60}s"
    return f"{s // 3600}h {(s % 3600) // 60}m"
