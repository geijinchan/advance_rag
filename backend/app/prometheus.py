"""Prometheus exposition — /metrics endpoint (ops observability, Part F+).

Renders the service telemetry in the Prometheus text exposition format
(text/plain; version=0.0.4) so a real Prometheus server (or `curl`) can
scrape the exact numbers /stats reports, plus per-endpoint HTTP counters
collected by a raw-ASGI middleware (label cardinality is bounded by using
the *route pattern*, not the raw path).

Design notes:
* Counters that survive restarts come from MetricsCollector's persisted
  totals; the HTTP request counter and latency histogram are in-process
  (a Prometheus server is the durable store in production — this is the
  documented deployment path).
* The latency histogram re-uses the same ring buffer /stats reports, so
  both surfaces can never disagree.
* Gauges (corpus size, sessions, cache, eval, benchmark) are rendered at
  scrape time from live singletons — no staleness.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

# ---------------------------------------------------------------- middleware


class HTTPMetricsMiddleware:
    """Raw ASGI middleware counting responses as http_requests_total.

    Label cardinality stays bounded: the *route pattern* (e.g.
    "/source/{chunk_id}") is used when Starlette matched one (it lands in
    scope["route"] during routing, i.e. before response.start reaches us);
    404s collapse to "unmatched"; anything else falls back to the raw path
    (rare: responses emitted before routing).

    State lives at class level so the /metrics endpoint can read the same
    counters the middleware writes, regardless of how Starlette instantiates
    the wrapper (it builds the ASGI stack at startup). The asyncio event
    loop serialises request handling between awaits, so the plain Counter
    updates are race-free in practice.
    """

    counts: Counter[tuple[str, str, int]] = Counter()
    _last_scrape: dict[str, float] = {"duration_ms": 0.0}

    def __init__(self, app, **kwargs: Any) -> None:
        self.app = app

    @classmethod
    def note_scrape(cls, duration_ms: float) -> None:
        cls._last_scrape["duration_ms"] = round(duration_ms, 1)

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        method = scope.get("method", "?")

        async def send_wrapper(message) -> None:
            if message["type"] == "http.response.start":
                status = message["status"]
                route = scope.get("route")
                path = getattr(route, "path", None)
                if not path:
                    path = "unmatched" if status == 404 else scope.get("path", "unknown")
                HTTPMetricsMiddleware.counts[(method, path, status)] += 1
            await send(message)

        await self.app(scope, receive, send_wrapper)


# ---------------------------------------------------------------- rendering

# seconds — spans 5 ms (cache-hit simulation) to 2.5 s (vLLM long form)
_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)


def _escape_label(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace('"', '\\"')
    )


def _labels(**pairs: Any) -> str:
    if not pairs:
        return ""
    inner = ",".join(f'{k}="{_escape_label(str(v))}"' for k, v in pairs.items())
    return "{" + inner + "}"


def _counter(name: str, help_: str, samples: dict[str, float]) -> list[str]:
    lines = [f"# HELP {name} {help_}", f"# TYPE {name} counter"]
    for labels_value, value in sorted(samples.items()):
        lines.append(f"{name}{labels_value} {value}")
    return lines


def _counter_value(name: str, help_: str, value: float) -> list[str]:
    """Label-free counter (single sample)."""
    return [
        f"# HELP {name} {help_}",
        f"# TYPE {name} counter",
        f"{name} {value}",
    ]


def _gauge(name: str, help_: str, value: float, labels_value: str = "") -> list[str]:
    return [
        f"# HELP {name} {help_}",
        f"# TYPE {name} gauge",
        f"{name}{labels_value} {value}",
    ]


def render_metrics(*, collector, http_middleware: type[HTTPMetricsMiddleware],
                   gauges: dict[str, Any]) -> str:
    """Full exposition document.

    `gauges` carries the app-level context: corpus, sessions, feedback,
    cache, serving, last eval / benchmark, version.
    """
    lines: list[str] = []

    # ---- build info + serving mode -----------------------------------
    version = gauges.get("version", "unknown")
    mode = gauges.get("serving_mode", "unknown")
    model = gauges.get("serving_model", "")
    lines += _gauge(
        "rag_build_info",
        "Service build metadata (version, serving mode, model).",
        1,
        _labels(version=version, mode=mode, model=model),
    )
    lines += _gauge("rag_uptime_seconds", "Process uptime in seconds.",
                    gauges.get("uptime_seconds", 0.0))

    # ---- ask counters (persisted totals; in-process detail) ----------
    raw = collector.prometheus_snapshot()
    lines += _counter(
        "rag_ask_total",
        "Questions answered by route (excludes pinned eval runs).",
        {
            _labels(route=route): count
            for route, count in raw["routes"].items()
        },
    )
    lines += _counter_value(
        "rag_ask_fallback_total",
        "Questions that ended in the honest fallback path.",
        raw["fallbacks"],
    )
    lines += _counter_value(
        "rag_ask_rewrites_total",
        "Query rewrites performed by the grader→rewriter retry loop.",
        raw["retries"],
    )
    lines += _counter_value(
        "rag_ask_followup_total",
        "Follow-up questions resolved with inherited entities.",
        raw["followups"],
    )
    lines += _counter_value(
        "rag_ask_streamed_total",
        "Answers delivered through the SSE stream endpoint.",
        raw["streamed"],
    )
    lines += _counter_value(
        "rag_cache_hits_total",
        "Semantic answer cache hits served.",
        raw["cache_hits"],
    )
    lines += _counter_value(
        "rag_cache_lookups_total",
        "Semantic answer cache lookups (hits + misses).",
        raw["cache_served"],
    )
    lines += _gauge(
        "rag_cache_hit_ratio",
        "Share of ask requests served from the semantic answer cache.",
        raw["cache_hit_ratio"],
    )

    # ---- latency histogram (seconds, matches /stats ring buffer) ------
    latencies_s = raw["latencies_seconds"]
    lines += [
        "# HELP rag_ask_latency_seconds End-to-end ask latency in seconds.",
        "# TYPE rag_ask_latency_seconds histogram",
    ]
    cumulative = 0
    for bound in _LATENCY_BUCKETS:
        cumulative = sum(1 for v in latencies_s if v <= bound)
        lines.append(
            f"rag_ask_latency_seconds_bucket{{le=\"{bound}\"}} {cumulative}"
        )
    lines.append(f'rag_ask_latency_seconds_bucket{{le="+Inf"}} {len(latencies_s)}')
    lines.append(f"rag_ask_latency_seconds_sum {round(sum(latencies_s), 6)}")
    lines.append(f"rag_ask_latency_seconds_count {len(latencies_s)}")

    # ---- HTTP traffic --------------------------------------------------
    http_samples = {
        _labels(method=method, path=path, status=status): count
        for (method, path, status), count in http_middleware.counts.items()
    }
    lines += _counter(
        "http_requests_total",
        "HTTP requests processed by status and route pattern.",
        http_samples,
    )
    lines += _gauge(
        "rag_metrics_scrape_duration_ms",
        "Duration of the most recent /metrics scrape in milliseconds.",
        http_middleware._last_scrape["duration_ms"],
    )

    # ---- corpus / index --------------------------------------------------
    corpus = gauges.get("corpus", {})
    lines += _gauge("rag_corpus_documents", "Indexed documents.", corpus.get("documents", 0))
    lines += _gauge("rag_corpus_chunks", "Indexed chunks.", corpus.get("chunks", 0))
    lines += _gauge("rag_embedding_dim", "Embedding vector dimensionality.",
                    corpus.get("embedding_dim", 0))

    # ---- sessions / feedback / cache state --------------------------------
    lines += _gauge("rag_sessions_total", "Persisted conversation sessions.",
                    gauges.get("sessions", 0))
    feedback = gauges.get("feedback", {})
    lines += _counter(
        "rag_feedback_total",
        "Answer ratings by kind.",
        {
            _labels(rating="up"): feedback.get("up", 0),
            _labels(rating="down"): feedback.get("down", 0),
        },
    )
    cache = gauges.get("cache", {})
    lines += _gauge("rag_cache_entries", "Semantic answer cache entries.",
                    cache.get("entries", 0))
    lines += _gauge("rag_cache_capacity", "Semantic answer cache capacity.",
                    cache.get("capacity", 0))

    # ---- cross-encoder rerank stage ---------------------------------------
    rerank = gauges.get("rerank") or {}
    if rerank:
        lines += _gauge(
            "rag_rerank_info",
            "Cross-encoder rerank stage metadata (mode, pool size).",
            1,
            _labels(mode=rerank.get("mode", "lexical"), n=rerank.get("n", 0)),
        )
        lines += _counter_value(
            "rag_rerank_candidates_total",
            "(Query, chunk) pairs cross-encoded by the rerank stage.",
            rerank.get("calls", 0),
        )
        lines += _counter_value(
            "rag_rerank_reorders_total",
            "Fusion candidates whose final position changed after cross-encoding.",
            rerank.get("reorders", 0),
        )
        lines += _gauge(
            "rag_rerank_latency_avg_ms",
            "Average rerank-stage latency in milliseconds.",
            rerank.get("avg_latency_ms", 0.0),
        )

    # ---- golden-set growth (feedback-driven eval coverage) ----------------
    golden = gauges.get("golden") or {}
    if golden:
        lines += _counter(
            "rag_golden_set_cases",
            "Golden-set eval cases by source (base vs user-grown).",
            {
                _labels(source="base"): golden.get("base", 0),
                _labels(source="user"): golden.get("user", 0),
            },
        )
    ledger = gauges.get("ledger") or {}
    if ledger:
        lines += _gauge(
            "rag_answer_ledger_entries",
            "Answer-ledger entries (request_id → answered content, joinable by feedback).",
            ledger.get("entries", 0),
        )
    eval_runs = gauges.get("eval_runs") or {}
    if eval_runs:
        lines += _counter_value(
            "rag_eval_runs_total",
            "Persisted golden-set evaluation runs (incl. ablation-tagged runs).",
            eval_runs.get("total", 0),
        )
        if eval_runs.get("last_accuracy") is not None:
            lines += _gauge(
                "rag_eval_last_run_accuracy",
                "Accuracy of the most recent persisted golden-set eval run.",
                eval_runs["last_accuracy"],
            )
    triage = gauges.get("triage") or {}
    if triage:
        lines += _gauge(
            "rag_triage_open",
            "Currently open (unresolved) downvote triage items.",
            triage.get("open", 0),
        )
        lines += _gauge(
            "rag_triage_resolved",
            "Resolved downvote triage items (marked fixed).",
            triage.get("resolved", 0),
        )
        if triage.get("oldest_open_hours") is not None:
            lines += _gauge(
                "rag_triage_oldest_open_hours",
                "Hours the oldest open (unresolved) triage item has waited (fix-list SLA).",
                triage["oldest_open_hours"],
            )

    # ---- eval / benchmark quality gauges ----------------------------------
    ev = gauges.get("last_eval") or {}
    if ev.get("accuracy") is not None:
        lines += _gauge("rag_eval_accuracy", "Golden-set accuracy (fraction passed).",
                        ev["accuracy"])
    if ev.get("faithfulness_avg") is not None:
        lines += _gauge("rag_eval_faithfulness",
                        "Average answer faithfulness from the last /eval run.",
                        ev["faithfulness_avg"])
    bm = gauges.get("last_benchmark") or {}
    if bm.get("tokens_per_sec") is not None:
        lines += _gauge("rag_benchmark_tokens_per_sec",
                        "Generation throughput from the last /benchmark run.",
                        bm["tokens_per_sec"])
    if bm.get("p95_ms") is not None:
        lines += _gauge("rag_benchmark_latency_p95_ms",
                        "p95 completion latency (ms) from the last /benchmark run.",
                        bm["p95_ms"])

    lines.append("")
    return "\n".join(lines)
