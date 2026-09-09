"""Grafana dashboard provisioning for the exposed Prometheus families.

``GET /ops/grafana`` returns an importable Grafana dashboard JSON (Grafana ≥9,
Prometheus datasource) covering the rag_* metric families the service
exposes at ``/metrics``:

* ask traffic: rate by route, fallbacks, rewrites, follow-ups, streamed answers
* latency: p50/p95/p99 from the rag_ask_latency_seconds histogram buckets
* answer cache: hit ratio, entries vs capacity
* quality: golden-set accuracy + faithfulness after each /eval run
* corpus + sessions + feedback counters
* per-endpoint HTTP traffic

The JSON is also written to download/ as an artifact (repo copy) — the same
document a `docker-compose` stack would mount into Grafana's provisioning
directory.
"""

from __future__ import annotations

from typing import Any


def _target(expr: str, legend: str, ref: str = "A", instant: bool = False) -> dict:
    return {
        "refId": ref,
        "expr": expr,
        "legendFormat": legend,
        "instant": instant,
    }


def _panel(pid: int, title: str, ptype: str, targets: list[dict],
           unit: str = "short", grid: dict | None = None,
           left_y: dict | None = None, description: str = "") -> dict:
    field_config = {
        "defaults": {"unit": unit},
        "overrides": [],
    }
    if left_y is not None:
        field_config = {
            "defaults": {"unit": unit, "custom": left_y},
            "overrides": [],
        }
    return {
        "id": pid,
        "title": title,
        "type": ptype,
        "description": description,
        "targets": targets,
        "fieldConfig": field_config,
        "gridPos": grid or {"h": 8, "w": 12, "x": 0, "y": 0},
        "options": (
            {"legend": {"displayMode": "list", "placement": "bottom",
                        "showLegend": True},
             "tooltip": {"mode": "multi", "sort": "desc"}}
            if ptype == "timeseries" else
            {"orientation": "auto", "reduceOptions": {"calcs": ["lastNotNull"],
                                                       "fields": "",
                                                       "values": False},
             "showThresholdLabels": False, "showThresholdMarkers": True}
        ),
    }


def build_dashboard() -> dict[str, Any]:
    latency = 'histogram_quantile(0.{q}, sum(rate(rag_ask_latency_seconds_bucket[5m])) by (le))'
    panels: list[dict] = [
        _panel(
            1, "Ask traffic by route", "timeseries",
            [_target('sum(rate(rag_ask_total[5m])) by (route)', "{{route}}")],
            unit="reqps",
            grid={"h": 8, "w": 8, "x": 0, "y": 0},
            description="Questions per second routed to document/out_of_scope/chitchat.",
        ),
        _panel(
            2, "Answer latency (quantiles)", "timeseries",
            [
                _target(latency.format(q=50), "p50", "A"),
                _target(latency.format(q=95), "p95", "B"),
                _target(latency.format(q=99), "p99", "C"),
            ],
            unit="s",
            grid={"h": 8, "w": 8, "x": 8, "y": 0},
            description="Full agent-graph latency quantiles from the ask histogram.",
        ),
        _panel(
            3, "Answer cache hit ratio", "gauge",
            [_target("rag_cache_hit_ratio", "hit ratio", "A", instant=True)],
            unit="percentunit",
            grid={"h": 8, "w": 8, "x": 16, "y": 0},
            description="Semantic answer cache hit ratio (grounded answers served from cache).",
        ),
        _panel(
            4, "Honesty signals: fallbacks, rewrites, follow-ups, streamed",
            "timeseries",
            [
                _target("sum(rate(rag_ask_fallback_total[5m]))", "fallbacks", "A"),
                _target("sum(rate(rag_ask_rewrites_total[5m]))", "rewrites", "B"),
                _target("sum(rate(rag_ask_followup_total[5m]))", "follow-ups", "C"),
                _target("sum(rate(rag_ask_streamed_total[5m]))", "streamed", "D"),
            ],
            unit="ops",
            grid={"h": 8, "w": 12, "x": 0, "y": 8},
            description="Agentic-loop counters — fallbacks should stay near zero on in-domain traffic.",
        ),
        _panel(
            5, "HTTP traffic by endpoint", "timeseries",
            [_target("sum(rate(http_requests_total[5m])) by (path)", "{{path}}")],
            unit="reqps",
            grid={"h": 8, "w": 12, "x": 12, "y": 8},
            description="Gateway-visible request rate per route pattern (cardinality-bounded).",
        ),
        _panel(
            6, "Golden-set accuracy & faithfulness", "stat",
            [
                _target("rag_eval_accuracy", "accuracy", "A", instant=True),
                _target("rag_eval_faithfulness", "faithfulness", "B", instant=True),
            ],
            unit="percentunit",
            grid={"h": 6, "w": 8, "x": 0, "y": 16},
            description="Updated after each POST /eval run (16 pinned cases incl. multi-hop).",
        ),
        _panel(
            7, "Answer cache pressure", "timeseries",
            [
                _target("rag_cache_entries", "entries", "A"),
                _target("rag_cache_capacity", "capacity", "B"),
            ],
            unit="short",
            grid={"h": 6, "w": 8, "x": 8, "y": 16},
            description="Live entries vs capacity — LRU eviction starts at capacity.",
        ),
        _panel(
            8, "Corpus & memory", "stat",
            [
                _target("rag_corpus_documents", "documents", "A", instant=True),
                _target("rag_corpus_chunks", "chunks", "B", instant=True),
                _target("rag_sessions_total", "sessions", "C", instant=True),
                _target("rag_feedback_total", "feedback", "D", instant=True),
            ],
            unit="short",
            grid={"h": 6, "w": 8, "x": 16, "y": 16},
            description="Index size, conversation sessions, and user feedback count.",
        ),
        _panel(
            9, "Triage fix-list SLA", "stat",
            [
                _target("rag_triage_open", "open items", "A", instant=True),
                _target("rag_triage_resolved", "resolved", "B", instant=True),
                _target("rag_triage_oldest_open_hours", "oldest open (h)", "C", instant=True),
            ],
            unit="short",
            grid={"h": 6, "w": 12, "x": 0, "y": 22},
            description="Downvote fix-list health — open vs resolved items and how long the oldest open item has waited (Round 20 SLA gauge).",
        ),
        _panel(
            10, "Eval run volume", "timeseries",
            [
                _target("rate(rag_eval_runs_total[1h])", "runs/h", "A"),
                _target("rag_eval_last_run_accuracy", "last accuracy", "B"),
            ],
            unit="short",
            grid={"h": 6, "w": 12, "x": 12, "y": 22},
            description="Persisted golden-set runs over time (incl. ablation) and the latest accuracy.",
        ),
    ]
    return {
        "title": "RAG Agentic Q&A — service observability",
        "uid": "rag-agent",
        "schemaVersion": 39,
        "version": 1,
        "refresh": "30s",
        "time": {"from": "now-1h", "to": "now"},
        "tags": ["rag", "agentic", "fastapi"],
        "templating": {"list": []},
        "panels": panels,
        "annotations": {"list": []},
    }
