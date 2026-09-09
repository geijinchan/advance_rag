"""Eval run history — persisted trend of golden-set evaluation runs.

Every ``POST /eval`` (and each ablation config run, tagged with its label)
appends a compact summary here, so the Eval tab can chart accuracy /
faithfulness / latency over time and show how golden-set growth (user-
promoted cases) changes the denominator. Capped ring of `cap` runs,
persisted to ``data/eval_history.json`` (atomic writes), newest first.

Ablation runs are recorded too but flagged ``ablation: true`` — the trend
chart default-filters them out (they run with the judge off), while the
table can show them.
"""

from __future__ import annotations

import time
from typing import Any

from app.persist import load_json, save_json


class EvalHistory:
    def __init__(self, cap: int = 40, persist_path: str | None = None) -> None:
        self._runs: list[dict[str, Any]] = []
        self._cap = cap
        self._persist_path = persist_path
        saved = load_json(persist_path) if persist_path else None
        if isinstance(saved, dict) and isinstance(saved.get("runs"), list):
            self._runs = [r for r in saved["runs"] if isinstance(r, dict)][:cap]

    def record(self, report: dict[str, Any]) -> dict[str, Any]:
        """Append one run summary (from an EvalRunner report dict)."""
        entry = {
            "ts": round(time.time(), 1),
            "label": str(report.get("label", "default")),
            "ablation": str(report.get("label", "")).startswith("ablation:"),
            "passed": int(report.get("passed", 0)),
            "total_cases": int(report.get("total_cases", 0)),
            "accuracy": report.get("accuracy"),
            "faithfulness_avg": report.get("faithfulness_avg"),
            "retrieval_hit_rate": report.get("retrieval_hit_rate"),
            "fact_support_rate": report.get("fact_support_rate"),
            "latency_p50_ms": report.get("latency_p50_ms"),
            "latency_p95_ms": report.get("latency_p95_ms"),
            "failed": int(report.get("failed", 0)),
        }
        self._runs.insert(0, entry)
        del self._runs[self._cap:]
        self._persist()
        return entry

    def trend(self, include_ablation: bool = False, limit: int = 20) -> dict[str, Any]:
        """Newest-first run list + summary stats for the trend chart."""
        runs = [r for r in self._runs if include_ablation or not r.get("ablation")]
        runs = runs[:limit]
        full = [r for r in self._runs if not r.get("ablation")]
        best_acc = max((r.get("accuracy") or 0 for r in full), default=None)
        last = full[0] if full else None
        first = full[-1] if full else None
        accuracy_trend = (
            "flat" if last and first and (last.get("accuracy") == first.get("accuracy"))
            else "changed"
        )
        return {
            "runs": runs,
            "total_runs": len(self._runs),
            "golden_runs": len(full),
            "ablation_runs": sum(1 for r in self._runs if r.get("ablation")),
            "last": last,
            "best_accuracy": best_acc,
            "accuracy_trend": accuracy_trend,
            "recorded_since": first.get("ts") if first else None,
        }

    def _persist(self) -> None:
        if self._persist_path:
            save_json(self._persist_path, {"runs": self._runs})
