"""Answer-feedback store — thumbs up/down telemetry per request.

Tiny by design: deduped by request_id (one rating per answer, later ratings
overwrite), aggregated on demand, capped in memory, and persisted to a JSON
file so ratings survive service restarts. Complements the golden-set
evaluator: the eval set measures pinned cases, feedback measures real usage.
Redis/DB-backed persistence is the production upgrade.
"""

from __future__ import annotations

import time
from typing import Any

from app.persist import load_json, save_json


class FeedbackStore:
    def __init__(self, cap: int = 500, persist_path: str | None = None) -> None:
        self._by_request: dict[str, dict[str, Any]] = {}
        self._cap = cap
        self._persist_path = persist_path
        saved = load_json(persist_path) if persist_path else None
        if isinstance(saved, dict):
            for rid, rec in saved.get("ratings", {}).items():
                if isinstance(rec, dict) and rec.get("rating") in ("up", "down"):
                    self._by_request[str(rid)] = rec

    def add(self, request_id: str, rating: str, comment: str | None = None) -> dict[str, Any]:
        if rating not in ("up", "down"):
            raise ValueError(f"rating must be 'up' or 'down', got {rating!r}")
        if not request_id:
            request_id = "anonymous"
        record = {
            "request_id": request_id,
            "rating": rating,
            "comment": (comment or "").strip()[:500] or None,
            "ts": round(time.time(), 1),
        }
        self._by_request[request_id] = record  # dedupe/overwrite per answer
        if len(self._by_request) > self._cap:
            oldest = min(self._by_request.values(), key=lambda r: r["ts"])
            self._by_request.pop(oldest["request_id"], None)
        self._persist()
        return record

    def _persist(self) -> None:
        if self._persist_path:
            save_json(self._persist_path, {"ratings": self._by_request})

    def total(self) -> int:
        return len(self._by_request)

    def all_records(self) -> list[dict[str, Any]]:
        """Every rating (not just the recent window) — golden-set growth
        joins these against the answer ledger by request_id."""
        return list(self._by_request.values())

    def stats(self) -> dict[str, Any]:
        records = list(self._by_request.values())
        up = sum(1 for r in records if r["rating"] == "up")
        down = len(records) - up
        return {
            "total": len(records),
            "up": up,
            "down": down,
            "approval": round(up / len(records), 4) if records else None,
            "recent": sorted(records, key=lambda r: r["ts"], reverse=True)[:10],
        }

    def clear(self) -> None:
        self._by_request.clear()
        self._persist()
