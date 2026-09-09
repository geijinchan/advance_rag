"""Triage resolution state — the 👎 fix-list lifecycle (resolve / reopen).

Round 18's quality radar only *listed* downvoted answers; this module gives
them a persisted workflow state so a triage item can be marked fixed (with an
optional resolution note) and later reopened. Mirrors ``FeedbackStore``:
keyed by request_id, capped in memory (500 — oldest evicted), persisted to
``data/triage_resolutions.json`` with atomic writes so resolutions survive
service restarts.

Semantics:
* Resolving requires a downvote to exist (the endpoint validates that against
  the feedback store); the tracker itself is deliberately dumb — it only
  remembers request_id → resolution record. Orphaned records (rating flipped
  to 👍 or evicted from the capped feedback ring) simply stop appearing in
  ``GET /quality/triage`` output, whose counts are computed from the join.
* Re-resolving the same request_id overwrites the record (new note + new
  timestamp), which is why ``resolve`` has no "already resolved" guard.
"""

from __future__ import annotations

import time
from typing import Any

from app.persist import load_json, save_json


class TriageTracker:
    """request_id → resolution record, capped + persisted."""

    def __init__(self, cap: int = 500, persist_path: str | None = None) -> None:
        self._resolutions: dict[str, dict[str, Any]] = {}
        self._cap = cap
        self._persist_path = persist_path
        saved = load_json(persist_path) if persist_path else None
        if isinstance(saved, dict):
            for rid, rec in saved.get("resolutions", {}).items():
                if isinstance(rec, dict) and rec.get("request_id"):
                    self._resolutions[str(rid)] = rec

    def resolve(self, request_id: str, note: str | None = None) -> dict[str, Any]:
        """Mark a triage item resolved (optionally noting HOW it was fixed).

        Re-resolving overwrites the previous record — the newest note and
        timestamp win. Returns the stored record.
        """
        record = {
            "request_id": request_id,
            "note": (note or "").strip()[:300] or None,
            "resolved_at": round(time.time(), 1),
        }
        self._resolutions[request_id] = record
        if len(self._resolutions) > self._cap:
            oldest = min(self._resolutions.values(), key=lambda r: r["resolved_at"])
            self._resolutions.pop(oldest["request_id"], None)
        self._persist()
        return record

    def reopen(self, request_id: str) -> bool:
        """Un-resolve an item — True if it was resolved and got removed."""
        if request_id in self._resolutions:
            self._resolutions.pop(request_id, None)
            self._persist()
            return True
        return False

    def is_resolved(self, request_id: str) -> bool:
        return request_id in self._resolutions

    def resolution_of(self, request_id: str) -> dict[str, Any] | None:
        return self._resolutions.get(request_id)

    def count(self) -> int:
        """Number of stored resolution records (incl. any orphaned ones)."""
        return len(self._resolutions)

    def _persist(self) -> None:
        if self._persist_path:
            save_json(self._persist_path, {"resolutions": self._resolutions})
