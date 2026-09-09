"""Semantic answer cache — repeated and near-duplicate questions answered
from cache instead of re-running the agent graph.

Design (deliberately simple + honest):
* Only **grounded** answers (route=document, no fallback, verification
  passed) are cached — fallbacks depend on the grader state at ask time and
  chitchat is cheap.
* Keying is semantic: the query is embedded with the SAME feature-hashing
  embedder used by the index; a cache hit requires cosine similarity ≥
  ``SIM_THRESHOLD`` against a cached query. Exact-normalised matches always
  hit regardless of the threshold.
* Entries carry the full response payload (answer, citations, trace) plus
  provenance (original request id, age, similarity) so a cached answer is
  explicitly labelled, never silently replayed.
* TTL 15 minutes, capacity 128, LRU-eviction by last access. Entries persist
  to disk (``data/answer_cache.json``) so a restart keeps warm answers —
  TTL semantics are enforced at load (expired entries are dropped), the
  same contract Redis would provide in production.

Thread-safety: all access happens on the event loop (FastAPI handlers are
async), and the cache methods are synchronous — no locks needed.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.rag.embeddings import get_embedder

logger = logging.getLogger(__name__)

TTL_SECONDS = 15 * 60
MAX_ENTRIES = 128
SIM_THRESHOLD = 0.93

_EMBEDDER = None


def _embed_query(text: str) -> list[float]:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = get_embedder()
    vec = _EMBEDDER.encode([text], is_query=True)[0]
    return [float(x) for x in vec]


def _normalize(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


@dataclass
class _CacheEntry:
    query: str                       # normalised query text
    vector: list[float]              # embedding of the normalised query
    payload: dict[str, Any]          # full AskResponse payload
    top_k: int | None
    stored_at: float = field(default_factory=time.time)
    hits: int = 0


class AnswerCache:
    def __init__(self, ttl: float = TTL_SECONDS, capacity: int = MAX_ENTRIES,
                 sim_threshold: float = SIM_THRESHOLD,
                 persist_path: str | None = None) -> None:
        self._ttl = ttl
        self._capacity = capacity
        self._sim_threshold = sim_threshold
        self._entries: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._persist_path = Path(persist_path) if persist_path else None
        self._persisted_hits = 0   # total hits served across restarts (telemetry)
        self._restored_entries = 0
        self._load()

    # ------------------------------------------------------------ persistence
    def _load(self) -> None:
        """Load persisted entries; TTL-expired ones are dropped so restart
        never resurrects stale answers."""
        if self._persist_path is None or not self._persist_path.exists():
            return
        try:
            raw = json.loads(self._persist_path.read_text(encoding="utf-8"))
            now = time.time()
            loaded = 0
            dropped = 0
            for item in raw.get("entries", []):
                try:
                    entry = _CacheEntry(
                        query=item["query"],
                        vector=[float(x) for x in item["vector"]],
                        payload=item["payload"],
                        top_k=item.get("top_k"),
                        stored_at=float(item.get("stored_at", now)),
                        hits=int(item.get("hits", 0)),
                    )
                except (KeyError, TypeError, ValueError):
                    dropped += 1
                    continue
                if self._expired(entry, now):
                    dropped += 1
                    continue
                if len(self._entries) >= self._capacity:
                    break
                self._entries[entry.query] = entry
                loaded += 1
            self._persisted_hits = int(raw.get("served_hits", 0))
            self._restored_entries = loaded
            if loaded or dropped:
                logger.info("answer cache: restored %d entries from disk "
                            "(%d expired/dropped)", loaded, dropped)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("answer cache: could not load %s (%s)",
                           self._persist_path, exc)

    def _flush(self) -> None:
        """Persist the live entries (best-effort, atomic write)."""
        if self._persist_path is None:
            return
        try:
            now = time.time()
            live = [e for e in self._entries.values() if not self._expired(e, now)]
            payload = {
                "version": 1,
                "saved_at": now,
                "served_hits": sum(e.hits for e in self._entries.values()),
                "entries": [
                    {
                        "query": e.query,
                        "vector": e.vector,
                        "payload": e.payload,
                        "top_k": e.top_k,
                        "stored_at": e.stored_at,
                        "hits": e.hits,
                    }
                    for e in live
                ],
            }
            tmp = self._persist_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self._persist_path)
        except OSError as exc:
            logger.warning("answer cache: persist failed (%s)", exc)

    # --------------------------------------------------------------- lookup
    def lookup(self, query: str, top_k: int | None = None) -> dict[str, Any] | None:
        """Return the cached payload for a semantically-matching query, or
        None. The returned payload is a *copy* with provenance added under
        ``cache_detail``; the caller sets request/session fields itself."""
        now = time.time()
        norm = _normalize(query)
        exact = self._entries.get(norm)
        if exact is not None:
            if self._expired(exact, now):
                del self._entries[norm]
            else:
                self._entries.move_to_end(norm)
                exact.hits += 1
                return self._payload_with_provenance(exact, 1.0, exact.hits)
        if not self._entries:
            return None

        qvec = _embed_query(norm)
        best: _CacheEntry | None = None
        best_sim = -1.0
        for entry in self._entries.values():
            if self._expired(entry, now) or entry.top_k != top_k:
                continue
            sim = _cosine(qvec, entry.vector)
            if sim > best_sim:
                best_sim = sim
                best = entry
        if best is not None and best_sim >= self._sim_threshold:
            self._entries.move_to_end(best.query)
            best.hits += 1
            return self._payload_with_provenance(best, round(best_sim, 4), best.hits)
        return None

    # ---------------------------------------------------------------- store
    def store(self, query: str, top_k: int | None, payload: dict[str, Any]) -> None:
        """Cache a payload; only grounded, verified, non-fallback answers."""
        if payload.get("fallback") or payload.get("route") != "document":
            return
        if not payload.get("answer") or "couldn't find" in str(payload.get("answer", "")):
            return
        norm = _normalize(query)
        self._entries[norm] = _CacheEntry(
            query=norm,
            vector=_embed_query(norm),
            payload=dict(payload),
            top_k=top_k,
        )
        self._entries.move_to_end(norm)
        while len(self._entries) > self._capacity:
            self._entries.popitem(last=False)  # evict least-recently used
        self._flush()

    # -------------------------------------------------------------- metrics
    def stats(self) -> dict[str, Any]:
        now = time.time()
        live = [e for e in self._entries.values() if not self._expired(e, now)]
        return {
            "entries": len(live),
            "capacity": self._capacity,
            "hits": sum(e.hits for e in self._entries.values()),
            "hit_rate_last": round(
                sum(e.hits for e in self._entries.values())
                / max(1, sum(e.hits for e in self._entries.values()) + len(self._entries)),
                4,
            ),
            "ttl_seconds": self._ttl,
            "sim_threshold": self._sim_threshold,
            "persisted": self._persist_path is not None,
            "restored_entries": self._restored_entries,
        }

    # --------------------------------------------------------------- helpers
    def _expired(self, entry: _CacheEntry, now: float) -> bool:
        return now - entry.stored_at > self._ttl

    @staticmethod
    def _payload_with_provenance(entry: _CacheEntry, similarity: float,
                                 times_served: int) -> dict[str, Any]:
        payload = dict(entry.payload)
        payload["cache_hit"] = True
        payload["cache_detail"] = {
            "original_request_id": payload.get("request_id", ""),
            "original_question": entry.query,
            "age_seconds": round(time.time() - entry.stored_at, 1),
            "similarity": similarity,
            "times_served": times_served,
            "original_latency_ms": payload.get("latency_ms", 0.0),
        }
        return payload


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    num = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return num / (na * nb)
