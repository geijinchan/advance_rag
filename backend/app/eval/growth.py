"""Feedback-driven golden-set growth — turns real usage into eval coverage.

The pipeline has three stages:

1. **AnswerLedger** — every production ask (graph answer or cache hit) is
   recorded under its ``request_id`` (question, answer, citations, route).
   This is the join table between /feedback ratings and actual content:
   ratings alone only know *that* a request was liked, the ledger knows
   *what* was answered.

2. **Candidate building** — upvoted requests are joined against the ledger;
   grounded answers with citations become golden-set *candidates*, each with
   auto-suggested fact patterns (numbers-with-units, entity codes, version
   strings) and expected source docs derived from the citations.

3. **GoldenSetManager** — promoted candidates (or manually authored cases)
   are persisted to ``data/golden_cases.json`` and merged with the base
   16-case GOLDEN_SET at eval time, so the regression harness grows with
   real usage. Base cases are immutable; user cases (``usr-`` prefix) can
   be listed, verified, and deleted.

The keyword extractor deliberately produces *conservative* regexes: numbers
keep their unit context (``36\\s*months``), tokens are escaped, and only
high-signal facts are kept — a reviewer can edit the patterns before
promoting via the API.

Round 19 adds the two ends of the loop's lifecycle: batch promotion
(``POST /eval/candidates/promote-batch``) and the triage resolve/reopen
workflow (``app/eval/triage.py`` joined in by ``build_triage``'s tracker
parameter).
"""

from __future__ import annotations

import re
import time
import uuid
from typing import TYPE_CHECKING, Any, Callable, Iterable

from app.eval.golden_set import GOLDEN_SET, EvalCase
from app.persist import load_json, save_json

if TYPE_CHECKING:  # triage state duck-typed at runtime (no import cycle)
    from app.eval.triage import TriageTracker

# ---------------------------------------------------------------- keyword mining

_NUM_UNIT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(°C|°F|C|F|Hz|kHz|MHz|ms|s|kg|g|mm|cm|m|V|A|W|"
    r"months?|years?|days?|hours?|minutes?|seconds?|%)",
    re.IGNORECASE,
)
_ENTITY_RE = re.compile(r"\b([XS]\d{3}|E\d{3}|IP\d{2}|v?\d+\.\d+\.\d+)\b")
# Distinctive technical tokens: 3+ chars, contains a digit or is rare-ish
# (uppercase-heavy product vocabulary). Used only as a last resort.
_TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z/-]{2,}\d[\w/-]*\b|\b\d{3,}\b")

_STOPWORDS = {
    "the", "and", "for", "with", "from", "this", "that", "are", "was", "were",
    "not", "must", "should", "when", "then", "than", "you", "your", "its",
    "into", "onto", "over", "under", "only", "also", "within", "provided",
    "requires", "required", "available", "service", "product", "controller",
    "sensor", "device", "device's", "following", "above", "below", "between",
}


def _unit_norm(unit: str, number: str) -> str:
    """Canonicalise number+unit spacing into a forgiving regex."""
    unit = unit.replace("months", "months?").replace("years", "years?")
    unit = unit.replace("days", "days?").replace("hours", "hours?")
    unit = unit.replace("minutes", "minutes?").replace("seconds", "seconds?")
    num = re.escape(number)
    if unit in ("°C", "°F"):
        return rf"{num}\s*°?\s*{unit[-1]}"
    return rf"{num}\s*{re.escape(unit)}"


def suggest_keywords(answer: str, limit: int = 3) -> list[str]:
    """Conservative fact-pattern suggestions mined from an answer text.

    Priority: numbers with units > entity/version codes > long digit runs.
    Each suggestion is a valid, escaped regex that ``_match_any`` in the
    evaluator can score case-insensitively.
    """
    patterns: list[str] = []
    seen: set[str] = set()

    def _add(p: str) -> None:
        if p not in seen and len(patterns) < limit:
            # must compile — never hand the evaluator a broken regex
            try:
                re.compile(p, re.IGNORECASE)
            except re.error:
                return
            seen.add(p)
            patterns.append(p)

    for m in _NUM_UNIT_RE.finditer(answer):
        _add(_unit_norm(m.group(2), m.group(1)))
    for m in _ENTITY_RE.finditer(answer):
        _add(re.escape(m.group(1)))
    if not patterns:
        for m in _TOKEN_RE.finditer(answer):
            tok = m.group(0)
            if tok.lower() not in _STOPWORDS:
                _add(re.escape(tok))
    return patterns


def _norm_question(q: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", q.lower()).strip()


# ---------------------------------------------------------------- answer ledger


class AnswerLedger:
    """request_id → answered content, capped + persisted.

    The ledger records what the service actually answered so a later rating
    (``POST /feedback``) can be resolved back to question/answer/citations.
    Redis/streaming store in production; JSON file here.
    """

    def __init__(self, cap: int = 300, persist_path: str | None = None) -> None:
        self._entries: dict[str, dict[str, Any]] = {}
        self._cap = cap
        self._persist_path = persist_path
        saved = load_json(persist_path) if persist_path else None
        if isinstance(saved, dict):
            for rid, rec in saved.get("entries", {}).items():
                if isinstance(rec, dict) and rec.get("answer"):
                    self._entries[str(rid)] = rec
        self.misses = 0  # ratings that couldn't be joined (pre-ledger asks)

    def record(self, request_id: str, payload: dict[str, Any]) -> None:
        entry = {
            "request_id": request_id,
            "question": str(payload.get("question", ""))[:1000],
            "effective_question": str(payload.get("effective_question", ""))[:1000],
            "answer": str(payload.get("answer", ""))[:4000],
            "route": str(payload.get("route", "")),
            "fallback": bool(payload.get("fallback", False)),
            "cache_hit": bool(payload.get("cache_hit", False)),
            "latency_ms": round(float(payload.get("latency_ms", 0.0)), 1),
            "citations": [
                {"doc": c.get("doc", ""), "page": c.get("page", 1),
                 "chunk_id": c.get("chunk_id", ""), "snippet": c.get("snippet", "")[:200]}
                for c in (payload.get("citations") or [])[:6]
            ],
            "ts": round(float(payload.get("ts") or time.time()), 1),
        }
        self._entries[request_id] = entry
        if len(self._entries) > self._cap:
            oldest = min(self._entries.values(), key=lambda r: r["ts"])
            self._entries.pop(oldest["request_id"], None)
        self._persist()

    def get(self, request_id: str) -> dict[str, Any] | None:
        return self._entries.get(request_id)

    def all_entries(self) -> list[dict[str, Any]]:
        """Every ledger entry (question/answer/route/latency/citations/ts) —
        powers the question-frequency analytics endpoint."""
        return list(self._entries.values())

    def stats(self) -> dict[str, Any]:
        return {
            "entries": len(self._entries),
            "capacity": self._cap,
            "grounded": sum(1 for e in self._entries.values()
                            if e["route"] == "document" and not e["fallback"]),
            "cache_served": sum(1 for e in self._entries.values() if e["cache_hit"]),
        }

    def _persist(self) -> None:
        if self._persist_path:
            save_json(self._persist_path, {"entries": self._entries})


# ---------------------------------------------------------------- golden growth


class GoldenSetManager:
    """Base golden set + persisted user-grown cases, merged for /eval."""

    def __init__(self, persist_path: str | None = None) -> None:
        self._persist_path = persist_path
        self._user_cases: dict[str, dict[str, Any]] = {}
        saved = load_json(persist_path) if persist_path else None
        if isinstance(saved, dict):
            for cid, rec in saved.get("cases", {}).items():
                if isinstance(rec, dict) and rec.get("question") and rec.get("id"):
                    self._user_cases[str(cid)] = rec

    # ------------------------------------------------------------ merge view
    def cases(self) -> list[EvalCase]:
        """Base cases + user cases, as EvalCase objects (eval-ready)."""
        out: list[EvalCase] = list(GOLDEN_SET)
        for rec in self._user_cases.values():
            out.append(EvalCase(
                id=str(rec["id"]),
                question=str(rec["question"]),
                expects=str(rec.get("expects", "grounded")),  # type: ignore[arg-type]
                answer_must_match_any=list(rec.get("keywords", [])),
                docs_any=list(rec.get("docs_any", [])),
                category=str(rec.get("category", "feedback")),
                note=str(rec.get("note", "")),
            ))
        return out

    def source_of(self, case_id: str) -> str:
        return "user" if case_id in self._user_cases else "base"

    def stats(self) -> dict[str, Any]:
        return {
            "base": len(GOLDEN_SET),
            "user": len(self._user_cases),
            "total": len(GOLDEN_SET) + len(self._user_cases),
        }

    # ------------------------------------------------------------ mutations
    def add_case(
        self,
        question: str,
        keywords: list[str],
        docs_any: list[str] | None = None,
        expects: str = "grounded",
        category: str = "feedback",
        note: str = "",
        origin: dict[str, Any] | None = None,
    ) -> EvalCase:
        if expects not in ("grounded", "fallback", "chitchat"):
            raise ValueError(f"expects must be grounded|fallback|chitchat, got {expects!r}")
        question = question.strip()
        if not (4 <= len(question) <= 1000):
            raise ValueError("question must be 4–1000 chars")
        cleaned: list[str] = []
        for p in keywords:
            p = str(p).strip()
            if not p or len(p) > 100 or p in cleaned:
                continue
            try:
                re.compile(p)
            except re.error:
                # escape instead of rejecting — stay permissive for manual input
                p = re.escape(p)
            cleaned.append(p)
            if len(cleaned) >= 6:
                break
        if expects == "grounded" and not cleaned:
            raise ValueError("grounded cases need at least one answer pattern")
        docs = [str(d).strip()[:80] for d in (docs_any or []) if str(d).strip()][:6]
        cid = f"usr-{uuid.uuid4().hex[:8]}"
        rec = {
            "id": cid,
            "question": question,
            "expects": expects,
            "keywords": cleaned,
            "docs_any": docs,
            "category": category,
            "note": (note or "").strip()[:300],
            "origin": origin or {},
            "promoted_at": round(time.time(), 1),
        }
        self._user_cases[cid] = rec
        self._persist()
        return self._to_case(rec)

    def remove_case(self, case_id: str) -> bool:
        if case_id in self._user_cases:
            self._user_cases.pop(case_id)
            self._persist()
            return True
        return False

    # ------------------------------------------------------------ candidates
    def build_candidates(
        self,
        ledger: AnswerLedger,
        feedback_records: Iterable[dict[str, Any]],
        limit: int = 12,
    ) -> dict[str, Any]:
        """Join upvoted feedback → ledger entries → golden-set candidates."""
        existing_qs = {_norm_question(c.question) for c in self.cases()}
        candidates: list[dict[str, Any]] = []
        unlinkable = 0
        skipped_route = 0
        skipped_dupe = 0
        seen_qs: set[str] = set()

        for rec in sorted(feedback_records, key=lambda r: r.get("ts", 0), reverse=True):
            if rec.get("rating") != "up":
                continue
            entry = ledger.get(str(rec.get("request_id", "")))
            if entry is None:
                unlinkable += 1
                continue
            if entry.get("route") != "document" or entry.get("fallback"):
                skipped_route += 1
                continue
            if not entry.get("answer") or not entry.get("citations"):
                skipped_route += 1
                continue
            nq = _norm_question(entry.get("question") or entry.get("effective_question", ""))
            if nq in existing_qs or nq in seen_qs or not nq:
                skipped_dupe += 1
                continue
            seen_qs.add(nq)
            keywords = suggest_keywords(entry["answer"])
            docs = []
            for c in entry["citations"]:
                d = c.get("doc", "")
                if d and d not in docs:
                    docs.append(d)
            candidates.append({
                "request_id": entry["request_id"],
                "question": entry.get("question") or entry.get("effective_question", ""),
                "answer_preview": entry["answer"][:400] + ("…" if len(entry["answer"]) > 400 else ""),
                "route": entry.get("route"),
                "cache_hit": entry.get("cache_hit", False),
                "latency_ms": entry.get("latency_ms", 0.0),
                "citations": [
                    {"doc": c.get("doc", ""), "page": c.get("page", 1)}
                    for c in entry["citations"][:4]
                ],
                "suggested_keywords": keywords,
                "suggested_docs": docs[:4],
                "keywords_editable": True,
                "feedback": {
                    "rating": rec.get("rating"),
                    "comment": rec.get("comment"),
                    "ts": rec.get("ts"),
                },
                "rated_at": rec.get("ts"),
            })
            if len(candidates) >= limit:
                break

        return {
            "candidates": candidates,
            "counts": {
                "shown": len(candidates),
                "unlinkable_ratings": unlinkable,
                "skipped_route": skipped_route,
                "skipped_duplicate": skipped_dupe,
            },
            "golden_set": self.stats(),
            "ledger": ledger.stats(),
        }

    # ------------------------------------------------------------ triage (👎)
    def build_triage(
        self,
        ledger: AnswerLedger,
        feedback_records: Iterable[dict[str, Any]],
        tracker: TriageTracker | None = None,
        limit: int = 12,
    ) -> dict[str, Any]:
        """Downvoted answers joined against the ledger — the quality radar.

        Where candidates turn 👍 into eval coverage, triage turns 👎 into a
        worklist: what did the service answer badly (fallback? weak
        citations? user comment reason?), newest first.

        Round 19 — **the fix-list is now open-only**: with a ``tracker``
        passed in, ``items`` contains only UNRESOLVED downvoted items (this
        is the deliberate breaking change coordinated with the frontend —
        resolved items move to ``resolved_items``). Every item — open or
        resolved — carries ``resolved``/``resolution_note``/``resolved_at``
        (False/None/None for open ones).

        ``counts``: ``shown`` now means *open* items shown (capped at
        ``limit``); ``open``/``resolved`` are accurate across ALL downvoted
        records, not just the capped lists. ``resolved`` counts every
        downvoted request_id that has a resolution record (even one whose
        ledger entry has since been evicted — it then also shows up in
        ``unlinkable_ratings``); ``open`` counts linkable downvoted items
        without a resolution. ``fallback_answers``/``uncited_answers``/
        ``with_reason`` stay computed over the shown open items.

        ``resolved_items``: resolved downvotes joined against the ledger,
        **newest resolved first**, capped at ``limit``, same fields as
        ``items`` plus the resolution fields. A resolution whose ledger
        entry was evicted is still listed — question ``"?"`` and empty
        citations — so the fixed history never silently disappears (the
        reason comment from feedback is still shown).

        ``limit=0`` returns counts only (both lists empty) — used by
        /stats + /metrics to avoid a second full join.
        """
        items: list[dict[str, Any]] = []
        resolved_items: list[dict[str, Any]] = []
        unlinkable = 0
        open_total = 0
        resolved_total = 0
        seen: set[str] = set()
        now = time.time()
        open_ages: list[float] = []  # hours, across ALL open items (Round 20 SLA)
        for rec in sorted(feedback_records, key=lambda r: r.get("ts", 0), reverse=True):
            if rec.get("rating") != "down":
                continue
            rid = str(rec.get("request_id", ""))
            if not rid or rid in seen:
                continue
            seen.add(rid)
            entry = ledger.get(rid)
            if entry is None:
                unlinkable += 1
            resolution = tracker.resolution_of(rid) if tracker is not None else None
            if resolution is not None:
                resolved_total += 1
                if entry is not None:
                    resolved_items.append({
                        "request_id": rid,
                        "question": entry.get("question") or entry.get("effective_question", ""),
                        "answer_preview": entry["answer"][:300] + ("…" if len(entry["answer"]) > 300 else ""),
                        "route": entry.get("route"),
                        "fallback": entry.get("fallback", False),
                        "cache_hit": entry.get("cache_hit", False),
                        "latency_ms": entry.get("latency_ms", 0.0),
                        "citations": [
                            {"doc": c.get("doc", ""), "page": c.get("page", 1)}
                            for c in entry.get("citations", [])[:3]
                        ],
                        "reason": rec.get("comment"),
                        "rated_at": rec.get("ts"),
                        "resolved": True,
                        "resolution_note": resolution.get("note"),
                        "resolved_at": resolution.get("resolved_at"),
                    })
                else:
                    # Ledger entry evicted: keep the fixed history visible
                    # (question "?", empty citations) rather than dropping it.
                    resolved_items.append({
                        "request_id": rid,
                        "question": "?",
                        "answer_preview": "",
                        "route": None,
                        "fallback": False,
                        "cache_hit": False,
                        "latency_ms": 0.0,
                        "citations": [],
                        "reason": rec.get("comment"),
                        "rated_at": rec.get("ts"),
                        "resolved": True,
                        "resolution_note": resolution.get("note"),
                        "resolved_at": resolution.get("resolved_at"),
                    })
                continue
            if entry is None:
                continue  # unlinkable open downvote — counted, not listable
            open_total += 1
            rated_ts = rec.get("ts")
            age_hours = (
                round(max(0.0, (now - float(rated_ts))) / 3600.0, 1)
                if isinstance(rated_ts, (int, float)) else None
            )
            if age_hours is not None:
                open_ages.append(age_hours)
            if len(items) < limit:
                items.append({
                    "request_id": rid,
                    "question": entry.get("question") or entry.get("effective_question", ""),
                    "answer_preview": entry["answer"][:300] + ("…" if len(entry["answer"]) > 300 else ""),
                    "route": entry.get("route"),
                    "fallback": entry.get("fallback", False),
                    "cache_hit": entry.get("cache_hit", False),
                    "latency_ms": entry.get("latency_ms", 0.0),
                    "citations": [
                        {"doc": c.get("doc", ""), "page": c.get("page", 1)}
                        for c in entry.get("citations", [])[:3]
                    ],
                    "reason": rec.get("comment"),
                    "rated_at": rec.get("ts"),
                    "age_hours": age_hours,
                    "resolved": False,
                    "resolution_note": None,
                    "resolved_at": None,
                })
        resolved_items.sort(key=lambda i: i.get("resolved_at") or 0.0, reverse=True)
        resolved_items = resolved_items[:limit]
        fallbacks = sum(1 for i in items if i["fallback"])
        no_citations = sum(1 for i in items if not i["citations"])
        return {
            "items": items,
            "resolved_items": resolved_items,
            "counts": {
                "shown": len(items),
                "open": open_total,
                "resolved": resolved_total,
                "unlinkable_ratings": unlinkable,
                "fallback_answers": fallbacks,
                "uncited_answers": no_citations,
                "with_reason": sum(1 for i in items if i["reason"]),
                # Round 20 SLA: age of the open fix-list (hours). oldest/mean
                # cover ALL open items, not just the capped `items` list.
                "oldest_open_hours": round(max(open_ages), 1) if open_ages else None,
                "mean_open_hours": (
                    round(sum(open_ages) / len(open_ages), 1) if open_ages else None
                ),
            },
        }

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _to_case(rec: dict[str, Any]) -> EvalCase:
        return EvalCase(
            id=str(rec["id"]),
            question=str(rec["question"]),
            expects=str(rec.get("expects", "grounded")),  # type: ignore[arg-type]
            answer_must_match_any=list(rec.get("keywords", [])),
            docs_any=list(rec.get("docs_any", [])),
            category=str(rec.get("category", "feedback")),
            note=str(rec.get("note", "")),
        )

    def _persist(self) -> None:
        if self._persist_path:
            save_json(self._persist_path, {"cases": self._user_cases})
