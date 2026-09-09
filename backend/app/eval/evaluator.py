"""Evaluation runner — executes the golden set through the FULL agentic
pipeline (router → retrieval → grading → generation → verification) and
scores four dimensions per case:

* **route_correct**     — did the router pick the expected path?
* **retrieval_hit**     — did at least one expected document surface in the
                          citations? (retrieval quality, hit@k)
* **fact_supported**    — does the answer contain at least one required
                          fact pattern? (answer correctness, RAGAS-style
                          "answer relevance" proxy)
* **faithfulness**      — RAGAS-style grounding score: how well is the answer
                          supported by its own citations? (LLM-judge when a
                          live model is served, deterministic lexical judge
                          otherwise; see app/eval/faithfulness.py)

A case passes when all applicable dimensions pass (fallback/chitchat cases
only need route + answer sanity; faithfulness is reported but does not gate
robustness/fallback cases). The report is returned to the caller and
cached for ``GET /eval/results``.
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from typing import Any, Callable

from app.eval.golden_set import GOLDEN_SET, EvalCase


def _match_any(patterns: list[str], text: str) -> str | None:
    """Return the first pattern that matches (case-insensitive), else None."""
    for p in patterns:
        try:
            if re.search(p, text, re.IGNORECASE):
                return p
            # plain substring for patterns entered without regex intent
            if p.lower() in text.lower():
                return p
        except re.error:
            if p.lower() in text.lower():
                return p
    return None


class EvalRunner:
    """Runs the golden set against a live `ask` function (the real graph)."""

    def __init__(self, ask_fn, faithfulness_judge=None, chunk_text_lookup=None,
                 cases_provider: Callable[[], list[EvalCase]] | None = None) -> None:
        self.ask_fn = ask_fn  # async (question, top_k) -> AgentState
        self.judge = faithfulness_judge   # FaithfulnessJudge | None
        self.chunk_text_lookup = chunk_text_lookup  # chunk_id -> full text | None
        self.cases_provider = cases_provider  # dynamic golden set (growth)
        self.last_report: dict[str, Any] | None = None
        # Ablation knobs (baseline: top_k=4, service default weights):
        # the ablation harness re-points these per configuration run.
        self.run_top_k: int | None = 4
        self._use_faithfulness: bool = True

    async def run_case(self, case: EvalCase) -> dict[str, Any]:
        t0 = time.perf_counter()
        st = await self.ask_fn(case.question, top_k=self.run_top_k)
        wall_ms = round((time.perf_counter() - t0) * 1000, 1)

        answer = st.answer or ""
        cited_docs = [c.get("doc", "") for c in st.citations]

        # -- route -----------------------------------------------------------
        if case.expects == "grounded":
            route_ok = st.route == "document" and not st.fallback
        elif case.expects == "fallback":
            route_ok = st.fallback or st.route == "out_of_scope"
        else:
            route_ok = st.route == "chitchat"

        # -- retrieval ---------------------------------------------------------
        retrieval_hit: bool | None = None
        hit_doc: str | None = None
        if case.docs_any and case.expects == "grounded":
            for d in cited_docs:
                if any(d.startswith(exp) or exp in d for exp in case.docs_any):
                    retrieval_hit = True
                    hit_doc = d
                    break
            retrieval_hit = bool(retrieval_hit)

        # -- answer fact ---------------------------------------------------------
        matched_pattern = _match_any(case.answer_must_match_any, answer)
        fact_supported = matched_pattern is not None

        # -- faithfulness (grounding vs own citations) ----------------------------
        # Judge sources = the deduped citation list PLUS every chunk the
        # generator actually used (citations dedupe by doc+page, so multiple
        # chunks from the same page collapse to one citation — the judge needs
        # the full set or verbatim quotes look "unsupported").
        faith_payload: dict[str, Any] | None = None
        if (
            case.expects == "grounded"
            and self.judge is not None
            and answer
            and getattr(self, "_use_faithfulness", True)
        ):
            try:
                judge_citations = list(st.citations)
                seen_ids = {c.get("chunk_id") for c in judge_citations}
                for hit in (st.relevant_chunks or [])[:4]:
                    cid = getattr(hit.chunk, "chunk_id", None)
                    if cid and cid not in seen_ids:
                        seen_ids.add(cid)
                        judge_citations.append({"chunk_id": cid})
                fr = await self.judge.score(
                    answer, judge_citations, question=case.question,
                    chunk_text_lookup=self.chunk_text_lookup,
                )
                faith_payload = fr.as_dict()
            except Exception as exc:  # noqa: BLE001 — judge is observability, not a gate
                faith_payload = {"score": None, "judge": "error", "error": str(exc)[:200]}

        checks = [route_ok, fact_supported]
        if retrieval_hit is not None:
            checks.append(retrieval_hit)
        # low faithfulness on grounded cases is a hard fail (groundedness gate)
        if faith_payload is not None and faith_payload.get("score") is not None:
            checks.append(float(faith_payload["score"]) >= 0.5)
        passed = all(checks)

        return {
            "id": case.id,
            "question": case.question,
            "category": case.category,
            "expects": case.expects,
            "passed": passed,
            "route_correct": route_ok,
            "route": st.route,
            "retrieval_hit": retrieval_hit,
            "retrieved_doc": hit_doc,
            "fact_supported": fact_supported,
            "matched": matched_pattern,
            "faithfulness": faith_payload,
            "answer_preview": answer[:220],
            "citations": cited_docs[:3],
            "latency_ms": wall_ms,
            "agent_path": st.agent_path,
            "note": case.note,
        }

    async def run(self, *, top_k: int | None = 4, with_faithfulness: bool = True,
                  label: str = "default") -> dict[str, Any]:
        """Run the golden set. `top_k` overrides the per-case retrieval depth;
        `with_faithfulness=False` skips the judge (used by the ablation loop
        where relative retrieval metrics, not grounding, are under test);
        `label` tags the report (default | ablation:<name>).

        The case list comes from ``self.cases_provider()`` when one is bound
        (golden-set growth: base + user-promoted cases), else the static
        module-level GOLDEN_SET."""
        self.run_top_k = top_k
        self._use_faithfulness = with_faithfulness
        cases = (list(self.cases_provider()) if self.cases_provider
                 else list(GOLDEN_SET))
        results = []
        for case in cases:
            try:
                results.append(await self.run_case(case))
            except Exception as exc:  # noqa: BLE001 — one bad case must not kill the run
                results.append({
                    "id": case.id,
                    "question": case.question,
                    "category": case.category,
                    "expects": case.expects,
                    "passed": False,
                    "route_correct": False,
                    "route": "error",
                    "retrieval_hit": None,
                    "retrieved_doc": None,
                    "fact_supported": False,
                    "matched": None,
                    "faithfulness": None,
                    "answer_preview": f"[eval error] {type(exc).__name__}: {exc}",
                    "citations": [],
                    "latency_ms": 0.0,
                    "agent_path": [],
                    "note": case.note,
                })

        total = len(results)
        passed = sum(1 for r in results if r["passed"])
        grounded = [r for r in results if r["expects"] == "grounded"]
        retrieval_hits = sum(1 for r in grounded if r["retrieval_hit"])
        fact_ok = sum(1 for r in results if r["fact_supported"])
        latencies = [r["latency_ms"] for r in results if r["latency_ms"] > 0]

        faith_scores = [
            float(r["faithfulness"]["score"])
            for r in results
            if isinstance(r.get("faithfulness"), dict)
            and r["faithfulness"].get("score") is not None
        ]
        faith_avg = round(sum(faith_scores) / len(faith_scores), 3) if faith_scores else None
        judge_mode = next(
            (r["faithfulness"].get("judge") for r in results
             if isinstance(r.get("faithfulness"), dict)),
            None,
        )

        by_category: dict[str, dict[str, int]] = {}
        for r in results:
            cat = by_category.setdefault(r["category"], {"total": 0, "passed": 0})
            cat["total"] += 1
            cat["passed"] += 1 if r["passed"] else 0

        report = {
            "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "label": label,
            "total_cases": total,
            "passed": passed,
            "failed": total - passed,
            "accuracy": round(passed / total, 4) if total else 0.0,
            "retrieval_hit_rate": round(retrieval_hits / len(grounded), 4) if grounded else None,
            "fact_support_rate": round(fact_ok / total, 4) if total else 0.0,
            "faithfulness_avg": faith_avg,
            "faithfulness_judge": judge_mode,
            "latency_p50_ms": _percentile(latencies, 50),
            "latency_p95_ms": _percentile(latencies, 95),
            "by_category": by_category,
            "results": results,
        }
        self.last_report = report
        return report


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return round(s[f], 1)
    return round(s[f] + (s[c] - s[f]) * (k - f), 1)
