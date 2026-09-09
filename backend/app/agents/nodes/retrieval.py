"""Retrieval, relevance-grading and query-rewrite nodes (Part C loop)."""

from __future__ import annotations

import logging

from app.agents.state import AgentState
from app.config import settings
from app.llm.brain import LLMBrain
from app.llm.simulation import HeuristicBrain
from app.rag.retriever import HybridRetriever

logger = logging.getLogger(__name__)

Brain = LLMBrain | HeuristicBrain
END = "__end__"


async def _merge_sub_query_hits(
    retriever: HybridRetriever,
    sub_queries: list[str],
    k: int,
) -> tuple[list, dict[str, str], list[dict]]:
    """Retrieve per sub-query and merge with entity balance.

    Each sub-query fetches ``max(3, k // 2 + 1)`` hits from a 3× wider
    candidate pool, with the entity's own code-named documents prioritised
    (``warranty X300`` → x300_controller_manual.md first — see the entity
    boost in HybridRetriever.search). A chunk surfaced by several
    sub-queries keeps the best score. The merged list is interleaved
    per sub-query (round-robin) so no single entity floods the context
    window — the exact failure mode decomposition exists to fix.

    Also returns the cross-encoder telemetry of every sub-search so the
    retriever node can fold the full rerank picture into its trace.
    """
    per_q = max(3, k // 2 + 1)
    per_query_hits: list[list] = []
    sub_query_map: dict[str, str] = {}
    rerank_telemetry: list[dict] = []
    seen: dict[str, float] = {}
    for sq in sub_queries:
        code = sq.rsplit(" ", 1)[-1].lower()
        pool = await retriever.search(sq, k=per_q * 3)
        if retriever.last_rerank:
            rerank_telemetry.append(retriever.last_rerank)
        # Hard entity priority: docs NAMED after the entity come first,
        # then chunks whose text carries the code, then the rest.
        pool = sorted(
            pool,
            key=lambda h: (code in h.chunk.doc.lower(), code in h.chunk.text.lower()),
            reverse=True,
        )[:per_q]
        kept = []
        for h in pool:
            prev = seen.get(h.chunk.chunk_id)
            if prev is None or h.score > prev:
                seen[h.chunk.chunk_id] = h.score
                kept.append(h)
                sub_query_map[h.chunk.chunk_id] = sq
        per_query_hits.append(kept)
    # Round-robin interleave: X-entity chunk, Y-entity chunk, X, Y, …
    merged: list = []
    exhausted = False
    while not exhausted:
        exhausted = True
        for hits in per_query_hits:
            if hits:
                merged.append(hits.pop(0))
                exhausted = False
    # Dedup (a chunk can appear in two lists when re-added by best-score)
    dedup: list = []
    ids: set[str] = set()
    for h in merged:
        if h.chunk.chunk_id not in ids:
            ids.add(h.chunk.chunk_id)
            dedup.append(h)
    return dedup[: max(k, len(sub_queries) * 3)], sub_query_map, rerank_telemetry


def make_retriever_node(retriever: HybridRetriever):
    async def retriever_node(state: AgentState) -> str | None:
        k = state.top_k or settings.retrieval_top_k
        query = state.effective_query or state.question
        rerank_detail: dict | None = None
        if state.sub_queries:
            hits, sub_query_map, telemetries = await _merge_sub_query_hits(
                retriever, state.sub_queries, k)
            state.sub_query_map = sub_query_map
            if telemetries:
                rerank_detail = {
                    "stage": "cross-encoder",
                    "sub_searches": len(telemetries),
                    "reranked": sum(t.get("reranked", 0) for t in telemetries),
                    "reorders": sum(t.get("reorders", 0) for t in telemetries),
                    "latency_ms": round(sum(t.get("latency_ms", 0.0) for t in telemetries), 2),
                    "top_scores": [
                        {"doc": s["doc"], "page": s["page"],
                         "score": s["rerank_score"], "fusion_rank": s["fusion_rank"]}
                        for t in telemetries for s in t.get("scores", [])[:3]
                    ],
                }
        else:
            hits = await retriever.search(query, k=k)
            t = retriever.last_rerank
            if t:
                rerank_detail = {
                    "stage": "cross-encoder",
                    "sub_searches": 0,
                    "reranked": t.get("reranked", 0),
                    "reorders": t.get("reorders", 0),
                    "latency_ms": t.get("latency_ms", 0.0),
                    "top_scores": [
                        {"doc": s["doc"], "page": s["page"],
                         "score": s["rerank_score"], "fusion_rank": s["fusion_rank"]}
                        for s in t.get("scores", [])[:3]
                    ],
                }
        state.retrieved = hits
        state.record(
            "retriever",
            query=query,
            top_k=k,
            decomposed=bool(state.sub_queries),
            rerank=rerank_detail,
            hits=[
                {"doc": h.chunk.doc, "page": h.chunk.page, "score": h.score,
                 "coverage": h.coverage,
                 "rerank_score": h.rerank_score, "fusion_rank": h.fusion_rank}
                for h in hits
            ],
        )
        return "grader"

    return retriever_node


def make_grader_node(brain: Brain, threshold: float | None = None):
    """Grade whether retrieved chunks actually answer the question.

    Per spec: if the best chunk is not relevant, rewrite the query and retry
    retrieval (bounded by MAX_RETRIEVAL_RETRIES); after that, honest fallback.
    """
    thr = threshold if threshold is not None else settings.grade_threshold

    async def grader_node(state: AgentState) -> str | None:
        grades = []
        for hit in state.retrieved:
            # Decomposed runs grade each chunk against the sub-query it was
            # fetched for — asking "is this relevant for X300?" about an
            # X300 chunk, instead of the full comparative question.
            grade_query = (
                state.sub_query_map.get(hit.chunk.chunk_id)
                or state.effective_query
                or state.question
            )
            grade = await brain.grade_chunk(
                grade_query, hit.chunk.text, hit.score
            )
            grades.append(
                {
                    "chunk_id": hit.chunk.chunk_id,
                    "doc": hit.chunk.doc,
                    "page": hit.chunk.page,
                    "grade": grade,
                    "retrieval_score": hit.score,
                }
            )
        state.grades = grades
        relevant = [
            hit
            for hit, g in zip(state.retrieved, grades)
            if g["grade"] >= thr
        ]
        relevant.sort(key=lambda h: h.score, reverse=True)
        state.relevant_chunks = relevant
        best = grades[0]["grade"] if grades else 0.0
        if relevant:
            state.grade_decision = "relevant"
            state.record(
                "grader",
                decision="relevant",
                best_grade=best,
                relevant_count=len(relevant),
                threshold=thr,
            )
        else:
            state.grade_decision = "not_relevant"
            state.record(
                "grader",
                decision="not_relevant",
                best_grade=best,
                threshold=thr,
                retries=state.retries,
            )
        return None  # conditional edges decide the next hop

    return grader_node


def grader_conditional(state: AgentState) -> str | None:
    if state.grade_decision == "relevant":
        return "generator"
    if state.retries < settings.max_retrieval_retries:
        return "rewriter"
    return "fallback"


def make_rewriter_node(brain: Brain):
    async def rewriter_node(state: AgentState) -> str | None:
        state.retries += 1
        rewritten = await brain.rewrite_query(state.effective_query or state.question)
        state.effective_query = rewritten
        state.record(
            "rewriter",
            status="retry",
            attempt=state.retries,
            original=state.question,
            rewritten=rewritten,
        )
        return "retriever"

    return rewriter_node
