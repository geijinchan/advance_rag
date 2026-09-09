"""Hybrid retrieval: BM25 (sparse) + dense vectors fused with RRF,
followed by a cross-encoder rerank stage (Part B "plus" item).

Pipeline: dense(MMR) ∪ BM25 → RRF fusion → heuristic precision boosts →
cross-encoder rerank of the top candidates (joint query-chunk scoring;
lexical surrogate in simulation mode, LLM-scored in vLLM mode).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.rag.bm25 import BM25Index
from app.rag.chunker import Chunk
from app.rag.embeddings import tokenize
from app.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float          # fused, normalised [0, 1]
    dense_rank: int
    bm25_rank: int        # 0 = not in BM25 top-k
    coverage: float       # fraction of query content terms present in chunk
    rerank_score: float | None = None   # cross-encoder score when the stage ran
    fusion_rank: int | None = None      # rank BEFORE the cross-encoder (1-based)


class HybridRetriever:
    """Pipeline: dense(MMR) ∪ BM25 → RRF fusion → cross-encoder rerank.

    RRF (Reciprocal Rank Fusion) is robust without score calibration:
        rrf(d) = Σ 1 / (k + rank_i(d))  over retrievers i

    The fusion heuristic then boosts chunks that literally contain more of
    the question's content terms, and the cross-encoder stage finally
    RE-SCORES the top candidates by reading the (query, chunk) pair jointly
    — the precision stage first-stage retrieval feeds (bge-reranker pattern;
    deterministic lexical surrogate in simulation mode, LLM judge in vLLM
    mode; see app/rag/reranker.py).

    Reranker weights (fused / coverage / agreement) are instance state with
    search-time overrides — the ablation harness and the retrieval playground
    exercise the exact same ranking code the agent graph uses, just with
    different parameters. ``rerank_enabled`` toggles the cross-encoder stage
    (ablation: fusion-only vs rerank).
    """

    DEFAULT_WEIGHTS = {"w_fused": 0.55, "w_coverage": 0.35, "w_agreement": 0.10}

    def __init__(self, store: VectorStore, rrf_k: int = 60, reranker=None) -> None:
        self.store = store
        self.rrf_k = rrf_k
        self._bm25 = BM25Index()
        self._bm25_chunk_ids: list[str] = []
        self.weights: dict[str, float] = dict(self.DEFAULT_WEIGHTS)
        self.reranker = reranker              # CrossEncoderReranker | None
        self.rerank_enabled = reranker is not None
        # Telemetry of the LAST search call (dict per rerank stage run) —
        # the retriever node folds it into its trace.
        self.last_rerank: dict | list | None = None

    # ------------------------------------------------------------- overrides
    def apply_weights(self, **overrides: float | None) -> dict[str, float]:
        """Set weight overrides (None/omitted → default). Returns the previous
        weights so callers can restore them (used by the ablation harness)."""
        previous = dict(self.weights)
        self.weights = {k: float(v) if v is not None else d
                        for k, d in self.DEFAULT_WEIGHTS.items()
                        for v in [overrides.get(k)]}
        return previous

    def restore_weights(self, previous: dict[str, float]) -> None:
        self.weights = dict(previous)

    # ------------------------------------------------------------------ sync
    def rebuild_lexical(self) -> None:
        """(Re)build the BM25 index from the current store contents."""
        self._bm25 = BM25Index()
        self._bm25_chunk_ids = []
        for c in self.store.snapshot():
            self._bm25.add(f"{c.text} {c.section} {c.doc}")
            self._bm25_chunk_ids.append(c.chunk_id)
        # Feed corpus IDF statistics to the cross-encoder surrogate so its
        # coverage feature weights rare terms like BM25 does.
        if self.reranker is not None:
            self.reranker.set_idf(self._bm25.idf_map())
        logger.debug("BM25 index rebuilt: %d docs", len(self._bm25_chunk_ids))

    # ---------------------------------------------------------------- search
    async def search(self, query: str, k: int = 6) -> list[RetrievedChunk]:
        if self.store.is_empty():
            return []
        if len(self._bm25) != self.store.size:
            self.rebuild_lexical()

        from app.llm.simulation import (
            _ENTITY_CODE_RE,
            _FIGURE_QUERY_RE,
            _QUANT_EVIDENCE_RE,
            _QUANT_QUERY_RE,
            _content_terms,
            _expanded_stems,
            _stem,
        )

        # Entity codes (X200, S350, E103 …) are the most discriminative query
        # terms but the hashing embedder is blind to them (char-n-gram overlap
        # makes x200 ≈ x300) and BM25 dilutes the code token that its own
        # doc-name injection spreads across the index. Codes are weighted
        # double in coverage, exact-code chunks get a ranking boost, and
        # code-named documents are added to the candidate pool.
        q_codes = {c.lower() for c in _ENTITY_CODE_RE.findall(query)}

        dense = self.store.dense_search(query, k=k)
        dense_ids = [c.chunk_id for c, _ in dense]
        bm25_hits = self._bm25.search_top(query, k * 3)

        id_to_chunk: dict[str, Chunk] = {c.chunk_id: c for c, _ in dense}
        bm25_rank_map: dict[str, int] = {}
        for doc_id, _ in bm25_hits:
            cid = self._bm25_chunk_ids[doc_id]
            if cid not in bm25_rank_map:
                bm25_rank_map[cid] = len(bm25_rank_map) + 1
            if cid not in id_to_chunk:
                match = self.store.get_by_chunk_id(cid)
                if match is not None:
                    id_to_chunk[cid] = match

        # ---- RRF fusion -----------------------------------------------------
        dense_rank_map = {cid: r + 1 for r, cid in enumerate(dense_ids)}
        rrf: dict[str, float] = {}
        for cid, rank in dense_rank_map.items():
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
        for cid, rank in bm25_rank_map.items():
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)

        # ---- entity-code pool expansion --------------------------------------
        # A doc whose NAME carries the asked code (x300_controller_manual.md)
        # is definitionally a candidate, even when both retrievers miss it
        # (hashing embedder is code-blind; BM25 dilutes the code token that
        # its own doc-name injection spreads across the index).
        if q_codes:
            for chunk in self.store.snapshot():
                if chunk.chunk_id in rrf:
                    continue
                doc_codes = {c.lower() for c in _ENTITY_CODE_RE.findall(chunk.doc)}
                if doc_codes & q_codes:
                    rrf[chunk.chunk_id] = 1.0 / (self.rrf_k + 40)  # modest synthetic rank
                    id_to_chunk[chunk.chunk_id] = chunk
        max_rrf = max(rrf.values(), default=0.0) or 1.0

        # ---- coverage reranker ----------------------------------------------
        # Stemmed + synonym-expanded coverage: "processing time" in a question
        # is covered by "evaluation takes 48 hours" in a chunk. Synonym hits
        # count 0.8 so literal matches still dominate; entity-code terms count
        # double (discriminative terms of technical questions).
        q_stems = {_stem(t) for t in _content_terms(query)}
        syn_stems = _expanded_stems(query) - q_stems
        quant_q = _QUANT_QUERY_RE.search(query)
        figure_q = _FIGURE_QUERY_RE.search(query)
        term_weight = {t: (2.0 if t in q_codes else 1.0) for t in q_stems}
        total_weight = sum(term_weight.values()) or 1.0
        scored: list[RetrievedChunk] = []
        for cid, fused in rrf.items():
            chunk = id_to_chunk.get(cid)
            if chunk is None:
                continue
            chunk_stems = set(tokenize(f"{chunk.text} {chunk.section} {chunk.doc}", drop_stopwords=False))
            chunk_stems = {_stem(t) for t in chunk_stems if len(t) > 2 or t.isdigit()}
            cov_hits = sum(term_weight.get(t, 1.0) for t in q_stems & chunk_stems) \
                + 0.8 * len(syn_stems & chunk_stems)
            coverage = min(1.0, cov_hits / total_weight) if q_stems else 0.0
            # Agreement bonus: surfaced by BOTH retrievers = strong signal.
            agreement = 1.0 if (cid in dense_rank_map and cid in bm25_rank_map) else 0.0
            w = self.weights
            final = w["w_fused"] * (fused / max_rrf) + w["w_coverage"] * coverage + w["w_agreement"] * agreement
            # Figure-caption pseudo-chunks (OCR'd chart text) are supporting
            # evidence, not primary prose: demote them unless the query is
            # explicitly about a visual.
            if chunk.is_figure_caption and not figure_q:
                final *= 0.72
            # Quantitative-question reranker: measurable questions deserve chunks
            # carrying numeric evidence ("24 months", "±0.3 °C", "48-hour").
            if quant_q and _QUANT_EVIDENCE_RE.search(chunk.text):
                final = min(1.0, final * 1.28)
            # Exact entity-code match: chunk text or doc name carries the very
            # code the query asks about (see comment above).
            if q_codes:
                chunk_codes = {c.lower() for c in _ENTITY_CODE_RE.findall(chunk.text)}
                chunk_codes |= {c.lower() for c in _ENTITY_CODE_RE.findall(chunk.doc)}
                if chunk_codes & q_codes:
                    final = min(1.0, final * 1.35)
            scored.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=round(min(final, 1.0), 4),
                    dense_rank=dense_rank_map.get(cid, 0),
                    bm25_rank=bm25_rank_map.get(cid, 0),
                    coverage=round(coverage, 4),
                )
            )
        scored.sort(key=lambda r: r.score, reverse=True)

        # ---- cross-encoder rerank stage -------------------------------------
        # The top max(k, n) fusion candidates are re-scored by jointly reading
        # the (query, chunk) pair; the stage is toggleable for ablation
        # (fusion-only) and skips itself when there is nothing to rerank.
        if (
            self.reranker is not None
            and self.rerank_enabled
            and scored
        ):
            pool = scored[: max(self.reranker.n, k)]
            ranked, telemetry = await self.reranker.rerank(query, pool)
            for i, rc in enumerate(pool):
                rc.fusion_rank = i + 1
            for item in telemetry.get("scores", []):
                rc = next((r for r in pool if r.chunk.chunk_id == item["chunk_id"]), None)
                if rc is not None:
                    rc.rerank_score = item["rerank_score"]
            self.last_rerank = telemetry
            return ranked[:k]
        self.last_rerank = None
        return scored[:k]
