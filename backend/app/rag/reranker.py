"""Cross-encoder reranking stage (Part B "plus" + Task-15 priority #3).

A bi-encoder pipeline (dense hashing embedder + BM25 + RRF fusion) scores the
query and the document *independently* — fast, but blind to how the two
actually interact. A cross-encoder jointly reads the (query, document) pair
and scores relevance directly; it is the standard precision stage after
first-stage retrieval (bge-reranker / ms-marco-MiniLM family).

This module implements the pattern in BOTH serving modes:

* **simulation** — a deterministic *lexical cross-encoder surrogate*: instead
  of a neural joint representation it computes pair-interaction features that
  only exist when query and chunk are read together (verbatim phrase
  containment, sentence-level answer-likeness, term-proximity window,
  IDF-weighted coverage, bigram Jaccard, entity-code joint match, numeric
  evidence for quantitative questions). Same interface, same call site —
  swap-in ready for the real model.
* **vLLM** — ``LLMCrossEncoder`` prompts the served chat model to score the
  pair 0–100 (the actual cross-attention lives in the LLM); any error
  degrades to the surrogate so the graph never fails on the reranker.

The stage runs AFTER RRF fusion on the top ``rerank_n`` candidates and
re-sorts them; ``fusion_rank`` vs final order makes every reorder observable
in the /search response and the retriever node trace.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from app.rag.chunker import Chunk

logger = logging.getLogger(__name__)

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+|\n+")
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


# --------------------------------------------------------------------- helpers
def _sentence_windows(text: str, min_len: int = 24) -> list[str]:
    """Sentences (and short-line units) of a chunk — the cross-encoder's
    'attention' targets: a chunk answers a question when one of its
    sentences does."""
    units = [u.strip() for u in _SENT_SPLIT_RE.split(text) if len(u.strip()) >= min_len]
    return units or ([text.strip()] if text.strip() else [])


@dataclass
class RerankResult:
    """Score + per-feature explanation for one (query, chunk) pair."""
    score: float                     # final cross-encoder score, [0, 1]
    features: dict[str, float] = field(default_factory=dict)
    best_sentence: str = ""          # the sentence that earned answer-likeness


class LexicalCrossEncoder:
    """Deterministic joint scorer — the simulation-mode cross-encoder.

    Features (all computed on the PAIR, not on either side alone):
      phrase      — verbatim query phrase containment in the chunk
      sentence    — best single sentence covering query content terms
                    (weighted by how much of the query it covers + numeric
                    evidence for measurable questions + entity codes)
      proximity   — how tightly the matched terms cluster in the chunk
      idf_cov     — IDF-weighted query-term coverage (rare terms matter)
      bigram      — bigram Jaccard between query and chunk
      code        — entity-code joint match (query code present in chunk/doc)
    """

    def __init__(self, idf: dict[str, float] | None = None) -> None:
        # BM25-style IDF map, injected by the retriever after lexical rebuild
        self.idf = idf or {}

    def score_sync(self, query: str, chunk: Chunk) -> RerankResult:
        """Deterministic joint scoring of the (query, chunk) pair."""
        from app.llm.simulation import (
            _ENTITY_CODE_RE,
            _QUANT_EVIDENCE_RE,
            _QUANT_QUERY_RE,
            _SYNONYMS,
            _content_terms,
            _expanded_stems,
            _stem,
        )
        from app.rag.embeddings import tokenize

        q_terms = _content_terms(query)
        q_stems = {_stem(t) for t in q_terms}
        # Synonym expansion mirrors the fusion coverage heuristic: a paraphrased
        # query ("processing time") must see chunks that say "evaluation takes".
        # Synonym hits count 0.5 in the joint scorer — literal matches and rare
        # (high-IDF) terms dominate, synonyms only rescue paraphrases.
        syn_stems = _expanded_stems(query) - q_stems
        q_codes = {c.lower() for c in _ENTITY_CODE_RE.findall(query)}
        quant_q = _QUANT_QUERY_RE.search(query)

        # Anchor term: the query's RAREST content stem — the question's focus.
        # A sentence is answer-like only when it carries the anchor (literally
        # or via the anchor's own synonyms): "accuracy of the S200 sensor" →
        # answer sentences contain "accuracy", not merely "S200" + "sensor"
        # (the maintenance schedule has those too, but about calibration).
        anchor = max(
            q_stems,
            key=lambda s: (self.idf.get(s, 1.5), s),
        ) if q_stems else ""
        anchor_syns = {
            _stem(w)
            for t in q_terms
            if _stem(t) == anchor
            for syn in _SYNONYMS.get(t, [])
            for w in syn.split()
        } - {anchor}
        total_idf = sum(self.idf.get(s, 1.5) for s in q_stems) or 1.0

        chunk_stems = {
            _stem(t)
            for t in tokenize(f"{chunk.text} {chunk.section} {chunk.doc}", drop_stopwords=False)
            if len(t) > 2 or t.isdigit()
        }
        chunk_lower = f"{chunk.text} {chunk.section} {chunk.doc}".lower()
        query_lower = " ".join(query.lower().split())

        f: dict[str, float] = {}

        # -- phrase containment: query substrings appearing verbatim ---------
        phrase_hits = 0
        phrase_total = 0
        for n in (4, 3):
            q_words = query_lower.split()
            for i in range(len(q_words) - n + 1):
                gram = " ".join(q_words[i:i + n])
                if any(w.isdigit() or len(w) > 3 for w in q_words[i:i + n]):
                    phrase_total += 1
                    if gram in chunk_lower:
                        phrase_hits += 1
        f["phrase"] = min(1.0, phrase_hits / phrase_total) if phrase_total else 0.0

        # -- sentence-level answer-likeness (the joint core) ------------------
        # IDF-weighted coverage over sentences that carry the anchor term.
        # This is the cross-attention surrogate: the pair (query, sentence)
        # is scored, not the chunk as a bag of words.
        best_cov = 0.0
        best_sentence = ""
        units = _sentence_windows(chunk.text)
        for unit in units:
            unit_stems = {
                _stem(t) for t in tokenize(unit, drop_stopwords=False)
                if len(t) > 2 or t.isdigit()
            }
            if not unit_stems:
                continue
            # Anchor gate: the sentence must address the question's focus.
            if anchor and anchor not in unit_stems and not (unit_stems & anchor_syns):
                continue
            hits = q_stems & unit_stems
            syn_hits = syn_stems & unit_stems
            if not hits and not syn_hits:
                continue
            cov = (
                sum(self.idf.get(t, 1.5) for t in hits)
                + 0.5 * sum(self.idf.get(s, 1.2) for s in syn_hits)
            ) / total_idf
            unit_codes = {c.lower() for c in _ENTITY_CODE_RE.findall(unit)}
            if q_codes and (unit_codes & q_codes):
                cov += 0.12
            if quant_q and _QUANT_EVIDENCE_RE.search(unit):
                cov += 0.10
            cov = min(1.0, cov)
            if cov > best_cov:
                best_cov = cov
                best_sentence = unit
        f["sentence"] = best_cov

        # -- proximity: span of matched stems in chunk token order -----------
        tokens = tokenize(chunk.text, drop_stopwords=False)
        positions = [
            i for i, t in enumerate(tokens)
            if _stem(t) in q_stems or _stem(t) in syn_stems
        ]
        if len(positions) >= 2:
            span = positions[-1] - positions[0] + 1
            density = len(positions) / span
            f["proximity"] = min(1.0, density)
        elif len(positions) == 1:
            f["proximity"] = 0.35
        else:
            f["proximity"] = 0.0

        # -- IDF-weighted coverage (synonym hits at 0.5 — literal dominates) --
        if q_stems:
            num = sum(self.idf.get(t, 1.5) for t in q_stems & chunk_stems) \
                + 0.5 * sum(self.idf.get(t, 1.2) for t in syn_stems & chunk_stems)
            den = sum(self.idf.get(t, 1.5) for t in q_stems)
            f["idf_cov"] = min(1.0, num / den) if den else 0.0
        else:
            f["idf_cov"] = 0.0

        # -- bigram Jaccard ----------------------------------------------------
        q_bigrams = {
            (q_words[i], q_words[i + 1])
            for q_words in [query_lower.split()]
            for i in range(len(q_words) - 1)
        }
        c_bigrams = set(zip(chunk_lower.split(), chunk_lower.split()[1:]))
        if q_bigrams:
            f["bigram"] = len(q_bigrams & c_bigrams) / len(q_bigrams)
        else:
            f["bigram"] = 0.0

        # -- entity-code joint match -------------------------------------------
        if q_codes:
            chunk_codes = {c.lower() for c in _ENTITY_CODE_RE.findall(chunk.text)}
            chunk_codes |= {c.lower() for c in _ENTITY_CODE_RE.findall(chunk.doc)}
            f["code"] = min(1.0, len(q_codes & chunk_codes) / len(q_codes))
        else:
            f["code"] = 0.5  # neutral when the query has no codes

        # -- weighted blend (phrase & sentence are the joint signals) ----------
        score = (
            0.26 * f["sentence"]
            + 0.20 * f["phrase"]
            + 0.18 * f["idf_cov"]
            + 0.14 * f["proximity"]
            + 0.10 * f["bigram"]
            + 0.12 * f["code"]
        )
        # A sentence that carries numeric evidence for a measurable query is
        # the classic "answer sentence" — small final nudge.
        if quant_q and best_sentence and _QUANT_EVIDENCE_RE.search(best_sentence):
            score = min(1.0, score + 0.06)
        return RerankResult(
            score=round(min(1.0, score), 4),
            features={k: round(v, 4) for k, v in f.items()},
            best_sentence=best_sentence[:200],
        )


class LLMCrossEncoder:
    """vLLM-served cross-encoder: the model jointly reads query + chunk and
    scores relevance 0–100. Degrades to the lexical surrogate on any error
    (network, parse, refusal) — the reranker must never break the graph.

    ``backend_provider`` is a callable returning the *active* LLM backend
    (or None while in simulation mode) so the stage follows the service's
    hot-swap between simulation and vLLM."""

    PROMPT = (
        "You are a relevance judge. Score how well the DOCUMENT answers the "
        "QUESTION. Answer with a single integer 0-100 only.\n\n"
        "QUESTION: {query}\n\nDOCUMENT: {chunk}"
    )

    def __init__(self, backend_provider, fallback: LexicalCrossEncoder) -> None:
        self.backend_provider = backend_provider
        self.fallback = fallback

    async def score(self, query: str, chunk: Chunk) -> RerankResult:
        backend = self.backend_provider()
        if backend is None:
            return self.fallback.score_sync(query, chunk)
        try:
            text = (chunk.text or "")[:1500]
            raw = await backend.chat(
                [{"role": "user", "content": self.PROMPT.format(query=query, chunk=text)}],
                temperature=0.0, max_tokens=8,
            )
            m = re.search(r"\b(100|[0-9]{1,2})\b", raw or "")
            if m is None:
                raise ValueError(f"unparseable rerank score: {raw!r}")
            value = float(m.group(1)) / 100.0
            return RerankResult(score=round(min(1.0, value), 4),
                                features={"llm": round(value, 4)})
        except Exception as exc:  # noqa: BLE001 — degrade, never fail
            logger.warning("LLM rerank failed (%s) — lexical surrogate used", exc)
            return self.fallback.score_sync(query, chunk)


class CrossEncoderReranker:
    """The stage the retriever calls: rerank top fusion candidates.

    ``backend_provider`` (optional) returns the active vLLM backend or None —
    in simulation mode the lexical surrogate scores every pair; when vLLM is
    reachable the LLM scores them (per-call fallback keeps the graph safe).
    ``n`` is how many fusion candidates get cross-encoded (standard reranker
    practice: recall@large → rerank@small). Latency + reorder telemetry is
    returned alongside the ranked chunks.
    """

    def __init__(self, n: int = 10, backend_provider=None) -> None:
        self.n = n
        self.lexical = LexicalCrossEncoder()
        self.backend_provider = backend_provider
        self.llm = LLMCrossEncoder(backend_provider, self.lexical)
        self.calls = 0
        self.total_ms = 0.0
        self.reorders = 0
        self.llm_scores = 0

    @property
    def mode(self) -> str:
        return (
            "vllm"
            if self.backend_provider is not None and self.backend_provider() is not None
            else "lexical"
        )

    def set_idf(self, idf: dict[str, float]) -> None:
        self.lexical.idf = idf

    async def rerank(self, query: str, candidates: list) -> tuple[list, dict[str, Any]]:
        """Cross-score ``candidates`` (RetrievedChunk, fusion order) and return
        (re-ranked list, telemetry). Non-candidates (beyond n) keep fusion
        order AFTER the reranked block."""
        import time as _time

        if not candidates:
            return [], {"reranked": 0, "reorders": 0, "latency_ms": 0.0, "scores": []}

        t0 = _time.perf_counter()
        pool = candidates[: self.n]
        tail = candidates[self.n:]
        scorer = self.llm if self.mode == "vllm" else self.lexical
        use_llm = self.mode == "vllm"
        scored: list[tuple[float, Any]] = []
        for rc in pool:
            result = (
                await scorer.score(query, rc.chunk) if use_llm
                else scorer.score_sync(query, rc.chunk)
            )
            scored.append((result.score, rc, result))
        self.calls += len(pool)
        if use_llm:
            self.llm_scores += len(pool)

        # Stable sort by score desc; fusion order breaks ties.
        order = sorted(range(len(scored)), key=lambda i: -scored[i][0])
        reranked = [scored[i][1] for i in order]
        reorders = sum(
            1 for new_pos, i in enumerate(order) if i != new_pos
        )
        self.reorders += reorders
        self.total_ms += (_time.perf_counter() - t0) * 1000

        telemetry = {
            "reranked": len(pool),
            "reorders": reorders,
            "latency_ms": round(((_time.perf_counter() - t0) * 1000), 2),
            "scores": [
                {
                    "chunk_id": scored[i][1].chunk.chunk_id,
                    "doc": scored[i][1].chunk.doc,
                    "page": scored[i][1].chunk.page,
                    "rerank_score": scored[i][0],
                    "fusion_rank": i + 1,
                    "features": scored[i][2].features,
                    "best_sentence": scored[i][2].best_sentence,
                }
                for i in order
            ],
        }
        return reranked + tail, telemetry

    def stats(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "n": self.n,
            "calls": self.calls,
            "llm_scores": self.llm_scores,
            "reorders": self.reorders,
            "avg_latency_ms": round(self.total_ms / self.calls, 3) if self.calls else 0.0,
        }
