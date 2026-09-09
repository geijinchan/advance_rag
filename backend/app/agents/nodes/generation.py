"""Generation, self-verification, citation-building and fallback nodes."""

from __future__ import annotations

import logging
import re

from app.agents.state import AgentState
from app.config import settings
from app.llm.brain import LLMBrain
from app.llm.simulation import HeuristicBrain

logger = logging.getLogger(__name__)

Brain = LLMBrain | HeuristicBrain
END = "__end__"

FALLBACK_TEMPLATE = (
    "I couldn't find this in the provided documents. "
    "{reason}"
)


def _build_citations(state: AgentState, chunks) -> list[dict]:
    """Citation payload for the API response, best-first, deduped by (doc,page).

    Decomposed (comparative) runs cite per-entity: the top two chunks of
    each sub-query group, so both sides of the comparison are traceable —
    rank-order [:4] would let one entity's documents crowd out the other's.
    """
    if state.decomposed and state.sub_queries:
        seen: set[tuple[str, int]] = set()
        citations: list[dict] = []
        for sq in state.sub_queries:
            group = [h for h in chunks
                     if state.sub_query_map.get(h.chunk.chunk_id) == sq]
            for hit in group[:2]:
                key = (hit.chunk.doc, hit.chunk.page)
                if key in seen:
                    continue
                seen.add(key)
                citations.append(
                    {
                        "doc": hit.chunk.doc,
                        "page": hit.chunk.page,
                        "chunk_id": hit.chunk.chunk_id,
                        "snippet": hit.chunk.preview(220),
                        "score": round(hit.score, 4),
                    }
                )
        return citations
    seen: set[tuple[str, int]] = set()
    citations: list[dict] = []
    for hit in chunks[:4]:
        key = (hit.chunk.doc, hit.chunk.page)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            {
                "doc": hit.chunk.doc,
                "page": hit.chunk.page,
                "chunk_id": hit.chunk.chunk_id,
                "snippet": hit.chunk.preview(220),
                "score": round(hit.score, 4),
            }
        )
    return citations


def _select_contexts(relevant_chunks, limit: int = 6, per_doc: int = 3):
    """Diversity-aware context window for generation/verification.

    Takes at most ``limit`` graded-relevant chunks with at most ``per_doc``
    chunks per document. Pure rank-order truncation ([:4]) lets chunks from
    one high-scoring doc crowd out the chunk that actually answers the
    question — observed live: "What is the RMA processing time?" at the
    default top-k=6 had rma_faq.txt at rank 5 behind two maintenance_schedule
    chunks and the answer came from the wrong document. per_doc=3 keeps the
    single-doc depth that table-heavy answers need (E103 lives in the 3rd
    troubleshooting_flowchart chunk) while still forcing cross-doc spread
    when one document floods the ranking.
    """
    picked = []
    per_doc_count: dict[str, int] = {}
    for hit in relevant_chunks:
        doc = hit.chunk.doc
        if per_doc_count.get(doc, 0) >= per_doc:
            continue
        per_doc_count[doc] = per_doc_count.get(doc, 0) + 1
        picked.append(hit)
        if len(picked) >= limit:
            break
    return picked


_ENTITY_CODE_LABEL_RE = re.compile(r"\b([xsc]\d{2,3})\b", re.IGNORECASE)


async def _generate_comparative(state: AgentState, brain: Brain) -> str:
    """Per-entity answer blocks for decomposed (comparative) questions.

    Each sub-query's graded-relevant chunks become that entity's context
    window; the brain answers per entity (so unit selection optimises for
    "standard warranty X200", not the diluted comparative question), and
    the blocks are joined with entity headers. Entities whose grading found
    nothing get an honest per-entity miss line instead of silence.

    When the run is being streamed, each entity block is emitted as answer
    deltas as soon as it is produced — a multi-hop answer visibly arrives
    entity by entity instead of all at once after verification.

    Blocks are capped at two extractive units: comparative questions get
    their signal from the FIRST facts per side — trailing units tend to be
    neighbour table rows (IP-rating matrices, cross-product summaries) that
    merely mention the entity code.
    """
    blocks: list[str] = []
    for sq in state.sub_queries:
        label = sq.rsplit(" ", 1)[-1].upper()
        code = label.lower()
        entity_hits = [
            h for h in state.relevant_chunks
            if state.sub_query_map.get(h.chunk.chunk_id) == sq
        ]
        # Entity-owned documents first: "warranty X300" answers best from
        # x300_controller_manual.md — entity-neutral docs (warranty_policy)
        # otherwise win the rank prior and their generic rows crowd out the
        # entity's own warranty table.
        entity_hits = sorted(
            entity_hits,
            key=lambda h: (code in h.chunk.doc.lower(), h.score),
            reverse=True,
        )
        contexts = [
            {
                "doc": h.chunk.doc,
                "page": h.chunk.page,
                "text": h.chunk.text,
                "score": h.score,
                "section": h.chunk.section,
            }
            for h in _select_contexts(entity_hits, limit=4, per_doc=2)
        ]
        if not contexts:
            miss = (
                f"{label} — no passage about this was found in the indexed documents."
            )
            blocks.append(miss)
            await state.emit_delta(("\n\n" if len(blocks) > 1 else "") + miss)
            continue
        header = f"{label} — "
        sep = "\n\n" if blocks else ""
        if state.stream_emitter is not None and hasattr(brain, "generate_answer_stream"):
            # Stream the entity block as it is produced; the final capped
            # block (below) is what the verifier gates and the final event
            # delivers, so a trimmed tail simply settles at finalisation.
            await state.emit_delta(sep + header)
            body = ""
            async for delta, ttft in brain.generate_answer_stream(sq, contexts):
                await state.emit_delta(delta, ttft_ms=ttft)
                body += delta
        else:
            body = await brain.generate_answer(sq, contexts)
        block = _cap_units(body, max_units=2)
        blocks.append(f"{label} — {block}")
    return "\n\n".join(blocks)


_FRAGMENT_TAIL_RE = re.compile(
    r"(?:\b(?:are|is|was|were|the|a|an|and|or|with|from|by|of|to|for|that|which|"
    r"not offered \(already|must|may|should|can)\W*$)|(?:[-–—]\s*$)",
    re.IGNORECASE,
)


def _cap_units(answer: str, max_units: int) -> str:
    """Keep only the first ``max_units`` citation-tagged units of an answer.

    An extractive answer is ``unit [Source: …] unit [Source: …] …`` — units
    are separated AFTER their trailing tag, so splitting on the gap between
    ``]`` and the next word keeps every unit paired with its own citation.
    Units that end mid-clause (a PDF-extraction artefact: table rows and
    sentences split at commas) are dropped — when every unit is fragment-y
    the first is kept so the block never loses its primary fact.
    """
    parts = [p for p in re.split(r"(?<=\])\s+", answer) if p.strip()]
    kept: list[str] = []
    for part in parts:
        if len(kept) >= max_units:
            break
        unit_text = re.sub(r"\s*\[Source:[^\]]*\]\s*$", "", part).strip()
        if len(unit_text) < 40 or _FRAGMENT_TAIL_RE.search(unit_text):
            continue  # fragment-y unit (truncated table row / mid-clause)
        kept.append(part)
    return " ".join(kept).strip() or (parts[0] if parts else answer.strip())


def make_generator_node(brain: Brain):
    async def generator_node(state: AgentState) -> str | None:
        if state.decomposed and state.sub_queries:
            # Comparative path: per-entity context windows + entity-labelled
            # blocks (see _generate_comparative).
            draft = await _generate_comparative(state, brain)
            state.draft_answer = draft
            state.record(
                "generator",
                chars=len(draft),
                decomposed=True,
                sub_queries=state.sub_queries,
                preview=draft[:160],
                streamed=state.deltas_emitted > 0,
            )
            return "verifier"
        contexts = [
            {
                "doc": h.chunk.doc,
                "page": h.chunk.page,
                "text": h.chunk.text,
                "score": h.score,
                "section": h.chunk.section,
            }
            for h in _select_contexts(state.relevant_chunks)
        ]
        # True incremental generation: while the run is streamed, deltas are
        # pushed to the client as they are produced (before verification).
        # The verifier still gates the FINAL answer — if it rejects the draft,
        # the final event replaces the streamed text with the honest fallback.
        if state.stream_emitter is not None and hasattr(brain, "generate_answer_stream"):
            parts: list[str] = []
            async for delta, ttft in brain.generate_answer_stream(state.question, contexts):
                await state.emit_delta(delta, ttft_ms=ttft)
                parts.append(delta)
            draft = "".join(parts)
        else:
            draft = await brain.generate_answer(state.question, contexts)
        state.draft_answer = draft
        state.record(
            "generator",
            chars=len(draft),
            contexts=[f"{c['doc']}#p{c['page']}" for c in contexts],
            preview=draft[:160],
            streamed=state.deltas_emitted > 0,
            ttft_ms=state.generation_ttft_ms,
        )
        return "verifier"

    return generator_node


def make_verifier_node(brain: Brain):
    """Hallucination gate: only answers supported by the context pass.

    Verification policy (documented in README): a failed verification sends
    the query to the honest fallback instead of a regeneration loop — for
    a grounded assistant, refusing beats a second guess.
    """

    async def verifier_node(state: AgentState) -> str | None:
        contexts = [
            {"doc": h.chunk.doc, "page": h.chunk.page, "text": h.chunk.text}
            for h in _select_contexts(state.relevant_chunks)
        ]
        verdict, support, issues = await brain.verify_answer(state.draft_answer, contexts)
        state.verdict = verdict
        state.support_score = support
        state.verification_issues = issues
        state.record(
            "verifier",
            verdict=verdict,
            support=round(support, 3),
            issues=issues[:3],
        )
        if verdict == "supported":
            state.answer = state.draft_answer
            state.citations = _build_citations(state, state.relevant_chunks)
            return END
        return "fallback"

    return verifier_node


def make_fallback_node():
    async def fallback_node(state: AgentState) -> str | None:
        state.fallback = True
        reasons = []
        if state.route == "out_of_scope":
            reasons.append("the question was classified as out-of-scope for this corpus")
        elif state.grades and state.grade_decision == "not_relevant":
            reasons.append(
                f"retrieval found no passage that the relevance grader accepted "
                f"(best grade {state.grades[0]['grade']:.2f}, "
                f"threshold {settings.grade_threshold}) after {state.retries} rewrite(s)"
            )
        elif state.verdict == "not_supported":
            reasons.append(
                f"the draft answer failed hallucination verification "
                f"(support {state.support_score:.2f} < {settings.verification_support_threshold})"
            )
        reason = (" Reason: " + "; ".join(reasons) + ".") if reasons else ""
        state.answer = FALLBACK_TEMPLATE.format(reason=reason)
        # Even on fallback we surface the best candidates found — helps users
        # debug why the assistant refused (Part D debuggability requirement).
        if state.retrieved:
            state.citations = _build_citations(state, state.retrieved[:2])
        state.record("fallback", reason="; ".join(reasons) or "unspecified")
        return END

    return fallback_node
