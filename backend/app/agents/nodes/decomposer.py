"""Query decomposition node — comparative / multi-entity questions (Part C+).

A question like "Compare the standard warranty of the X200 and the X300" is
multi-hop for lexical retrieval: no single chunk mentions both models, and a
fused ranking of the raw query tends to flood with chunks from whichever
document scores highest, starving the other entity of context. The decomposer
detects such questions and splits them into one sub-query per entity so the
retriever can gather balanced evidence per side, the grader can judge each
chunk against the sub-query it was fetched for, and the generator can emit a
structured per-entity answer.

Detection is deliberately deterministic (regex over comparative cues + entity
codes), so the behaviour is explainable and regression-testable; in vLLM mode
the same decomposition feeds prompt-driven generation, where the LLM is asked
to answer per entity with the balanced context blocks.
"""

from __future__ import annotations

import logging
import re

from app.agents.state import AgentState

logger = logging.getLogger(__name__)

_COMPARATIVE_RE = re.compile(
    r"\b(compare|comparison|difference|differences|differ|versus|vs\.?|"
    r"which of|both|either|respective)\b",
    re.IGNORECASE,
)
# e.g. "between the X200 and the X300" / "of the X100 vs the S200"
_BETWEEN_RE = re.compile(r"\bbetween\s+(?:the\s+)?([a-z]+\d+)\s+(?:and|vs\.?|versus|or)\s+(?:the\s+)?([a-z]+\d+)\b", re.IGNORECASE)

_ENTITY_RE = re.compile(r"\b(?:[xsc]\d{2,3})\b", re.IGNORECASE)

_MAX_SUB_QUERIES = 3


def _extract_entities(question: str) -> list[str]:
    """Distinct entity codes in first-appearance order (x200, s350 …)."""
    seen: set[str] = set()
    entities: list[str] = []
    for m in _ENTITY_RE.finditer(question):
        code = m.group(0).lower()
        if code not in seen:
            seen.add(code)
            entities.append(m.group(0))
    return entities


_TOPIC_STOPWORDS = re.compile(
    r"\b(and|or|the|a|an|of|to|for|with|is|are|do|does|did|their|each|"
    r"what|which|who|how|when|where|why|between|on|in|at|from|by|"
    r"model|models|controller|sensor|device|controllers|sensors|devices|"
    r"both|either|give|tell|list|compare|comparison|differ|different|differences|"
    r"long|much|many|there|that|this|these|those|it|its)\b",
    re.IGNORECASE,
)


def _topic_phrase(question: str, entities: list[str]) -> str:
    """The asked-about topic with comparative glue and entity mentions removed.

    "Compare the standard warranty of the X200 and the X300" →
    "standard warranty" — then each sub-query becomes e.g.
    "standard warranty X200". Repeated topic words are collapsed so
    "the warranty on the X100 and the warranty on the X300" → "warranty".
    """
    q = question
    q = _BETWEEN_RE.sub(" ", q)
    for code in entities:
        q = re.sub(re.escape(code), " ", q, flags=re.IGNORECASE)
    q = _COMPARATIVE_RE.sub(" ", q)
    q = _TOPIC_STOPWORDS.sub(" ", q)
    words: list[str] = []
    seen: set[str] = set()
    for w in q.split():
        if len(w) > 2 and w.lower() not in seen:
            seen.add(w.lower())
            words.append(w)
    phrase = " ".join(words).strip()
    return phrase or "specifications"


def detect_decomposition(question: str) -> list[str] | None:
    """Return per-entity sub-queries, or None when the question is single-hop.

    Rules (all must hold):
      * a comparative cue is present (compare / difference / versus / …), OR
        an explicit "between A and B" pattern with two entity codes;
      * ≥ 2 distinct entity codes appear in the question;
      * the result would be 2–3 sub-queries (more entities → capped, the
        rare 4-entity question falls back to normal retrieval).
    """
    if not _COMPARATIVE_RE.search(question) and not _BETWEEN_RE.search(question):
        return None
    entities = _extract_entities(question)
    if len(entities) < 2 or len(entities) > _MAX_SUB_QUERIES:
        return None
    topic = _topic_phrase(question, entities)
    return [f"{topic} {code}" for code in entities]


# ---- LLM-driven decomposition (vLLM mode) ---------------------------------
#
# The regex detector is explainable and regression-safe, but it only fires on
# known comparative cues + corpus entity codes. When a real LLM is served, we
# ask it to decompose instead — this generalises multi-hop detection to
# paraphrases and concept-pair comparisons the regexes do not know. The output
# is validated hard (2–3 non-trivial sub-queries) and any doubt falls back to
# the deterministic path, so the LLM can only ADD decompositions, never break
# the guaranteed ones.

_LLM_DECOMPOSER_SYSTEM = """You decompose multi-hop questions for a retrieval
pipeline over industrial documents (controllers X100/X200/X300, sensors
S200/S350, warranty/RMA, firmware, maintenance, safety, networking).

If the question compares or combines DISTINCT entities/topics that no single
document would cover together, split it into 2 or 3 self-contained sub-questions,
each naming ONE entity and the shared topic (e.g. "standard warranty X200").
If the question is single-hop, answer exactly: [].

Respond ONLY with minified JSON: {"sub_queries": ["...", "..."]}"""

_MIN_SUB_QUERY_LEN = 12


async def llm_decompose(brain, question: str) -> list[str] | None:
    """Prompt-driven decomposition with strict validation; None = unusable."""
    from app.llm.brain import _parse_json  # shared tolerant JSON extraction

    try:
        raw = await brain.backend.chat(
            [
                {"role": "system", "content": _LLM_DECOMPOSER_SYSTEM},
                {"role": "user", "content": question},
            ],
            max_tokens=160,
            temperature=0.0,
        )
    except Exception:  # noqa: BLE001 — unreachable vLLM degrades to regex
        return None
    parsed = _parse_json(raw)
    if not parsed or not isinstance(parsed.get("sub_queries"), list):
        return None
    subs = [str(s).strip() for s in parsed["sub_queries"] if str(s).strip()]
    if not 2 <= len(subs) <= _MAX_SUB_QUERIES:
        return None
    if any(len(s) < _MIN_SUB_QUERY_LEN for s in subs):
        return None
    if len(set(s.lower() for s in subs)) != len(subs):
        return None
    return subs


def broad_decomposition_signal(question: str) -> bool:
    """Cheaper high-recall gate used by the router in vLLM mode only.

    Sends any question mentioning ≥ 2 distinct entity codes (or a strong
    comparative cue) to the decomposer node, where the LLM decides whether
    a decomposition actually helps. Single-hop verdicts ([] from the LLM)
    fall straight back to normal retrieval, so the broad gate can only
    add decompositions, never remove the regex-guaranteed ones."""
    if len(_extract_entities(question)) >= 2:
        return True
    return bool(_COMPARATIVE_RE.search(question) and len(question.split()) >= 5)


def make_decomposer_node(brain=None):
    """brain (optional): the LLMBrain when serving in vLLM mode — enables
    prompt-driven decomposition; simulation mode keeps the regex path."""

    async def decomposer_node(state: AgentState) -> str | None:
        query = state.effective_query or state.question
        sub_queries = detect_decomposition(query)
        method = "regex"
        if sub_queries is None and brain is not None and getattr(brain, "backend", None):
            llm_subs = await llm_decompose(brain, query)
            if llm_subs:
                sub_queries = llm_subs
                method = "llm"
        if not sub_queries:
            # The router only sends us here when detection fired at route
            # time; a follow-up-resolved query can differ — degrade gracefully.
            state.record("decomposer", status="skipped", reason="no decomposition detected")
            return "retriever"
        state.sub_queries = sub_queries
        state.decomposed = True
        state.record(
            "decomposer",
            sub_queries=sub_queries,
            entities=[sq.rsplit(" ", 1)[-1].upper() for sq in sub_queries],
            topic=sub_queries[0].rsplit(" ", 1)[0] if " " in sub_queries[0] else "",
            method=method,
        )
        return "retriever"

    return decomposer_node
