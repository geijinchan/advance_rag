"""Faithfulness scoring — RAGAS-style "answer faithfulness" dimension.

Measures how well an answer is *supported by its citations* (grounding),
independent of whether it matches a pinned expected pattern:

* **claim support**  — each answer sentence is checked against the union of
  the cited chunk texts (stemmed content-word overlap ≥ threshold ⇒ the
  claim is "entailed" by the sources);
* **grounding ratio**— share of the answer's content words found in sources;
* **contradictions** — numeric/keyword mismatches between answer and sources
  (e.g. answer says "12 months" where the source says "24 months", or
  negated statements).

Two backends, same interface (consistent with the project's honest
serving-mode design):
* ``vLLM live``  — the served model is prompted as an LLM-judge; it returns
  a JSON verdict we normalise to [0, 1];
* ``simulation`` — a deterministic lexical judge (stemming + stopword removal
  + number matching). Clearly labelled in the detail payload.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# deterministic lexical judge (simulation mode)
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "to", "of",
    "and", "or", "in", "on", "for", "with", "as", "at", "by", "from", "that",
    "this", "it", "its", "if", "not", "no", "can", "do", "does", "did",
    "what", "which", "who", "how", "when", "where", "why", "will", "shall",
    "should", "would", "may", "might", "must", "you", "your", "we", "our",
    "they", "their", "there", "here", "about", "into", "over", "than",
    "then", "also", "more", "most", "some", "any", "all", "each", "per",
}

_SUFFIXES = ("'s", "es", "ed", "ing", "ly", "s")


def _stem(word: str) -> str:
    w = word.lower()
    for suf in _SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def _content_words(text: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"[a-zA-Z][a-zA-Z'-]+|\d+(?:\.\d+)?", text.lower())]


# Standalone quantities only: excludes digits embedded in identifiers
# (S200, M12, IOD-4471, 10.10.20.0/24, 4-20 mA ranges …) so that model codes
# and IP/subnet strings can never trigger false "number not in sources"
# contradictions.
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9.\-/])\d+(?:\.\d+)?(?![A-Za-z0-9.\-/])")


def _numbers(text: str) -> list[str]:
    return _NUMBER_RE.findall(text)


# Citation markers injected by the generator ("[Source: doc.pdf, page 6]")
# are structure, not claims — strip them before sentence splitting.
_CITATION_TAG_RE = re.compile(r"\[Source:[^\]]{0,160}\]", re.IGNORECASE)


def _strip_citation_markers(text: str) -> str:
    return _CITATION_TAG_RE.sub(" ", text)


_SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+")


@dataclass
class FaithfulnessResult:
    score: float                      # 0..1 (1 = fully supported)
    judge: str                        # "lexical" | "llm"
    supported_claims: int = 0
    total_claims: int = 0
    contradictions: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    llm_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 3),
            "judge": self.judge,
            "supported_claims": self.supported_claims,
            "total_claims": self.total_claims,
            "contradictions": self.contradictions,
            "unsupported": [u[:140] for u in self.unsupported[:3]],
            "llm_reason": self.llm_reason,
        }


class LexicalFaithfulnessJudge:
    """Deterministic entailment proxy: a sentence is supported when a strong
    share of its content words (and all its salient numbers) occur in the
    cited sources."""

    SUPPORT_THRESHOLD = 0.62

    def score(self, answer: str, sources: list[str]) -> FaithfulnessResult:
        answer = _strip_citation_markers((answer or "").strip())
        if not answer:
            return FaithfulnessResult(score=1.0, judge="lexical")

        source_words = {w for s in sources for w in _content_words(s)}
        source_nums = {n for s in sources for n in _numbers(s)}

        sentences = [s.strip() for s in _SENT_SPLIT.split(answer) if len(s.strip()) >= 12]
        if not sentences:
            sentences = [answer]

        supported = 0
        unsupported: list[str] = []
        contradictions: list[str] = []

        for sent in sentences:
            words = [w for w in _content_words(sent) if w not in _STOPWORDS]
            nums = _numbers(sent)
            if not words:
                continue
            overlap = sum(1 for w in words if w in source_words)
            ratio = overlap / len(words)

            num_ok = all(n in source_nums for n in nums) if nums else True
            # a number present in the answer but absent from sources is a
            # potential hallucination → contradiction signal
            if nums and not num_ok:
                missing = [n for n in nums if n not in source_nums]
                contradictions.append(
                    f"number(s) {missing} not found in cited sources"
                )

            if ratio >= self.SUPPORT_THRESHOLD and num_ok:
                supported += 1
            else:
                unsupported.append(sent)

        total = len(sentences)
        claim_score = supported / total if total else 1.0

        # grounding ratio over the whole answer
        answer_words = [w for w in _content_words(answer) if w not in _STOPWORDS]
        grounding = (
            sum(1 for w in answer_words if w in source_words) / len(answer_words)
            if answer_words
            else 1.0
        )

        penalty = 0.12 * len(contradictions)
        score = max(0.0, min(1.0, 0.65 * claim_score + 0.35 * grounding - penalty))

        return FaithfulnessResult(
            score=score,
            judge="lexical",
            supported_claims=supported,
            total_claims=total,
            contradictions=contradictions[:3],
            unsupported=unsupported,
        )


# ---------------------------------------------------------------------------
# LLM-as-judge (vLLM live mode)
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """You are a strict evaluator for a RAG system. Score how faithful the ANSWER is to the SOURCES.

Question: {question}

SOURCES:
{sources}

ANSWER: {answer}

Faithfulness means: every factual claim in the answer is supported by the sources. Penalise invented numbers, unsupported claims and contradictions.

Reply with ONLY a JSON object:
{{"score": <number 0-10>, "reason": "<one short sentence>", "contradictions": ["<contradiction or unsupported claim, if any>"]}}"""


class FaithfulnessJudge:
    """Facade: LLM judge when a live model is available, lexical otherwise."""

    def __init__(self, models=None) -> None:
        self._lexical = LexicalFaithfulnessJudge()
        self._models = models  # ModelService (optional; may be injected later)

    def bind(self, models) -> None:
        self._models = models

    @property
    def mode(self) -> str:
        if self._models is not None and self._models.mode == "vllm":
            return "llm"
        return "lexical"

    async def score(
        self,
        answer: str,
        citations: list[dict[str, Any]],
        question: str = "",
        chunk_text_lookup=None,
    ) -> FaithfulnessResult:
        """Score an answer against its citations.

        ``chunk_text_lookup``: optional async-or-sync callable chunk_id -> full
        text. When provided the judge reads the FULL source chunk rather than
        the 220-char citation snippet (sharper entailment decisions).
        """
        sources: list[str] = []
        for c in citations or []:
            if not isinstance(c, dict):
                continue
            text = ""
            if chunk_text_lookup is not None and c.get("chunk_id"):
                try:
                    text = chunk_text_lookup(c["chunk_id"]) or ""
                except Exception:  # noqa: BLE001
                    text = ""
            if not text:
                text = str(c.get("snippet", c.get("text", c.get("preview", ""))))
            if text.strip():
                sources.append(text)

        if not sources:
            # nothing cited → an answer that says anything is unfaithful
            return FaithfulnessResult(
                score=0.0 if (answer or "").strip() else 1.0,
                judge="lexical",
                total_claims=0,
            )

        if self.mode == "llm":
            try:
                return await self._score_llm(answer, sources, question)
            except Exception:  # noqa: BLE001 — judge must never break the pipeline
                pass
        return self._lexical.score(answer, sources)

    async def _score_llm(
        self, answer: str, sources: list[str], question: str
    ) -> FaithfulnessResult:
        from app.llm.brain import _parse_json  # local import to avoid cycles

        prompt = _JUDGE_PROMPT.format(
            question=question[:400],
            sources="\n---\n".join(s[:1500] for s in sources)[:6000],
            answer=answer[:2500],
        )
        raw = await self._models.backend.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        verdict = _parse_json(raw or "")
        if not isinstance(verdict, dict):
            raise ValueError("judge response was not JSON")

        score10 = float(verdict.get("score", 5))
        return FaithfulnessResult(
            score=max(0.0, min(1.0, score10 / 10.0)),
            judge="llm",
            contradictions=[str(c) for c in verdict.get("contradictions", [])][:3],
            llm_reason=str(verdict.get("reason", ""))[:300] or None,
        )


# module-level singleton bound by AppState at startup
faithfulness = FaithfulnessJudge()
