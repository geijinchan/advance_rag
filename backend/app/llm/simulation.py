"""Deterministic simulation layer (offline fallback when no vLLM is served).

Two classes live here:

``SimulationBackend``  — implements the :class:`~app.llm.base.LLMBackend`
protocol with extractive answers and simulated token streaming, so the
benchmark harness and SSE endpoint exercise the same code paths offline.

``HeuristicBrain``     — classical-NLP implementations of the agentic
decision points (route / grade / rewrite / generate / verify / image-QA).
Every decision is deterministic and explainable; answers are assembled
strictly from retrieved text, so grounding and citations hold by
construction. Used by the agent nodes when the service runs in simulation
mode, and also as the *fallback parser* for malformed LLM output in vLLM
mode (defence in depth).
"""

from __future__ import annotations

import asyncio
import random
import re
import time
from typing import Any, AsyncIterator

from app.llm.base import ChatMessage, StreamEvent
from app.rag.embeddings import tokenize

# --------------------------------------------------------------------------- 
# Lexicons & pattern banks
# ---------------------------------------------------------------------------
_CHITCHAT_PATTERNS = [
    r"^(hi|hello|hey|yo|good (morning|afternoon|evening))\b",
    r"^(thanks|thank you|cheers|great|awesome|nice)\b",
    r"^(how are you|how('s| is) it going|what('s| is) up)\b",
    r"\b(tell me a joke|who are you|what can you do|introduce yourself)\b",
]
_CHITCHAT_RE = [re.compile(p, re.IGNORECASE) for p in _CHITCHAT_PATTERNS]

_WORLD_KNOWLEDGE_HINTS = [
    "world cup", "olympics", "president", "prime minister", "capital of",
    "who discovered", "who wrote romeo", "population of", "weather", "stock market",
    "bitcoin price", "latest news", "movie", "celebrity", "nobel prize",
]

# Domain synonym table for query rewriting (corpus: industrial device docs).
# Also used to expand term coverage: "processing time" in a question is
# satisfied by "evaluation takes 48 hours" in a chunk (paraphrase gap).
_SYNONYMS: dict[str, list[str]] = {
    "warranty": ["guarantee", "coverage period", "warranty period"],
    "guarantee": ["warranty"],
    "repair": ["fix", "rma", "service", "maintenance"],
    "broken": ["failure", "fault", "error", "defect"],
    "error": ["error code", "fault", "failure"],
    "temperature": ["operating temperature", "thermal", "heat"],
    "install": ["installation", "deployment", "commissioning", "setup"],
    "setup": ["installation", "configuration"],
    "connect": ["network", "integration", "wiring", "modbus", "opc"],
    "network": ["ethernet", "ip", "modbus", "opc ua", "mqtt", "vlan"],
    "sensor": ["s200", "s350", "measurement", "calibration"],
    "battery": ["power supply", "voltage", "24vdc", "48vdc"],
    "upgrade": ["firmware", "update", "flash", "version"],
    "version": ["firmware", "release", "changelog"],
    "led": ["indicator", "status", "light", "pattern"],
    "safe": ["safety", "sil", "iso 13849", "ppe"],
    "return": ["rma", "return merchandise", "warranty claim"],
    "processing": ["evaluation", "turnaround", "handling"],
    "time": ["hours", "duration", "clock", "interval"],
    "mean": ["meaning", "indication", "signify", "indicate"],
    "indicate": ["meaning", "indication", "signal"],
    "long": ["duration", "hours", "time"],
    "take": ["takes", "duration", "evaluation"],
    "due": ["overdue", "interval", "maintenance"],
    "facility": ["rotterdam", "address", "location", "site"],
    "fix": ["fixed", "resolved", "patch", "corrected"],
}

_FILLER_WORDS = {"please", "could", "would", "can", "you", "tell", "me", "about", "for", "the", "a", "an", "is", "what", "which", "how", "does", "do", "did", "any"}


def _content_terms(question: str) -> set[str]:
    """Content-bearing terms: words (len>2) plus numbers (model codes, 24, 36)."""
    return {t for t in tokenize(question) if len(t) > 2 or t.isdigit()} - _FILLER_WORDS


_SUFFIXES = ("ations", "ation", "ional", "ions", "ion", "ities", "ies", "ing", "ed", "es", "s")


def _stem(t: str) -> str:
    """One-pass light suffix stripper (inspection→inspect, inspected→inspect).
    Used ONLY for coverage matching — the index itself stays unstemmed."""
    for suf in _SUFFIXES:
        if t.endswith(suf) and len(t) - len(suf) >= 4:
            return t[: -len(suf)]
    return t


def _stemmed(text: str) -> set[str]:
    return {_stem(t) for t in tokenize(text)}


# Entity codes (model / error codes) in a question are its most
# discriminative terms — units that literally carry the asked-for code
# ("E103 (Code); Ambient …") must lead the assembled answer, not merely be
# included after higher-coverage neighbour rows from the same table.
_ENTITY_CODE_RE = re.compile(r"\b(?:[xsc]\d{2,3}|e1[0-9]{2})\b", re.IGNORECASE)


_QUANT_QUERY_RE = re.compile(
    r"\b(max|maximum|long|many|much|period|months?|years?|days?|hours?|accuracy|precise|precision|tolerance|range|often|frequent|time|takes?|duration|turnaround|wait|due|interval)\b",
    re.I,
)
_QUANT_EVIDENCE_RE = re.compile(
    r"(\d+\s*[- ]?\s*(month|year|day|hour|week|hz)|±\s*[\d.]+|\b\d+(?:\.\d+)?\s*(°c|%|v|ma|bar|mhz|gb|mhz)\b)",
    re.I,
)

# Version questions deserve version-pattern evidence ("v3.2.1");
# negative contexts ("v3.2.0 or earlier must not be used") name the WRONG version.
_VERSION_QUERY_RE = re.compile(r"\b(which|what)\s+(firmware\s+)?version\b|\bversion\b", re.I)
_VERSION_EVIDENCE_RE = re.compile(r"\bv\d+\.\d+(?:\.\d+)?\b", re.I)
_NEG_VERSION_RE = re.compile(
    r"(must\s+not|or\s+earlier|prior\s+to|older\s+than|do\s+not\s+use|not\s+be\s+used|before\s+v?\d)",
    re.I,
)

# Queries explicitly about visuals keep figure-caption chunks competitive.
_FIGURE_QUERY_RE = re.compile(r"\b(figure|chart|diagram|graph|image|plot|illustration|graph)\b", re.I)

# OCR-extracted chart text is noisy (axis labels merge into digit runs);
# such units are supporting evidence, never quotable prose.
_OCR_MARKER_RE = re.compile(r"Visible text:|^\[Figure", re.I)


def _digit_ratio(unit: str) -> float:
    toks = unit.split()
    if not toks:
        return 0.0
    numeric = sum(
        1 for t in toks if t.lstrip("±+-").replace(",", "").replace(".", "").replace("%", "").isdigit()
    )
    return numeric / len(toks)


def _expanded_stems(question: str) -> set[str]:
    """Stems of question content terms PLUS their domain synonyms.

    Paraphrase gap filler: "RMA processing time" should match chunks that
    say "evaluation takes 48 hours" — the synonym expansion makes the
    coverage-based grader and unit scorer see through the phrasing.
    """
    stems: set[str] = set()
    for t in _content_terms(question):
        stems.add(_stem(t))
        for syn in _SYNONYMS.get(t, []):
            for w in syn.split():
                stems.add(_stem(w))
    return stems


def _affirmative_version_in(text: str) -> str | None:
    """First version string ("v3.2.1") in `text` whose surrounding window is
    not a negated mention ("v3.2.0 or earlier must not be used")."""
    for m in _VERSION_EVIDENCE_RE.finditer(text or ""):
        start = max(0, m.start() - 70)
        window = text[start : m.end() + 30]
        if not _NEG_VERSION_RE.search(window):
            return m.group(0)
    return None


class SimulationBackend:
    """LLMBackend-compatible offline engine (extractive + simulated stream)."""

    def __init__(self, knowledge: "HeuristicBrain | None" = None) -> None:
        self.name = "simulation:extractive-v1"
        self.mode = "simulation"
        self._brain = knowledge
        self._requests = 0
        self._tokens = 0

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        self._requests += 1
        prompt = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
        # In agent context the last system message carries the structured task;
        # extractive synthesis keeps behaviour grounded offline.
        context = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "system"), "")
        if self._brain is not None and context:
            answer = await self._brain.extract_from_context(prompt, context)
        else:
            answer = self._generic_reply(prompt)
        self._tokens += len(answer.split())
        await asyncio.sleep(random.uniform(0.01, 0.03))  # simulate RTT
        return answer

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamEvent]:
        text = await self.chat(messages, max_tokens=max_tokens, temperature=temperature)
        t0 = time.perf_counter()
        await asyncio.sleep(random.uniform(0.02, 0.06))  # simulated TTFT
        yield StreamEvent(ttft_ms=round((time.perf_counter() - t0) * 1000, 1))
        # stream in 3-4 token chunks
        words = text.split(" ")
        i = 0
        while i < len(words):
            take = words[i : i + 3]
            self._tokens += len(take)
            yield StreamEvent(token=" ".join(take) + (" " if i + 3 < len(words) else ""))
            i += 3
            await asyncio.sleep(random.uniform(0.005, 0.015))

    async def chat_vision(self, prompt: str, image_b64: str, *, mime: str = "image/png") -> str:
        self._requests += 1
        if self._brain is not None:
            return self._brain.describe_image_text_only(prompt)
        return "No vision model is served; OCR fallback should be used instead."

    async def probe(self) -> bool:
        return True  # always "up"

    def metrics(self) -> dict[str, Any]:
        return {"requests": self._requests, "completion_tokens": self._tokens}

    def _generic_reply(self, prompt: str) -> str:
        return (
            f"[simulation] Offline deterministic engine. Received prompt: "
            f"'{prompt[:180]}'. This backend only produces grounded extractive "
            f"answers inside the agent pipeline; start a vLLM server for "
            f"generative behaviour."
        )


class HeuristicBrain:
    """Decision engine used by the agent graph in simulation mode."""

    def __init__(self, corpus_vocab_provider=None) -> None:
        self._vocab_provider = corpus_vocab_provider  # callable -> set[str]

    # ------------------------------------------------------------- router
    async def classify_query(self, question: str) -> tuple[str, float, str]:
        """Return (route, confidence, reason).

        Routes: 'document' | 'chitchat' | 'out_of_scope'.
        """
        q = question.strip().lower()
        for rx in _CHITCHAT_RE:
            if rx.search(q):
                return "chitchat", 0.95, "greeting/social pattern matched"
        if len(q) < 4:
            return "chitchat", 0.8, "too short to be a document query"

        terms = _content_terms(question)
        vocab = self._corpus_vocab()
        if terms:
            overlap = len(terms & vocab) / len(terms)
            if any(h in q for h in _WORLD_KNOWLEDGE_HINTS) and overlap < 0.34:
                return (
                    "out_of_scope",
                    round(overlap, 2),
                    f"world-knowledge cue + low corpus overlap ({overlap:.0%})",
                )
            if overlap < 0.17 and not any(h in q for h in _WORLD_KNOWLEDGE_HINTS):
                return (
                    "out_of_scope",
                    round(overlap, 2),
                    f"query shares almost no vocabulary with the corpus ({overlap:.0%})",
                )
            return "document", round(overlap, 2), f"corpus term overlap {overlap:.0%}"
        return "out_of_scope", 0.3, "no usable content terms"

    def _corpus_vocab(self) -> set[str]:
        if self._vocab_provider is not None:
            try:
                return self._vocab_provider() or set()
            except Exception:
                return set()
        return set()

    # ------------------------------------------------------------- grader
    async def grade_chunk(self, question: str, chunk_text: str, retrieval_score: float) -> float:
        """Relevance grade in [0, 1]: retrieval signal + stemmed term coverage.

        Coverage is synonym-expanded (weight 0.8 for synonym hits) so a
        paraphrased question still grades its evidence chunk as relevant.
        """
        terms = {_stem(t) for t in _content_terms(question)}
        if not terms:
            return 0.0
        chunk_terms = _stemmed(chunk_text)
        literal = len(terms & chunk_terms)
        syn = _expanded_stems(question) - terms
        coverage = min(1.0, (literal + 0.8 * len(syn & chunk_terms)) / len(terms))
        # 60% coverage-weighted + 40% retriever score, coverage dominant because
        # the retriever score is already rank-normalised.
        grade = 0.6 * coverage + 0.4 * max(0.0, min(1.0, retrieval_score))
        return round(min(grade, 1.0), 4)

    # ----------------------------------------------------------- rewriter
    async def rewrite_query(self, question: str) -> str:
        """Expand with domain synonyms; drop filler; keep original terms."""
        terms = sorted(_content_terms(question))
        extras: list[str] = []
        for t in terms:
            for syn in _SYNONYMS.get(t, []):
                if syn not in extras and syn not in terms:
                    extras.append(syn)
        base = " ".join(terms) if terms else question
        if extras:
            return f"{base} {' '.join(extras[:6])}".strip()
        return base or question

    # ---------------------------------------------------------- generator
    _MD_IMAGE_RE = re.compile(r"^!\[[^\]]*\]\([^)]*\)\s*$")

    @staticmethod
    def _extract_answer_units(text: str) -> list[str]:
        """Split a chunk into citable "answer units": prose sentences plus
        markdown-table rows rendered as header-annotated pseudo-sentences
        ("X200 controller (Model); 24 months (Standard); ..."). Table facts
        (warranty matrices, error codes, LED patterns) are otherwise
        unreachable by sentence extraction.

        Header detection is conservative: the first row is only treated as a
        header when a ``|---|`` separator follows it; otherwise rows are
        joined as plain values (avoids pairing cells with a wrong "header").
        """
        units: list[str] = []
        lines = text.split("\n")
        i = 0

        def is_table_line(ln: str) -> bool:
            return ln.startswith("|") and ln.count("|") >= 3

        def is_separator(cells: list[str]) -> bool:
            return all(set(c) <= {"-", ":", " ", ""} for c in cells)

        def split_row(ln: str) -> list[str]:
            return [c.strip() for c in ln.strip("|").split("|")]

        while i < len(lines):
            line = lines[i].strip()
            if is_table_line(line):
                first_cells = split_row(line)
                j = i + 1
                has_header = (
                    j < len(lines)
                    and is_table_line(lines[j].strip())
                    and is_separator(split_row(lines[j].strip()))
                )
                header = first_cells if has_header else None
                # Without a header row the FIRST line is data too — process it
                # instead of silently dropping it as a phantom header.
                i = j if has_header else i
                while i < len(lines):
                    row = lines[i].strip()
                    if not is_table_line(row):
                        break
                    cells = split_row(row)
                    i += 1
                    if is_separator(cells):
                        continue
                    if header is not None:
                        parts = []
                        for h, c in zip(header, cells):
                            if c and h and not is_separator([h]):
                                parts.append(f"{c} ({h})")
                            elif c:
                                parts.append(c)
                        if parts:
                            units.append("; ".join(parts[:7]))
                    else:
                        vals = [c for c in cells if c]
                        if vals:
                            units.append(" — ".join(vals[:7]))
                continue
            if line and "```" not in line and not HeuristicBrain._MD_IMAGE_RE.match(line):
                for s in re.split(r"(?<=[.!?])\s+", line):
                    s = s.strip()
                    if len(s) > 30 and s.count("|") < 3:
                        units.append(s)
            i += 1
        return units

    async def generate_answer(self, question: str, contexts: list[dict]) -> str:
        """Extractive, citation-grounded answer.

        `contexts`: [{"doc","page","text","score"}] sorted best-first.
        Unit selection: IDF-flavoured term weighting — model codes and
        numbers (x200, e112, 24, 36) count double because they are the
        discriminative terms of technical questions; question-form and
        ALL-CAPS header lines are demoted; coverage dominates the score,
        with mild rank/position priors. Citations formatted
        ``[Source: doc, page N]`` per spec.
        """
        if not contexts:
            return (
                "I couldn't find this in the provided documents. "
                "No passage in the indexed corpus appears relevant to this question."
            )
        terms = _content_terms(question)

        def term_weight(t: str) -> float:
            return 2.0 if t.isdigit() or re.match(r"^[a-z]+\d+", t) else 1.0

        def form_penalty(unit: str) -> float:
            pen = 1.0
            if unit.rstrip().endswith("?"):
                pen *= 0.7  # FAQ question lines echo the query, not answers
            letters = [ch for ch in unit if ch.isalpha()]
            if letters and sum(ch.isupper() for ch in letters) / len(letters) > 0.6:
                pen *= 0.85  # section headers / title lines
            if unit.startswith("#"):
                pen *= 0.5  # markdown headings
            # OCR chart-caption garble ("Coverage (months) 40 354 304 …") is
            # supporting evidence at best — never quote it as the answer.
            if _OCR_MARKER_RE.search(unit):
                pen *= 0.45
            elif _digit_ratio(unit) > 0.35:
                pen *= 0.7
            return pen

        total_weight = sum(term_weight(t) for t in terms) or 1.0
        q_stem = {_stem(t) for t in terms}
        syn_stems = _expanded_stems(question) - q_stem
        quant_q = _QUANT_QUERY_RE.search(question)
        version_q = _VERSION_QUERY_RE.search(question)
        q_codes = {c.lower() for c in _ENTITY_CODE_RE.findall(question)}
        scored_units: list[tuple[float, int, int, dict, str]] = []
        # Contexts arrive diversity-selected (≤3 per doc) from the generator
        # node — consider up to 6 so a low-fusion-rank chunk carrying the
        # actual answer can still win on unit-level evidence.
        for rank, ctx in enumerate(contexts[:6]):
            # Units inherit the doc-name terms (x200_manual.pdf → {x200, manual})
            # so table rows/summary lines from the right document match model-code
            # queries even when the unit itself never repeats the model name.
            doc_terms = _stemmed(ctx["doc"])
            for pos, unit in enumerate(self._extract_answer_units(ctx["text"])[:20]):
                ut = _stemmed(unit) | doc_terms
                hits = sum(term_weight(t) for t in q_stem & ut) + 0.8 * len(syn_stems & ut)
                coverage = min(1.0, hits / total_weight)
                positional = 1.0 / (1 + pos)
                score = (1.6 * coverage + 0.5 / (1 + rank) + 0.15 * positional) * form_penalty(unit)
                # Quantitative-evidence bonus: measurable questions deserve
                # measurable units ("24 months", "±0.3°C", "48-hour").
                if quant_q and _QUANT_EVIDENCE_RE.search(unit):
                    score *= 1.35
                # Exact entity-code match: the unit carries the very code the
                # question asks about (E103, X200, S200 …) — boost it above
                # higher-coverage neighbours that never name the code.
                if q_codes and (q_codes & ut):
                    score *= 1.5
                # Version questions deserve units carrying "vX.Y.Z" — but
                # only when the version is stated affirmatively (negated
                # mentions like "v3.2.0 or earlier must not be used" name
                # the wrong version and must not win the boost).
                if (
                    version_q
                    and _VERSION_EVIDENCE_RE.search(unit)
                    and not _NEG_VERSION_RE.search(unit)
                ):
                    score *= 1.4
                if coverage == 0:
                    continue
                scored_units.append((score, rank, pos, ctx, unit))
        scored_units.sort(key=lambda x: -x[0])

        if not scored_units or scored_units[0][0] < 0.6:
            best = contexts[0]
            snippet = best["text"][:280].strip()
            return f"{snippet} [Source: {best['doc']}, page {best['page']}]"

        # Assemble 1-3 units, at most one per source, dedup near-identical.
        chosen: list[str] = []
        used_ranks: set[int] = set()
        seen_norm: set[str] = set()
        for score, rank, _pos, ctx, unit in scored_units:
            norm = re.sub(r"\W+", "", unit.lower())[:80]
            if norm in seen_norm or rank in used_ranks:
                continue  # dedup near-identical units / one per source
            if len(chosen) >= 3:
                break
            unit_text = unit
            # Version framing: "which version fixed X?" answers usually sit
            # next to the version string rather than inside the fix sentence
            # ("Install firmware v3.2.1 … This release fixes the MODBUS-TCP
            # timeout bug"). Prefix the affirmative version found in the
            # section or the same chunk so the answer names it.
            if version_q and not _VERSION_EVIDENCE_RE.search(unit_text):
                ver = _affirmative_version_in(ctx.get("section", "")) or _affirmative_version_in(
                    ctx.get("text", "")
                )
                if ver:
                    unit_text = f"{ver} — {unit_text}"
            chosen.append(f"{unit_text} [Source: {ctx['doc']}, page {ctx['page']}]")
            seen_norm.add(norm)
            used_ranks.add(rank)
        if chosen:
            return " ".join(chosen)
        return (
            "I couldn't find this in the provided documents. "
            "No usable sentence could be extracted from the retrieved passages."
        )

    async def extract_from_context(self, question: str, context_block: str) -> str:
        """Answer from a raw system-prompt context block (benchmark path)."""
        parts = re.split(r"\[CTX \d+\]", context_block)
        contexts = []
        for i, part in enumerate(parts[1:], start=1):
            m = re.search(r"\((.*?)\)", part)
            doc = m.group(1) if m else "context"
            contexts.append({"doc": doc, "page": 1, "text": part.strip(), "score": 1.0})
        return await self.generate_answer(question, contexts)

    async def generate_answer_stream(self, question: str, contexts: list[dict]):
        """Incremental variant of generate_answer for SSE streaming.

        The deterministic engine assembles the full answer first (that IS
        the generation step here), then yields it in 3–4 word groups with
        the same simulated decode cadence as SimulationBackend.chat_stream,
        reporting time-to-first-token on the first chunk. In vLLM mode the
        LLMBrain counterpart streams real tokens as the server emits them.
        """
        import asyncio as _asyncio
        import random as _random

        t0 = time.perf_counter()
        answer = await self.generate_answer(question, contexts)
        words = answer.split(" ")
        first = True
        for i in range(0, len(words), 4):
            take = words[i : i + 4]
            delta = " ".join(take) + (" " if i + 4 < len(words) else "")
            if first:
                yield delta, round((time.perf_counter() - t0) * 1000, 1)
                first = False
            else:
                yield delta, None
            await _asyncio.sleep(_random.uniform(0.004, 0.012))


    # ---------------------------------------------------------- verifier
    async def verify_answer(self, answer: str, contexts: list[dict]) -> tuple[str, float, list[str]]:
        """Hallucination check: fraction of answer content n-grams supported
        by the retrieved context. Returns (verdict, support_score, issues).
        """
        supported_text = " ".join(c["text"] for c in contexts)
        supported_terms = _stemmed(supported_text)
        answer_clean = re.sub(r"\[Source:[^\]]*\]", "", answer)
        answer_terms = [t for t in _content_terms(answer_clean) if len(t) > 2]
        if not answer_terms:
            return "not_supported", 0.0, ["answer contains no checkable claims"]
        hits = [t for t in answer_terms if _stem(t) in supported_terms]
        support = len(hits) / len(answer_terms)
        issues = []
        missing = [t for t in answer_terms if t not in supported_terms][:8]
        if missing:
            issues.append(f"terms not found in context: {', '.join(missing)}")
        verdict = "supported" if support >= 0.5 else "not_supported"
        return verdict, round(support, 4), issues

    # -------------------------------------------------------- image (OCR)
    def describe_image_text_only(self, prompt: str) -> str:
        return (
            "[simulation] No vision model is served in simulation mode; "
            "the /ask-image endpoint falls back to OCR + image analysis."
        )
