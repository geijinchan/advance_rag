"""LLM-powered brain: prompt engineering for each agentic decision point
when a real vLLM server is behind the service. Robust JSON parsing with
heuristic fallback so a malformed generation degrades instead of crashing."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.llm.base import ChatMessage
from app.llm.simulation import HeuristicBrain

logger = logging.getLogger(__name__)

_ROUTER_SYSTEM = """You are a query router for a corporate document-assistant.
Classify the user's question into exactly one route:
- "document": the question might be answerable from internal product manuals,
  policies, datasheets, firmware notes, maintenance docs, resumes, or portfolios.
  Route any requests for contact info, emails, phone numbers, or details about people here.
- "chitchat": greetings, thanks, or meta questions about the assistant.
- "out_of_scope": general world knowledge or clearly unrelated topics
  (sports, politics, news, celebrities, weather...).

Internal corpus topics: X100/X200/X300 industrial controllers, S200/S350
sensors, warranty & RMA policy, firmware releases, safety compliance,
maintenance, networking/fieldbus, troubleshooting, deployment, portfolios, and contact info.

Respond ONLY with minified JSON: {"route":"...","confidence":0.0-1.0,"reason":"..."}"""

_GRADER_SYSTEM = """You are a relevance grader for a retrieval-augmented assistant.
Given a question and a retrieved passage, decide whether the passage helps
answer the question. Be strict: generic or adjacent content is NOT relevant.

Respond ONLY with minified JSON: {"relevant":true|false,"score":0.0-1.0,"reason":"..."}"""

_REWRITER_SYSTEM = """You rewrite a question that failed retrieval. Keep the
original intent, add likely corpus vocabulary (model codes, technical terms,
synonyms). Output ONLY the rewritten question on a single line, no quotes."""

_ANSWER_SYSTEM = """You are a grounded document assistant. Answer STRICTLY from
the provided context passages. Rules:
1. Cite every fact inline as [Source: <doc>, page <N>] using the source tags
   given with each passage — never invent documents, pages or facts.
2. If the context does not contain the answer, reply exactly:
   I couldn't find this in the provided documents.
3. Be concise: 1-4 sentences, no preamble, no repetition of the question."""

_VERIFIER_SYSTEM = """You are a strict hallucination checker. Given the
retrieved context and a draft answer, verify every claim in the answer is
supported by the context. Citations must point at real passages.

Respond ONLY with minified JSON:
{"verdict":"supported"|"not_supported","support":0.0-1.0,"issues":["..."]}"""

_VISION_PROMPT = """You are looking at an image (a chart, diagram, screenshot
or photo possibly related to industrial equipment documents). Question: {question}

Describe what the image shows and answer the question if the image contains
enough information. If not, say what is missing. Mention any text, units,
axes or values you can read."""


def _parse_json(raw: str) -> dict[str, Any] | None:
    """Tolerant JSON extraction: handles code fences and leading prose."""
    raw = raw.strip()
    if not raw:
        return None
    if raw.startswith("```"):
        raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


class LLMBrain:
    """Prompt-driven implementation of the brain interface (vLLM mode)."""

    def __init__(self, backend, fallback: HeuristicBrain) -> None:
        self.backend = backend  # VLLMBackend
        self.fallback = fallback

    # -------------------------------------------------------------- router
    async def classify_query(self, question: str) -> tuple[str, float, str]:
        raw = await self.backend.chat(
            [ChatMessage.system(_ROUTER_SYSTEM), ChatMessage.user(question)],
            max_tokens=120,
            temperature=0.0,
        )
        parsed = _parse_json(raw)
        if parsed and parsed.get("route") in {"document", "chitchat", "out_of_scope"}:
            return (
                parsed["route"],
                float(parsed.get("confidence", 0.5)),
                str(parsed.get("reason", "llm classification")),
            )
        route, conf, reason = await self.fallback.classify_query(question)
        return route, conf, f"fallback heuristic ({reason})"

    # ------------------------------------------------------------- grader
    async def grade_chunk(self, question: str, chunk_text: str, retrieval_score: float) -> float:
        passage = chunk_text[:1200]
        raw = await self.backend.chat(
            [
                ChatMessage.system(_GRADER_SYSTEM),
                ChatMessage.user(f"Question: {question}\n\nPassage:\n{passage}"),
            ],
            max_tokens=140,
            temperature=0.0,
        )
        parsed = _parse_json(raw)
        if parsed and isinstance(parsed.get("score"), (int, float)):
            return float(min(max(parsed["score"], 0.0), 1.0))
        return await self.fallback.grade_chunk(question, chunk_text, retrieval_score)

    # ----------------------------------------------------------- rewriter
    async def rewrite_query(self, question: str) -> str:
        raw = await self.backend.chat(
            [ChatMessage.system(_REWRITER_SYSTEM), ChatMessage.user(question)],
            max_tokens=120,
            temperature=0.2,
        )
        rewritten = raw.strip().strip('"').splitlines()[0] if raw.strip() else ""
        if len(rewritten) >= 8:
            return rewritten
        return await self.fallback.rewrite_query(question)

    # ---------------------------------------------------------- generator
    async def generate_answer(self, question: str, contexts: list[dict]) -> str:
        ctx_block = "\n\n".join(
            f"[CTX {i+1}] (source: {c['doc']}, page {c['page']}):\n{c['text'][:1400]}"
            for i, c in enumerate(contexts[:5])
        )
        messages = [
            ChatMessage.system(_ANSWER_SYSTEM),
            ChatMessage.user(f"Question: {question}\n\nContext passages:\n{ctx_block}"),
        ]
        return await self.backend.chat(messages, temperature=0.1)

    async def generate_answer_stream(self, question: str, contexts: list[dict]):
        """True incremental generation: consumes the vLLM token stream and
        yields (delta, ttft_ms) pairs — ttft_ms is set on the first chunk
        only. Falls back to the non-streaming call + chunked replay when
        the backend errors mid-stream (honest degradation, same answer)."""
        ctx_block = "\n\n".join(
            f"[CTX {i+1}] (source: {c['doc']}, page {c['page']}):\n{c['text'][:1400]}"
            for i, c in enumerate(contexts[:5])
        )
        messages = [
            ChatMessage.system(_ANSWER_SYSTEM),
            ChatMessage.user(f"Question: {question}\n\nContext passages:\n{ctx_block}"),
        ]
        try:
            async for event in self.backend.chat_stream(messages, temperature=0.1):
                if event.token:
                    yield event.token, event.get("ttft_ms")
        except Exception as exc:  # noqa: BLE001 — degrade to non-streaming
            logger.warning("chat_stream failed (%s) — falling back to full completion", exc)
            full_text = await self.backend.chat(messages, temperature=0.1)
            yield full_text, None

    # ---------------------------------------------------------- verifier
    async def verify_answer(self, answer: str, contexts: list[dict]) -> tuple[str, float, list[str]]:
        ctx_block = "\n\n".join(
            f"[CTX {i+1}] (source: {c['doc']}, page {c['page']}):\n{c['text'][:1200]}"
            for i, c in enumerate(contexts[:5])
        )
        raw = await self.backend.chat(
            [
                ChatMessage.system(_VERIFIER_SYSTEM),
                ChatMessage.user(f"Context:\n{ctx_block}\n\nDraft answer: {answer}"),
            ],
            max_tokens=200,
            temperature=0.0,
        )
        parsed = _parse_json(raw)
        if parsed and parsed.get("verdict") in {"supported", "not_supported"}:
            return (
                parsed["verdict"],
                float(parsed.get("support", 0.5)),
                list(parsed.get("issues", []))[:5],
            )
        return await self.fallback.verify_answer(answer, contexts)

    # -------------------------------------------------------- image (VLM)
    async def answer_image(self, question: str, image_b64: str, mime: str = "image/png") -> str:
        return await self.backend.chat_vision(_VISION_PROMPT.format(question=question), image_b64, mime=mime)
