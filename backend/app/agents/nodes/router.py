"""Router node — classify and dispatch (Part C)."""

from __future__ import annotations

import logging

from app.agents.state import AgentState
from app.llm.base import ChatMessage
from app.llm.brain import LLMBrain
from app.llm.simulation import HeuristicBrain

logger = logging.getLogger(__name__)

Brain = LLMBrain | HeuristicBrain
END = "__end__"

_CHITCHAT_REPLIES = {
    "greeting": (
        "Hello! I'm the IndustriOS document assistant. Ask me anything about the "
        "indexed corpus — specs, warranty, firmware, troubleshooting — and I'll "
        "answer with citations."
    ),
    "thanks": "You're welcome! Ask away if you need anything else from the documents.",
    "capabilities": (
        "I answer questions grounded in the indexed internal documents, cite the exact "
        "source document and page, verify answers against the retrieved context, and "
        "honestly say when something isn't covered. Try me with a warranty or firmware question."
    ),
}


def _chitchat_kind(question: str) -> str:
    q = question.lower()
    if "who are you" in q or "what can you do" in q or "introduce" in q:
        return "capabilities"
    if "thank" in q:
        return "thanks"
    return "greeting"


def make_router_node(brain: Brain):
    async def router_node(state: AgentState) -> str | None:
        route, confidence, reason = await brain.classify_query(state.question)
        state.route = route
        state.route_reason = reason
        state.effective_query = state.question
        state.record("router", route=route, confidence=confidence, reason=reason)
        if route == "chitchat":
            return "chitchat"
        if route == "out_of_scope":
            return "fallback"
        # Comparative / multi-entity questions take the decomposition path so
        # retrieval gathers balanced evidence per entity (see decomposer.py).
        from app.agents.nodes.decomposer import broad_decomposition_signal, detect_decomposition

        if detect_decomposition(state.question):
            return "decomposer"
        # vLLM mode: broad recall gate — the LLM decomposer decides inside the
        # node and gracefully skips single-hop questions ([] → retriever).
        if getattr(brain, "backend", None) is not None and broad_decomposition_signal(
            state.question
        ):
            return "decomposer"
        return "retriever"

    return router_node


def make_chitchat_node(mode: str, backend):
    async def chitchat_node(state: AgentState) -> str | None:
        if mode == "vllm":
            answer = await backend.chat(
                [
                    ChatMessage.system(
                        "You are a friendly document assistant. Reply in one short "
                        "sentence, then invite a question about the internal documents."
                    ),
                    ChatMessage.user(state.question),
                ],
                max_tokens=80,
                temperature=0.4,
            )
        else:
            answer = _CHITCHAT_REPLIES[_chitchat_kind(state.question)]
        state.answer = answer
        # Chit-chat is not verified, so its deltas ARE the final text —
        # streamed word-group by word-group for the same live feel.
        if state.stream_emitter is not None:
            words = answer.split(" ")
            for i in range(0, len(words), 4):
                delta = " ".join(words[i : i + 4]) + (" " if i + 4 < len(words) else "")
                await state.emit_delta(delta, ttft_ms=1.0 if i == 0 else None)
        state.record("chitchat", reply=answer[:120], streamed=state.deltas_emitted > 0)
        return END

    return chitchat_node
