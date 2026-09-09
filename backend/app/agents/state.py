"""Agent state — one typed object threaded through the graph, capturing the
full execution trace for debuggability (Part D requirement)."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from app.rag.retriever import RetrievedChunk


@dataclass
class TraceRecord:
    node: str
    status: str            # ok | retry | fallback | skipped | error
    detail: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    sequence: int = 0


@dataclass
class AgentState:
    """Mutable state passed node-to-node by the graph runner."""

    question: str
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    top_k: int | None = None

    # router
    route: str = "document"            # document | chitchat | out_of_scope
    route_reason: str = ""

    # query decomposition (comparative / multi-entity questions)
    sub_queries: list[str] = field(default_factory=list)
    sub_query_map: dict[str, str] = field(default_factory=dict)  # chunk_id → sub-query
    decomposed: bool = False

    # retrieval / grading loop
    effective_query: str = ""          # possibly rewritten
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    grades: list[dict[str, Any]] = field(default_factory=list)
    relevant_chunks: list[RetrievedChunk] = field(default_factory=list)
    retries: int = 0
    grade_decision: str = "not_relevant"

    # generation / verification
    draft_answer: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    verdict: str = ""                  # supported | not_supported
    support_score: float = 0.0
    verification_issues: list[str] = field(default_factory=list)

    # output
    answer: str = ""
    fallback: bool = False

    # semantic answer cache
    cache_hit: bool = False
    cache_detail: dict[str, Any] = field(default_factory=dict)

    # true incremental streaming (transient — never serialized)
    # stream_emitter is the API-layer SSE callback; the generator and chitchat
    # nodes push answer deltas through it DURING generation, before the
    # verifier runs. deltas_emitted gates the endpoint's post-hoc reveal.
    stream_emitter: Any = None
    deltas_emitted: int = 0
    generation_ttft_ms: float | None = None

    # observability
    trace: list[TraceRecord] = field(default_factory=list)
    started_at: float = field(default_factory=time.perf_counter)

    async def emit_delta(self, delta: str, *, ttft_ms: float | None = None) -> None:
        """Push an answer_delta SSE event if this run is being streamed."""
        if self.stream_emitter is None or not delta:
            return
        event: dict[str, Any] = {"type": "answer_delta", "delta": delta}
        if ttft_ms is not None:
            event["ttft_ms"] = round(ttft_ms, 1)
            self.generation_ttft_ms = round(ttft_ms, 1)
        self.deltas_emitted += 1
        await self.stream_emitter(event)

    def record(self, node: str, status: str = "ok", **detail: Any) -> None:
        self.trace.append(
            TraceRecord(node=node, status=status, detail=detail, sequence=len(self.trace))
        )

    def elapsed_ms(self) -> float:
        return round((time.perf_counter() - self.started_at) * 1000, 1)

    @property
    def agent_path(self) -> list[str]:
        return [t.node for t in self.trace]

    def to_response_payload(self, serving_mode: str) -> dict[str, Any]:
        from app.schemas import AskResponse  # local import avoids cycle

        return AskResponse(
            question=self.question,
            answer=self.answer,
            citations=self.citations,
            route=self.route,  # type: ignore[arg-type]
            agent_path=self.agent_path,  # type: ignore[arg-type]
            trace=[
                {"node": t.node, "status": t.status, "detail": t.detail,
                 "latency_ms": t.latency_ms, "sequence": t.sequence}
                for t in self.trace
            ],
            retries=self.retries,
            fallback=self.fallback,
            latency_ms=self.elapsed_ms(),
            serving_mode=serving_mode,  # type: ignore[arg-type]
            request_id=self.request_id,
            sub_queries=self.sub_queries,
            decomposed=self.decomposed,
            cache_hit=self.cache_hit,
            cache_detail=self.cache_detail,
        ).model_dump()
