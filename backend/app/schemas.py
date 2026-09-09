"""Pydantic API schemas — the contract shared with the Next.js frontend."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

Route = Literal["document", "chitchat", "out_of_scope"]
NodeName = Literal[
    "router",
    "decomposer",
    "retriever",
    "grader",
    "rewriter",
    "generator",
    "verifier",
    "chitchat",
    "fallback",
    "session",
    "cache",
]
Decision = Literal["relevant", "not_relevant"]
Verdict = Literal["supported", "not_supported"]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=1000, description="User question")
    top_k: Optional[int] = Field(
        None, ge=1, le=20, description="Override retrieval top-k for this request"
    )
    session_id: Optional[str] = Field(
        None, max_length=64,
        description="Conversation session id; omitted → a new session is created",
    )


class Citation(BaseModel):
    """A single grounded citation pointing into the indexed corpus."""

    doc: str = Field(..., description="Source document name, e.g. x200_manual.pdf")
    page: int = Field(..., description="1-based page number (1 for flat text files)")
    chunk_id: str
    snippet: str = Field(..., description="Short supporting excerpt from the chunk")
    score: float = Field(..., description="Retrieval relevance score in [0, 1]")


class TraceEntry(BaseModel):
    """One step of the agent graph execution — for debuggability (Part D)."""

    node: NodeName
    status: Literal["ok", "retry", "fallback", "skipped", "error"]
    detail: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0
    sequence: int = 0


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    route: Route
    agent_path: list[NodeName] = Field(default_factory=list)
    trace: list[TraceEntry] = Field(default_factory=list)
    retries: int = 0
    fallback: bool = Field(False, description="True when the honest fallback path was taken")
    latency_ms: float = 0.0
    serving_mode: Literal["vllm", "simulation", "groq"] = "simulation"
    request_id: str = ""
    session_id: str = Field("", description="Conversation session this turn belongs to")
    effective_question: str = Field("", description="Query actually executed (follow-up resolved)")
    follow_up_applied: bool = Field(False, description="True when conversation context was injected")
    inherited_entities: list[str] = Field(default_factory=list)
    turn_index: int = 0
    sub_queries: list[str] = Field(
        default_factory=list,
        description="Sub-queries when a comparative question was decomposed",
    )
    decomposed: bool = Field(
        False, description="True when the question was split into per-entity sub-queries"
    )
    cache_hit: bool = Field(
        False, description="True when the answer was served from the semantic cache"
    )
    cache_detail: dict[str, Any] = Field(
        default_factory=dict,
        description="Cache provenance: original request, age, similarity",
    )


class IngestedDocument(BaseModel):
    doc: str
    kind: Literal["pdf", "text"]
    pages: int
    chunks: int
    figures: int
    words: int
    duration_ms: float


class IngestResponse(BaseModel):
    ingested: list[IngestedDocument]
    total_chunks: int
    total_documents: int
    duration_ms: float


class CorpusDocument(BaseModel):
    doc: str
    kind: str
    pages: int
    chunks: int
    figures: int
    words: int


class CorpusResponse(BaseModel):
    documents: list[CorpusDocument]
    total_chunks: int
    total_documents: int
    embedding_dim: int
    embedding_backend: str


class ServingInfo(BaseModel):
    mode: Literal["vllm", "simulation", "groq"]
    model: str
    vision_model: str
    vllm_base_url: str
    vllm_reachable: bool
    embedding_backend: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    serving: ServingInfo
    corpus: dict[str, Any]
    version: str


class BenchmarkRequest(BaseModel):
    concurrency: int = Field(8, ge=1, le=64)
    requests: int = Field(24, ge=1, le=500)
    question: str | None = Field(
        None, description="Optional custom question; defaults to a rotating sample set"
    )


class BenchmarkResponse(BaseModel):
    mode: Literal["vllm", "simulation", "groq"]
    concurrency: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    ttft_p50_ms: float
    ttft_p95_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    tokens_per_sec: float
    total_tokens: int
    duration_s: float
    notes: list[str] = Field(default_factory=list)


class ImageAskResponse(BaseModel):
    question: str
    answer: str
    mode: Literal["vllm-vision", "ocr-fallback"]
    ocr_text: str = Field("", description="Text detected in the image (truncated)")
    image_info: dict[str, Any] = Field(default_factory=dict)
    citations: list[Citation] = Field(default_factory=list)
    latency_ms: float


class Suggestion(BaseModel):
    question: str
    label: str
    expects: Literal["grounded", "fallback", "chitchat"]


class SuggestionsResponse(BaseModel):
    suggestions: list[Suggestion]


# ------------------------------------------------------------------ eval
class EvalCaseResult(BaseModel):
    id: str
    question: str
    category: str
    expects: str
    passed: bool
    route_correct: bool
    route: str
    retrieval_hit: Optional[bool] = None
    retrieved_doc: Optional[str] = None
    fact_supported: bool
    matched: Optional[str] = None
    answer_preview: str
    citations: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    agent_path: list[str] = Field(default_factory=list)
    note: str = ""


class EvalReport(BaseModel):
    ran_at: str
    total_cases: int
    passed: int
    failed: int
    accuracy: float
    retrieval_hit_rate: Optional[float] = None
    fact_support_rate: float
    latency_p50_ms: float
    latency_p95_ms: float
    by_category: dict[str, dict[str, int]] = Field(default_factory=dict)
    results: list[EvalCaseResult] = Field(default_factory=list)


# ------------------------------------------------------------ conversation
class SessionSummary(BaseModel):
    id: str
    turns: int
    preview: str
    created_at: float
    last_activity: float
    entities: list[str] = Field(default_factory=list)


class SessionListResponse(BaseModel):
    sessions: list[SessionSummary]
    stats: dict[str, Any]


class TurnPayload(BaseModel):
    question: str
    effective_question: str
    answer: str
    route: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    agent_path: list[str] = Field(default_factory=list)
    ts: float
    follow_up_applied: bool


class SessionHistoryResponse(BaseModel):
    session_id: str
    turns: list[TurnPayload]


# ------------------------------------------------------------- retrieval API
class SearchResultItem(BaseModel):
    chunk_id: str
    doc: str
    page: int
    section: str
    score: float
    coverage: float
    dense_rank: int
    bm25_rank: int
    is_figure: bool
    preview: str


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultItem]
    weights: dict[str, float] = Field(
        default_factory=dict,
        description="Reranker weights actually applied (fusion / coverage / agreement)",
    )


# ------------------------------------------------------------------- ablation
class AblationConfig(BaseModel):
    """One retriever configuration under test (name + parameter overrides)."""

    name: str
    top_k: Optional[int] = None
    w_fused: Optional[float] = None
    w_coverage: Optional[float] = None
    w_agreement: Optional[float] = None


class AblationResult(BaseModel):
    name: str
    config: dict[str, Any] = Field(default_factory=dict)
    accuracy: float
    retrieval_hit_rate: Optional[float] = None
    fact_support_rate: float
    faithfulness_avg: Optional[float] = None
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    passed: int = 0
    total: int = 0
    delta_accuracy: Optional[float] = None


class AblationReport(BaseModel):
    ran_at: str
    baseline: str
    configs: list[AblationResult] = Field(default_factory=list)
    winner: str = ""
    duration_ms: float = 0.0
    notes: list[str] = Field(default_factory=list)


class SourceNeighbor(BaseModel):
    chunk_id: str
    page: int
    section: str
    preview: str
    relation: str


class SourceResponse(BaseModel):
    chunk_id: str
    doc: str
    doc_kind: str
    page: int
    section: str
    text: str
    token_count: int
    is_figure_caption: bool
    neighbors: list[dict[str, Any]] = Field(default_factory=list)


# ----------------------------------------------------------------- feedback
class FeedbackRequest(BaseModel):
    request_id: str = Field("", max_length=64, description="Answer request id being rated")
    rating: Literal["up", "down"]
    comment: Optional[str] = Field(None, max_length=500, description="Optional free-text note")
