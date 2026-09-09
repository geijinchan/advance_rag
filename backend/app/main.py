"""FastAPI application — the public surface of the service (Part D).

Endpoints:
  GET  /health              — liveness + serving mode + corpus stats
  POST /ask                 — agentic Q&A (full trace returned)
  POST /ask/stream          — same, streamed as SSE node events (bonus)
  POST /ask-image           — visual Q&A via served VLM / OCR fallback (Part E)
  POST /ingest              — upload + index documents (multipart)
  POST /ingest/sample       — (re)ingest the bundled demo corpus
  GET  /corpus              — indexed document inventory
  POST /reset               — wipe the index
  POST /benchmark           — throughput/latency benchmark (Part A)
  GET  /benchmark/results   — last benchmark result
  GET  /suggestions         — demo questions for the UI
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import signal
import threading
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from app.agents.state import AgentState
from app.agents.workflow import build_graph
from app.benchmark import BenchmarkRunner
from app.config import settings
from app.conversation import SessionStore, Turn, resolve_followup
from app.eval.evaluator import EvalRunner
from app.eval.faithfulness import FaithfulnessJudge
from app.eval.growth import AnswerLedger, GoldenSetManager
from app.eval.history import EvalHistory
from app.eval.triage import TriageTracker
from app.feedback import FeedbackStore
from app.llm.service import ModelService
from app.metrics import MetricsCollector
from app.multimodal.image_qa import ImageQAService
from app.parsing.pdf_parser import SUPPORTED_EXTENSIONS
from app.prometheus import HTTPMetricsMiddleware, render_metrics
from app.rag.answer_cache import AnswerCache
from app.rag.reranker import CrossEncoderReranker
from app.rag.ingestion import IngestionPipeline
from app.rag.retriever import HybridRetriever
from app.rag.vector_store import VectorStore
from app.schemas import (
    AskRequest,
    BenchmarkRequest,
    FeedbackRequest,
    HealthResponse,
    SuggestionsResponse,
)

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
logger = logging.getLogger("rag-agent")

VERSION = "1.0.0"

SUGGESTIONS = [
    {"question": "What is the maximum warranty period mentioned for the X200 model?",
     "label": "X200 warranty", "expects": "grounded"},
    {"question": "Which firmware version fixed the MODBUS-TCP timeout bug and when was it released?",
     "label": "Firmware fix", "expects": "grounded"},
    {"question": "How long does an RMA evaluation take and where is the facility?",
     "label": "RMA timing", "expects": "grounded"},
    {"question": "What does a blinking red LED at 2 Hz indicate?",
     "label": "LED pattern", "expects": "grounded"},
    {"question": "Compare the standard warranty of the X200 and the X300.",
     "label": "Compare (multi-hop)", "expects": "grounded"},
    {"question": "Who won the 2022 World Cup?", "label": "Out of scope",
     "expects": "fallback"},
    {"question": "What is the airspeed velocity of an unladen swallow?",
     "label": "Out of scope #2", "expects": "fallback"},
    {"question": "hello there!", "label": "Chit-chat", "expects": "chitchat"},
]


class AppState:
    """Container for singletons + graph lifecycle management."""

    def __init__(self) -> None:
        self.store = VectorStore(settings.index_file)
        # Cross-encoder rerank stage: backend_provider follows the model
        # service hot-swap (lexical surrogate in simulation, LLM in vLLM mode).
        self.reranker = CrossEncoderReranker(
            n=settings.rerank_n,
            backend_provider=lambda: self.models.backend if self.models.vllm_available else None,
        )
        self.retriever = HybridRetriever(self.store, rrf_k=settings.rrf_k,
                                         reranker=self.reranker)
        self.retriever.rerank_enabled = settings.rerank_enabled
        self.models = ModelService(corpus_vocab_provider=self.store.corpus_vocabulary)
        self.pipeline = IngestionPipeline(self.store)
        self.image_qa = ImageQAService(self.models, self.retriever)
        self.benchmark = BenchmarkRunner(self.models)
        self.sessions = SessionStore(persist_path=str(settings.data_dir / "sessions.json"))
        self.feedback = FeedbackStore(persist_path=str(settings.data_dir / "feedback.json"))
        self.metrics = MetricsCollector(persist_path=str(settings.data_dir / "metrics.json"))
        self.cache = AnswerCache(persist_path=str(settings.data_dir / "answer_cache.json"))
        # Golden-set growth: ledger joins request_ids → answered content,
        # manager merges base + user-promoted cases at eval time.
        self.ledger = AnswerLedger(persist_path=str(settings.data_dir / "answer_ledger.json"))
        self.golden = GoldenSetManager(persist_path=str(settings.data_dir / "golden_cases.json"))
        # Trend of eval runs (accuracy/faithfulness/latency over time).
        self.eval_history = EvalHistory(persist_path=str(settings.data_dir / "eval_history.json"))
        # Triage lifecycle: 👎 fix-list items can be marked resolved (note +
        # timestamp) and reopened; persisted, joined by build_triage.
        self.triage = TriageTracker(persist_path=str(settings.data_dir / "triage_resolutions.json"))
        self.faithfulness = FaithfulnessJudge()
        self.faithfulness.bind(self.models)
        self.eval_runner = EvalRunner(
            self._eval_ask,
            faithfulness_judge=self.faithfulness,
            chunk_text_lookup=self._chunk_text,
            cases_provider=self.golden.cases,
        )
        self._graph = build_graph(self.models, self.retriever, self.models.mode)
        self._graph_mode = self.models.mode

    @staticmethod
    def _chunk_text(chunk_id: str) -> str | None:
        """Full source text for the faithfulness judge (chunk_id -> text)."""
        chunk = state.store.get_by_chunk_id(chunk_id) if state is not None else None
        return chunk.text if chunk is not None else None

    def graph(self):
        """Current graph, rebuilt if the serving mode flipped since build."""
        if self._graph_mode != self.models.mode:
            logger.info("Serving mode changed %s → %s: rebuilding agent graph",
                        self._graph_mode, self.models.mode)
            self._graph = build_graph(self.models, self.retriever, self.models.mode)
            self._graph_mode = self.models.mode
        return self._graph

    async def ask(self, question: str, top_k: int | None = None, on_event=None):
        state = AgentState(question=question, top_k=top_k)
        return await self.graph().run(state, on_event=on_event)

    async def _eval_ask(self, question: str, top_k: int | None = None):
        """Ask-path used by the eval harness (no session side-effects)."""
        return await self.ask(question, top_k)

    async def ask_with_session(self, request: AskRequest, on_event=None):
        """Ask-path with conversation memory + semantic answer cache.

        Follow-ups are resolved first ("what about the X300?" inherits
        entities), then the effective query is checked against the semantic
        cache — a hit skips the agent graph entirely and returns the cached
        grounded answer with explicit provenance (original request id, age,
        similarity). Misses run the agent graph, and grounded answers are
        stored for future lookups."""
        session = self.sessions.get_or_create(request.session_id)
        effective_q, applied, inherited = resolve_followup(session, request.question)

        async def _emit(ev: dict) -> None:
            if on_event is not None:
                await on_event(ev)

        if applied:
            await _emit({
                "node": "session", "status": "ok", "detail": {
                    "follow_up": True,
                    "inherited_entities": inherited,
                    "effective_question": effective_q,
                }, "latency_ms": 0.1,
            })

        # ---- semantic cache lookup ------------------------------------
        cached = self.cache.lookup(effective_q, request.top_k)
        if cached is not None:
            detail = cached.get("cache_detail", {})
            result = AgentState(question=request.question, top_k=request.top_k)
            result.route = cached.get("route", "document")
            result.answer = cached.get("answer", "")
            result.citations = cached.get("citations", [])
            result.fallback = False
            result.cache_hit = True
            result.cache_detail = detail
            result.sub_queries = cached.get("sub_queries", [])
            result.decomposed = bool(cached.get("decomposed", False))
            result.record(
                "cache",
                original_request_id=detail.get("original_request_id", ""),
                similarity=detail.get("similarity", 1.0),
                age_seconds=detail.get("age_seconds", 0.0),
                times_served=detail.get("times_served", 1),
                original_latency_ms=detail.get("original_latency_ms", 0.0),
            )
            await _emit({
                "node": "cache", "status": "ok",
                "detail": {"similarity": detail.get("similarity", 1.0),
                           "age_seconds": detail.get("age_seconds", 0.0)},
                "latency_ms": 0.2,
                "route": result.route, "retries": 0,
            })
            turn = Turn(
                question=request.question,
                effective_question=effective_q,
                answer=result.answer,
                route=result.route,
                citations=result.citations,
                agent_path=result.agent_path,
                follow_up_applied=applied,
            )
            self.sessions.append_turn(session.id, turn)
            self.metrics.observe_ask(
                question=request.question,
                route=result.route,
                latency_ms=result.elapsed_ms(),
                fallback=False,
                retries=0,
                follow_up=applied,
                streamed=on_event is not None,
                session_id=session.id,
                cache_hit=True,
            )
            self.ledger.record(result.request_id, {
                "question": request.question,
                "effective_question": effective_q,
                "answer": result.answer,
                "citations": result.citations,
                "route": result.route,
                "fallback": False,
                "cache_hit": True,
                "latency_ms": result.elapsed_ms(),
            })
            return result, session, effective_q, applied, inherited

        result = await self.ask(effective_q, request.top_k, on_event=on_event)
        # Cache only grounded, verified answers (AnswerCache.store filters).
        self.cache.store(
            effective_q, request.top_k,
            result.to_response_payload(self.models.mode),
        )
        turn = Turn(
            question=request.question,
            effective_question=effective_q,
            answer=result.answer,
            route=result.route,
            citations=result.citations,
            agent_path=result.agent_path,
            follow_up_applied=applied,
        )
        self.sessions.append_turn(session.id, turn)
        # usage telemetry (eval runs are excluded; they call `ask` directly)
        self.metrics.observe_ask(
            question=request.question,
            route=result.route,
            latency_ms=result.elapsed_ms(),
            fallback=bool(result.fallback),
            retries=int(result.retries or 0),
            follow_up=applied,
            streamed=on_event is not None,
            session_id=session.id,
            cache_hit=False,
        )
        self.ledger.record(result.request_id, {
            "question": request.question,
            "effective_question": effective_q,
            "answer": result.answer,
            "citations": result.citations,
            "route": result.route,
            "fallback": bool(result.fallback),
            "cache_hit": False,
            "latency_ms": result.elapsed_ms(),
        })
        return result, session, effective_q, applied, inherited


state: AppState | None = None


def _start_orphan_watchdog() -> None:
    """Exit the server if the bun watcher process dies.

    When uvicorn is spawned by index.js it sets RAG_AGENT_WATCHED=1. A normal
    uvicorn SIGKILL/SIGTERM of the watcher leaves the Python child running and
    holding port 3003 — every later respawn then fails with "address already
    in use". The watchdog polls the parent pid; if the process is reparented
    (watcher gone), we SIGTERM ourselves for a graceful uvicorn shutdown.
    """
    if os.environ.get("RAG_AGENT_WATCHED") != "1":
        return

    original_ppid = os.getppid()

    def _watch() -> None:
        import time

        while True:
            time.sleep(2.0)
            try:
                if os.getppid() != original_ppid:
                    logging.getLogger("rag-agent.watchdog").warning(
                        "watcher (ppid=%s) disappeared — shutting down to free :%s",
                        original_ppid, settings.api_port,
                    )
                    os.kill(os.getpid(), signal.SIGTERM)
                    return
            except Exception:  # pragma: no cover - defensive
                return

    thread = threading.Thread(target=_watch, name="orphan-watchdog", daemon=True)
    thread.start()
    logging.getLogger("rag-agent.watchdog").info(
        "orphan watchdog armed (watcher ppid=%s)", original_ppid
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global state
    state = AppState()
    _start_orphan_watchdog()
    await state.models.refresh()  # probe vLLM, possibly switch to the real backend
    if settings.auto_ingest_on_boot:
        if not state.store.load():
            logger.info("No usable persisted index — ingesting corpus from %s", settings.corpus_dir)
            results = state.pipeline.ingest_corpus_dir(settings.corpus_dir)
            total = sum(r.chunks for r in results)
            logger.info("Ingested %d documents, %d chunks", len(results), total)
        else:
            # BM25 index is not persisted — rebuild after load.
            state.retriever.rebuild_lexical()
    else:
        state.store.load()
    yield
    await state.models.aclose()


app = FastAPI(
    title="RAG-Powered Agentic Q&A Service",
    description="Document Intelligence Assistant — vLLM-served, agentic, citation-grounded.",
    version=VERSION,
    lifespan=lifespan,
)

# Prometheus-ready HTTP counters (route-pattern labels; see app/prometheus.py)
app.add_middleware(HTTPMetricsMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _state() -> AppState:
    if state is None:  # pragma: no cover - lifespan guarantees init
        raise HTTPException(status_code=503, detail="service not initialised")
    return state


# --------------------------------------------------------------------- health
@app.get("/health", response_model=HealthResponse)
async def health(refresh: bool = Query(False, description="Re-probe the vLLM server")):
    st = _state()
    if refresh:
        await st.models.refresh()
    reachable = st.models.vllm_available
    corpus = st.store.stats()
    return HealthResponse(
        status="ok" if corpus["total_chunks"] > 0 or reachable else "degraded",
        serving=st.models.serving_info(),  # type: ignore[arg-type]
        corpus={
            "total_documents": corpus["total_documents"],
            "total_chunks": corpus["total_chunks"],
            "embedding_dim": corpus["embedding_dim"],
        },
        version=VERSION,
    )


@app.get("/")
async def root():
    st = _state()
    return {
        "service": "rag-agent",
        "version": VERSION,
        "mode": st.models.mode,
        "docs": "/docs",
        "endpoints": ["/health", "/ask", "/ask/stream", "/ask-image", "/ingest",
                      "/ingest/sample", "/corpus", "/reset", "/benchmark",
                      "/eval", "/eval/ablation", "/eval/compare", "/eval/history",
                      "/search", "/source", "/feedback", "/eval/candidates",
                      "/eval/candidates/promote-batch", "/eval/cases",
                      "/quality/triage", "/quality/triage/resolve",
                      "/quality/triage/reopen", "/sessions", "/stats",
                      "/suggestions", "/metrics", "/ops/grafana"],
    }


def _session_payload(st: AppState, result: AgentState, session, effective_q: str,
                     applied: bool, inherited: list[str]) -> dict:
    payload = result.to_response_payload(st.models.mode)
    payload.update({
        "session_id": session.id,
        "effective_question": effective_q,
        "follow_up_applied": applied,
        "inherited_entities": inherited,
        "turn_index": len(session.turns),
    })
    return payload


# ------------------------------------------------------------------------ ask
@app.post("/ask")
async def ask(request: AskRequest):
    st = _state()
    result, session, effective_q, applied, inherited = await st.ask_with_session(request)
    return _session_payload(st, result, session, effective_q, applied, inherited)


@app.post("/ask/stream")
async def ask_stream(request: AskRequest):
    st = _state()

    async def event_stream():
        queue: asyncio.Queue = asyncio.Queue()
        DONE = object()

        async def on_event(ev: dict) -> None:
            # Generator/chitchat nodes push SSE-shaped answer_delta events
            # DURING generation (true incremental streaming); graph trace
            # records arrive as plain node dicts and get wrapped here.
            if ev.get("type") == "answer_delta":
                await queue.put(ev)
            else:
                await queue.put({"type": "node", **ev})

        async def runner():
            try:
                result, session, effective_q, applied, inherited = await st.ask_with_session(
                    request, on_event=on_event
                )
                payload = _session_payload(st, result, session, effective_q, applied, inherited)
                # True incremental generation: answer deltas were already
                # emitted by the generator node while it produced the draft
                # (before verification). The post-hoc reveal below now only
                # covers paths that did NOT stream:
                #   * cache hits — replayed at near-instant cadence, because
                #     the value of a hit is precisely that nothing is recomputed;
                #   * honest fallbacks — short refusal text, fast cadence.
                if getattr(result, "deltas_emitted", 0) == 0:
                    words = payload["answer"].split(" ")
                    cadence = 0.002 if payload.get("cache_hit") else 0.01
                    for i in range(0, len(words), 4):
                        delta = " ".join(words[i : i + 4]) + (" " if i + 4 < len(words) else "")
                        await queue.put({"type": "answer_delta", "delta": delta})
                        await asyncio.sleep(cadence)
                await queue.put({"type": "final", "payload": payload})
            except Exception as exc:  # noqa: BLE001 — stream errors must reach client
                logger.exception("stream failure")
                await queue.put({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
            finally:
                await queue.put(DONE)

        task = asyncio.create_task(runner())
        try:
            while True:
                item = await queue.get()
                if item is DONE:
                    break
                kind = item.pop("type")
                yield f"event: {kind}\ndata: {json.dumps(item, ensure_ascii=False)}\n\n"
        finally:
            await task

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ------------------------------------------------------------------ ask-image
@app.post("/ask-image")
async def ask_image(file: UploadFile = File(...), question: str = Form(...)):
    st = _state()
    data = await file.read()
    try:
        return await st.image_qa.answer(data, question, file.filename or "image")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# -------------------------------------------------------------------- ingest
@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)):
    st = _state()
    results = []
    for f in files:
        suffix = "." + (f.filename or "").rsplit(".", 1)[-1].lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"unsupported file type: {f.filename}")
        data = await f.read()
        try:
            results.append(st.pipeline.ingest_bytes(f.filename or f.filename, data))
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"failed to ingest {f.filename}: {exc}") from exc
    st.retriever.rebuild_lexical()
    stats = st.store.stats()
    return {
        "ingested": [r.__dict__ for r in results],
        "total_chunks": stats["total_chunks"],
        "total_documents": stats["total_documents"],
        "duration_ms": round(sum(r.duration_ms for r in results), 1),
    }


@app.post("/ingest/sample")
async def ingest_sample():
    st = _state()
    results = st.pipeline.ingest_corpus_dir(settings.corpus_dir)
    st.retriever.rebuild_lexical()
    stats = st.store.stats()
    return {
        "ingested": [r.__dict__ for r in results],
        "total_chunks": stats["total_chunks"],
        "total_documents": stats["total_documents"],
        "duration_ms": round(sum(r.duration_ms for r in results), 1),
    }


@app.get("/corpus")
async def corpus():
    return _state().store.stats()


@app.post("/reset")
async def reset():
    st = _state()
    st.store.clear()
    st.retriever.rebuild_lexical()
    return {"status": "index cleared", "total_chunks": 0}


# ----------------------------------------------------------------- benchmark
@app.post("/benchmark")
async def run_benchmark(request: BenchmarkRequest):
    st = _state()
    result = await st.benchmark.run(
        concurrency=request.concurrency,
        total_requests=request.requests,
        question=request.question,
    )
    st.metrics.observe_benchmark(result)
    return result


@app.get("/benchmark/results")
async def benchmark_results():
    st = _state()
    if st.benchmark.last_result is None:
        raise HTTPException(status_code=404, detail="no benchmark has been run yet")
    return st.benchmark.last_result


@app.get("/suggestions", response_model=SuggestionsResponse)
async def suggestions():
    return SuggestionsResponse(suggestions=SUGGESTIONS)  # type: ignore[arg-type]


# --------------------------------------------------------------------- stats
@app.get("/stats")
async def stats():
    """Service-wide metrics snapshot: usage (questions by route, latency
    p50/p95, fallbacks, follow-ups), sessions, feedback approval, corpus and
    serving context, rerank stage telemetry, last eval/benchmark. Powers the
    UI overview panel."""
    st = _state()
    try:
        info = st.models.serving_info()
        serving = {"mode": info.get("mode"), "model": info.get("model")}
    except Exception:  # noqa: BLE001
        serving = {"mode": st.models.mode, "model": None}
    corpus_stats = st.store.stats()
    snap = st.metrics.snapshot(
        sessions=st.sessions.stats(),
        feedback={k: v for k, v in st.feedback.stats().items() if k != "recent"},
        corpus={
            "documents": corpus_stats.get("total_documents"),
            "chunks": corpus_stats.get("total_chunks"),
        },
        serving=serving,
        cache=st.cache.stats(),
    )
    snap["rerank"] = st.reranker.stats()
    snap["golden"] = {
        **st.golden.stats(),
        "ledger_entries": st.ledger.stats()["entries"],
        "upvoted": st.feedback.stats()["up"] or 0,
        "downvoted": st.feedback.stats()["down"] or 0,
    }
    hist = st.eval_history.trend(limit=1)
    snap["eval_runs"] = {
        "total": hist["total_runs"],
        "golden": hist["golden_runs"],
        "ablation": hist["ablation_runs"],
        "last_accuracy": (hist.get("last") or {}).get("accuracy"),
    }
    snap["triage"] = _triage_counts(st)
    return snap


# ------------------------------------------------------------------- metrics
@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus text exposition (version=0.0.4) of the service telemetry:
    ask counters by route, end-to-end latency histogram, cache/fallback/
    rewrite counters, per-route-pattern HTTP traffic, corpus/session/
    feedback/cache gauges and the last eval + benchmark results, plus the
    triage lifecycle (open/resolved fix-list items).

    Scrapable by a stock Prometheus server (`curl localhost:3003/metrics`
    works too) — the exact same numbers /stats reports, plus HTTP counters
    collected by the ASGI middleware."""
    import time as _time

    st = _state()
    t0 = _time.perf_counter()
    try:
        info = st.models.serving_info()
        serving_mode, serving_model = info.get("mode"), info.get("model")
    except Exception:  # noqa: BLE001
        serving_mode, serving_model = st.models.mode, None
    _hist = st.eval_history.trend(limit=1)
    snap_eval_runs = {
        "total": _hist["total_runs"],
        "golden": _hist["golden_runs"],
        "ablation": _hist["ablation_runs"],
        "last_accuracy": (_hist.get("last") or {}).get("accuracy"),
    }
    corpus_stats = st.store.stats()
    raw = st.metrics.prometheus_snapshot()
    doc = render_metrics(
        collector=st.metrics,
        http_middleware=HTTPMetricsMiddleware,
        gauges={
            "version": VERSION,
            "serving_mode": serving_mode,
            "serving_model": serving_model or "",
            "uptime_seconds": raw["uptime_seconds"],
            "corpus": {
                "documents": corpus_stats.get("total_documents", 0),
                "chunks": corpus_stats.get("total_chunks", 0),
                "embedding_dim": corpus_stats.get("embedding_dim", 0),
            },
            "sessions": st.sessions.stats().get("sessions", 0),
            "feedback": {
                k: v for k, v in st.feedback.stats().items()
                if k in {"up", "down"}
            },
            "cache": st.cache.stats(),
            "last_eval": raw["last_eval"],
            "last_benchmark": raw["last_benchmark"],
            "rerank": st.reranker.stats(),
            "golden": st.golden.stats(),
            "ledger": st.ledger.stats(),
            "eval_runs": snap_eval_runs,
            "triage": _triage_counts(st),
        },
    )
    HTTPMetricsMiddleware.note_scrape((_time.perf_counter() - t0) * 1000)
    return Response(
        content=doc,
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


# ------------------------------------------------------------- ops/grafana
@app.get("/ops/grafana")
async def grafana_dashboard():
    """Grafana-importable dashboard JSON provisioning the rag_* metric
    families exposed at /metrics (traffic, latency quantiles, cache, honesty
    signals, eval quality, corpus/memory). Import via Grafana's dashboard
    JSON import, or scrape this endpoint directly."""
    from app.grafana import build_dashboard

    return build_dashboard()


# ----------------------------------------------------------------------- eval
@app.post("/eval")
async def run_eval():
    """Run the golden Q&A set through the full agentic pipeline (Part F bonus:
    RAGAS-style evaluation harness — route correctness, retrieval hit@k,
    answer fact-support, latency)."""
    st = _state()
    report = await st.eval_runner.run()
    st.metrics.observe_eval(report)
    st.eval_history.record(report)
    return report


# Retriever configurations under ablation: top-k depth, reranker weights,
# and the cross-encoder stage itself. The baseline mirrors the shipped
# defaults; every config runs the SAME golden set through the SAME agent
# graph — only retrieval parameters vary.
ABLATION_CONFIGS: list[dict] = [
    {"name": "baseline (top-k 4, default weights, cross-encoder)"},
    {"name": "cross-encoder off (fusion only)", "rerank": False},
    {"name": "cross-encoder broad (n=14)", "rerank_n": 14},
    {"name": "top-k 3 (narrow)", "top_k": 3},
    {"name": "top-k 8 (broad)", "top_k": 8},
    {"name": "coverage-heavy (w 0.35/0.55/0.10)",
     "weights": {"w_fused": 0.35, "w_coverage": 0.55, "w_agreement": 0.10}},
    {"name": "fusion-heavy (w 0.75/0.15/0.10)",
     "weights": {"w_fused": 0.75, "w_coverage": 0.15, "w_agreement": 0.10}},
]


@app.post("/eval/ablation")
async def run_ablation():
    """Retrieval ablation study: run the golden set under each configuration
    (top-k depth, reranker weight profile) and report accuracy / hit@k /
    fact-support / latency per config with deltas vs the baseline.

    This is the ML-engineering experimentation loop made reproducible: the
    exact code the agent graph uses is re-parameterised per run, so the table
    shows what each retrieval knob actually buys on a fixed eval set.
    """
    import time as _time

    st = _state()
    t0 = _time.perf_counter()
    saved_report = st.eval_runner.last_report  # keep /eval/results stable
    baseline_accuracy: float | None = None
    configs_out: list[dict] = []
    for cfg in ABLATION_CONFIGS:
        weights = cfg.get("weights")
        previous = st.retriever.apply_weights(**weights) if weights else None
        saved_rr = st.retriever.rerank_enabled
        saved_n = st.reranker.n if st.reranker else None
        if "rerank" in cfg and st.reranker is not None:
            st.retriever.rerank_enabled = bool(cfg["rerank"])
        if cfg.get("rerank_n") and st.reranker is not None:
            st.reranker.n = int(cfg["rerank_n"])
        try:
            report = await st.eval_runner.run(
                top_k=cfg.get("top_k") or 4,
                with_faithfulness=False,
                label=f"ablation:{cfg['name']}",
            )
            st.eval_history.record(report)
        finally:
            if previous is not None:
                st.retriever.restore_weights(previous)
            st.retriever.rerank_enabled = saved_rr
            if saved_n is not None and st.reranker is not None:
                st.reranker.n = saved_n
        accuracy = report["accuracy"]
        if baseline_accuracy is None:
            baseline_accuracy = accuracy
        configs_out.append({
            "name": cfg["name"],
            "config": {k: v for k, v in cfg.items() if k != "name"},
            "accuracy": accuracy,
            "retrieval_hit_rate": report["retrieval_hit_rate"],
            "fact_support_rate": report["fact_support_rate"],
            "faithfulness_avg": None,
            "latency_p50_ms": report["latency_p50_ms"],
            "latency_p95_ms": report["latency_p95_ms"],
            "passed": report["passed"],
            "total": report["total_cases"],
            "delta_accuracy": round(accuracy - (baseline_accuracy or 0.0), 4),
        })
    st.eval_runner.last_report = saved_report
    winner = max(
        configs_out,
        key=lambda c: (c["accuracy"], -c["latency_p50_ms"]),
    )["name"]
    from datetime import datetime, timezone

    return {
        "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "baseline": ABLATION_CONFIGS[0]["name"],
        "configs": configs_out,
        "winner": winner,
        "duration_ms": round((_time.perf_counter() - t0) * 1000, 1),
        "notes": [
            "each config runs the full golden set through the real agent graph",
            "faithfulness judged only on the default /eval run (kept off here for speed)",
            "weights (fused/coverage/agreement) are the fusion heuristic profile",
            "cross-encoder rows toggle the rerank stage (off) or widen its pool (n=14)",
        ],
    }


@app.get("/eval/results")
async def eval_results():
    st = _state()
    if st.eval_runner.last_report is None:
        raise HTTPException(status_code=404, detail="no evaluation has been run yet")
    return st.eval_runner.last_report


@app.get("/eval/history")
async def eval_history(include_ablation: bool = Query(
        False, description="include ablation-config runs (judge off)")):
    """Persisted trend of eval runs: accuracy / faithfulness / latency over
    time + how golden-set growth changed the case count. Each POST /eval
    (and each ablation config) appends a compact summary — newest first."""
    st = _state()
    return st.eval_history.trend(include_ablation=include_ablation, limit=20)


def _triage_counts(st: "AppState") -> dict[str, Any]:
    """Open/resolved triage counts + SLA ages for /stats + /metrics.

    Single join (limit=0 → counts only, both lists empty) so the two
    surfaces always agree with GET /quality/triage without running the
    item-building work twice. Round 20 adds the open-fix-list ages:
    ``oldest_open_hours`` / ``mean_open_hours`` (None when nothing is open).
    """
    counts = st.golden.build_triage(
        st.ledger, st.feedback.all_records(), st.triage, limit=0
    )["counts"]
    return {
        "open": int(counts.get("open", 0)),
        "resolved": int(counts.get("resolved", 0)),
        "oldest_open_hours": counts.get("oldest_open_hours"),
        "mean_open_hours": counts.get("mean_open_hours"),
    }


@app.get("/quality/triage")
async def quality_triage():
    """Downvote triage — the quality radar. 👎 answers joined against the
    answer ledger: question, answer preview, fallback flag, citation count,
    optional user reason. Where /eval/candidates turns upvotes into eval
    coverage, this turns downvotes into a fix-list.

    Round 19: ``items`` is the OPEN fix-list only (unresolved downvotes);
    resolved ones move to ``resolved_items`` (newest resolved first, with
    resolution note + timestamp). ``counts`` gains ``open``/``resolved``
    (accurate across all downvoted records). POST /quality/triage/resolve
    and /quality/triage/reopen move items between the two lists."""
    st = _state()
    return st.golden.build_triage(st.ledger, st.feedback.all_records(), st.triage)


class TriageResolveRequest(BaseModel):
    """Mark a downvoted answer as fixed (optionally noting how)."""
    request_id: str = Field(..., min_length=1, max_length=64,
                            description="Request id of the downvoted answer")
    note: str = Field("", max_length=300,
                      description="How it was fixed (optional, ≤300 chars)")


class TriageReopenRequest(BaseModel):
    """Reopen a previously resolved triage item."""
    request_id: str = Field(..., min_length=1, max_length=64)


@app.post("/quality/triage/resolve")
async def resolve_triage_item(request: TriageResolveRequest):
    """Resolve a 👎 triage item — the fix-list's "done" action. Validates
    that the request_id currently carries a downvote (404 otherwise), then
    records the resolution (note stripped to ≤300 chars, None-able) in the
    persisted tracker. The item leaves ``items`` and lands in
    ``resolved_items`` on the next GET /quality/triage; counts are returned
    inline so the UI can update without a refetch."""
    st = _state()
    has_downvote = any(
        rec.get("request_id") == request.request_id and rec.get("rating") == "down"
        for rec in st.feedback.all_records()
    )
    if not has_downvote:
        raise HTTPException(status_code=404, detail="request_id has no downvote")
    record = st.triage.resolve(request.request_id, request.note)
    return {
        "status": "resolved",
        "request_id": request.request_id,
        "note": record["note"],
        "resolved_at": record["resolved_at"],
        "counts": _triage_counts(st),
    }


@app.post("/quality/triage/reopen")
async def reopen_triage_item(request: TriageReopenRequest):
    """Reopen a resolved triage item — the resolution record is deleted and
    the downvote re-enters the open fix-list. 404 if the item is not
    currently resolved. Useful when a "fix" turns out not to hold."""
    st = _state()
    if not st.triage.reopen(request.request_id):
        raise HTTPException(status_code=404, detail="request_id is not currently resolved")
    return {
        "status": "reopened",
        "request_id": request.request_id,
        "counts": _triage_counts(st),
    }


@app.get("/eval/cases")
async def eval_cases():
    """The golden set itself (question + expectations), for the UI —
    base cases plus user-grown cases promoted from feedback."""
    st = _state()
    cases = st.golden.cases()
    return {
        "total": len(cases),
        "stats": st.golden.stats(),
        "cases": [
            {
                "id": c.id,
                "question": c.question,
                "category": c.category,
                "expects": c.expects,
                "note": c.note,
                "source": st.golden.source_of(c.id),
                "keywords": c.answer_must_match_any,
                "docs_any": c.docs_any,
                "deletable": st.golden.source_of(c.id) == "user",
            }
            for c in cases
        ],
    }


class PromoteRequest(BaseModel):
    """Promote an upvoted answer into the golden set."""
    request_id: str = Field(..., min_length=1, max_length=64,
                            description="Ledger/request id of the upvoted answer")
    keywords: Optional[list[str]] = Field(
        None, max_length=6,
        description="Answer fact patterns; defaults to the auto-suggestions")
    docs_any: Optional[list[str]] = Field(
        None, max_length=6,
        description="Expected source docs; defaults to the answer's citations")
    note: str = Field("", max_length=300)


class BatchPromoteRequest(BaseModel):
    """Promote several upvoted answers into the golden set in one call."""
    request_ids: list[str] = Field(
        ..., min_length=1, max_length=12,
        description="Ledger/request ids of upvoted answers (1–12; duplicates "
                    "in the input are silently de-duplicated, order kept)")
    keywords_map: Optional[dict[str, list[str]]] = Field(
        None,
        description="Per-request edited fact patterns, keyed by request_id "
                    "(same effect as PromoteRequest.keywords per item)")
    note: str = Field("", max_length=300)


@app.get("/eval/candidates")
async def eval_candidates():
    """Golden-set growth candidates: upvoted answers joined against the
    answer ledger, each with auto-suggested fact patterns + source docs.
    Review, edit keywords, promote via POST /eval/candidates/promote (one)
    or POST /eval/candidates/promote-batch (many at once)."""
    st = _state()
    return st.golden.build_candidates(st.ledger, st.feedback.all_records())


async def _promote_one(
    st: AppState,
    request_id: str,
    keywords: Optional[list[str]] = None,
    docs_any: Optional[list[str]] = None,
    note: str = "",
) -> dict:
    """Promote ONE upvoted ledger answer into the golden set + verify it.

    Shared by the single and batch promote endpoints so the two paths can
    never diverge. Returns ``{"status": "promoted", "case": {...},
    "verification": {...}}`` on success, or ``{"status": "failed",
    "error": str, "code": 404|422}`` for the known rejection cases —
    unknown ledger id (404), non-grounded answer (422), or case-validation
    error (422). All rejections happen BEFORE the case is persisted, so a
    failed promote never leaves partial state behind; a successful one
    persists the case (then verifies — a verification that *raises* leaves
    the promoted case in place, mirroring the single-promote behavior that
    /eval surfaces as a red case).
    """
    entry = st.ledger.get(request_id)
    if entry is None:
        return {"status": "failed",
                "error": "request_id not found in the answer ledger", "code": 404}
    if entry.get("route") != "document" or entry.get("fallback") or not entry.get("answer"):
        return {"status": "failed",
                "error": "only grounded answers can become golden cases", "code": 422}
    from app.eval.growth import suggest_keywords
    kw = keywords or suggest_keywords(entry["answer"])
    docs = docs_any
    if docs is None:
        docs = []
        for c in entry.get("citations", []):
            d = c.get("doc", "")
            if d and d not in docs:
                docs.append(d)
    try:
        case = st.golden.add_case(
            question=entry.get("question") or entry.get("effective_question", ""),
            keywords=kw,
            docs_any=docs,
            expects="grounded",
            category="feedback",
            note=note or "promoted from 👍 feedback",
            origin={
                "request_id": entry.get("request_id"),
                "answer_chars": len(entry.get("answer", "")),
                "cache_hit": entry.get("cache_hit", False),
            },
        )
    except ValueError as exc:
        return {"status": "failed", "error": str(exc), "code": 422}
    verification = await st.eval_runner.run_case(case)
    return {
        "status": "promoted",
        "case": {
            "id": case.id,
            "question": case.question,
            "keywords": case.answer_must_match_any,
            "docs_any": case.docs_any,
            "category": case.category,
            "note": case.note,
        },
        "verification": verification,
    }


@app.post("/eval/candidates/promote")
async def promote_candidate(request: PromoteRequest):
    """Turn an upvoted answer into a golden eval case. The ledger provides
    question/answer/citations; keywords default to the miner's suggestions.
    The new case is immediately dry-run through the full agent graph and the
    verification result is returned (promoted cases that fail show up red in
    /eval — delete them or re-promote with edited keywords)."""
    st = _state()
    result = await _promote_one(
        st, request.request_id, request.keywords, request.docs_any, request.note
    )
    if result["status"] == "failed":
        raise HTTPException(status_code=result.get("code", 422), detail=result["error"])
    return {
        "status": "promoted",
        "case": result["case"],
        "golden_set": st.golden.stats(),
        "verification": result["verification"],
    }


@app.post("/eval/candidates/promote-batch")
async def promote_candidates_batch(request: BatchPromoteRequest):
    """Promote a batch of upvoted answers (1–12) into the golden set.

    Each id goes through the exact same single-promote logic
    (``_promote_one``): ledger lookup → grounded check → case creation →
    immediate verification through the full agent graph (sequential — each
    verification runs the graph, ~1 s in simulation mode). Per-item
    failures are reported in ``results`` with their ``error`` and do NOT
    abort the batch. Duplicate request_ids in the input list are silently
    de-duplicated (order preserved) — promoting the same id in two
    separate batches would create a second user case, which the candidates
    join normally screens out; the de-duped batch never does.
    ``keywords_map`` carries per-request edited patterns (entries for ids
    not in the batch are ignored)."""
    st = _state()
    seen: set[str] = set()
    ids: list[str] = []
    for rid in request.request_ids:
        rid = str(rid).strip()
        if rid and rid not in seen:
            seen.add(rid)
            ids.append(rid)
    keywords_map = request.keywords_map or {}
    results: list[dict] = []
    promoted = 0
    failed = 0
    for rid in ids:
        entry = st.ledger.get(rid)
        try:
            outcome = await _promote_one(
                st, rid, keywords_map.get(rid), None, request.note
            )
        except Exception as exc:  # noqa: BLE001 — one bad item must not kill the batch
            logger.exception("batch promote: verification failed for %s", rid)
            outcome = {"status": "failed",
                       "error": f"{type(exc).__name__}: {exc}", "code": 500}
        row: dict = {
            "request_id": rid,
            "question": (entry or {}).get("question")
                        or (entry or {}).get("effective_question"),
            "status": outcome["status"],
        }
        if outcome["status"] == "promoted":
            promoted += 1
            verification = outcome["verification"]
            row["case_id"] = outcome["case"]["id"]
            row["passed"] = verification.get("passed")
            row["matched"] = verification.get("matched")
        else:
            failed += 1
            row["error"] = outcome["error"]
        results.append(row)
    return {
        "status": "batch-complete",
        "promoted": promoted,
        "failed": failed,
        "results": results,
        "golden_set": st.golden.stats(),
    }


class ManualCaseRequest(BaseModel):
    """Hand-authored golden case (no feedback required)."""
    question: str = Field(..., min_length=4, max_length=1000)
    keywords: list[str] = Field(..., min_length=1, max_length=6)
    docs_any: Optional[list[str]] = Field(None, max_length=6)
    expects: str = Field("grounded")
    category: str = Field("manual")
    note: str = Field("", max_length=300)


@app.post("/eval/cases")
async def add_eval_case(request: ManualCaseRequest):
    """Add a manually authored golden case (question + expected fact
    patterns). Runs through the same validation + immediate verification as
    promoted candidates."""
    st = _state()
    try:
        case = st.golden.add_case(
            question=request.question,
            keywords=request.keywords,
            docs_any=request.docs_any,
            expects=request.expects,
            category=request.category,
            note=request.note or "manually authored",
            origin={"manual": True},
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    verification = await st.eval_runner.run_case(case)
    return {
        "status": "added",
        "case": {
            "id": case.id,
            "question": case.question,
            "keywords": case.answer_must_match_any,
            "docs_any": case.docs_any,
            "category": case.category,
            "note": case.note,
        },
        "golden_set": st.golden.stats(),
        "verification": verification,
    }


@app.delete("/eval/cases/{case_id}")
async def delete_eval_case(case_id: str):
    """Remove a user-grown golden case (base cases are immutable)."""
    st = _state()
    if not st.golden.remove_case(case_id):
        raise HTTPException(status_code=404, detail=f"case {case_id!r} not found or is a base case")
    return {"status": "deleted", "case_id": case_id, "golden_set": st.golden.stats()}


# ----------------------------------------------------------- A/B comparison
class CompareRequest(BaseModel):
    """Side-by-side answer comparison under two retrieval configurations."""
    question: str = Field(..., min_length=2, max_length=1000)
    config_a: Optional[str] = Field(
        None, max_length=80,
        description="Named ablation config for side A (default: baseline)")
    config_b: Optional[str] = Field(
        None, max_length=80,
        description="Named ablation config for side B (default: cross-encoder off)")


async def _apply_config(st: AppState, cfg: dict):
    """Apply one ablation config to the live retriever; returns a
    restore-closure used in the finally block (same pattern as the
    /eval/ablation harness)."""
    weights = cfg.get("weights")
    previous = st.retriever.apply_weights(**weights) if weights else None
    saved_rr = st.retriever.rerank_enabled
    saved_n = st.reranker.n if st.reranker else None
    if "rerank" in cfg and st.reranker is not None:
        st.retriever.rerank_enabled = bool(cfg["rerank"])
    if cfg.get("rerank_n") and st.reranker is not None:
        st.reranker.n = int(cfg["rerank_n"])

    async def restore() -> None:
        if previous is not None:
            st.retriever.restore_weights(previous)
        st.retriever.rerank_enabled = saved_rr
        if saved_n is not None and st.reranker is not None:
            st.reranker.n = saved_n

    return restore


def _find_config(name: str | None, default: dict) -> dict:
    if not name:
        return default
    for cfg in ABLATION_CONFIGS:
        if cfg["name"] == name:
            return cfg
    raise HTTPException(
        status_code=422,
        detail=f"unknown config {name!r}; choose from: "
               + ", ".join(c["name"] for c in ABLATION_CONFIGS),
    )


@app.post("/eval/compare")
async def compare_answers(request: CompareRequest):
    """Answer-quality A/B: run ONE question through the full agent graph
    under two retrieval configurations (named ablation configs) and return
    both answers with citations, faithfulness and latency, plus a verdict
    block (answer agreement, citation overlap, deltas). The graph, cache and
    sessions are untouched — each side runs the cache-free eval ask path."""
    import time as _time

    st = _state()
    cfg_a = _find_config(request.config_a, ABLATION_CONFIGS[0])
    cfg_b = _find_config(request.config_b, ABLATION_CONFIGS[1])
    if cfg_a["name"] == cfg_b["name"]:
        raise HTTPException(status_code=422, detail="config_a and config_b must differ")

    async def _run_side(cfg: dict) -> dict:
        restore = await _apply_config(st, cfg)
        try:
            t0 = _time.perf_counter()
            result = await st._eval_ask(request.question, top_k=cfg.get("top_k") or 4)
            wall_ms = round((_time.perf_counter() - t0) * 1000, 1)
        finally:
            await restore()
        answer = result.answer or ""
        faith = None
        if answer and result.citations and not result.fallback:
            try:
                fr = await st.faithfulness.score(
                    answer, list(result.citations), question=request.question,
                    chunk_text_lookup=st._chunk_text,
                )
                faith = fr.as_dict()
            except Exception:  # noqa: BLE001 — judge is observability, not a gate
                faith = None
        docs_a = sorted({c.get("doc", "") for c in result.citations})
        rerank_detail = None
        for t in result.trace:
            node = getattr(t, "node", None)
            detail = getattr(t, "detail", None)
            if node == "retriever" and isinstance(detail, dict):
                rerank_detail = detail.get("rerank")
                break
        return {
            "config": cfg["name"],
            "config_detail": {k: v for k, v in cfg.items() if k != "name"},
            "answer": answer,
            "citations": [
                {"doc": c.get("doc", ""), "page": c.get("page", 1),
                 "snippet": c.get("snippet", "")[:160]}
                for c in result.citations[:6]
            ],
            "route": result.route,
            "fallback": bool(result.fallback),
            "decomposed": bool(result.decomposed),
            "agent_path": result.agent_path,
            "latency_ms": wall_ms,
            "faithfulness": faith,
            "distinct_docs": docs_a,
            "rerank": rerank_detail,
        }

    side_a = await _run_side(cfg_a)
    side_b = await _run_side(cfg_b)

    def _norm(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    docs_a, docs_b = set(side_a["distinct_docs"]), set(side_b["distinct_docs"])
    union = docs_a | docs_b or {""}
    overlap = round(len(docs_a & docs_b) / len(union), 4) if docs_a or docs_b else 0.0
    fa = (side_a.get("faithfulness") or {}).get("score")
    fb = (side_b.get("faithfulness") or {}).get("score")
    verdict = {
        "answers_identical": _norm(side_a["answer"]) == _norm(side_b["answer"]),
        "answer_a_chars": len(side_a["answer"]),
        "answer_b_chars": len(side_b["answer"]),
        "citation_overlap": overlap,
        "shared_docs": sorted(docs_a & docs_b),
        "docs_only_a": sorted(docs_a - docs_b),
        "docs_only_b": sorted(docs_b - docs_a),
        "latency_delta_ms": round(side_b["latency_ms"] - side_a["latency_ms"], 1),
        "faithfulness_delta": (
            round(float(fb) - float(fa), 3)
            if fa is not None and fb is not None else None
        ),
    }
    from datetime import datetime, timezone

    return {
        "question": request.question,
        "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "side_a": side_a,
        "side_b": side_b,
        "verdict": verdict,
        "available_configs": [c["name"] for c in ABLATION_CONFIGS],
    }


# ------------------------------------------------------------------ sessions
@app.get("/sessions")
async def list_sessions():
    st = _state()
    return {"sessions": st.sessions.list_sessions(), "stats": st.sessions.stats()}


@app.get("/sessions/{session_id}")
async def session_history(session_id: str):
    st = _state()
    history = st.sessions.history(session_id)
    if history is None:
        raise HTTPException(status_code=404, detail="unknown session")
    return {"session_id": session_id, "turns": history}


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    st = _state()
    if not st.sessions.delete(session_id):
        raise HTTPException(status_code=404, detail="unknown session")
    return {"status": "deleted", "session_id": session_id}


# -------------------------------------------------------------------- search
@app.get("/search")
async def search(
    q: str = Query(..., min_length=2, max_length=500, description="Search query"),
    k: int = Query(5, ge=1, le=20, description="Number of results"),
    wf: float | None = Query(None, ge=0.0, le=1.0,
                             description="Reranker weight: RRF fusion (default 0.55)"),
    wc: float | None = Query(None, ge=0.0, le=1.0,
                             description="Reranker weight: query coverage (default 0.35)"),
    wa: float | None = Query(None, ge=0.0, le=1.0,
                             description="Reranker weight: retriever agreement (default 0.10)"),
    rr: bool = Query(True, description="Cross-encoder rerank stage on/off"),
):
    """Interactive hybrid retrieval (BM25 + dense + RRF fusion + cross-encoder
    rerank) over the corpus. Exposes per-result dense/bm25 ranks AND the
    cross-encoder score/fusion-rank pair so the ranking is fully observable —
    a live demo of the retrieval layer behind the agent graph. Optional
    weight parameters and the ``rr`` flag power the retrieval playground:
    the SAME ranking code the agent uses, re-parameterised per request."""
    st = _state()
    weights_applied: dict[str, float]
    saved_rr = st.retriever.rerank_enabled
    st.retriever.rerank_enabled = rr and st.retriever.reranker is not None
    try:
        if wf is not None or wc is not None or wa is not None:
            previous = st.retriever.apply_weights(w_fused=wf, w_coverage=wc, w_agreement=wa)
            weights_applied = dict(st.retriever.weights)
            try:
                hits = await st.retriever.search(q, k=k)
            finally:
                st.retriever.restore_weights(previous)
        else:
            weights_applied = dict(st.retriever.weights)
            hits = await st.retriever.search(q, k=k)
    finally:
        st.retriever.rerank_enabled = saved_rr
    rerank_t = st.retriever.last_rerank
    rerank_info: dict | None = None
    if rerank_t:
        rerank_info = {
            "stage": "cross-encoder",
            "mode": st.reranker.mode if st.reranker else None,
            "reranked": rerank_t.get("reranked", 0),
            "reorders": rerank_t.get("reorders", 0),
            "latency_ms": rerank_t.get("latency_ms", 0.0),
        }
    return {
        "query": q,
        "weights": weights_applied,
        "rerank": rerank_info,
        "results": [
            {
                "chunk_id": h.chunk.chunk_id,
                "doc": h.chunk.doc,
                "page": h.chunk.page,
                "section": h.chunk.section,
                "score": h.score,
                "coverage": h.coverage,
                "dense_rank": h.dense_rank,
                "bm25_rank": h.bm25_rank,
                "rerank_score": h.rerank_score,
                "fusion_rank": h.fusion_rank,
                "is_figure": h.chunk.is_figure_caption,
                "preview": h.chunk.preview(200),
            }
            for h in hits
        ],
    }


# -------------------------------------------------------------------- source
@app.get("/source/{chunk_id}")
async def get_source(chunk_id: str):
    """Full chunk text for citation verification, with adjacent chunks from
    the same document as reading context."""
    st = _state()
    chunk = st.store.get_by_chunk_id(chunk_id)
    if chunk is None:
        raise HTTPException(status_code=404, detail="unknown chunk_id")
    snap = st.store.snapshot()
    idx = next((i for i, c in enumerate(snap) if c.chunk_id == chunk_id), None)
    neighbors = []
    if idx is not None:
        for off in (-1, 1):
            j = idx + off
            if 0 <= j < len(snap) and snap[j].doc == chunk.doc:
                neighbors.append(
                    {
                        "chunk_id": snap[j].chunk_id,
                        "page": snap[j].page,
                        "section": snap[j].section,
                        "preview": snap[j].preview(160),
                        "relation": "previous" if off < 0 else "next",
                    }
                )
    return {
        "chunk_id": chunk.chunk_id,
        "doc": chunk.doc,
        "doc_kind": chunk.doc_kind,
        "page": chunk.page,
        "section": chunk.section,
        "text": chunk.text,
        "token_count": chunk.token_count,
        "is_figure_caption": chunk.is_figure_caption,
        "neighbors": neighbors,
    }


# ------------------------------------------------------------------ feedback
@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Answer rating (thumbs up/down) tied to a request_id — lightweight
    answer-quality telemetry that complements the golden-set eval."""
    st = _state()
    record = st.feedback.add(request.request_id, request.rating, request.comment)
    return {"status": "recorded", "rating": record["rating"], "total": st.feedback.total()}


@app.get("/feedback/stats")
async def feedback_stats():
    st = _state()
    return st.feedback.stats()


# --------------------------------------------------------------- analytics
@app.get("/analytics/questions")
async def analytics_questions(limit: int = Query(8, ge=1, le=25, description="Top-N questions to return")):
    """Question analytics — the usage lens over the answer ledger.

    Mines every recorded ask (capped 300-entry ring) for question
    *frequencies*: normalised (lower-cased, whitespace-collapsed) question
    text → ask count, last-asked time, routes seen, fallback rate,
    citation rate, mean latency and the 👍/👎 votes joined in from the
    feedback store. Sorted by count, then recency. Complements /stats
    (which reports aggregates) with *what* users actually ask — the input
    signal for golden-set growth and corpus coverage decisions."""
    st = _state()
    rating_by_rid: dict[str, str] = {}
    for rec in st.feedback.all_records():
        rid = str(rec.get("request_id", ""))
        if rid and rec.get("rating") in ("up", "down"):
            rating_by_rid[rid] = str(rec["rating"])

    def _norm(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    groups: dict[str, dict[str, Any]] = {}
    total_answered = 0
    for entry in st.ledger.all_entries():
        question = (entry.get("question") or entry.get("effective_question") or "").strip()
        if not question:
            continue
        total_answered += 1
        key = _norm(question)
        g = groups.setdefault(key, {
            "question": question[:200],
            "count": 0,
            "last_ts": 0.0,
            "routes": set(),
            "fallbacks": 0,
            "cited": 0,
            "latency_sum": 0.0,
            "cache_hits": 0,
            "ups": 0,
            "downs": 0,
        })
        g["count"] += 1
        g["last_ts"] = max(g["last_ts"], entry.get("ts") or 0.0)
        route = entry.get("route") or "unknown"
        g["routes"].add(route)
        if entry.get("fallback"):
            g["fallbacks"] += 1
        if entry.get("citations"):
            g["cited"] += 1
        if entry.get("cache_hit"):
            g["cache_hits"] += 1
        g["latency_sum"] += entry.get("latency_ms", 0.0) or 0.0
        vote = rating_by_rid.get(str(entry.get("request_id", "")))
        if vote == "up":
            g["ups"] += 1
        elif vote == "down":
            g["downs"] += 1

    ranked = sorted(
        groups.values(), key=lambda g: (g["count"], g["last_ts"]), reverse=True
    )[:limit]
    top = [{
        "question": g["question"],
        "count": g["count"],
        "last_ts": g["last_ts"],
        "routes": sorted(g["routes"]),
        "fallback_rate": round(g["fallbacks"] / g["count"], 3) if g["count"] else 0.0,
        "citation_rate": round(g["cited"] / g["count"], 3) if g["count"] else 0.0,
        "cache_rate": round(g["cache_hits"] / g["count"], 3) if g["count"] else 0.0,
        "avg_latency_ms": round(g["latency_sum"] / g["count"], 1) if g["count"] else 0.0,
        "ups": g["ups"],
        "downs": g["downs"],
    } for g in ranked]
    return {
        "top": top,
        "totals": {
            "distinct_questions": len(groups),
            "answered_asks": total_answered,
            "ledger_capacity": st.ledger.stats()["capacity"],
        },
    }
