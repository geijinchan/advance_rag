"""
================================================================================
ADVANCED RAG BACKEND
LangGraph orchestration + Groq LLMs + Hybrid Retrieval (BM25 + Dense)
PDF upload, chunking, RAG-Fusion, MMR, LLM reranking, self-correcting loop.
================================================================================

PIPELINE (matches the architecture diagram):
    question
      -> query_transform (rewrite + multi-query expansion)   [Groq]
      -> retrieve        (BM25 ∥ Dense, per query, parallel)
      -> rag_fusion      (Reciprocal Rank Fusion)
      -> mmr             (diversity selection)
      -> rerank          (LLM cross-encoder style scoring)   [Groq]
      -> build_context   (dedup + assembly)
      -> generate        (grounded answer w/ citations)      [Groq]
      -> validate        (grounding judge)                   [Groq]
           ├─ grounded or retries exhausted -> END
           └─ else -> back to query_transform with feedback  (CYCLE)

--------------------------------------------------------------------------------
SETUP
--------------------------------------------------------------------------------
# requirements.txt
    fastapi>=0.110
    uvicorn[standard]>=0.29
    python-multipart>=0.0.9
    pydantic>=2.6
    pypdf>=4.2
    numpy>=1.26
    faiss-cpu>=1.8
    rank-bm25>=0.2.2
    sentence-transformers>=2.7
    langchain>=0.2
    langchain-core>=0.2
    langchain-groq>=0.1.6
    langchain-huggingface>=0.0.3
    langchain-text-splitters>=0.2
    langgraph>=0.2

# .env
    GROQ_API_KEY=gsk_...

# run  (put GROQ_API_KEY in a .env beside this file, then:)
    python app.py                              # -> http://localhost:8000
    #  or, with auto-reload for development:
    uvicorn app:api --host 0.0.0.0 --port 8000 --reload

# usage
    curl -F "file=@paper.pdf" http://localhost:8000/upload
    curl -X POST http://localhost:8000/query \
         -H "Content-Type: application/json" \
         -d '{"question": "What loss function does the model use?"}'

NOTE ON EMBEDDINGS: Groq hosts LLMs only (no embedding endpoint), so dense
vectors use a local sentence-transformers model (all-MiniLM-L6-v2, ~80MB,
downloads once). Swap `build_embedder()` if you prefer another provider.
================================================================================
"""

from __future__ import annotations

import io
import os
import re
import json
import time
import uuid
import asyncio
import logging
import threading
import operator
from typing import Any, Annotated, TypedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import faiss
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import StateGraph, START, END

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("rag")

# Load GROQ_API_KEY (and model overrides) from the .env sitting beside this file,
# so the app runs no matter which directory it's launched from. override=True means
# this .env is authoritative — it wins over any stale key already in the shell env.
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env", override=True)
except Exception:  # python-dotenv is optional; env vars may already be exported
    pass


# ==============================================================================
# 1. CONFIG
# ==============================================================================

class Settings:
    GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")

    # Groq-hosted models (see https://console.groq.com/docs/models)
    LLM_FAST: str = os.environ.get("LLM_FAST", "llama-3.1-8b-instant")        # query transform, validation
    LLM_SMART: str = os.environ.get("LLM_SMART", "llama-3.3-70b-versatile")   # reranking, generation

    EMBED_MODEL: str = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # Chunking
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150

    # Retrieval breadth
    PER_QUERY_TOP_K: int = 8        # per (query x retriever)
    NUM_EXPANDED_QUERIES: int = 3
    FUSION_TOP_K: int = 40          # candidates after RRF
    RRF_K: int = 60

    # Diversity / precision
    MMR_TOP_K: int = 15
    MMR_LAMBDA: float = 0.65        # 1.0 = pure relevance, 0.0 = pure diversity
    FINAL_TOP_K: int = 6            # docs sent to the generator

    # Self-correction
    MAX_RETRIES: int = 2

    # Context safety
    MAX_DOC_CHARS: int = 1500       # truncate chunks shown to reranker/generator


SETTINGS = Settings()
if not SETTINGS.GROQ_API_KEY:
    log.warning("GROQ_API_KEY is not set — LLM calls will fail.")


# ==============================================================================
# 2. MODELS (lazy singletons)
# ==============================================================================

_llm_fast: ChatGroq | None = None
_llm_smart: ChatGroq | None = None
_embedder: HuggingFaceEmbeddings | None = None


def get_llm_fast() -> ChatGroq:
    global _llm_fast
    if _llm_fast is None:
        _llm_fast = ChatGroq(model=SETTINGS.LLM_FAST, temperature=0, api_key=SETTINGS.GROQ_API_KEY)
    return _llm_fast


def get_llm_smart() -> ChatGroq:
    global _llm_smart
    if _llm_smart is None:
        _llm_smart = ChatGroq(model=SETTINGS.LLM_SMART, temperature=0, api_key=SETTINGS.GROQ_API_KEY)
    return _llm_smart


def get_embedder() -> HuggingFaceEmbeddings:
    global _embedder
    if _embedder is None:
        log.info(f"Loading embedding model: {SETTINGS.EMBED_MODEL} (first run downloads it)")
        _embedder = HuggingFaceEmbeddings(
            model_name=SETTINGS.EMBED_MODEL,
            encode_kwargs={"normalize_embeddings": True},  # cosine == inner product
        )
    return _embedder


# ==============================================================================
# 3. DOCUMENT STORE  (BM25 index + FAISS index + docstore, thread-safe)
# ==============================================================================

def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class DocumentStore:
    """
    Holds all indexed chunks. Dual index:
      - BM25Okapi over tokenized chunks (sparse/lexical)
      - FAISS IndexFlatIP over normalized embeddings (dense/cosine)
    """

    def __init__(self) -> None:
        self.docs: dict[str, dict] = {}        # doc_id -> {id, text, metadata}
        self.id_order: list[str] = []          # faiss row -> doc_id
        self.bm25: BM25Okapi | None = None
        self.bm25_ids: list[str] = []
        self.faiss_index: faiss.IndexFlatIP | None = None
        self.lock = threading.Lock()

    # ---------------- ingestion ----------------

    def add_documents(self, chunks: list[dict]) -> int:
        if not chunks:
            return 0
        embedder = get_embedder()
        texts = [c["text"] for c in chunks]
        vectors = np.array(embedder.embed_documents(texts), dtype=np.float32)
        # normalize_embeddings=True already unit-normalizes; re-normalize defensively
        faiss.normalize_L2(vectors)

        with self.lock:
            if self.faiss_index is None:
                self.faiss_index = faiss.IndexFlatIP(vectors.shape[1])

            for chunk, vec in zip(chunks, vectors):
                doc_id = chunk["id"]
                self.docs[doc_id] = chunk
                self.id_order.append(doc_id)
                self.faiss_index.add(vec.reshape(1, -1))

            # BM25 has no incremental add -> rebuild (fine for moderate corpora)
            self.bm25_ids = list(self.docs.keys())
            self.bm25 = BM25Okapi([tokenize(self.docs[i]["text"]) for i in self.bm25_ids])

        log.info(f"Indexed {len(chunks)} chunks (total: {len(self.docs)})")
        return len(chunks)

    def reset(self) -> None:
        with self.lock:
            self.docs, self.id_order, self.bm25_ids = {}, [], []
            self.bm25, self.faiss_index = None, None

    @property
    def size(self) -> int:
        return len(self.docs)

    # ---------------- retrieval ----------------

    def bm25_search(self, query: str, k: int) -> list[dict]:
        with self.lock:
            if self.bm25 is None:
                return []
            scores = self.bm25.get_scores(tokenize(query))
            top = np.argsort(scores)[::-1][:k]
            return [
                {**self.docs[self.bm25_ids[i]], "bm25_score": float(scores[i])}
                for i in top if scores[i] > 0
            ]

    def dense_search(self, query: str, k: int) -> list[dict]:
        with self.lock:
            if self.faiss_index is None or self.faiss_index.ntotal == 0:
                return []
            vec = np.array(get_embedder().embed_query(query), dtype=np.float32).reshape(1, -1)
            faiss.normalize_L2(vec)
            scores, idxs = self.faiss_index.search(vec, min(k, self.faiss_index.ntotal))
            return [
                {**self.docs[self.id_order[i]], "dense_score": float(s)}
                for s, i in zip(scores[0], idxs[0]) if i >= 0
            ]


STORE = DocumentStore()


# ==============================================================================
# 4. INGESTION  (PDF -> text -> chunks)
# ==============================================================================

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=SETTINGS.CHUNK_SIZE,
    chunk_overlap=SETTINGS.CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def pdf_to_chunks(file_bytes: bytes, filename: str) -> list[dict]:
    """Extract per-page text from a PDF and split into overlapping chunks."""
    reader = PdfReader(io.BytesIO(file_bytes))
    chunks: list[dict] = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        for i, piece in enumerate(_splitter.split_text(text)):
            chunks.append({
                "id": str(uuid.uuid4()),
                "text": piece,
                "metadata": {"source": filename, "page": page_num, "chunk": i},
            })

    return chunks


# ==============================================================================
# 5. LLM HELPERS
# ==============================================================================

def llm_text(llm: ChatGroq, prompt: str) -> str:
    return llm.invoke(prompt).content.strip()


def parse_json(raw: str) -> Any:
    """Tolerant JSON extraction from LLM output (handles code fences/prose)."""
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"(\{.*\}|\[.*\])", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
    return None


# ==============================================================================
# 6. LANGGRAPH STATE
# ==============================================================================

class RAGState(TypedDict, total=False):
    # input
    question: str

    # query transformation
    queries: list[str]

    # retrieval: one ranked list per (query x retriever)
    retrieval_lists: list[list[dict]]

    # fusion -> diversity -> precision
    fused: list[dict]
    mmr_docs: list[dict]
    reranked: list[dict]

    # final context + generation
    context: list[dict]
    answer: str

    # validation / control
    grounded: bool
    confidence: float
    validation_reason: str
    retry_count: int

    # observability (append-only via reducer)
    trace: Annotated[list[dict], operator.add]


def trace_step(step: str, **info) -> dict:
    return {"step": step, "ts": round(time.time(), 3), **info}


# ==============================================================================
# 7. GRAPH NODES
# ==============================================================================

# ---------- 7.1 Query transformation (rewrite + multi-query expansion) --------

def node_query_transform(state: RAGState) -> dict:
    question = state["question"]
    retry = state.get("retry_count", 0)
    feedback = state.get("validation_reason", "")

    if retry == 0:
        prompt = f"""You are a search query expansion expert for a retrieval system.

Given the user question, generate {SETTINGS.NUM_EXPANDED_QUERIES} alternative
search queries that would help retrieve relevant documents. Use different
angles: synonyms, more specific phrasing, and a more general phrasing.

User question: {question}

Return ONLY the alternative queries, one per line, no numbering, no bullets."""

        raw = llm_text(get_llm_fast(), prompt)
        expanded = [q.strip("-• 0123456789.").strip() for q in raw.splitlines() if q.strip()]
        queries = [question] + expanded[: SETTINGS.NUM_EXPANDED_QUERIES]

    else:
        # Self-correction: rewrite using the validator's feedback
        prompt = f"""A previous retrieval attempt for this question failed grounding validation.

User question: {question}
Why it failed: {feedback}

Generate {SETTINGS.NUM_EXPANDED_QUERIES} NEW search queries, rephrased to find
the missing evidence. Return ONLY the queries, one per line."""

        raw = llm_text(get_llm_fast(), prompt)
        expanded = [q.strip("-• 0123456789.").strip() for q in raw.splitlines() if q.strip()]
        queries = [question] + expanded[: SETTINGS.NUM_EXPANDED_QUERIES]

    log.info(f"[query_transform] retry={retry} queries={queries}")
    return {
        "queries": queries,
        "trace": [trace_step("query_transform", retry=retry, queries=queries)],
    }


# ---------- 7.2 Parallel hybrid retrieval -------------------------------------

def node_retrieve(state: RAGState) -> dict:
    queries = state["queries"]
    k = SETTINGS.PER_QUERY_TOP_K

    def work(q: str, mode: str) -> list[dict]:
        if mode == "bm25":
            return STORE.bm25_search(q, k)
        return STORE.dense_search(q, k)

    lists: list[list[dict]] = []
    with ThreadPoolExecutor(max_workers=len(queries) * 2) as pool:
        futures = []
        for q in queries:
            futures.append(pool.submit(work, q, "bm25"))
            futures.append(pool.submit(work, q, "dense"))
        for f in futures:
            result = f.result()
            if result:
                lists.append(result)

    total = sum(len(l) for l in lists)
    log.info(f"[retrieve] {len(lists)} ranked lists, {total} raw hits")
    return {
        "retrieval_lists": lists,
        "trace": [trace_step("retrieve", ranked_lists=len(lists), raw_hits=total)],
    }


# ---------- 7.3 RAG Fusion (Reciprocal Rank Fusion) ---------------------------

def node_fusion(state: RAGState) -> dict:
    rrf_scores: dict[str, float] = {}
    doc_lookup: dict[str, dict] = {}

    for ranked in state["retrieval_lists"]:
        for rank, doc in enumerate(ranked):
            doc_id = doc["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (SETTINGS.RRF_K + rank + 1)
            doc_lookup.setdefault(doc_id, doc)

    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[: SETTINGS.FUSION_TOP_K]
    fused_docs = [{**doc_lookup[did], "rrf_score": score} for did, score in fused]

    log.info(f"[fusion] {len(fused_docs)} fused candidates")
    return {
        "fused": fused_docs,
        "trace": [trace_step("rag_fusion", candidates=len(fused_docs))],
    }


# ---------- 7.4 MMR (Maximal Marginal Relevance — diversity) -------------------

def node_mmr(state: RAGState) -> dict:
    candidates = state["fused"]
    question = state["question"]
    lam, top_k = SETTINGS.MMR_LAMBDA, SETTINGS.MMR_TOP_K

    if len(candidates) <= top_k:
        selected = candidates
    else:
        embedder = get_embedder()
        q_vec = np.array(embedder.embed_query(question), dtype=np.float32)
        d_vecs = np.array(embedder.embed_documents([d["text"] for d in candidates]), dtype=np.float32)
        # embeddings already normalized by encode_kwargs -> dot product = cosine

        relevance = d_vecs @ q_vec
        selected_idx: list[int] = []
        remaining = set(range(len(candidates)))

        # seed with the most relevant doc
        first = int(np.argmax(relevance))
        selected_idx.append(first)
        remaining.discard(first)

        while remaining and len(selected_idx) < top_k:
            best, best_score = None, -np.inf
            sel_vecs = d_vecs[selected_idx]
            for i in remaining:
                redundancy = float(np.max(sel_vecs @ d_vecs[i]))
                score = lam * float(relevance[i]) - (1 - lam) * redundancy
                if score > best_score:
                    best, best_score = i, score
            selected_idx.append(best)
            remaining.discard(best)

        selected = [candidates[i] for i in selected_idx]

    log.info(f"[mmr] selected {len(selected)} diverse docs")
    return {
        "mmr_docs": selected,
        "trace": [trace_step("mmr", selected=len(selected), lambda_=lam)],
    }


# ---------- 7.5 LLM Reranker (cross-encoder style, batched scoring) ------------

RERANK_BATCH = 5

def node_rerank(state: RAGState) -> dict:
    docs = state["mmr_docs"]
    question = state["question"]
    llm = get_llm_smart()

    def score_batch(batch: list[dict]) -> list[float]:
        doc_block = "\n\n".join(
            f"[{i}] {d['text'][:SETTINGS.MAX_DOC_CHARS]}" for i, d in enumerate(batch)
        )
        prompt = f"""You are a relevance scoring function. Score how relevant each
passage is for answering the question, from 0 (irrelevant) to 10 (directly
answers it). Judge ONLY the information content, not position.

Question: {question}

Passages:
{doc_block}

Return ONLY a JSON array of {len(batch)} numbers, in order. Example: [3, 9, 0]"""

        parsed = parse_json(llm_text(llm, prompt))
        if isinstance(parsed, list) and len(parsed) == len(batch):
            try:
                return [float(x) for x in parsed]
            except (TypeError, ValueError):
                pass
        log.warning("[rerank] unparseable scores for a batch; defaulting to 0")
        return [0.0] * len(batch)

    batches = [docs[i: i + RERANK_BATCH] for i in range(0, len(docs), RERANK_BATCH)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        batch_scores = list(pool.map(score_batch, batches))

    flat = [s for scores in batch_scores for s in scores]
    scored = [{**d, "rerank_score": s} for d, s in zip(docs, flat)]
    scored.sort(key=lambda d: d["rerank_score"], reverse=True)
    top = scored[: SETTINGS.FINAL_TOP_K]

    log.info(f"[rerank] top-{len(top)} scores={[d['rerank_score'] for d in top]}")
    return {
        "reranked": top,
        "trace": [trace_step("rerank", kept=len(top), top_scores=[d["rerank_score"] for d in top])],
    }


# ---------- 7.6 Context assembly (dedup + truncate) ----------------------------

def node_build_context(state: RAGState) -> dict:
    seen: set[str] = set()
    context: list[dict] = []

    for doc in state["reranked"]:
        key = doc["text"][:120]  # cheap near-dup guard
        if key in seen:
            continue
        seen.add(key)
        context.append({**doc, "text": doc["text"][:SETTINGS.MAX_DOC_CHARS]})

    return {
        "context": context,
        "trace": [trace_step("build_context", docs=len(context))],
    }


# ---------- 7.7 Generation -----------------------------------------------------

def node_generate(state: RAGState) -> dict:
    question = state["question"]
    context = state["context"]

    ctx_block = "\n\n".join(
        f"[{i+1}] (source: {d['metadata'].get('source')}, page {d['metadata'].get('page')})\n{d['text']}"
        for i, d in enumerate(context)
    )

    prompt = f"""You are a precise assistant. Answer the question using ONLY the
context below. Cite sources inline as [1], [2], etc. If the context is
insufficient, say exactly what is missing — do not invent facts.

Context:
{ctx_block}

Question: {question}

Answer:"""

    answer = llm_text(get_llm_smart(), prompt)
    log.info(f"[generate] answer length={len(answer)}")
    return {
        "answer": answer,
        "trace": [trace_step("generate", answer_chars=len(answer))],
    }


# ---------- 7.8 Validation (grounding judge) -----------------------------------

def node_validate(state: RAGState) -> dict:
    prompt = f"""You are a strict fact-checking judge. Decide whether the ANSWER
is fully supported by the CONTEXT (no unsupported claims) and adequately
addresses the QUESTION.

QUESTION: {state['question']}

ANSWER: {state['answer']}

CONTEXT:
{chr(10).join(f"[{i+1}] {d['text'][:800]}" for i, d in enumerate(state['context']))}

Return ONLY JSON: {{"grounded": true/false, "confidence": 0.0-1.0, "reason": "short explanation"}}"""

    parsed = parse_json(llm_text(get_llm_fast(), prompt)) or {}
    grounded = bool(parsed.get("grounded", True))           # fail-open on parse errors
    confidence = float(parsed.get("confidence", 0.5))
    reason = str(parsed.get("reason", ""))
    retry_count = state.get("retry_count", 0) + (0 if grounded else 1)

    log.info(f"[validate] grounded={grounded} confidence={confidence} retries={retry_count}")
    return {
        "grounded": grounded,
        "confidence": confidence,
        "validation_reason": reason,
        "retry_count": retry_count,
        "trace": [trace_step("validate", grounded=grounded, confidence=confidence, reason=reason)],
    }


# ---------- 7.9 Conditional edge (the LangGraph superpower: cycles) ------------

def route_after_validation(state: RAGState) -> str:
    if state["grounded"] or state.get("retry_count", 0) >= SETTINGS.MAX_RETRIES:
        return END
    log.info("[router] insufficient grounding -> looping back to query_transform")
    return "query_transform"


# ==============================================================================
# 8. GRAPH ASSEMBLY
# ==============================================================================

def build_graph():
    g = StateGraph(RAGState)

    g.add_node("query_transform", node_query_transform)
    g.add_node("retrieve", node_retrieve)
    g.add_node("rag_fusion", node_fusion)
    g.add_node("mmr", node_mmr)
    g.add_node("rerank", node_rerank)
    g.add_node("build_context", node_build_context)
    g.add_node("generate", node_generate)
    g.add_node("validate", node_validate)

    g.add_edge(START, "query_transform")
    g.add_edge("query_transform", "retrieve")
    g.add_edge("retrieve", "rag_fusion")
    g.add_edge("rag_fusion", "mmr")
    g.add_edge("mmr", "rerank")
    g.add_edge("rerank", "build_context")
    g.add_edge("build_context", "generate")
    g.add_edge("generate", "validate")

    # self-correcting cycle
    g.add_conditional_edges("validate", route_after_validation, {END: END, "query_transform": "query_transform"})

    return g.compile()


GRAPH = build_graph()


# ==============================================================================
# 9. FASTAPI BACKEND
# ==============================================================================

api = FastAPI(title="Advanced RAG API", version="1.0.0")
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ---- serve the front-end (vanilla HTML/CSS/JS in ./static) ----
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

_STATIC_DIR = Path(__file__).resolve().parent / "static"


@api.get("/", include_in_schema=False)
def _index():
    """Single-page UI."""
    return FileResponse(_STATIC_DIR / "index.html")


api.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# ingestion job tracking
JOBS: dict[str, dict] = {}


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    grounded: bool
    confidence: float
    retries: int
    sources: list[dict]
    trace: list[dict]


# ---------------- health / info ----------------

@api.get("/health")
def health():
    return {"status": "ok", "indexed_chunks": STORE.size, "llm": SETTINGS.LLM_SMART}


@api.get("/documents")
def documents():
    sources: dict[str, dict] = {}
    for d in STORE.docs.values():
        s = d["metadata"]["source"]
        sources.setdefault(s, {"source": s, "chunks": 0, "pages": set()})
        sources[s]["chunks"] += 1
        sources[s]["pages"].add(d["metadata"]["page"])
    return [
        {"source": v["source"], "chunks": v["chunks"], "pages": len(v["pages"])}
        for v in sources.values()
    ]


# ---------------- ingestion ----------------

def _ingest_job(job_id: str, file_bytes: bytes, filename: str) -> None:
    try:
        JOBS[job_id] = {"status": "processing", "filename": filename}
        chunks = pdf_to_chunks(file_bytes, filename)
        if not chunks:
            JOBS[job_id] = {"status": "failed", "error": "no extractable text"}
            return
        n = STORE.add_documents(chunks)
        JOBS[job_id] = {"status": "done", "filename": filename, "chunks": n}
    except Exception as e:  # noqa: BLE001
        log.exception("ingestion failed")
        JOBS[job_id] = {"status": "failed", "error": str(e)}


@api.post("/upload")
async def upload(background: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")
    data = await file.read()
    job_id = str(uuid.uuid4())
    background.add_task(_ingest_job, job_id, data, file.filename)
    return {"job_id": job_id, "status": "queued", "filename": file.filename}


@api.get("/ingest-status/{job_id}")
def ingest_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "unknown job_id")
    return JOBS[job_id]


@api.post("/reset")
def reset():
    STORE.reset()
    JOBS.clear()
    return {"status": "cleared"}


# ---------------- query ----------------

@api.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if STORE.size == 0:
        raise HTTPException(400, "No documents indexed. Upload a PDF first.")

    result = GRAPH.invoke({
        "question": req.question,
        "retry_count": 0,
        "trace": [trace_step("start", question=req.question)],
    })

    sources = [
        {
            "rank": i + 1,
            "source": d["metadata"].get("source"),
            "page": d["metadata"].get("page"),
            "rerank_score": d.get("rerank_score"),
            "snippet": d["text"][:300],
        }
        for i, d in enumerate(result.get("context", []))
    ]

    return QueryResponse(
        question=req.question,
        answer=result["answer"],
        grounded=result.get("grounded", False),
        confidence=result.get("confidence", 0.0),
        retries=result.get("retry_count", 0),
        sources=sources,
        trace=result.get("trace", []),
    )


# ==============================================================================
# 10. ENTRYPOINT
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    print(f"\n  Advanced RAG  ->  http://localhost:{port}\n")
    uvicorn.run(api, host="0.0.0.0", port=port)