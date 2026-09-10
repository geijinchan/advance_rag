# Advance RAG — Document Intelligence Assistant

A **RAG-Powered Agentic Q&A** platform that answers questions grounded in your document corpus, with inline citations, an agentic verification graph, and honest fallback when the corpus doesn't contain the answer.

Built with **Next.js** (frontend) + **FastAPI** (backend) + **vLLM** (optional GPU-accelerated inference), with a **Groq** fallback and a GPU-free **simulation** mode.

---

## Features

- **Hybrid Retrieval** — BM25 + dense embeddings with Reciprocal Rank Fusion (RRF) and MMR diversity
- **Agentic Workflow** — Router → Retriever → Grader → Rewriter → Generator → Verifier pipeline
- **Citation-Grounded Answers** — Every claim links back to `[Source: doc, page N]`
- **Honest Fallback** — Transparently says "I couldn't find this in the provided documents"
- **Multimodal** — Image Q&A via VLM or OCR fallback
- **SSE Streaming** — Real-time streamed responses via Server-Sent Events
- **Simulation Mode** — Full-featured demo without GPU (deterministic extractive engine)
- **Cross-Encoder Reranking** — Optional neural reranking for higher precision
- **Conversation Memory** — Multi-turn follow-up questions with session tracking

---

## Architecture

```
┌─────────────────────────┐     ┌──────────────────────────────┐
│   Next.js Frontend      │────▶│   FastAPI Backend (:3003)     │
│   (:3000)               │     │   /ask  /ask/stream  /health  │
└─────────────────────────┘     └──────────────┬───────────────┘
                                               │ Agent Graph
                                               ▼
  ┌────────┐  route  ┌────────────┐  grade  ┌─────────┐  rewrite  ┌──────────┐
  │ ROUTER │───────▶ │ RETRIEVER  │────────▶│ GRADER  │──────────▶│ REWRITER │
  └───┬────┘         │ BM25+Dense │         └────┬────┘           └────┬─────┘
      │ chitchat     │ RRF + MMR  │         relevant│◀──────────────────┘
      ▼              └────────────┘         ┌──────▼──────┐
 ┌──────────┐                               │  GENERATOR  │──▶ verified ──▶ END
 │ CHITCHAT │                               └─────────────┘
 └──────────┘
```

---

## Run with Docker (recommended)

Requires **Docker Desktop** (with Docker Compose). This builds and runs both the
frontend and the backend together.

### 1. Configure environment

```bash
cp .env.example .env
```

Then edit `.env` and set your `GROQ_API_KEY`. Without a reachable vLLM server and
without a Groq key, the backend still boots in **simulation mode** (deterministic,
no external LLM calls).

### 2. Build and start

```bash
docker compose up -d --build
```

That's it. The stack comes up on:

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:3003
- **Health check**: http://localhost:3003/health

On the **first** boot the backend ingests the documents in `backend/corpus/` and
downloads the embedding model (`BAAI/bge-base-en-v1.5`, ~400 MB). This can take a
couple of minutes — the frontend waits for the backend to report healthy before
it starts. The model is cached in a Docker volume, so subsequent restarts are fast.

### Useful commands

```bash
docker compose ps                 # service status
docker compose logs -f backend    # follow backend logs
docker compose logs -f frontend   # follow frontend logs
docker compose restart backend    # restart a single service
docker compose down               # stop and remove the containers
docker compose down -v            # also remove volumes (index + model cache)
```

---

## Run Locally (development)

Use this for iterating on the code without Docker.

### Prerequisites

- **Node.js** ≥ 18.x and **npm** ≥ 9.x
- **Python** ≥ 3.11

### 1. Install dependencies

```bash
# Frontend
npm install

# Backend
cd backend
python -m venv .venv

# Activate the venv:
#   Windows:      .venv\Scripts\activate
#   macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
cd ..
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 3. Ingest the corpus

```bash
npm run backend:ingest
# Or directly:  cd backend && python scripts/ingest.py
```

### 4. Run the two services

```bash
# Terminal 1 — Frontend
npm run dev

# Terminal 2 — Backend
npm run backend:dev
# Or directly:  cd backend && python -m uvicorn app.main:app --host 0.0.0.0 --port 3003 --reload
```

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:3003

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness + serving mode + corpus stats |
| `POST` | `/ask` | Agentic Q&A (full trace returned) |
| `POST` | `/ask/stream` | SSE-streamed agentic Q&A |
| `POST` | `/ask-image` | Visual Q&A via VLM / OCR fallback |
| `POST` | `/ingest` | Upload & index documents (multipart) |
| `POST` | `/ingest/sample` | Re-ingest the bundled demo corpus |
| `GET` | `/corpus` | Indexed document inventory |
| `POST` | `/reset` | Wipe the index |
| `POST` | `/benchmark` | Throughput/latency benchmark |
| `GET` | `/suggestions` | Demo questions for the UI |

---

## Example Request

```bash
curl -s -X POST http://localhost:3003/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is the maximum warranty period for the X200?"}'
```

---

## Project Structure

```
advance-rag/
├── src/                        # Next.js frontend
│   ├── app/                    # App router pages
│   ├── components/             # React components (shadcn/ui)
│   ├── hooks/                  # Custom React hooks
│   └── lib/                    # Utilities
├── backend/                    # Python FastAPI backend
│   ├── app/
│   │   ├── agents/             # Agentic workflow (graph, nodes)
│   │   ├── llm/                # LLM backends (vLLM, Groq, simulation)
│   │   ├── multimodal/         # Image Q&A
│   │   ├── parsing/            # PDF/document parsing
│   │   ├── rag/                # RAG pipeline (retriever, embeddings, etc.)
│   │   ├── eval/               # Evaluation framework
│   │   ├── main.py             # FastAPI application
│   │   ├── config.py           # Settings (env vars)
│   │   └── schemas.py          # Pydantic models
│   ├── corpus/                 # Demo document corpus (ingested on boot)
│   ├── scripts/                # CLI tools (ingest, ask, benchmark)
│   └── Dockerfile              # Backend image (built by the root compose)
├── Dockerfile                  # Frontend image
├── docker-compose.yml          # Runs frontend + backend
├── package.json
├── next.config.ts
├── tailwind.config.ts
├── tsconfig.json
└── .env.example
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | Groq API key (fallback when vLLM is unreachable) |
| `GROQ_MODEL_NAME` | `openai/gpt-oss-20b` | Groq text model |
| `GROQ_VISION_MODEL_NAME` | `llama-3.2-11b-vision-preview` | Groq vision model |
| `VLLM_BASE_URL` | `http://localhost:8001/v1` | vLLM server URL |
| `VLLM_MODEL_NAME` | `Qwen/Qwen2.5-7B-Instruct` | Text model name |
| `VLLM_VISION_MODEL_NAME` | `Qwen/Qwen2.5-VL-7B-Instruct` | Vision model name |
| `API_PORT` | `3003` | Backend API port |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---

## License

MIT
