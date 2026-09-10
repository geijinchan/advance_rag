# Advanced RAG — Hybrid Retrieval (BM25 + Semantic)

A self-contained UI + server for an advanced LangGraph RAG pipeline.

**Pipeline:** query rewrite + multi-query → parallel hybrid retrieval (BM25 ∥ dense/FAISS) →
RAG-Fusion (RRF) → MMR diversity → LLM rerank → grounded generation → grounding validation
with a self-correcting loop. LLMs run on **Groq**; embeddings are local (`all-MiniLM-L6-v2`).

## Run

```bash
# 1. install deps
pip install -r requirements.txt

# 2. set your key in .env  (GROQ_API_KEY=gsk_...)  — get one at https://console.groq.com/keys

# 3. launch
python app.py                 # -> http://localhost:8000
#   or, with auto-reload:  uvicorn app:api --reload
```

Then open **http://localhost:8000**, drop in a PDF, wait for indexing, and ask.

## Flow (what the UI shows)

1. **Documents** — upload a PDF → it's chunked, embedded, and indexed into both BM25 and FAISS.
2. **Ask** — type a question (⌘/Ctrl+Enter to send).
3. **Pipeline flow** — each of the 8 LangGraph stages lights up with live stats and timing;
   a `↻ passes` badge appears when the self-correcting loop re-ran retrieval.
4. **Answer** — grounded answer with `[n]` citations, a grounded/confidence/retries badge row,
   and the ranked source passages. Expand **Raw trace** for the full step log.

## Files

| file | role |
|------|------|
| `app.py` | the whole backend — pipeline + FastAPI routes; loads `.env`, serves the UI, runs uvicorn |
| `.env` | `GROQ_API_KEY` + optional model overrides |
| `requirements.txt` | Python dependencies |
| `static/index.html · style.css · app.js` | the vanilla HTML/CSS/JS front-end |

## Endpoints

`GET /health` · `POST /upload` · `GET /ingest-status/{id}` · `GET /documents` · `POST /query` · `POST /reset`

## Notes

- **Models** (set in `.env`): `LLM_FAST` handles query rewriting + validation; `LLM_SMART` handles
  reranking + generation. Each question fires several `LLM_SMART` calls (rerank batches + generate,
  ×retries), so if you use a low-rate-limit model there you may hit Groq 429s — switch to a smaller
  model in `.env`.
- The first query downloads the embedding model (~80 MB) once.
- State is **in-memory** — restarting the server clears all indexed documents.
