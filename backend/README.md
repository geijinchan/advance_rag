# RAG-Powered Agentic Q&A Service — served via vLLM

**Document Intelligence Assistant** for a fictional internal-corpus
(IndustriOS Dynamics Ltd. — 16 documents: controller manuals, sensor
datasheets, warranty policy, firmware notes, …). Answers plain-English
questions **grounded in those documents only**, with inline citations
(`[Source: doc, page N]`), an **agentic verification graph** between the
question and the answer, and an **honest fallback** when the corpus does
not contain the answer.

Role: AI/ML Engineer take-home · Python 3.12 · FastAPI · self-hosted vLLM.

---

## 1. Architecture

```
                                     ┌──────────────────────────────┐
        POST /ask ──────────────────▶│  FastAPI app (app/main.py)   │
        POST /ask/stream (SSE)       │  /health /corpus /ingest     │
        POST /ask-image              │  /benchmark /reset           │
                                     └──────────────┬───────────────┘
                                                    │ AgentGraph (own mini-LangGraph)
                                                    ▼
   ┌────────┐   route   ┌────────────┐  hits  ┌─────────┐ not rel. ┌──────────┐
   │ ROUTER │──doc q──▶ │ RETRIEVER  │───────▶│ GRADER  │──retry──▶│ REWRITER│
   └───┬────┘           │ BM25+dense │        │ threshold│  (<2)    └────┬─────┘
       │ chitchat       │ RRF + MMR  │        └────┬────┘◀──────────────┘
       ▼                └────────────┘     relevant│
  ┌──────────┐                              ┌──────▼──────┐  supported ┌──────┐
  │ CHITCHAT │                              │  GENERATOR  │──────────▶│ END  │
  └──────────┘                              └──────┬──────┘            └──────┘
       │ out-of-scope / failed grade /              │ not supported
       ▼                failed verification         ▼
  ┌──────────────────────────────────────────────────────────┐
  │ FALLBACK: "I couldn't find this in the provided documents"│
  └──────────────────────────────────────────────────────────┘

  Model layer:  VLLMBackend (OpenAI-compatible /chat/completions, streaming,
                vision payloads)  ⇄  SimulationBackend (deterministic
                extractive engine, used when no vLLM server is reachable —
                e.g. this CPU-only sandbox; every response labels its mode).
  Embeddings:   sentence-transformers (bge-base-en-v1.5) when installed,
                else feature-hashing n-gram vectors (384-d, L2-normalised).
  Vector store: in-process numpy store (cosine + MMR) persisted to JSON;
                Qdrant profile provided in docker-compose.
```

**Part coverage**

| Assignment part | Where |
|---|---|
| A — vLLM serving | `app/llm/vllm_backend.py` (OpenAI-compat client, streaming, retries), `docker-compose.yml` (GPU service), `/benchmark` + `scripts/benchmark.py` |
| B — RAG pipeline | `app/parsing/pdf_parser.py`, `app/rag/{chunker,embeddings,bm25,retriever,vector_store,ingestion}.py` |
| C — Agentic workflow | `app/agents/graph.py` (state-machine engine), `app/agents/nodes/*` (router/grader/rewriter/generator/verifier/fallback) |
| D — API & packaging | `app/main.py`, `app/schemas.py`, `Dockerfile`, `docker-compose.yml`, this README |
| E — Multimodal | `app/multimodal/image_qa.py` (`POST /ask-image`, VLM via vLLM; OCR fallback offline) **and** image-aware ingestion (figure extraction + captioned pseudo-chunks) |
| F — Bonus | SSE streaming endpoint `/ask/stream`, benchmark harness, this minimal chat UI |

## 2. Quick start

### 2.1 Sandbox / CPU-only demo (simulation mode)

```bash
cd mini-services/rag-agent
pip install -r requirements.txt          # or: use the project venv
python scripts/ingest.py                 # parse → chunk → embed → index
python scripts/ask.py --all              # smoke test all agent paths
python -m uvicorn app.main:app --port 3003
curl -s localhost:3003/health | jq
```

Without a vLLM server the service runs in **simulation mode** (labelled in
`/health` and in every response): a deterministic extractive engine powers
every decision point, so the full graph, citations, fallbacks, streaming and
benchmark plumbing are demonstrable offline. Answers are assembled strictly
from retrieved text — the simulation never fabricates.

### 2.2 Full stack with vLLM (requires NVIDIA GPU)

```bash
cp .env.example .env
docker compose up -d vllm        # Qwen2.5-VL-7B: serves text AND vision
docker compose up --build api    # FastAPI, waits for vLLM health
curl -s "localhost:3003/health?refresh=true" | jq     # mode flips to "vllm"
```

Bare-metal vLLM alternative:

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --max-model-len 8192 \
  --gpu-memory-utilization 0.85 --dtype auto --port 8001
VLLM_BASE_URL=http://localhost:8001/v1 python -m uvicorn app.main:app --port 3003
```

### 2.3 API sketch

```bash
# Agentic Q&A — full trace included (session_id enables follow-ups)
curl -s -X POST localhost:3003/ask -H 'content-type: application/json' \
  -d '{"question":"What is the maximum warranty period for the X200?", "session_id":"s1"}' | jq

# Follow-up: short/elliptical questions inherit entities from the last turn
curl -s -X POST localhost:3003/ask -H 'content-type: application/json' \
  -d '{"question":"and the extended warranty?", "session_id":"s1"}' | jq

# Comparative / multi-hop question — decomposed per entity, grounded answer
# blocks per side (sub_queries + decomposed fields in the response)
curl -s -X POST localhost:3003/ask -H 'content-type: application/json' \
  -d '{"question":"Compare the standard warranty of the X200 and the X300."}' | jq '.sub_queries, .answer'

# Repeat ask → semantic answer cache hit (cache_hit + cache_detail provenance;
# agent_path becomes ["cache"] and the SSE stream emits a cache node event)
curl -s -X POST localhost:3003/ask -H 'content-type: application/json' \
  -d '{"question":"Compare the standard warranty of the X200 and the X300."}' | jq '.cache_hit, .cache_detail'

# Live agent trace + progressive answer reveal (SSE)
curl -N -X POST localhost:3003/ask/stream -H 'content-type: application/json' \
  -d '{"question":"Which firmware fixed the MODBUS-TCP bug?"}'

# Interactive hybrid retrieval (BM25 + dense + RRF fusion + cross-encoder
# rerank; per-result dense/BM25 ranks AND rerank_score/fusion_rank)
curl -s 'localhost:3003/search?q=LED%20blinking%20warning&k=3' | jq

# Retrieval playground: reranker weight overrides + cross-encoder on/off,
# same ranking code the agent graph uses — response echoes the weights
# actually applied and a rerank telemetry block (mode/reorders/latency)
curl -s 'localhost:3003/search?q=warranty&k=5&wf=0.9&wc=0.3&rr=false' | jq '.weights, .rerank, .results[0]'

# Full source chunk behind a citation (+ adjacent chunks)
curl -s 'localhost:3003/source/troubleshooting_flowchart.md::p1::6B12' | jq '.doc, .section'

# Visual Q&A (Part E)
curl -s -X POST localhost:3003/ask-image -F file=@chart.png \
  -F 'question=What does this chart show?'

# Index new documents
curl -s -X POST localhost:3003/ingest -F files=@your_manual.pdf

# Answer feedback telemetry
curl -s -X POST localhost:3003/feedback -H 'content-type: application/json' \
  -d '{"request_id":"abc123", "rating":"up"}' | jq

# Golden-set evaluation (16 base cases + any user-grown cases: route,
# retrieval hit@k, fact support, faithfulness per case + aggregate)
curl -s -X POST localhost:3003/eval | jq '.accuracy, .faithfulness_avg, .faithfulness_judge'
curl -s localhost:3003/sessions | jq '.stats'

# Feedback-driven golden-set growth: upvoted answers joined against the
# answer ledger → candidates with auto-mined fact patterns + source docs
curl -s localhost:3003/eval/candidates | jq '.counts, .candidates[0]'

# Promote an upvoted answer into the golden set (auto-verifies the new
# case through the full graph); list the grown set; delete a user case
curl -s -X POST localhost:3003/eval/candidates/promote -H 'content-type: application/json' \
  -d '{"request_id":"<ledger-id>"}' | jq '.case, .verification.passed'
curl -s localhost:3003/eval/cases | jq '.stats, .cases[-1]'
curl -s -X DELETE localhost:3003/eval/cases/usr-00000000 | jq

# Batch promote: several upvoted answers in one call (1–12 ids, each
# auto-verified through the full graph; per-item failures land in .results
# and never abort the batch; optional per-request keyword edits via
# keywords_map)
curl -s -X POST localhost:3003/eval/candidates/promote-batch -H 'content-type: application/json' \
  -d '{"request_ids":["<id-1>","<id-2>"], "note":"round-19 growth"}' \
  | jq '.promoted, .failed, .results[], .golden_set'

# Answer-quality A/B: one question through the full agent graph under two
# named retrieval configs — both answers + citations + faithfulness +
# verdict (agreement, citation overlap, latency/faithfulness deltas)
curl -s -X POST localhost:3003/eval/compare -H 'content-type: application/json' \
  -d '{"question":"What is the standard warranty on the X200?"}' | jq '.verdict'

# Persisted trend of eval runs (accuracy/faithfulness/latency over time;
# ablation-tagged runs optionally included)
curl -s localhost:3003/eval/history | jq '.total_runs, .last'

# Downvote triage — the quality radar: 👎 answers joined against the answer
# ledger (question, preview, fallback flag, reason comment). items = the OPEN
# fix-list; resolved_items = fixed ones (newest resolved first); counts.open /
# counts.resolved are accurate across all downvoted records
curl -s localhost:3003/quality/triage | jq '.counts, .items[0], .resolved_items[0]'

# Resolve a triage item (the fix-list's "done" action, optional ≤300-char
# note) and reopen one — the 👎 lifecycle, persisted across restarts
curl -s -X POST localhost:3003/quality/triage/resolve -H 'content-type: application/json' \
  -d '{"request_id":"<rid>", "note":"fixed by promoting a golden case"}' | jq
curl -s -X POST localhost:3003/quality/triage/reopen -H 'content-type: application/json' \
  -d '{"request_id":"<rid>"}' | jq

# Retrieval ablation study: golden set under 7 retriever configs
# (top-k depth × reranker weight profiles × cross-encoder on/off/n) with
# deltas vs baseline
curl -s -X POST localhost:3003/eval/ablation | jq '.winner, .configs[] | {name, accuracy, delta_accuracy}'

# Grafana-importable dashboard JSON provisioning the rag_* metric families
# (traffic, latency quantiles, cache, honesty signals, eval quality)
curl -s localhost:3003/ops/grafana | jq '.title, .panels[].title'

# Service-wide metrics (usage by route, latency p50/p95, follow-ups,
# feedback approval, uptime, triage open/resolved + SLA ages, last eval)
curl -s localhost:3003/stats | jq '.questions, .triage'

# Question analytics — frequency/route/citation/latency/votes per distinct
# question mined from the answer ledger (usage lens for corpus growth)
curl -s "localhost:3003/analytics/questions?limit=8" | jq '.totals, .top[0]'

# Prometheus exposition (text/plain, version=0.0.4) — scrape-ready:
# ask counters by route, latency histogram, cache/fallback/rewrite totals,
# per-route-pattern http_requests_total, corpus/session/eval/triage gauges
# (incl. the rag_triage_oldest_open_hours fix-list SLA gauge)
curl -s localhost:3003/metrics | grep rag_triage
```

Interactive docs: `http://localhost:3003/docs`.

## 3. Design decisions & trade-offs

**Serving (Part A).** One GPU budget → serve a single Qwen2.5-VL-7B for both
text and vision (the assignment's Step-5 suggestion) rather than two models.
Parameters: `--max-model-len 8192` (chunk context + answer headroom),
`--gpu-memory-utilization 0.85` (KV-cache space vs. co-located services),
AWQ-quantised variant when memory is tight. The client (`vllm_backend.py`)
speaks the OpenAI-compatible API with exponential-backoff retries so a
restarting vLLM container degrades gracefully to simulation mode instead of
hard-failing requests — and `/health?refresh=true` hot-swaps the backend back
when vLLM returns.

**Retrieval (Part B).** Hybrid BM25 (own implementation, k1=1.5/b=0.75) +
dense vectors fused with Reciprocal Rank Fusion, then a coverage/agreement
heuristic with a quantitative-evidence boost for measurable questions
("24 months", "±0.3 °C") so numeric answers outrank topical prose. Dense
search uses MMR (λ=0.7) to avoid five near-duplicate chunks from one page.
Coverage matching is **stemmed and synonym-expanded** (weight 0.8 for synonym
hits): "RMA processing time" matches chunks that say "evaluation takes
48 hours". Figure-caption pseudo-chunks (OCR'd chart text) are demoted ×0.72
unless the query is explicitly about a visual — OCR axis-label garble is
supporting evidence, never quotable prose.

**Cross-encoder rerank stage (new).** After fusion, the top candidates are
re-scored by **jointly reading the (query, chunk) pair** — the standard
precision stage first-stage retrieval feeds (bge-reranker pattern).
`app/rag/reranker.py` implements the pattern in both modes: in simulation a
deterministic **lexical cross-encoder surrogate** computes pair-interaction
features that only exist when query and chunk are read together — verbatim
phrase containment, sentence-level answer-likeness gated on the query's
rarest (max-IDF) *anchor* term with IDF-weighted coverage and synonym
rescue, matched-term proximity density, IDF-weighted coverage, bigram
Jaccard, entity-code joint match, and numeric-evidence for measurable
questions; in vLLM mode the served model scores each pair 0–100 via a
strict prompt (degrading per-pair to the surrogate on any error, so the
graph never fails on the reranker). The stage is toggleable
(`RERANK_ENABLED`, `/search?rr=false`, ablation rows) and fully observable:
per-result `rerank_score` + `fusion_rank` (order changes), retriever-node
trace telemetry (reranked/reorders/latency/top-scores), `/stats` +
`rag_rerank_*` Prometheus families. Golden set: 16/16 with the stage on;
the ablation table shows it costs ~8 ms p50 in simulation mode while
keeping 100% accuracy (the LLM-scored path is where it pays off).

**Chunking (Part B).** Sentence-aware, structure-respecting chunker
(~380-token target, 1-sentence overlap, headings start new chunks) instead of
fixed 512-token windows: citations then point at semantically complete
passages, and pages stay truthful because chunking is per-page. Newlines are
preserved so markdown tables stay row-aligned — and **table rows are atomic
units that are never split mid-table** (flush is deferred while a table run
continues, safety valve at 2× target): a semicolon inside a table cell
("Wait 8 minutes; never cut power") must not break the row structure, or
downstream header pairing is lost and the wrong table row gets quoted. Table
rows are turned into header-annotated pseudo-sentences by the extractive
generator, making table-resident facts (warranty matrices, LED patterns)
answerable.

**Agentic graph (Part C).** A hand-rolled mini-LangGraph
(`app/agents/graph.py`: nodes, conditional edges, entry point, bounded steps,
per-node timing, exception-safe routing) — deliberately, to demonstrate the
workflow rather than import it. Retry budget: up to 2 query rewrites before
the honest fallback. Verification failure → fallback (refusing beats a second
guess for a grounded assistant). Every step is timed and returned in
`trace[]` — debuggability was an explicit Part D requirement.

**Query decomposition (new, multi-hop).** Comparative questions ("Compare the
warranty of the X200 and the X300") are multi-hop for lexical retrieval: no
single chunk mentions both entities and a fused ranking of the raw query lets
whichever document scores highest flood the context. The router detects a
comparative cue + ≥2 entity codes and dispatches to a **decomposer** node
(`app/agents/nodes/decomposer.py`) that splits the question into per-entity
sub-queries ("standard warranty X200", "standard warranty X300"). Retrieval
then runs per sub-query over a 3× wider pool with the entity's own code-named
documents hard-prioritised, merges round-robin (entity-balanced context), the
grader judges each chunk against the sub-query it was fetched for, and the
generator emits entity-labelled answer blocks with per-entity citations.
Two multi-hop cases in the golden set pin the behaviour — both entities'
facts must appear, both entities' documents must be cited.
In **vLLM mode** the decomposer additionally asks the served LLM to split
the question (strict-JSON, hard-validated: 2–3 non-trivial, de-duplicated
sub-queries) with the regex detector as both the pre-gate and the fallback —
the LLM can only *add* decompositions (paraphrases and concept pairs the
regexes don't know), never break the guaranteed ones. The trace records
`method: "regex" | "llm"`.

**Conversation memory (new).** `app/conversation.py` keeps in-memory session
threads (capped at 20 turns). Elliptical follow-ups — "and the extended
warranty?" — inherit entity codes (x200, e103, v3.2.1) from the last grounded
turn; the resolved effective query is returned in the response and emitted as
a `session` node event in the SSE stream, so the UI can show exactly what was
understood. Sessions are inspectable (`GET /sessions`, `GET /sessions/{id}`)
and deletable. The mechanism is deliberately heuristic and observable rather
than a hidden embedding-based memory — appropriate for an explainability-
first demo, swap for a summary-buffer memory in production.

**Evaluation harness (new, Part F bonus).** `app/eval/` holds a 16-case golden
set (factoid / lookup / table / behaviour / robustness / **multihop**
categories — the multihop cases pin comparative-question decomposition) and
an asynchronous runner that executes each case through the **full agent
pipeline** and scores three dimensions: route correctness, retrieval hit@k
against pinned expected documents, and answer fact-support (required pattern
present). `POST /eval` returns the report (accuracy, hit-rate, p50/p95
latency, per-case results); `scripts/eval.py` runs it from CI and exits
non-zero on failure — this is the regression gate that caught (and now
guards) the table-splitting and OCR-garble bugs below.

**Retrieval ablation study (new).** `POST /eval/ablation` re-runs the golden
set under seven retriever configurations (top-k 3/4/8 × coverage-heavy /
fusion-heavy reranker weight profiles × cross-encoder off/on/broad-n)
through the SAME agent graph —
`EvalRunner` is parametrised (top_k, judge on/off) and `HybridRetriever`
gains apply/restore weight overrides, so each run exercises the exact
ranking code the agent uses, re-parameterised. The report (per-config
accuracy / hit@k / fact-support / p50 / p95 / Δ-accuracy vs baseline +
winner) is the ML-engineering experimentation loop made reproducible; the
UI's Eval tab renders the comparison table. Live result on this corpus:
broad top-k=8 **hurts** (−6.3 pts — more same-doc chunks crowd the answer),
narrow top-k=3 matches the baseline at lower latency.

**Feedback-driven golden-set growth (new).** `app/eval/growth.py` closes
the loop between real usage and the regression harness. Every production
ask is recorded in an **answer ledger** (`data/answer_ledger.json`, capped,
persisted) keyed by request_id; upvoted ratings are joined against it to
produce review candidates — each with auto-mined fact patterns (numbers
with units like `36\s*months`, entity codes like `E103`, version strings)
and expected docs derived from the answer's citations. `GET
/eval/candidates` lists them; `POST /eval/candidates/promote` turns one
into a golden case (`usr-` prefix, persisted in `data/golden_cases.json`)
and immediately dry-runs it through the full agent graph — the
verification result is returned so a bad promotion is visible on arrival.
`POST /eval/cases` authors cases by hand; `DELETE /eval/cases/{id}` removes
user cases (base cases are immutable). `/eval`, `/eval/ablation` and
`/eval/compare` all run the **merged** set (16 base + N user), and
`rag_golden_set_cases{source}` in `/metrics` tracks the split.

**Answer-quality A/B (new).** `POST /eval/compare` runs ONE question
through the full agent graph under two named retrieval configurations and
returns both sides (answer, citations, route, fallback, decomposed,
faithfulness, latency, rerank telemetry) plus a verdict block — answer
agreement, citation Jaccard overlap, per-side-only docs, latency and
faithfulness deltas. Same save/restore discipline as the ablation harness;
sessions and the answer cache are untouched (cache-free eval ask path).
The UI renders the two answers side by side with word-level diff
highlighting so answer-quality differences between retrieval configs are
readable at a glance.

**Eval run history (new).** `app/eval/history.py` persists a capped ring of
run summaries (data/eval_history.json): every `POST /eval` and every
ablation config appends label/passed/total/accuracy/faithfulness/hit-rate/
p50/p95. `GET /eval/history` returns the newest-first trend (ablation runs
flagged and filtered by default) + summary stats, so the Eval tab charts
quality over time and shows how golden-set growth changed the denominator.
`rag_eval_runs_total` + `rag_eval_last_run_accuracy` land in `/metrics`.

**Downvote triage / quality radar (Round 18).** `GET /quality/triage` joins 👎
ratings against the answer ledger (same join as the candidates flow, other
polarity): question, answer preview, fallback flag, citations, and the
user's reason comment. Upvotes grow the eval set; downvotes become a
fix-list. Round 19 completed that loop's lifecycle (next paragraph).

**Feedback-loop closure (new, Round 19).** The two polarities now have a
complete lifecycle. **👍 batch promotion:** `POST
/eval/candidates/promote-batch` promotes up to 12 upvoted answers in one
call — every id runs the exact same single-promote path (refactored into
one shared `_promote_one` helper: ledger lookup → grounded check → `usr-`
case creation → immediate verification through the full agent graph,
sequentially, ~1 s each in simulation mode). Per-item failures are reported
in `results` with their `error` and never abort the batch; duplicate ids in
the input are silently de-duplicated; optional `keywords_map` carries
per-request edited fact patterns. The single `POST /eval/candidates/promote`
is behavior-identical to before. **👎 resolve/reopen lifecycle:** `GET
/quality/triage`'s `items` is now the **open** fix-list only, resolved items
move to `resolved_items` (newest resolved first, with resolution note +
timestamp); `POST /quality/triage/resolve` marks an item fixed (optional
≤300-char note, requires a current downvote), `POST /quality/triage/reopen`
puts it back — both persisted in `data/triage_resolutions.json` (capped
ring, atomic writes). `counts.open`/`counts.resolved` are accurate across
all downvoted records and land in `/stats` (`triage` block) and `/metrics`
(`rag_triage_open`/`rag_triage_resolved` gauges).

**Streaming (SSE) — true incremental generation.** `/ask/stream` emits
three event kinds: `node` (per-graph-step progress with per-node latency),
`answer_delta` (generated tokens pushed **while the generator node is still
producing** — the first delta carries `ttft_ms`; the verifier has not run
yet), and `final` (the full verified payload with citations and trace).
Deltas come from `brain.generate_answer_stream` (real vLLM token stream in
vLLM mode, simulated decode cadence in simulation mode) through a transient
`state.stream_emitter` the graph injects — the same SSE callback, so no
second plumbing path. Multi-hop questions stream **entity block by entity
block**. Honest-streaming semantics: the verifier still gates the final
answer, so a rejected draft is replaced by the fallback in `final` (the UI
shows a "draft failed verification" notice). Cache hits and fallbacks —
which produce no incremental tokens — are replayed post-hoc at near-instant
cadence.

**Observability (Prometheus).** `GET /metrics` renders the service
telemetry in the standard text exposition format (v0.0.4): `rag_ask_total`
per route, a `rag_ask_latency_seconds` histogram (buckets 5 ms–2.5 s),
cache/fallback/rewrite/follow-up counters, `http_requests_total` by
**route pattern** (label cardinality is bounded by using the matched
Starlette route, 404s collapse to `unmatched`), plus corpus/session/
feedback/cache/eval/benchmark gauges and `rag_build_info{version,mode}`.
A raw-ASGI middleware collects the HTTP counters; the same MetricsCollector
powers `/stats`, so both surfaces can never disagree. The UI's Pulse tab
scrapes it live (headline values + raw viewer). Scrapes are measured
(`rag_metrics_scrape_duration_ms`). In-process counters (histogram, HTTP)
reset on restart — a Prometheus server is the durable store in production.
`GET /ops/grafana` returns a **Grafana-importable dashboard JSON**
(8 panels: ask traffic by route, latency quantiles from the histogram,
cache hit ratio + pressure, honesty signals, per-endpoint HTTP traffic,
golden-set quality, corpus/memory stats) so the ops story is one import
away from live dashboards; a copy ships as
`download/rag-grafana-dashboard.json`.

**Retrieval introspection (new).** `GET /search?q=…&k=…` runs the hybrid
retriever directly and returns per-result **dense rank, BM25 rank, fused
score and coverage** — the RRF fusion becomes observable in the UI's Corpus
tab, whose **retrieval playground** exposes top-k and the three reranker
weights (wf/wc/wa) as live controls that re-rank with the same code the
agent graph uses. `GET /source/{chunk_id}` returns the full indexed chunk
behind any citation plus adjacent chunks from the same document, so every
answer can be verified against its source in one click (citation cards in
the UI open this dialog). `POST /feedback` records thumbs up/down per
request_id (deduped, in-memory) with `GET /feedback/stats` aggregating
approval.

**Semantic answer cache (new).** `app/rag/answer_cache.py` caches only
**grounded, verified** answers (never fallbacks), keyed by the embedded
effective query with cosine ≥ 0.93 plus exact-normalised always-hit; TTL
15 min, 128-entry LRU. A hit skips the agent graph entirely — the response
carries `cache_hit` + `cache_detail` provenance (original request id, age,
similarity, times served, original latency), the trace is a single honest
`cache` node, and the SSE path streams the cached answer at near-instant
cadence. Cache telemetry lands in `/stats` (entries, hits, hit-rate, TTL,
restored count). **Entries persist to `data/answer_cache.json`** (atomic
writes; TTL-expired entries dropped at load) so warm answers survive
restarts — verified by asking, restarting the service, and hitting from the
restored cache. In simulation mode the hashing embedder makes the similarity gate
near-exact (paraphrases honestly miss); with sentence-transformers the same
code becomes a true semantic cache. Production shape: Redis with the same
semantics.

**Brains as strategies.** `LLMBrain` (prompt-driven, strict-JSON outputs with
tolerant parsing) and `HeuristicBrain` (deterministic classical NLP) share
one async interface. The heuristic brain doubles as the fallback parser for
malformed LLM output — defence in depth.

**Multimodal (Part E).** Option 2 (`/ask-image` through the served VLM) as
primary, **plus** Option 1's ingestion half: PDF figures are extracted at
ingest time, OCR-captioned offline (or VLM-captioned when served) and indexed
as pseudo-chunks so chart questions are answerable from the text store too.
Offline, the VLM path degrades to OCR + PIL analysis and says so explicitly
in the response (`mode: "ocr-fallback"`).

**Simulation mode.** The sandbox has no GPU, so a deterministic extractive
engine stands in for the LLM at every decision point (routing via
pattern/vocab-overlap, grading via stemmed coverage, verification via claim
support ratio, generation via weighted unit selection). This keeps the
assignment fully demonstrable end-to-end while honestly reporting its mode —
and it is the same code path the vLLM backend takes, so swapping brains
changes nothing structurally.

## 4. Benchmark results (Part A)

Run: `POST /benchmark {"concurrency": 8, "requests": 24}` or
`python scripts/benchmark.py`. The harness streams from the active backend
and reports TTFT, P50/P95 latency and tokens/sec.

| Metric | Simulation mode (this sandbox) | vLLM mode (Qwen2.5-VL-7B, RTX 4090, AWQ) |
|---|---|---|
| TTFT p50 / p95 | 47 / 60 ms | *(fill from your GPU run)* |
| Latency p50 / p95 | 284 / 448 ms | *(fill from your GPU run)* |
| Throughput | ~970 tok/s (synthetic) | *(fill from your GPU run)* |
| Success | 24/24 | — |

> The sandbox numbers exercise the full benchmark plumbing but the
> "backend" is the deterministic engine — they are **not** GPU serving
> numbers. With `docker compose up vllm` the same endpoint measures the real
> server; sample expectations for Qwen2.5-VL-7B-AWQ on a 4090: TTFT p95
> < 400 ms, ~60–120 tok/s per stream, >1k tok/s aggregate at concurrency 8.

## 5. Demo corpus

16 documents (`corpus/`) — 12 markdown + 1 plain-text FAQ + 3 typeset PDFs
(7 pages each, real page numbers, one embedded matplotlib chart). Facts are
deliberately cross-referenced (warranty windows appear in 4 docs, RMA timing
in 3) so retrieval quality and citation fidelity are observable. Regenerate
or replace with your own files, then `POST /ingest/sample` or
`python scripts/ingest.py <dir>`.

## 6. Evaluation results (golden set)

`POST /eval` or `python scripts/eval.py` (current sandbox, simulation mode):

| Metric | Result |
|---|---|
| Accuracy | **16/16 (100%)** |
| Retrieval hit-rate | 100% |
| Fact-support rate | 94–100% (multihop cases require BOTH entities' facts) |
| Faithfulness (grounding judge) | **99.9%** (lexical judge; LLM-as-judge in vLLM mode) |
| Latency p50 / p95 | ~10 / ~21 ms |

`POST /eval/ablation` (retriever configs on the same golden set, current):

| Config | Accuracy | hit@k | p50 | Δ accuracy |
|---|---|---|---|---|
| baseline (top-k 4, default weights) | 100% | 100% | 10.1 ms | — |
| top-k 3 (narrow) | 100% | 100% | 8.3 ms | 0 (faster) |
| top-k 8 (broad) | 93.8% | 92.3% | 14.0 ms | **−6.2 pts** |
| coverage-heavy (0.35/0.55/0.10) | 100% | 100% | 10.5 ms | 0 |
| fusion-heavy (0.75/0.15/0.10) | 100% | 100% | 9.8 ms | 0 |

Takeaway: on this corpus, broad retrieval **hurts** (extra same-doc chunks
crowd out the answering chunk for table-resident facts), and the shipped
defaults are already on the Pareto front — which is exactly the kind of
statement an ablation table should be able to back with numbers.

Per-category bars and per-case pass/fail with answer previews + per-case
faithfulness chips are rendered in the UI's **Eval** tab. The harness doubles
as the regression gate: any change to chunking, retrieval, or answer
composition that regresses a pinned case fails the run (exit code 1 from
`scripts/eval.py`). The faithfulness dimension (RAGAS-style) scores how well
each answer is supported by its own citations — claim-level entailment,
grounding ratio and numeric-contradiction detection in the deterministic
judge, a strict prompted judge when the model is served live.

## 7. Known limitations & next steps

- **Simulation ≠ semantics.** The offline brain is lexical (stems, synonyms,
  coverage) — paraphrases that share no tokens will fall back honestly.
  Installing `sentence-transformers` upgrades dense retrieval; a real vLLM
  server upgrades every decision point.
- **The hashing embedder is entity-code-blind** (x200 ≈ x300 under char
  n-grams) — compensated at the ranking stage (code-weighted coverage,
  exact-code boost, code-doc pool expansion), but a neural embedder removes
  the need for the compensation.
- **Query decomposition is regex-detected in simulation mode** (comparative
  cue + 2–3 entity codes): deterministic and explainable, but limited to the
  corpus's code vocabulary. The LLM decomposer path (vLLM mode, validated +
  regex fallback) now generalises this — untested against live vLLM here
  (no GPU).
- **Semantic cache is near-exact in simulation mode** (hashing-embedder
  cosine ≥ 0.93 ≈ duplicate detection — paraphrases honestly miss); with
  sentence-transformers the same code becomes a true semantic cache.
- **Verifier is term-based offline** (claim-support ratio over stemmed terms);
  the eval harness now carries a dedicated faithfulness judge (lexical
  offline / LLM-as-judge live).
- **Cross-encoder is a lexical surrogate in simulation mode** (pair features
  instead of neural joint attention); the real model path exists for vLLM
  (LLM-scored 0–100, per-pair fallback) but is untested against a live
  server (no GPU here). The surrogate's anchor gate is corpus-IDF-driven —
  robust on this corpus, tuned via the ablation harness for anything bigger.
- **Chit-chat detection is pattern-based** offline; trivially prompt-driven
  in vLLM mode.
- **Session memory is heuristic entity-carrying** (regex entity codes), not
  semantic — deliberate for observability; a summary-buffer memory backed by
  the LLM is the production upgrade. Sessions, feedback and usage counters
  now persist to `data/*.json` (atomic writes) and survive restarts; Redis
  remains the production upgrade (as does the answer cache).
- **Triage resolutions are a capped ring (500)** like the feedback store —
  oldest resolutions are silently evicted. Resolving requires the request_id
  to currently carry a downvote; a 👍 flip or feedback eviction leaves an
  orphaned resolution record that simply stops appearing in the triage join.
  A 👎 whose ledger entry was evicted (ledger caps at 300) still resolves and
  shows in `resolved_items` with question `"?"`; a 👎 with no ledger entry at
  all (pre-ledger ask) cannot be listed, only counted (`unlinkable_ratings`).
  Batch promote is sequential and capped at 12 ids per call (each
  verification runs the full agent graph — a large batch takes ~N seconds).
- With more time: LoRA fine-tune + serve via vLLM `--lora-modules`.

## 8. Repository layout

```
mini-services/rag-agent/
├── app/
│   ├── main.py               # FastAPI endpoints (ask, stream, image, ingest, eval, ablation…)
│   ├── config.py             # 12-factor settings
│   ├── schemas.py            # Pydantic API contract
│   ├── conversation.py       # session store + follow-up resolution
│   ├── feedback.py           # thumbs up/down telemetry store
│   ├── benchmark.py          # concurrency/TTFT/P95 harness
│   ├── metrics.py · persist.py  # usage telemetry + atomic persistence
│   ├── prometheus.py         # /metrics exposition + HTTP counter middleware
│   ├── grafana.py            # /ops/grafana importable dashboard JSON
│   ├── eval/                 # golden set + runner + faithfulness judge +
│   │                         # growth + triage + triage.py resolve tracker
│   │                         # + history + A/B compare (F)
│   ├── llm/                  # vLLM client, LLM brain, simulation brain
│   ├── rag/                  # chunk→embed→index→retrieve + answer_cache
│   │   ├── reranker.py       # cross-encoder stage (lexical surrogate / LLM)
│   ├── agents/               # graph engine + nodes (router/decomposer/grader/…)
│   ├── parsing/              # pdfplumber + text loaders, figure extraction
│   └── multimodal/           # /ask-image (VLM + OCR fallback)
├── scripts/                  # ingest.py · ask.py · benchmark.py · eval.py · generate_corpus.py
├── corpus/                   # 16-document demo knowledge base
├── Dockerfile · docker-compose.yml · .env.example · requirements.txt
└── index.js                  # dev process manager (watch + restart)
```
