/* ============================================================
   Advanced RAG — front-end logic
   Talks to the FastAPI backend (same origin): /health /upload
   /ingest-status/:id /documents /query /reset
   ============================================================ */

// The 8 LangGraph stages, in order. Keys match the backend `trace` step names.
const STAGES = [
  { key: "query_transform", icon: "✎", name: "Query Transform", hint: "rewrite + multi-query" },
  { key: "retrieve",        icon: "⇉", name: "Retrieve",        hint: "BM25 ∥ Dense", pills: true },
  { key: "rag_fusion",      icon: "⊕", name: "RAG-Fusion",      hint: "reciprocal rank fusion" },
  { key: "mmr",             icon: "◈", name: "MMR",             hint: "diversity select" },
  { key: "rerank",          icon: "☰", name: "Rerank",          hint: "LLM relevance score" },
  { key: "build_context",   icon: "▤", name: "Context",         hint: "dedup + assemble" },
  { key: "generate",        icon: "✦", name: "Generate",        hint: "grounded answer" },
  { key: "validate",        icon: "✓", name: "Validate",        hint: "grounding judge" },
];
const STAGE_KEYS = new Set(STAGES.map((s) => s.key));

const LEGEND = [
  ["Query Transform", "rewrites the question into several search queries"],
  ["Retrieve", "runs BM25 (lexical) and dense (semantic) search in parallel"],
  ["RAG-Fusion", "merges every ranked list with Reciprocal Rank Fusion"],
  ["MMR", "picks a diverse, non-redundant candidate set"],
  ["Rerank", "an LLM scores each passage 0–10 for relevance"],
  ["Generate", "writes a grounded answer with inline [n] citations"],
  ["Validate", "a judge checks grounding; if weak, it loops back"],
];

// ---- element handles ----
const $ = (id) => document.getElementById(id);
const statusEl = $("status"), statusText = $("statusText");
const fileInput = $("fileInput"), dropzone = $("dropzone"), uploadStatus = $("uploadStatus");
const docList = $("docList"), resetBtn = $("resetBtn");
const questionEl = $("question"), askBtn = $("askBtn"), askHint = $("askHint");
const pipelineEl = $("pipeline"), cycleBadge = $("cycleBadge");
const answerCard = $("answerCard"), answerEl = $("answer"), badgesEl = $("badges");
const sourcesEl = $("sources"), srcCount = $("srcCount");
const traceCard = $("traceCard"), traceToggle = $("traceToggle"), traceEl = $("trace");

// ============================================================
//  Boot
// ============================================================
function init() {
  // legend
  $("flowLegend").innerHTML = LEGEND.map(([b, t]) => `<li><b>${b}</b> — ${t}</li>`).join("");
  // pipeline skeleton
  renderPipelineSkeleton();

  // events
  fileInput.addEventListener("change", (e) => uploadFile(e.target.files[0]));
  wireDropzone();
  resetBtn.addEventListener("click", resetAll);
  askBtn.addEventListener("click", ask);
  questionEl.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") ask();
  });
  traceToggle.addEventListener("click", () => {
    const open = traceEl.hidden === false;
    traceEl.hidden = open;
    traceToggle.textContent = (open ? "▸" : "▾") + " Raw trace";
  });

  refreshStatus();
  refreshDocs();
  setInterval(refreshStatus, 5000); // keep the status pill live
}

// ============================================================
//  Status + documents
// ============================================================
async function refreshStatus() {
  try {
    const j = await (await fetch("/health")).json();
    setStatus("ok", `connected · ${j.indexed_chunks} chunks · ${j.llm}`);
    updateAskState(j.indexed_chunks);
  } catch {
    setStatus("bad", "backend offline");
    updateAskState(0, true);
  }
}

function setStatus(kind, text) {
  statusEl.className = "status status--" + (kind === "ok" ? "ok" : kind === "bad" ? "bad" : "wait");
  statusText.textContent = text;
}

function updateAskState(chunks, offline) {
  const ready = chunks > 0 && !offline;
  askBtn.disabled = !ready;
  askHint.textContent = offline
    ? "Backend is offline — start app.py."
    : ready
    ? "⌘/Ctrl + Enter to ask."
    : "Upload a PDF first to enable querying.";
}

async function refreshDocs() {
  try {
    const docs = await (await fetch("/documents")).json();
    if (!docs.length) {
      docList.innerHTML = `<li class="doc-empty">No documents yet.</li>`;
      return;
    }
    docList.innerHTML = docs
      .map(
        (d) => `<li>
          <span class="doc-name" title="${esc(d.source)}">${esc(d.source)}</span>
          <span class="doc-meta">${d.chunks} chunks · ${d.pages}p</span>
        </li>`
      )
      .join("");
  } catch {
    /* offline — leave as is */
  }
}

// ============================================================
//  Upload  ->  poll ingest status
// ============================================================
function wireDropzone() {
  ["dragenter", "dragover"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropzone.classList.add("drag");
    })
  );
  ["dragleave", "drop"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropzone.classList.remove("drag");
    })
  );
  dropzone.addEventListener("drop", (e) => {
    if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
  });
}

async function uploadFile(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    showUpload("Only PDF files are supported.", "err");
    return;
  }
  showUpload(`Uploading ${file.name}…`, "load");
  try {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch("/upload", { method: "POST", body: fd });
    if (!res.ok) throw new Error(await res.text());
    const { job_id } = await res.json();
    pollIngest(job_id, file.name);
  } catch (e) {
    showUpload("Upload failed: " + e.message, "err");
  } finally {
    fileInput.value = "";
  }
}

function pollIngest(jobId, name) {
  showUpload(`Indexing ${name} — chunk → embed → BM25 + FAISS…`, "load");
  const timer = setInterval(async () => {
    try {
      const j = await (await fetch(`/ingest-status/${jobId}`)).json();
      if (j.status === "done") {
        clearInterval(timer);
        showUpload(`✓ Indexed ${name} — ${j.chunks} chunks`, "ok");
        refreshDocs();
        refreshStatus();
      } else if (j.status === "failed") {
        clearInterval(timer);
        showUpload(`✗ ${j.error || "ingestion failed"}`, "err");
      }
    } catch {
      /* transient — keep polling */
    }
  }, 1000);
}

function showUpload(msg, kind) {
  uploadStatus.hidden = false;
  uploadStatus.className = "upload-status" + (kind === "err" ? " err" : kind === "ok" ? " ok" : "");
  uploadStatus.innerHTML = (kind === "load" ? `<span class="spinner"></span>` : "") + `<span>${esc(msg)}</span>`;
}

async function resetAll() {
  if (!confirm("Remove every indexed document?")) return;
  await fetch("/reset", { method: "POST" });
  uploadStatus.hidden = true;
  answerCard.hidden = true;
  traceCard.hidden = true;
  renderPipelineSkeleton();
  refreshDocs();
  refreshStatus();
}

// ============================================================
//  Ask
// ============================================================
async function ask() {
  const q = questionEl.value.trim();
  if (!q || askBtn.disabled) return;

  askBtn.disabled = true;
  askBtn.textContent = "Thinking…";
  renderPipelineSkeleton(); // dim all stages while running
  answerCard.hidden = true;
  traceCard.hidden = true;

  try {
    const res = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    renderPipeline(data.trace, data.retries);
    renderAnswer(data);
    renderTrace(data.trace);
  } catch (e) {
    showError(e.message);
  } finally {
    askBtn.disabled = false;
    askBtn.textContent = "Ask";
    refreshStatus();
  }
}

// ============================================================
//  Render: answer + badges + sources
// ============================================================
function renderAnswer(d) {
  answerCard.hidden = false;
  const conf = Math.round((d.confidence || 0) * 100);
  badgesEl.innerHTML = [
    d.grounded
      ? `<span class="badge ok">✓ grounded</span>`
      : `<span class="badge bad">✗ ungrounded</span>`,
    `<span class="badge info">confidence ${conf}%</span>`,
    `<span class="badge info">${d.retries} ${d.retries === 1 ? "retry" : "retries"}</span>`,
  ].join("");

  answerEl.innerHTML = citations(esc(d.answer || ""));

  srcCount.textContent = d.sources.length ? `(${d.sources.length})` : "";
  sourcesEl.innerHTML = d.sources
    .map(
      (s) => `<li>
        <div class="src-meta">
          <span><b>${esc(s.source || "?")}</b></span>
          <span>page ${s.page ?? "—"}</span>
          ${s.rerank_score != null ? `<span class="src-score">score ${(+s.rerank_score).toFixed(1)}</span>` : ""}
        </div>
        <div class="src-snip">${esc(s.snippet || "")}…</div>
      </li>`
    )
    .join("");
  answerCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function showError(msg) {
  answerCard.hidden = false;
  badgesEl.innerHTML = `<span class="badge bad">error</span>`;
  answerEl.innerHTML = `<span style="color:var(--bad)">${esc(msg)}</span>`;
  srcCount.textContent = "";
  sourcesEl.innerHTML = "";
}

// wrap [1], [2] (or full-width 【1】 some models emit) citations in styled badges
function citations(html) {
  return html.replace(/[\[【]\s*(\d+)\s*[\]】]/g, '<span class="cite">$1</span>');
}

// ============================================================
//  Render: pipeline flow
// ============================================================
function renderPipelineSkeleton() {
  cycleBadge.hidden = true;
  pipelineEl.innerHTML = STAGES.map(
    (s) => `<div class="stage" data-key="${s.key}">
      <div class="s-time"></div>
      <div class="s-top"><span class="s-icon">${s.icon}</span><span class="s-name">${s.name}</span></div>
      <div class="s-hint">${s.hint}</div>
      ${s.pills ? `<div class="retr-pills"><span class="pill bm25">BM25</span><span class="pill dense">Dense</span></div>` : ""}
      <div class="s-stat"></div>
    </div>`
  ).join("");
}

function renderPipeline(trace, retries) {
  renderPipelineSkeleton();
  if (!Array.isArray(trace)) return;

  // per-stage: last observed step (for stats) + summed duration (across retries)
  const last = {}, durMs = {};
  let prevTs = null, cycles = 0;
  for (const t of trace) {
    const dur = prevTs == null ? 0 : Math.max(0, (t.ts - prevTs) * 1000);
    prevTs = t.ts;
    if (t.step === "query_transform") cycles++;
    if (STAGE_KEYS.has(t.step)) {
      last[t.step] = t;
      durMs[t.step] = (durMs[t.step] || 0) + dur;
    }
  }

  for (const s of STAGES) {
    const t = last[s.key];
    if (!t) continue;
    const card = pipelineEl.querySelector(`.stage[data-key="${s.key}"]`);
    card.classList.add("on");
    card.querySelector(".s-stat").textContent = statText(s.key, t);
    const ms = durMs[s.key] || 0;
    if (ms) card.querySelector(".s-time").textContent = ms >= 1000 ? (ms / 1000).toFixed(1) + "s" : Math.round(ms) + "ms";
  }

  if (cycles > 1 || retries > 0) {
    cycleBadge.hidden = false;
    cycleBadge.textContent = `↻ ${cycles} passes · self-corrected`;
  }
}

function statText(key, t) {
  switch (key) {
    case "query_transform": return `${(t.queries || []).length} queries`;
    case "retrieve":        return `${t.raw_hits} hits · ${t.ranked_lists} lists`;
    case "rag_fusion":      return `${t.candidates} candidates`;
    case "mmr":             return `${t.selected} kept`;
    case "rerank":          return `top ${t.kept}`;
    case "build_context":   return `${t.docs} docs`;
    case "generate":        return `${t.answer_chars} chars`;
    case "validate":        return `${t.grounded ? "grounded" : "weak"} · ${Math.round((t.confidence || 0) * 100)}%`;
    default: return "";
  }
}

function renderTrace(trace) {
  traceCard.hidden = false;
  traceEl.hidden = true;
  traceToggle.textContent = "▸ Raw trace";
  traceEl.textContent = JSON.stringify(trace, null, 2);
}

// ============================================================
//  util
// ============================================================
function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}

init();
