"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence, useReducedMotion } from "framer-motion";
import { useTheme } from "next-themes";
import {
  Activity,
  AlertTriangle,
  ArrowDownUp,
  ArrowUp,
  BarChart3,
  BookOpen,
  Bot,
  CheckCheck,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  CircleAlert,
  ClipboardCheck,
  Clock,
  Copy,
  Database,
  Download,
  FileDown,
  FileJson,
  FileText,
  FlaskConical,
  Gauge,
  GitCompare,
  HeartPulse,
  History,
  ImageIcon,
  Layers,
  LayoutDashboard,
  Link2,
  ListChecks,
  Loader2,
  Lock,
  MessageSquare,
  MessageSquareQuote,
  Moon,
  MousePointerClick,
  MoveHorizontal,
  RefreshCw,
  RotateCcw,
  Search,
  Send,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Split,
  Sprout,
  Sun,
  TextSearch,
  ThumbsDown,
  ThumbsUp,
  Timer,
  Trash2,
  TrendingDown,
  TrendingUp,
  Trophy,
  Undo2,
  Upload,
  User,
  X,
  XCircle,
  ZoomIn,
  Zap,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useToast } from "@/hooks/use-toast";
import { cn } from "@/lib/utils";

const API_PORT = 3003;
const api = (path: string) => `${path}${path.includes("?") ? "&" : "?"}XTransformPort=${API_PORT}`;

// Retrieval playground defaults (match the retriever's DEFAULT_WEIGHTS).
const DEFAULT_WF = 0.55;
const DEFAULT_WC = 0.35;
const DEFAULT_WA = 0.10;

// ----------------------------------------------------------------- types
type Citation = {
  doc: string;
  page: number;
  chunk_id: string;
  snippet: string;
  score: number;
};

type TraceEntry = {
  node: string;
  status: string;
  detail: Record<string, unknown>;
  latency_ms: number;
  sequence: number;
};

type CacheDetail = {
  original_request_id: string;
  original_question: string;
  age_seconds: number;
  similarity: number;
  times_served: number;
  original_latency_ms: number;
};

// Cross-encoder rerank telemetry — retriever node trace detail and /search block.
type RerankInfo = {
  stage: string;
  mode?: string;
  sub_searches?: number;
  reranked: number;
  reorders: number;
  latency_ms: number;
  top_scores?: { doc: string; page: number; score: number; fusion_rank: number | null }[];
};

type AskResponse = {
  question: string;
  answer: string;
  citations: Citation[];
  route: "document" | "chitchat" | "out_of_scope";
  agent_path: string[];
  trace: TraceEntry[];
  retries: number;
  fallback: boolean;
  latency_ms: number;
  serving_mode: "vllm" | "simulation";
  request_id: string;
  session_id: string;
  effective_question: string;
  follow_up_applied: boolean;
  inherited_entities: string[];
  turn_index: number;
  sub_queries?: string[];
  decomposed?: boolean;
  cache_hit?: boolean;
  cache_detail?: CacheDetail | null;
};

type HealthResponse = {
  status: "ok" | "degraded";
  serving: {
    mode: "vllm" | "simulation";
    model: string;
    vllm_base_url: string;
    vllm_reachable: boolean;
    embedding_backend: string;
  };
  corpus: { total_documents: number; total_chunks: number; embedding_dim: number };
  version: string;
};

type CorpusDoc = {
  doc: string;
  kind: string;
  pages: number;
  chunks: number;
  figures: number;
  words: number;
};

type CorpusResponse = {
  documents: CorpusDoc[];
  total_chunks: number;
  total_documents: number;
  embedding_dim: number;
  embedding_backend: string;
};

type BenchmarkResult = {
  mode: string;
  concurrency: number;
  total_requests: number;
  successful_requests: number;
  failed_requests: number;
  ttft_p50_ms: number;
  ttft_p95_ms: number;
  latency_p50_ms: number;
  latency_p95_ms: number;
  tokens_per_sec: number;
  total_tokens: number;
  duration_s: number;
};

type ImageAskResponse = {
  question: string;
  answer: string;
  mode: "vllm-vision" | "ocr-fallback";
  ocr_text: string;
  image_info: {
    filename: string;
    format: string;
    width: number;
    height: number;
    dominant_colors: string[];
  };
  citations: Citation[];
  latency_ms: number;
};

type Suggestion = { question: string; label: string; expects: "grounded" | "fallback" | "chitchat" };

type EvalCaseResult = {
  id: string;
  question: string;
  category: string;
  expects: string;
  passed: boolean;
  route_correct: boolean;
  route: string;
  retrieval_hit: boolean | null;
  retrieved_doc: string | null;
  fact_supported: boolean;
  matched: string | null;
  faithfulness: {
    score: number | null;
    judge: string;
    supported_claims: number;
    total_claims: number;
    contradictions: string[];
    unsupported: string[];
    llm_reason: string | null;
  } | null;
  answer_preview: string;
  citations: string[];
  latency_ms: number;
  agent_path: string[];
  note: string;
};

type EvalReport = {
  ran_at: string;
  total_cases: number;
  passed: number;
  failed: number;
  accuracy: number;
  retrieval_hit_rate: number | null;
  fact_support_rate: number;
  faithfulness_avg: number | null;
  faithfulness_judge: string | null;
  latency_p50_ms: number;
  latency_p95_ms: number;
  by_category: Record<string, { total: number; passed: number }>;
  results: EvalCaseResult[];
};

type StatsResponse = {
  service: { uptime_seconds: number; uptime_human: string; started_at: string };
  questions: {
    total: number;
    by_route: Record<string, number>;
    fallbacks: number;
    fallback_rate: number | null;
    query_rewrites: number;
    follow_ups_resolved: number;
    streamed_answers: number;
    cache_hits: number;
    cache_hit_rate: number | null;
    latency_avg_ms: number | null;
    latency_p50_ms: number;
    latency_p95_ms: number;
    recent: { question: string; route: string; latency_ms: number; fallback: boolean; ts: number }[];
  };
  sessions: { sessions: number; total_turns: number; follow_ups_resolved: number };
  feedback: { total: number; up: number; down: number; approval: number | null };
  corpus: { documents: number | null; chunks: number | null };
  serving: { mode: string; model: string | null };
  cache:
    | {
        entries: number;
        capacity: number;
        hits: number;
        hit_rate_last: number;
        ttl_seconds: number;
        sim_threshold: number;
        persisted?: boolean;
        restored_entries?: number;
      }
    | null;
  rerank?: { mode: string; n: number; calls: number; llm_scores: number; reorders: number; avg_latency_ms: number | null };
  golden?: { base: number; user: number; total: number; ledger_entries: number; upvoted: number; downvoted?: number };
  eval_runs?: { total: number; golden: number; ablation: number; last_accuracy: number };
  triage?: { open: number; resolved: number; oldest_open_hours?: number | null; mean_open_hours?: number | null };
  last_eval: { accuracy: number; faithfulness_avg: number | null; ran_at: string; passed: number; total: number } | null;
  last_benchmark: { requests: number; ok: number; tokens_per_sec: number | null; p95_ms: number | null } | null;
};

type SessionSummary = {
  id: string;
  turns: number;
  preview: string;
  created_at: number;
  last_activity: number;
  entities: string[];
};

type SessionTurn = {
  question: string;
  effective_question: string;
  answer: string;
  route: string;
  agent_path: string[];
  ts: number;
  follow_up_applied: boolean;
  citations?: Citation[];
};

type SourceDetail = {
  chunk_id: string;
  doc: string;
  doc_kind: string;
  page: number;
  section: string;
  text: string;
  token_count: number;
  is_figure_caption: boolean;
  neighbors: { chunk_id: string; page: number; section: string; preview: string; relation: string }[];
};

type SearchHit = {
  chunk_id: string;
  doc: string;
  page: number;
  section: string;
  score: number;
  coverage: number;
  dense_rank: number;
  bm25_rank: number;
  is_figure: boolean;
  preview: string;
  rerank_score?: number | null;
  fusion_rank?: number | null;
};

type SearchWeights = { w_fused: number; w_coverage: number; w_agreement: number };

type SearchResponse = {
  query: string;
  weights: SearchWeights;
  results: SearchHit[];
  rerank?: { stage: string; mode: string; reranked: number; reorders: number; latency_ms: number } | null;
};

type AblationConfigResult = {
  name: string;
  config: Record<string, unknown>;
  accuracy: number;
  retrieval_hit_rate: number | null;
  fact_support_rate: number;
  faithfulness_avg: number | null;
  latency_p50_ms: number;
  latency_p95_ms: number;
  passed: number;
  total: number;
  delta_accuracy: number;
};

type AblationReport = {
  ran_at: string;
  baseline: string;
  configs: AblationConfigResult[];
  winner: string;
  duration_ms: number;
  notes: string[];
};

// ------------------------------------------- golden-set growth (Task 17-b)
// Feedback-driven growth: upvoted answers are mined into candidates that can
// be promoted into user golden cases; the eval run evaluates the merged set.
type GoldenSetCounts = { base: number; user: number; total: number };

type EvalCandidate = {
  request_id: string;
  question: string;
  answer_preview: string;
  route: string;
  cache_hit: boolean;
  latency_ms: number;
  citations: { doc: string; page: number }[];
  suggested_keywords: string[];
  suggested_docs: string[];
  keywords_editable: boolean;
  feedback: { rating: string; comment: string | null; ts: number };
  rated_at: number;
};

type CandidatesResponse = {
  candidates: EvalCandidate[];
  counts: { shown: number; unlinkable_ratings: number; skipped_route: number; skipped_duplicate: number };
  golden_set: GoldenSetCounts;
  ledger: { entries: number; capacity: number; grounded: number; cache_served: number };
};

type PromoteResponse = {
  status: string;
  case: { id: string; question: string; keywords: string[]; docs_any: string[]; category: string; note: string };
  golden_set: GoldenSetCounts;
  verification: EvalCaseResult | null;
};

type PromoteOutcome = {
  requestId: string;
  passed: boolean;
  matched: string | null;
  goldenSet: GoldenSetCounts;
  caseId: string;
};

// Batch promotion (Task 19-b): 1–12 request_ids → per-item results.
type BatchPromoteResult = {
  request_id: string;
  question: string | null;
  status: "promoted" | "failed";
  case_id?: string; // promoted only
  passed?: boolean; // promoted only
  matched?: string; // promoted only
  error?: string; // failed only
};

type BatchPromoteResponse = {
  status: string;
  promoted: number;
  failed: number;
  results: BatchPromoteResult[];
  golden_set: GoldenSetCounts;
};

type BatchOutcome = {
  promoted: number;
  failed: number;
  results: BatchPromoteResult[];
  goldenSet: GoldenSetCounts;
};

type GoldenCase = {
  id: string;
  question: string;
  category: string;
  expects: string;
  note: string;
  source: "base" | "user";
  keywords?: string[];
  docs_any?: string[];
  deletable: boolean;
};

type CasesResponse = {
  total: number;
  stats: GoldenSetCounts;
  cases: GoldenCase[];
};

type CompareSide = {
  config: string;
  config_detail: Record<string, unknown>;
  answer: string;
  citations: { doc: string; page: number; snippet: string }[];
  route: string;
  fallback: boolean;
  decomposed: boolean;
  agent_path: string[];
  latency_ms: number;
  faithfulness: EvalCaseResult["faithfulness"] | null;
  distinct_docs: string[];
  rerank: RerankInfo | null;
};

type CompareVerdict = {
  answers_identical: boolean;
  answer_a_chars: number;
  answer_b_chars: number;
  citation_overlap: number;
  shared_docs: string[];
  docs_only_a: string[];
  docs_only_b: string[];
  latency_delta_ms: number;
  faithfulness_delta: number | null;
};

type CompareResponse = {
  question: string;
  ran_at: string;
  side_a: CompareSide;
  side_b: CompareSide;
  verdict: CompareVerdict;
  available_configs: string[];
};

// ------------------------------- eval run history + quality triage (18-b)
// Persisted run ring (data/eval_history.json): every POST /eval and every
// ablation config appends a summary; the trend powers the history chart.
type EvalRunSummary = {
  ts: number;
  label: string;
  ablation: boolean;
  passed: number;
  total_cases: number;
  accuracy: number;
  faithfulness_avg: number | null;
  retrieval_hit_rate: number | null;
  fact_support_rate: number;
  latency_p50_ms: number;
  latency_p95_ms: number;
  failed: number;
};

type EvalHistoryResponse = {
  runs: EvalRunSummary[]; // newest first
  total_runs: number;
  golden_runs: number;
  ablation_runs: number;
  last: EvalRunSummary | null;
  best_accuracy: number;
  accuracy_trend: string; // "flat" | "changed"
  recorded_since: number;
};

// Downvote triage: 👎 answers joined against the answer ledger. Task 19-b:
// `items` is the OPEN fix-list; resolved items live in `resolved_items`
// (newest resolved first) with their persisted resolution notes.
type TriageItem = {
  request_id: string;
  question: string;
  answer_preview: string;
  route: string;
  fallback: boolean;
  cache_hit: boolean;
  latency_ms: number;
  citations: { doc: string; page: number }[];
  reason: string | null;
  rated_at: number;
  age_hours?: number | null; // Round 20 SLA — hours since the downvote (open items)
  resolved: boolean;
  resolution_note: string | null;
  resolved_at: number | null;
};

type TriageResponse = {
  items: TriageItem[]; // open (unresolved) fix-list
  resolved_items: TriageItem[]; // newest resolved first
  counts: {
    shown: number; // open items shown
    open: number; // all linkable unresolved
    resolved: number;
    unlinkable_ratings: number;
    fallback_answers: number;
    uncited_answers: number;
    with_reason: number;
    oldest_open_hours?: number | null; // Round 20 SLA — across ALL open items
    mean_open_hours?: number | null;
  };
};

// Round 20 — question analytics (GET /analytics/questions): frequency +
// quality aggregates per distinct normalised question from the answer ledger.
type TopQuestion = {
  question: string;
  count: number;
  last_ts: number;
  routes: string[];
  fallback_rate: number;
  citation_rate: number;
  cache_rate: number;
  avg_latency_ms: number;
  ups: number;
  downs: number;
};

type AnalyticsResponse = {
  top: TopQuestion[];
  totals: { distinct_questions: number; answered_asks: number; ledger_capacity: number };
};

// Fallback config list for the A/B compare dropdowns — the endpoint returns
// the live list in `available_configs`, this only bootstraps the defaults.
const COMPARE_CONFIGS = [
  "baseline (top-k 4, default weights, cross-encoder)",
  "cross-encoder off (fusion only)",
  "cross-encoder broad (n=14)",
  "top-k 3 (narrow)",
  "top-k 8 (broad)",
  "coverage-heavy (w 0.35/0.55/0.10)",
  "fusion-heavy (w 0.75/0.15/0.10)",
];

// ------------------------------------------------ prometheus text parsing
// /metrics serves the Prometheus text exposition format (text/plain;
// version=0.0.4). These tiny parsers pull headline values client-side.
const PROM_NUM = "-?\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?";

function promValue(text: string, name: string): number | null {
  const re = new RegExp(`^${name}(?:\\{[^}]*\\})?\\s+(${PROM_NUM})\\s*$`, "m");
  const m = re.exec(text);
  return m ? Number(m[1]) : null;
}

function promSum(text: string, family: string): number | null {
  const re = new RegExp(`^${family}(?:\\{[^}]*\\})?\\s+(${PROM_NUM})\\s*$`, "gm");
  let sum = 0;
  let seen = false;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    sum += Number(m[1]);
    seen = true;
  }
  return seen ? sum : null;
}

type PromHeadline = {
  asks: number | null;
  latencyObs: number | null;
  httpReqs: number | null;
  cacheHitRatio: number | null;
};

function parsePromHeadline(text: string): PromHeadline {
  return {
    asks: promSum(text, "rag_ask_total"),
    latencyObs: promValue(text, "rag_ask_latency_seconds_count"),
    httpReqs: promSum(text, "http_requests_total"),
    cacheHitRatio: promValue(text, "rag_cache_hit_ratio"),
  };
}

function promInt(n: number | null): string {
  return n === null ? "—" : Math.round(n).toLocaleString();
}

// ------------------------------------------------------------- agent steps
const NODE_META: Record<string, { label: string; icon: typeof Zap; tone: string }> = {
  session: { label: "Memory", icon: History, tone: "text-violet-600 dark:text-violet-400" },
  router: { label: "Router", icon: Sparkles, tone: "text-emerald-600 dark:text-emerald-400" },
  retriever: { label: "Retriever", icon: Database, tone: "text-teal-600 dark:text-teal-400" },
  grader: { label: "Grader", icon: CircleAlert, tone: "text-amber-600 dark:text-amber-400" },
  rewriter: { label: "Rewriter", icon: RefreshCw, tone: "text-orange-600 dark:text-orange-400" },
  decomposer: { label: "Decomposer", icon: Split, tone: "text-teal-600 dark:text-teal-400" },
  cache: { label: "Cache", icon: Zap, tone: "text-amber-600 dark:text-amber-400" },
  generator: { label: "Generator", icon: Bot, tone: "text-emerald-600 dark:text-emerald-400" },
  verifier: { label: "Verifier", icon: ShieldCheck, tone: "text-emerald-600 dark:text-emerald-400" },
  chitchat: { label: "Chit-chat", icon: MessageSquare, tone: "text-rose-600 dark:text-rose-400" },
  fallback: { label: "Fallback", icon: CircleAlert, tone: "text-amber-600 dark:text-amber-400" },
};

const CATEGORY_TONES: Record<string, string> = {
  factoid: "border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-900 dark:bg-emerald-950 dark:text-emerald-400",
  lookup: "border-teal-200 bg-teal-50 text-teal-700 dark:border-teal-900 dark:bg-teal-950 dark:text-teal-400",
  table: "border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-400",
  behaviour: "border-rose-200 bg-rose-50 text-rose-700 dark:border-rose-900 dark:bg-rose-950 dark:text-rose-400",
  robustness: "border-orange-200 bg-orange-50 text-orange-700 dark:border-orange-900 dark:bg-orange-950 dark:text-orange-400",
};

const ROUTE_TONES: Record<string, string> = {
  document: "border-emerald-300 text-emerald-700 dark:border-emerald-800 dark:text-emerald-400",
  chitchat: "border-rose-300 text-rose-700 dark:border-rose-800 dark:text-rose-400",
  out_of_scope: "border-amber-300 text-amber-700 dark:border-amber-800 dark:text-amber-400",
};

const ROUTE_MAX = { document: 30, chitchat: 40, out_of_scope: 50 };

function RouteBadge({ route }: { route: string }) {
  return (
    <Badge variant="outline" className={cn("gap-1", ROUTE_TONES[route] ?? "")}>
      {route === "document" && <FileText className="h-3 w-3" />}
      {route === "chitchat" && <MessageSquare className="h-3 w-3" />}
      {route === "out_of_scope" && <CircleAlert className="h-3 w-3" />}
      {route}
    </Badge>
  );
}

function AgentPathBar({
  path,
  active,
  fallback,
}: {
  path: string[];
  active: boolean;
  fallback: boolean;
}) {
  const steps = path.length ? path : ["router"];
  return (
    <div className="flex flex-wrap items-center gap-1.5" aria-label="Agent execution path">
      {steps.map((node, i) => {
        const meta = NODE_META[node] ?? { label: node, icon: Layers, tone: "text-muted-foreground" };
        const Icon = meta.icon;
        const isLast = i === steps.length - 1;
        const isActiveNode = active && isLast;
        return (
          <motion.span
            key={`${node}-${i}`}
            initial={{ opacity: 0, scale: 0.85 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.18 }}
            className="flex items-center gap-1.5"
          >
            <span
              className={cn(
                "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-medium transition-all",
                node === "fallback" || fallback
                  ? "border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-400"
                  : "border-border bg-muted/70",
                isActiveNode && "scale-105 border-emerald-300 bg-emerald-50 dark:border-emerald-800 dark:bg-emerald-950"
              )}
            >
              <Icon className={cn("h-3 w-3", meta.tone)} />
              <span className={cn(node === "fallback" || fallback ? "" : meta.tone)}>{meta.label}</span>
            </span>
            {!isLast && <ChevronRight className="h-3 w-3 shrink-0 text-muted-foreground/40" />}
          </motion.span>
        );
      })}
    </div>
  );
}

function SubQueryPills({ subQueries }: { subQueries: string[] }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2, delay: 0.05 }}
      className="flex flex-wrap items-center gap-1.5"
      aria-label="Decomposed sub-queries"
    >
      <span className="inline-flex shrink-0 items-center gap-1 text-[10px] font-medium uppercase tracking-wide text-teal-600 dark:text-teal-400">
        <Split className="h-3 w-3" />
        sub-queries
      </span>
      {subQueries.map((sq, i) => (
        <TooltipProvider key={sq + i} delayDuration={150}>
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                type="button"
                className="inline-flex max-w-56 items-center gap-1 rounded-full border border-teal-200 bg-teal-50/70 px-2 py-0.5 font-mono text-[10.5px] text-teal-700 transition-all hover:border-teal-400 hover:bg-teal-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500/60 focus-visible:ring-offset-1 dark:border-teal-900 dark:bg-teal-950/50 dark:text-teal-300 dark:hover:border-teal-700 dark:hover:bg-teal-900"
                title={sq}
              >
                <span className="font-bold opacity-60">{i + 1}</span>
                <span className="truncate">{sq}</span>
              </button>
            </TooltipTrigger>
            <TooltipContent side="bottom" className="max-w-64">
              <p className="font-mono text-[11px]">retrieved & graded as an independent query</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      ))}
    </motion.div>
  );
}

function CacheChip({ detail }: { detail: CacheDetail | null | undefined }) {
  const tips = detail
    ? [
        `similarity: ${detail.similarity.toFixed(2)}`,
        `age: ${detail.age_seconds < 90 ? `${detail.age_seconds.toFixed(0)}s` : `${Math.round(detail.age_seconds / 60)}m`}`,
        `times served: ${detail.times_served}`,
        `original latency: ${detail.original_latency_ms.toFixed(1)} ms`,
        `original: “${detail.original_question}”`,
      ]
    : ["served from the semantic answer cache"];
  return (
    <TooltipProvider delayDuration={150}>
      <Tooltip>
        <TooltipTrigger asChild>
          <Badge
            variant="outline"
            className="cursor-help gap-1 border-amber-300 bg-amber-50 text-amber-700 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-400"
          >
            <Zap className="h-3 w-3" />
            cached
          </Badge>
        </TooltipTrigger>
        <TooltipContent side="top" className="max-w-72">
          <p className="flex items-center gap-1 text-[11px] font-semibold">
            <Zap className="h-3 w-3 text-amber-500" />
            served from cache
          </p>
          {tips.map((t) => (
            <p key={t} className="mt-0.5 font-mono text-[10px] leading-snug text-muted-foreground">
              {t}
            </p>
          ))}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

function DecomposedChip({ count }: { count: number }) {
  return (
    <TooltipProvider delayDuration={150}>
      <Tooltip>
        <TooltipTrigger asChild>
          <Badge
            variant="outline"
            className="cursor-help gap-1 border-emerald-300 bg-emerald-50 text-emerald-700 dark:border-emerald-800 dark:bg-emerald-950 dark:text-emerald-400"
          >
            <Split className="h-3 w-3" />
            decomposed ×{count}
          </Badge>
        </TooltipTrigger>
        <TooltipContent side="top" className="max-w-64">
          <p className="text-[11px]">Multi-hop question — split into {count} independent sub-queries, retrieved and graded separately, then merged.</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

// Pull the rerank telemetry out of the retriever node's trace detail (Task 1a).
function traceRerank(trace: TraceEntry[]): RerankInfo | null {
  for (const t of trace) {
    if (t.node === "retriever" && t.detail && typeof t.detail === "object") {
      const rr = (t.detail as { rerank?: RerankInfo | null }).rerank;
      if (rr && typeof rr === "object" && typeof rr.reranked === "number") return rr;
    }
  }
  return null;
}

function RerankChip({ info, servingMode }: { info: RerankInfo; servingMode: string }) {
  const modeLabel = servingMode === "vllm" ? "llm scoring" : "lexical surrogate";
  const tips = [
    `reranked: ${info.reranked} candidates`,
    `reorders: ${info.reorders}`,
    `latency: ${info.latency_ms.toFixed(1)} ms`,
    `mode: ${modeLabel}`,
    ...(info.sub_searches && info.sub_searches > 0 ? [`sub-searches: ${info.sub_searches} (merged pools)`] : []),
  ];
  return (
    <motion.span
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.18 }}
    >
      <TooltipProvider delayDuration={150}>
        <Tooltip>
          <TooltipTrigger asChild>
            <Badge
              variant="outline"
              className="cursor-help gap-1 border-teal-300 bg-teal-50 text-teal-700 transition-colors hover:bg-teal-100 dark:border-teal-800 dark:bg-teal-950 dark:text-teal-400 dark:hover:bg-teal-900"
            >
              <ArrowDownUp className="h-3 w-3" />
              cross-encoder
            </Badge>
          </TooltipTrigger>
          <TooltipContent side="top" className="max-w-72">
            <p className="flex items-center gap-1 text-[11px] font-semibold">
              <ArrowDownUp className="h-3 w-3 text-teal-500" />
              cross-encoder rerank stage
            </p>
            {tips.map((t) => (
              <p key={t} className="mt-0.5 font-mono text-[10px] leading-snug text-muted-foreground">
                {t}
              </p>
            ))}
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    </motion.span>
  );
}

// Rank shift chip for the retrieval playground — makes the cross-encoder
// reordering visible: amber for promotions, muted for demotions (Task 1b/3).
function RankShiftChip({ from, to }: { from: number; to: number }) {
  const changed = from !== to;
  const promoted = changed && from > to;
  const label = changed ? `#${from}→#${to}` : `#${from}`;
  return (
    <TooltipProvider delayDuration={150}>
      <Tooltip>
        <TooltipTrigger asChild>
          <span
            className={cn(
              "inline-flex cursor-help items-center gap-0.5 rounded px-1 py-px font-mono text-[9px]",
              promoted
                ? "bg-amber-500/15 text-amber-700 dark:text-amber-400"
                : "bg-muted text-muted-foreground"
            )}
            aria-label={
              changed
                ? `Cross-encoder ${promoted ? "promoted" : "demoted"} this chunk from fusion rank ${from} to ${to}`
                : `Fusion rank ${from}, unchanged by the rerank stage`
            }
          >
            {promoted ? (
              <TrendingUp className="h-2.5 w-2.5" aria-hidden />
            ) : changed ? (
              <TrendingDown className="h-2.5 w-2.5 opacity-70" aria-hidden />
            ) : null}
            {label}
          </span>
        </TooltipTrigger>
        <TooltipContent side="top">
          <p className="max-w-56 text-[10px]">
            {changed
              ? `cross-encoder ${promoted ? "promoted" : "demoted"} this chunk: fusion rank #${from} → #${to}`
              : `fusion rank #${from} — unchanged by the rerank stage`}
          </p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

// ------------------------------------------------------------- message UI
type ChatMessage =
  | { id: string; role: "user"; text: string }
  | {
      id: string;
      role: "assistant";
      question: string;
      streaming: boolean;
      path: string[];
      streamedAnswer?: string;
      ttftMs?: number;
      data?: AskResponse;
    };

function FaithChip({ f }: { f: EvalCaseResult["faithfulness"] }) {
  if (!f || f.score === null) return null;
  const pct = Math.round(f.score * 100);
  const tone =
    f.score >= 0.8
      ? "border-emerald-300 text-emerald-700 dark:border-emerald-800 dark:text-emerald-400"
      : f.score >= 0.5
        ? "border-amber-300 text-amber-700 dark:border-amber-800 dark:text-amber-400"
        : "border-red-300 text-red-700 dark:border-red-800 dark:text-red-400";
  const tips = [
    `judge: ${f.judge}`,
    `claims: ${f.supported_claims}/${f.total_claims} supported`,
    f.contradictions.length ? `contradictions: ${f.contradictions.join("; ")}` : null,
    f.unsupported.length ? `unsupported: ${f.unsupported.join(" | ")}` : null,
    f.llm_reason ? `reason: ${f.llm_reason}` : null,
  ].filter(Boolean);
  return (
    <TooltipProvider delayDuration={150}>
      <Tooltip>
        <TooltipTrigger asChild>
          <Badge variant="outline" className={cn("shrink-0 cursor-help gap-0.5 font-mono text-[9px]", tone)}>
            <ShieldCheck className="h-2.5 w-2.5" />
            {pct}%
          </Badge>
        </TooltipTrigger>
        <TooltipContent side="left" className="max-w-72">
          <p className="text-[11px] font-semibold">answer faithfulness</p>
          {tips.map((t) => (
            <p key={t} className="mt-0.5 font-mono text-[10px] leading-snug text-muted-foreground">
              {t}
            </p>
          ))}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

// -------------------------------------------- A/B word-level diff (18-b)
// Pure LCS word diff between the two A/B answers. Tokens keep their
// whitespace runs so paragraph breaks survive rendering; consecutive tokens
// on the same side are grouped to keep the DOM small. Cap: if either answer
// exceeds 400 words the O(n·m) DP is skipped and both cards fall back to the
// plain AnswerText rendering (a tooltip explains why).
const DIFF_WORD_CAP = 400;

type DiffSide = "same" | "a" | "b";
type DiffToken = { word: string; side: DiffSide };
type DiffGroup = { text: string; side: DiffSide };
type AnswerDiff = {
  groups: DiffGroup[];
  skipped: boolean; // cap exceeded — render plain
  onlyAWords: number;
  onlyBWords: number;
};

function tokenizeKeepSpace(text: string): string[] {
  return text.match(/\S+|\s+/g) ?? [];
}

function countWords(text: string): number {
  return (text.match(/\S+/g) ?? []).length;
}

// LCS diff over the full token arrays (words + whitespace runs) — each
// returned entry is one token tagged with the side it belongs to.
function diffWords(a: string, b: string): DiffToken[] {
  const ta = tokenizeKeepSpace(a);
  const tb = tokenizeKeepSpace(b);
  const n = ta.length;
  const m = tb.length;
  // dp[i][j] = LCS length of ta[i:] × tb[j:] (bottom-up, Int32 rows).
  const dp: Int32Array[] = Array.from({ length: n + 1 }, () => new Int32Array(m + 1));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      dp[i][j] = ta[i] === tb[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }
  const tokens: DiffToken[] = [];
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (ta[i] === tb[j]) {
      tokens.push({ word: ta[i], side: "same" });
      i++;
      j++;
    } else if (dp[i + 1][j] >= dp[i][j + 1]) {
      tokens.push({ word: ta[i], side: "a" });
      i++;
    } else {
      tokens.push({ word: tb[j], side: "b" });
      j++;
    }
  }
  while (i < n) tokens.push({ word: ta[i++], side: "a" });
  while (j < m) tokens.push({ word: tb[j++], side: "b" });
  return tokens;
}

function groupDiffTokens(tokens: DiffToken[]): DiffGroup[] {
  const groups: DiffGroup[] = [];
  for (const t of tokens) {
    const last = groups[groups.length - 1];
    if (last && last.side === t.side) last.text += t.word;
    else groups.push({ text: t.word, side: t.side });
  }
  return groups;
}

function buildAnswerDiff(a: string, b: string): AnswerDiff {
  if (countWords(a) > DIFF_WORD_CAP || countWords(b) > DIFF_WORD_CAP) {
    return { groups: [], skipped: true, onlyAWords: 0, onlyBWords: 0 };
  }
  const tokens = diffWords(a, b);
  let onlyA = 0;
  let onlyB = 0;
  for (const t of tokens) {
    if (/\S/.test(t.word)) {
      if (t.side === "a") onlyA++;
      else if (t.side === "b") onlyB++;
    }
  }
  return { groups: groupDiffTokens(tokens), skipped: false, onlyAWords: onlyA, onlyBWords: onlyB };
}

// One same-side text group — [Source: doc, page N] markers render as the
// established inline citation chips (mirrors AnswerParagraph styling).
function DiffSameSegment({ text }: { text: string }) {
  const regex = /\[Source:\s*([^,\]]+),\s*page\s*(\d+)\]/g;
  const parts: Array<{ type: "text" | "cite"; value: string; page?: number }> = [];
  let lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = regex.exec(text)) !== null) {
    if (m.index > lastIndex) parts.push({ type: "text", value: text.slice(lastIndex, m.index) });
    parts.push({ type: "cite", value: m[1].trim(), page: Number(m[2]) });
    lastIndex = m.index + m[0].length;
  }
  if (lastIndex < text.length) parts.push({ type: "text", value: text.slice(lastIndex) });
  return (
    <>
      {parts.map((p, i) =>
        p.type === "text" ? (
          <span key={i}>{p.value}</span>
        ) : (
          <TooltipProvider key={i} delayDuration={200}>
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="mx-0.5 inline-flex cursor-default items-center gap-1 rounded-md border border-emerald-200 bg-emerald-50 px-1.5 py-0.5 align-baseline font-mono text-[11px] font-medium text-emerald-700 transition-colors hover:bg-emerald-100 hover:border-emerald-300 dark:border-emerald-800 dark:bg-emerald-950 dark:text-emerald-400 dark:hover:bg-emerald-900 dark:hover:border-emerald-700">
                  <Link2 className="h-3 w-3 shrink-0" />
                  {p.value.length > 26 ? p.value.slice(0, 24) + "…" : p.value}
                  {typeof p.page === "number" && <span className="opacity-70">p.{p.page}</span>}
                </span>
              </TooltipTrigger>
              <TooltipContent side="top" className="max-w-64">
                <p className="font-mono text-[11px]">{p.value} · page {p.page}</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        )
      )}
    </>
  );
}

// Diff-highlighted answer body for the A/B cards — card A renders same+a
// groups (a-only words emerald), card B renders same+b (b-only amber).
function DiffAnswerText({ groups, label }: { groups: DiffGroup[]; label: "A" | "B" }) {
  const own = label === "A" ? "a" : "b";
  const visible = groups.filter((g) => g.side !== (label === "A" ? "b" : "a"));
  return (
    <div className="whitespace-pre-wrap text-sm leading-relaxed" aria-label={`Answer side ${label} with word-level diff highlights`}>
      {visible.map((g, i) =>
        g.side === "same" ? (
          <DiffSameSegment key={i} text={g.text} />
        ) : (
          <TooltipProvider key={i} delayDuration={200}>
            <Tooltip>
              <TooltipTrigger asChild>
                <span
                  className={cn(
                    "cursor-default rounded-sm px-0.5",
                    own === "a"
                      ? "bg-emerald-500/15 text-emerald-900 hover:bg-emerald-500/25 dark:bg-emerald-500/20 dark:text-emerald-100 dark:hover:bg-emerald-500/30"
                      : "bg-amber-500/15 text-amber-900 hover:bg-amber-500/25 dark:bg-amber-500/20 dark:text-amber-100 dark:hover:bg-amber-500/30"
                  )}
                  title={own === "a" ? "only in A — this wording is absent from the B answer" : "only in B — this wording is absent from the A answer"}
                >
                  {g.text}
                </span>
              </TooltipTrigger>
              <TooltipContent side="top" className="max-w-64">
                <p className="text-[10px]">
                  {own === "a"
                    ? "word-level diff: this text appears only in the A answer"
                    : "word-level diff: this text appears only in the B answer"}
                </p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        )
      )}
    </div>
  );
}

function AnswerText({ text }: { text: string }) {
  // Decomposed answers contain "\n\n" between entity blocks — split into
  // separate paragraphs with proper spacing (citation chips parse per paragraph).
  const paragraphs = useMemo(
    () => text.split(/\n{2,}/).map((p) => p.trim()).filter((p) => p.length > 0),
    [text]
  );
  return (
    <div className="space-y-2.5 text-sm leading-relaxed">
      {paragraphs.map((para, pi) => (
        <AnswerParagraph key={pi} text={para} last={pi === paragraphs.length - 1} />
      ))}
    </div>
  );
}

function AnswerParagraph({ text, last }: { text: string; last: boolean }) {
  const parts = useMemo(() => {
    // Render [Source: doc, page N] inline citations as styled chips.
    const regex = /\[Source:\s*([^,\]]+),\s*page\s*(\d+)\]/g;
    const out: Array<{ type: "text" | "cite"; value: string; page?: number }> = [];
    let lastIndex = 0;
    let m: RegExpExecArray | null;
    while ((m = regex.exec(text)) !== null) {
      if (m.index > lastIndex) out.push({ type: "text", value: text.slice(lastIndex, m.index) });
      out.push({ type: "cite", value: m[1].trim(), page: Number(m[2]) });
      lastIndex = m.index + m[0].length;
    }
    if (lastIndex < text.length) out.push({ type: "text", value: text.slice(lastIndex) });
    return out;
  }, [text]);
  return (
    <p className={last ? undefined : "min-w-0"}>
      {parts.map((p, i) =>
        p.type === "text" ? (
          <span key={i}>{p.value}</span>
        ) : (
          <TooltipProvider key={i} delayDuration={200}>
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="mx-0.5 inline-flex cursor-default items-center gap-1 rounded-md border border-emerald-200 bg-emerald-50 px-1.5 py-0.5 align-baseline font-mono text-[11px] font-medium text-emerald-700 transition-colors hover:bg-emerald-100 hover:border-emerald-300 dark:border-emerald-800 dark:bg-emerald-950 dark:text-emerald-400 dark:hover:bg-emerald-900 dark:hover:border-emerald-700">
                  <Link2 className="h-3 w-3 shrink-0" />
                  {p.value.length > 26 ? p.value.slice(0, 24) + "…" : p.value}
                  {typeof p.page === "number" && <span className="opacity-70">p.{p.page}</span>}
                </span>
              </TooltipTrigger>
              <TooltipContent side="top" className="max-w-64">
                <p className="font-mono text-[11px]">{p.value} · page {p.page}</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        )
      )}
    </p>
  );
}

function TraceDetail({ entry }: { entry: TraceEntry }) {
  const detail = entry.detail ?? {};
  const rows = Object.entries(detail).filter(([, v]) => v !== null && v !== undefined);
  if (!rows.length) return null;
  return (
    <div className="space-y-0.5 font-mono text-[11px] leading-relaxed text-muted-foreground">
      {rows.map(([k, v]) => (
        <div key={k} className="flex gap-2 break-all">
          <span className="shrink-0 text-foreground/60">{k}:</span>
          <span>
            {Array.isArray(v)
              ? v.every((x) => typeof x === "string" || typeof x === "number")
                ? v.join(", ")
                : JSON.stringify(v)
              : typeof v === "object"
                ? JSON.stringify(v)
                : String(v)}
          </span>
        </div>
      ))}
    </div>
  );
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <Button
      variant="ghost"
      size="sm"
      className="h-7 gap-1 px-2 text-[11px] text-muted-foreground hover:text-foreground"
      onClick={() => {
        void navigator.clipboard?.writeText(text).then(() => {
          setCopied(true);
          setTimeout(() => setCopied(false), 1600);
        });
      }}
      aria-label="Copy answer"
    >
      {copied ? <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600" /> : <Copy className="h-3.5 w-3.5" />}
      {copied ? "Copied" : "Copy"}
    </Button>
  );
}

const FOLLOWUP_SUGGESTIONS = ["and the extended warranty?", "what about its accuracy?", "what about the maintenance schedule?"];

// Fade-swap value for Pulse stat tiles — animates on stats refresh (Task 3c).
function AnimatedStat({ value }: { value: string }) {
  const reduced = useReducedMotion();
  return (
    <span className="relative inline-flex">
      <AnimatePresence mode="popLayout" initial={false}>
        <motion.span
          key={value}
          initial={reduced ? undefined : { opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          exit={reduced ? undefined : { opacity: 0, y: -4 }}
          transition={{ duration: 0.15, ease: "easeOut" }}
          className="inline-block"
        >
          {value}
        </motion.span>
      </AnimatePresence>
    </span>
  );
}

// Harmonized tab-panel swap: 150ms fade + 2px slide (Task 3h).
function TabPanel({ children }: { children: React.ReactNode }) {
  const reduced = useReducedMotion();
  return (
    <motion.div
      initial={reduced ? undefined : { opacity: 0, y: 2 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.15, ease: "easeOut" }}
    >
      {children}
    </motion.div>
  );
}

function FeedbackButtons({
  requestId,
  onRate,
}: {
  requestId: string;
  onRate: (requestId: string, rating: "up" | "down") => void;
}) {
  const [rated, setRated] = useState<"up" | "down" | null>(null);
  return (
    <span className="flex items-center gap-0.5" aria-label="Rate this answer">
      <Button
        variant="ghost"
        size="icon"
        className={cn(
          "h-6 w-6 rounded-full transition-all",
          rated === "up"
            ? "text-emerald-600"
            : "text-muted-foreground/60 hover:text-emerald-600"
        )}
        onClick={() => {
          setRated("up");
          onRate(requestId, "up");
        }}
        disabled={rated !== null}
        aria-label="Helpful"
        title="Helpful"
      >
        <ThumbsUp className={cn("h-3.5 w-3.5 transition-transform", rated === "up" && "scale-110 fill-emerald-500/30")} />
      </Button>
      <Button
        variant="ghost"
        size="icon"
        className={cn(
          "h-6 w-6 rounded-full transition-all",
          rated === "down"
            ? "text-amber-600"
            : "text-muted-foreground/60 hover:text-amber-600"
        )}
        onClick={() => {
          setRated("down");
          onRate(requestId, "down");
        }}
        disabled={rated !== null}
        aria-label="Not helpful"
        title="Not helpful"
      >
        <ThumbsDown className={cn("h-3.5 w-3.5 transition-transform", rated === "down" && "scale-110 fill-amber-500/30")} />
      </Button>
    </span>
  );
}

function AssistantMessage({
  msg,
  onAsk,
  onOpenSource,
  onRate,
}: {
  msg: Extract<ChatMessage, { role: "assistant" }>;
  onAsk: (q: string) => void;
  onOpenSource: (chunkId: string) => void;
  onRate: (requestId: string, rating: "up" | "down") => void;
}) {
  const [traceOpen, setTraceOpen] = useState(false);
  const done = !msg.streaming && msg.data;
  const data = msg.data;
  const showFollowups = done && data && !data.fallback && data.route === "document";
  // Cross-encoder rerank telemetry from the retriever node trace (Task 1a).
  const rerankInfo = data ? traceRerank(data.trace) : null;
  // Live pipeline phase: deltas stream during generation; once the verifier
  // node event arrives the draft is complete and being checked (Task 1b).
  const lastNode = msg.path.length ? msg.path[msg.path.length - 1] : null;
  const verifying = msg.streaming && lastNode === "verifier";
  const streamedChars = msg.streamedAnswer?.length ?? 0;
  // A streamed draft that the final payload replaced with the honest fallback (Task 1c).
  const draftReplaced =
    done && data?.fallback === true && (msg.streamedAnswer ?? "").trim().length > 0;

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, ease: "easeOut" }}
      className="flex gap-3"
    >
      <div className="relative mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-emerald-500 to-teal-600 text-white shadow-sm">
        <Bot className="h-4 w-4" />
        {msg.streaming && (
          <span className="absolute -right-0.5 -top-0.5 flex h-2.5 w-2.5">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
            <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-emerald-500" />
          </span>
        )}
      </div>
      <Card className="min-w-0 flex-1 border-border/70 bg-card shadow-sm transition-shadow hover:shadow-md">
        <CardContent className="space-y-3 p-4">
          <AgentPathBar
            path={done ? data.agent_path : msg.path.length ? msg.path : ["router"]}
            active={msg.streaming}
            fallback={done ? data.fallback : false}
          />

          {done && data.decomposed && data.sub_queries && data.sub_queries.length > 0 && (
            <SubQueryPills subQueries={data.sub_queries} />
          )}

          {msg.streaming && msg.streamedAnswer ? (
            <div className="rounded-lg border border-border border-l-4 border-l-emerald-500 bg-background p-3">
              <AnswerText text={msg.streamedAnswer} />
              <span
                className="ml-1 inline-block h-3.5 w-1.5 animate-pulse rounded-sm bg-emerald-500 align-text-bottom"
                aria-hidden
              />
              <span className="sr-only">answer is streaming</span>
              <div
                className="mt-1.5 flex items-center gap-2 border-t border-border/60 pt-1.5"
                aria-live="polite"
              >
                {verifying ? (
                  <span className="flex items-center gap-1 text-[10px] font-semibold text-emerald-600 dark:text-emerald-400">
                    <ShieldCheck className="h-3 w-3 animate-pulse" aria-hidden />
                    verifying…
                  </span>
                ) : (
                  <span className="shimmer-text text-[10px] font-semibold tracking-wide">generating…</span>
                )}
                <span className="ml-auto font-mono text-[10px] tabular-nums text-muted-foreground/70">
                  {streamedChars} chars
                </span>
              </div>
            </div>
          ) : done ? (
            <>
              {data.follow_up_applied && (
                <div className="flex flex-wrap items-center gap-1.5 rounded-md border border-violet-200 bg-violet-50/70 px-2.5 py-1.5 text-[11px] text-violet-700 dark:border-violet-900 dark:bg-violet-950/50 dark:text-violet-300">
                  <History className="h-3.5 w-3.5 shrink-0" />
                  <span className="font-medium">Follow-up resolved</span>
                  <span className="text-violet-600/80 dark:text-violet-400/80">
                    inherited {data.inherited_entities.map((e) => `“${e}”`).join(", ")} from the conversation
                  </span>
                  <TooltipProvider delayDuration={150}>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Badge variant="outline" className="cursor-help border-violet-300 font-mono text-[10px] dark:border-violet-800">
                          effective query
                        </Badge>
                      </TooltipTrigger>
                      <TooltipContent side="top" className="max-w-72">
                        <p className="font-mono text-[11px]">{data.effective_question}</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
              )}

              {draftReplaced && (
                <div
                  role="note"
                  className="flex items-center gap-1.5 text-[11px] font-medium text-amber-600 dark:text-amber-400"
                >
                  <AlertTriangle className="h-3.5 w-3.5 shrink-0" aria-hidden />
                  Draft failed verification — replaced by the honest fallback.
                </div>
              )}

              <div
                className={cn(
                  "rounded-lg border-l-4 border p-3 shadow-xs",
                  data.fallback
                    ? "border-l-amber-400 border-amber-200 bg-amber-50/60 text-amber-900 dark:border-l-amber-600 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-200"
                    : data.route === "chitchat"
                      ? "border-l-border border-border bg-muted/50"
                      : "border-l-emerald-500 border-border bg-background"
                )}
              >
                <AnswerText text={data.answer} />
              </div>
              {data.citations.length > 0 && (
                <div className="space-y-2">
                  <p className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
                    <FileText className="h-3 w-3" />
                    Sources ({data.citations.length})
                  </p>
                  <div className="grid gap-1.5 sm:grid-cols-1">
                    {data.citations.map((c, i) => (
                      <TooltipProvider key={c.chunk_id + i} delayDuration={150}>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <button
                              type="button"
                              onClick={() => onOpenSource(c.chunk_id)}
                              className="hover-lift flex w-full cursor-pointer items-start gap-2 rounded-md border border-border/60 bg-muted/40 px-2.5 py-1.5 text-left hover:border-emerald-300 hover:bg-emerald-50/50 hover:shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500/60 focus-visible:ring-offset-1 dark:hover:border-emerald-800 dark:hover:bg-emerald-950/40"
                              aria-label={`View source: ${c.doc} page ${c.page}`}
                            >
                              <span className="mt-0.5 flex h-4.5 w-4.5 shrink-0 items-center justify-center rounded bg-emerald-600/10 font-mono text-[10px] font-bold text-emerald-700 dark:text-emerald-400">
                                {i + 1}
                              </span>
                              <div className="min-w-0 flex-1">
                                <p className="truncate font-mono text-xs font-medium">
                                  {c.doc}
                                  <span className="text-muted-foreground"> · p.{c.page}</span>
                                </p>
                                <p className="line-clamp-2 text-[11px] leading-snug text-muted-foreground">{c.snippet}</p>
                              </div>
                              <div className="flex shrink-0 items-center gap-1.5">
                                <div className="hidden h-1 w-10 overflow-hidden rounded-full bg-muted sm:block" aria-hidden>
                                  <div
                                    className="h-full rounded-full bg-emerald-500"
                                    style={{ width: `${Math.round(Math.min(1, c.score) * 100)}%` }}
                                  />
                                </div>
                                <Badge variant="secondary" className="font-mono text-[10px]">
                                  {c.score.toFixed(2)}
                                </Badge>
                              </div>
                            </button>
                          </TooltipTrigger>
                          <TooltipContent side="left" className="max-w-80">
                            <p className="font-mono text-[10px] text-muted-foreground">{c.chunk_id}</p>
                            <p className="mt-1 text-[11px] leading-relaxed">{c.snippet}</p>
                            <p className="mt-1 text-[10px] italic text-muted-foreground">Click to view full source chunk</p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    ))}
                  </div>
                </div>
              )}

              <div className="flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground">
                <RouteBadge route={data.route} />
                {data.decomposed && data.sub_queries && data.sub_queries.length > 0 && (
                  <DecomposedChip count={data.sub_queries.length} />
                )}
                {data.cache_hit && <CacheChip detail={data.cache_detail} />}
                {rerankInfo && data && <RerankChip info={rerankInfo} servingMode={data.serving_mode} />}
                {data.retries > 0 && (
                  <Badge variant="outline" className="gap-1 border-orange-300 text-orange-700 dark:border-orange-800 dark:text-orange-400">
                    <RefreshCw className="h-3 w-3" />
                    rewrites: {data.retries}
                  </Badge>
                )}
                <TooltipProvider delayDuration={150}>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Badge
                        variant="outline"
                        className={cn(
                          "gap-1 font-mono",
                          data.latency_ms < ROUTE_MAX["document"]
                            ? "border-emerald-300 text-emerald-700 dark:border-emerald-800 dark:text-emerald-400"
                            : "border-amber-300 text-amber-700 dark:border-amber-800 dark:text-amber-400"
                        )}
                      >
                        <Clock className="h-3 w-3" />
                        {data.latency_ms.toFixed(0)} ms
                      </Badge>
                    </TooltipTrigger>
                    <TooltipContent side="top">
                      <p className="font-mono text-[11px]">end-to-end agent latency</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                {typeof msg.ttftMs === "number" && (
                  <TooltipProvider delayDuration={150}>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Badge
                          variant="outline"
                          className="cursor-help gap-1 border-teal-300 font-mono text-teal-700 dark:border-teal-800 dark:text-teal-400"
                        >
                          <Gauge className="h-3 w-3" />
                          TTFT {msg.ttftMs.toFixed(1)}ms
                        </Badge>
                      </TooltipTrigger>
                      <TooltipContent side="top">
                        <p className="font-mono text-[11px]">
                          time to first token — when the first generated token reached the client
                        </p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                )}
                <Badge variant="outline" className="font-mono">
                  {data.serving_mode}
                </Badge>
                <span className="ml-auto flex items-center gap-1">
                  <TooltipProvider delayDuration={150}>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-7 gap-1 px-2 text-[11px] text-muted-foreground hover:text-foreground"
                          onClick={() => onAsk(msg.question)}
                          aria-label="Regenerate answer"
                        >
                          <RotateCcw className="h-3.5 w-3.5" />
                          Retry
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent side="top">
                        <p className="text-[11px]">Re-run the agent graph on this question</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                  {data.request_id && (
                    <FeedbackButtons requestId={data.request_id} onRate={onRate} />
                  )}
                  <CopyButton text={data.answer} />
                </span>
              </div>

              <Collapsible open={traceOpen} onOpenChange={setTraceOpen}>
                <CollapsibleTrigger asChild>
                  <Button variant="ghost" size="sm" className="h-7 gap-1 px-2 text-[11px] text-muted-foreground">
                    {traceOpen ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
                    Execution trace ({data.trace.length} steps)
                  </Button>
                </CollapsibleTrigger>
                <CollapsibleContent>
                  <div className="thin-scrollbar max-h-72 overflow-y-auto rounded-md border border-border/60 bg-muted/30 p-2">
                    <div className="relative space-y-2 before:absolute before:left-[7px] before:top-2 before:bottom-2 before:w-px before:bg-border">
                      {data.trace.map((t, i) => (
                        <div key={i} className="relative space-y-1 rounded-md bg-background/60 p-2 pl-4">
                          <span
                            className={cn(
                              "absolute left-[3px] top-3 h-2 w-2 rounded-full border border-background ring-2 ring-background/60",
                              t.status === "ok" && "bg-emerald-500",
                              t.status === "error" && "bg-red-500",
                              (t.status === "fallback" || t.status === "retry") && "bg-amber-500",
                              t.status === "skipped" && "bg-muted-foreground/40"
                            )}
                            aria-hidden
                          />
                          <div className="flex items-center justify-between gap-2">
                            <span className="font-mono text-[11px] font-semibold">
                              {t.sequence}. {t.node}
                              <span
                                className={cn(
                                  "ml-2 font-normal",
                                  t.status === "ok" && "text-emerald-600",
                                  t.status === "error" && "text-red-600",
                                  (t.status === "fallback" || t.status === "retry") && "text-amber-600"
                                )}
                              >
                                {t.status}
                              </span>
                            </span>
                            <span className="font-mono text-[10px] text-muted-foreground">
                              {t.latency_ms?.toFixed(1)} ms
                            </span>
                          </div>
                          <TraceDetail entry={t} />
                        </div>
                      ))}
                    </div>
                  </div>
                </CollapsibleContent>
              </Collapsible>

              {showFollowups && (
                <div className="flex flex-wrap items-center gap-1.5 pt-1">
                  <span className="text-[10px] uppercase tracking-wide text-muted-foreground/70">Try a follow-up</span>
                  {FOLLOWUP_SUGGESTIONS.map((q) => (
                    <button
                      key={q}
                      onClick={() => onAsk(q)}
                      className="inline-flex items-center gap-1 rounded-full border border-violet-200 bg-violet-50/60 px-2.5 py-1 text-[11px] text-violet-700 transition-all hover:border-violet-400 hover:bg-violet-100 hover:shadow-xs focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500/60 focus-visible:ring-offset-1 dark:border-violet-900 dark:bg-violet-950/50 dark:text-violet-300 dark:hover:bg-violet-900"
                    >
                      <History className="h-3 w-3" />
                      {q}
                    </button>
                  ))}
                </div>
              )}
            </>
          ) : (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin text-emerald-600" />
              Agents working…
            </div>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );
}

// ----------------------------------------------------------------- score ring
function ThemeToggle() {
  const { setTheme } = useTheme();
  // Icons are swapped purely via CSS (class-based dark mode): no hydration
  // mismatch, no mounted-state effect.
  return (
    <Button
      variant="ghost"
      size="icon"
      className="h-8 w-8"
      onClick={() => {
        const isDark = document.documentElement.classList.contains("dark");
        setTheme(isDark ? "light" : "dark");
      }}
      aria-label="Toggle color theme"
      title="Toggle light / dark theme"
    >
      <Moon className="h-4 w-4 transition-transform duration-300 dark:hidden" />
      <Sun className="hidden h-4 w-4 transition-transform duration-300 dark:block" />
    </Button>
  );
}

function ScoreRing({ value, size = 88 }: { value: number; size?: number }) {
  const r = (size - 12) / 2;
  const c = 2 * Math.PI * r;
  const pct = Math.max(0, Math.min(1, value));
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="shrink-0" role="img" aria-label={`accuracy ${Math.round(pct * 100)} percent`}>
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" strokeWidth={9} className="stroke-muted" />
      <motion.circle
        cx={size / 2}
        cy={size / 2}
        r={r}
        fill="none"
        strokeWidth={9}
        strokeLinecap="round"
        className="stroke-emerald-500"
        strokeDasharray={c}
        initial={{ strokeDashoffset: c }}
        animate={{ strokeDashoffset: c * (1 - pct) }}
        transition={{ duration: 0.9, ease: "easeOut" }}
        transform={`rotate(-90 ${size / 2} ${size / 2})`}
      />
      <text x="50%" y="47%" textAnchor="middle" dominantBaseline="central" className="fill-foreground font-mono text-[15px] font-bold">
        {Math.round(pct * 100)}%
      </text>
      <text x="50%" y="63%" textAnchor="middle" dominantBaseline="central" className="fill-muted-foreground text-[8px] uppercase tracking-wider">
        accuracy
      </text>
    </svg>
  );
}

function ablationConfigSummary(config: Record<string, unknown>): string {
  if (config.rerank === false) return "cross-encoder off";
  if (typeof config.rerank_n === "number") return `rerank pool n=${config.rerank_n}`;
  if (config.top_k !== undefined) return `top-k ${String(config.top_k)}`;
  const w = config.weights as { w_fused?: number; w_coverage?: number; w_agreement?: number } | undefined;
  if (w) return `w ${w.w_fused ?? "—"}/${w.w_coverage ?? "—"}/${w.w_agreement ?? "—"}`;
  if (Object.keys(config).length === 0) return "default";
  return JSON.stringify(config);
}

function AblationTable({ report }: { report: AblationReport }) {
  return (
    <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} className="space-y-2">
      <div className="flex flex-wrap items-center justify-between gap-1.5">
        <p className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
          <FlaskConical className="h-3 w-3" />
          retriever ablation
        </p>
        <Badge
          variant="outline"
          className="gap-1 border-emerald-300 bg-emerald-50 text-[9.5px] text-emerald-700 dark:border-emerald-800 dark:bg-emerald-950 dark:text-emerald-400"
        >
          <Trophy className="h-2.5 w-2.5" />
          {report.winner}
        </Badge>
      </div>
      <div className="overflow-hidden rounded-md border">
        <Table>
          <TableHeader>
            <TableRow className="hover:bg-transparent">
              <TableHead className="h-8 px-2 text-[10px] uppercase tracking-wide">config</TableHead>
              <TableHead className="h-8 px-2 text-[10px] uppercase tracking-wide text-right">accuracy</TableHead>
              <TableHead className="h-8 px-2 text-[10px] uppercase tracking-wide text-right">hit@k</TableHead>
              <TableHead className="h-8 px-2 text-[10px] uppercase tracking-wide text-right">fact</TableHead>
              <TableHead className="h-8 px-2 text-[10px] uppercase tracking-wide text-right">p50</TableHead>
              <TableHead className="h-8 px-2 text-[10px] uppercase tracking-wide text-right">Δ vs base</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {report.configs.map((c) => {
              const isWinner = c.name === report.winner;
              const isBaseline = c.name === report.baseline;
              const delta = c.delta_accuracy;
              return (
                <TableRow
                  key={c.name}
                  className={cn(
                    isWinner &&
                      "bg-emerald-50/70 ring-1 ring-inset ring-emerald-400/70 hover:bg-emerald-100/60 dark:bg-emerald-950/40 dark:ring-emerald-700/70 dark:hover:bg-emerald-900/40"
                  )}
                >
                  <TableCell className="max-w-40 p-2">
                    <p className="flex items-center gap-1 truncate text-[11px] font-medium">
                      {isWinner && <Trophy className="h-3 w-3 shrink-0 text-emerald-600 dark:text-emerald-400" aria-label="winner config" />}
                      {c.name}
                      {isBaseline && (
                        <Badge variant="secondary" className="h-4 px-1 text-[8px] uppercase">
                          base
                        </Badge>
                      )}
                    </p>
                    <p className="truncate font-mono text-[9px] text-muted-foreground">{ablationConfigSummary(c.config)}</p>
                  </TableCell>
                  <TableCell className="p-2 text-right">
                    <span className={cn("font-mono text-[11px] font-semibold", c.passed === c.total ? "text-emerald-600 dark:text-emerald-400" : "text-amber-600 dark:text-amber-400")}>
                      {(c.accuracy * 100).toFixed(0)}%
                    </span>
                    <p className="font-mono text-[9px] text-muted-foreground">{c.passed}/{c.total}</p>
                  </TableCell>
                  <TableCell className="p-2 text-right font-mono text-[11px]">
                    {c.retrieval_hit_rate !== null ? `${(c.retrieval_hit_rate * 100).toFixed(0)}%` : "—"}
                  </TableCell>
                  <TableCell className="p-2 text-right font-mono text-[11px]">
                    {(c.fact_support_rate * 100).toFixed(0)}%
                  </TableCell>
                  <TableCell className="p-2 text-right font-mono text-[11px] text-muted-foreground">
                    {c.latency_p50_ms.toFixed(1)}ms
                  </TableCell>
                  <TableCell className="p-2 text-right">
                    <span
                      className={cn(
                        "inline-flex items-center gap-0.5 rounded px-1 py-0.5 font-mono text-[10px] font-semibold",
                        delta > 0
                          ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/60 dark:text-emerald-400"
                          : delta < 0
                            ? "bg-red-100 text-red-700 dark:bg-red-900/60 dark:text-red-400"
                            : "bg-muted text-muted-foreground"
                      )}
                    >
                      {delta > 0 ? "+" : ""}
                      {(delta * 100).toFixed(1)}
                    </span>
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </div>
      {report.notes.length > 0 && (
        <ul className="space-y-0.5">
          {report.notes.map((n) => (
            <li key={n} className="flex items-start gap-1.5 text-[10px] leading-snug text-muted-foreground">
              <span className="mt-px shrink-0 opacity-60">·</span>
              {n}
            </li>
          ))}
        </ul>
      )}
      <p className="text-right font-mono text-[9px] text-muted-foreground">
        ran {new Date(report.ran_at).toLocaleTimeString()} · {(report.duration_ms / 1000).toFixed(1)}s
      </p>
    </motion.div>
  );
}

// ------------------------------------------- golden-set growth components
// Task 17-b: candidates review UI, golden cases management, and the A/B
// answer-quality comparison — all reusing the established chip/tooltip/
// framer-motion conventions (teal/amber accents, dark: variants, 44px
// targets on primary buttons, thin-scrollbar lists).

// Source badge for golden cases — muted "base" (immutable) vs teal "user".
function SourceBadge({ source }: { source: "base" | "user" }) {
  const isUser = source === "user";
  return (
    <TooltipProvider delayDuration={150}>
      <Tooltip>
        <TooltipTrigger asChild>
          <Badge
            variant="outline"
            className={cn(
              "cursor-help gap-0.5 px-1 text-[8.5px] uppercase tracking-wide transition-colors",
              isUser
                ? "border-teal-300 bg-teal-50 text-teal-700 hover:bg-teal-100 dark:border-teal-800 dark:bg-teal-950 dark:text-teal-400 dark:hover:bg-teal-900"
                : "border-border bg-muted/60 text-muted-foreground hover:bg-muted"
            )}
            aria-label={isUser ? "User-grown case — deletable" : "Base case — immutable"}
          >
            {isUser ? <Sprout className="h-2.5 w-2.5" /> : <Lock className="h-2.5 w-2.5" />}
            {source}
          </Badge>
        </TooltipTrigger>
        <TooltipContent side="top">
          <p className="max-w-52 text-[10px]">
            {isUser
              ? "grown from 👍 feedback — verify, then delete if it drifts"
              : "base cases are immutable — shipped with the eval suite"}
          </p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

// Monospace fact-pattern chip — the regex a promoted answer must match.
function KeywordChip({ kw }: { kw: string }) {
  return (
    <TooltipProvider delayDuration={150}>
      <Tooltip>
        <TooltipTrigger asChild>
          <span
            className="inline-flex max-w-36 cursor-help items-center rounded border border-teal-200 bg-teal-50/70 px-1.5 py-px font-mono text-[9.5px] text-teal-700 transition-colors hover:border-teal-400 hover:bg-teal-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500/60 dark:border-teal-900 dark:bg-teal-950/50 dark:text-teal-300 dark:hover:border-teal-700 dark:hover:bg-teal-900"
            aria-label={`fact pattern ${kw}`}
          >
            <span className="truncate">{kw}</span>
          </span>
        </TooltipTrigger>
        <TooltipContent side="top" className="max-w-64">
          <p className="text-[10px]">regex fact pattern the answer must match</p>
          <p className="mt-0.5 break-all font-mono text-[10px] text-muted-foreground">{kw}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

function formatRatedAt(ts: number): string {
  const d = new Date(ts * 1000);
  return Number.isNaN(d.getTime()) ? "—" : d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

// Round 20 SLA — compact human age for hours ("48m", "3.2h", "2.1d", "1.4w").
function formatAgeHours(hours: number): string {
  if (!Number.isFinite(hours) || hours < 0) return "—";
  if (hours < 1) return `${Math.max(1, Math.round(hours * 60))}m`;
  if (hours < 48) return `${hours.toFixed(hours < 10 ? 1 : 0)}h`;
  if (hours < 24 * 14) return `${(hours / 24).toFixed(1)}d`;
  return `${(hours / (24 * 7)).toFixed(1)}w`;
}

// SLA escalation tone for an age in hours: neutral → amber → rose.
function ageTone(hours: number): "muted" | "amber" | "rose" {
  if (hours >= 48) return "rose";
  if (hours >= 6) return "amber";
  return "muted";
}

const AGE_TONE_CLASSES: Record<ReturnType<typeof ageTone>, string> = {
  muted: "border-border/70 bg-muted/40 text-muted-foreground dark:text-muted-foreground",
  amber: "border-amber-300 bg-amber-50 text-amber-700 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-400",
  rose: "border-rose-300 bg-rose-50 text-rose-700 dark:border-rose-800 dark:bg-rose-950 dark:text-rose-400",
};

// One upvoted-answer candidate: preview + editable fact patterns + promote.
// Task 19-b: checkbox in the header row selects it for batch promotion.
function CandidateCard({
  candidate,
  keywords,
  onKeywordsChange,
  onPromote,
  promoting,
  selected,
  onToggleSelect,
  selectDisabled,
}: {
  candidate: EvalCandidate;
  keywords: string;
  onKeywordsChange: (value: string) => void;
  onPromote: () => void;
  promoting: boolean;
  selected: boolean;
  onToggleSelect: () => void;
  selectDisabled: boolean;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      className={cn(
        "hover-lift space-y-2 rounded-lg border border-border/70 bg-muted/20 p-3 transition-colors hover:border-teal-300/60 dark:hover:border-teal-800/60",
        selected &&
          "border-teal-400/60 bg-teal-50/30 ring-1 ring-teal-400/50 dark:border-teal-700/60 dark:bg-teal-950/30 dark:ring-teal-700/50"
      )}
    >
      <div className="flex items-start gap-2">
        <TooltipProvider delayDuration={200}>
          <Tooltip>
            <TooltipTrigger asChild>
              <Checkbox
                checked={selected}
                onCheckedChange={() => onToggleSelect()}
                disabled={selectDisabled || promoting}
                className="mt-0.5 data-[state=checked]:border-teal-600 data-[state=checked]:bg-teal-600 focus-visible:ring-teal-500/40 dark:data-[state=checked]:border-teal-500 dark:data-[state=checked]:bg-teal-500"
                aria-label={`Select “${candidate.question}” for batch promotion`}
              />
            </TooltipTrigger>
            <TooltipContent side="top" className="max-w-56">
              <p className="text-[10px] leading-relaxed">
                tick to include this candidate in the batch promotion below — up to 12 per run
              </p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
        <p className="min-w-0 flex-1 text-xs font-medium leading-snug">{candidate.question}</p>
      </div>
      <p className="line-clamp-3 text-[11px] leading-relaxed text-muted-foreground">{candidate.answer_preview}</p>

      <div className="flex flex-wrap items-center gap-1.5">
        <TooltipProvider delayDuration={150}>
          <Tooltip>
            <TooltipTrigger asChild>
              <span
                className="inline-flex cursor-help items-center gap-1 rounded-full border border-emerald-300 bg-emerald-50 px-2 py-0.5 text-[10px] font-medium text-emerald-700 transition-colors hover:bg-emerald-100 dark:border-emerald-800 dark:bg-emerald-950 dark:text-emerald-400 dark:hover:bg-emerald-900"
                aria-label={`Upvoted at ${formatRatedAt(candidate.rated_at)}`}
              >
                <ThumbsUp className="h-3 w-3" aria-hidden />
                upvoted
              </span>
            </TooltipTrigger>
            <TooltipContent side="top" className="max-w-64">
              <p className="text-[10px]">
                👍 rated {formatRatedAt(candidate.rated_at)}
                {candidate.feedback.comment ? ` — “${candidate.feedback.comment}”` : ""}
              </p>
              <p className="mt-0.5 font-mono text-[10px] text-muted-foreground">
                {candidate.route} · {candidate.latency_ms.toFixed(0)} ms{candidate.cache_hit ? " · served from cache" : ""}
              </p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
        {candidate.citations.slice(0, 3).map((c, i) => (
          <Badge key={`${c.doc}-${c.page}-${i}`} variant="secondary" className="max-w-40 truncate px-1.5 font-mono text-[9px]">
            {c.doc} · p.{c.page}
          </Badge>
        ))}
        {candidate.citations.length > 3 && (
          <Badge variant="outline" className="px-1.5 font-mono text-[9px] text-muted-foreground">
            +{candidate.citations.length - 3}
          </Badge>
        )}
      </div>

      <div className="space-y-1">
        <label
          htmlFor={`kw-${candidate.request_id}`}
          className="flex cursor-help items-center gap-1 text-[10px] uppercase tracking-wide text-muted-foreground"
          title="Comma-separated regex fact patterns — mined from the answer, editable before promoting"
        >
          fact patterns
          <span className="font-mono normal-case opacity-70">(editable)</span>
        </label>
        <Input
          id={`kw-${candidate.request_id}`}
          value={keywords}
          onChange={(e) => onKeywordsChange(e.target.value)}
          disabled={promoting}
          className="h-8 border-teal-200/60 font-mono text-[10.5px] focus-visible:ring-teal-500/40 dark:border-teal-900/60"
          placeholder="no patterns — backend will re-suggest"
          aria-label={`Editable fact patterns for: ${candidate.question}`}
        />
        <p className="truncate font-mono text-[9px] text-muted-foreground/80">
          docs: {candidate.suggested_docs.length > 0 ? candidate.suggested_docs.join(", ") : "—"}
        </p>
      </div>

      <TooltipProvider delayDuration={200}>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              size="sm"
              className="h-11 w-full gap-1.5 bg-teal-600 text-xs font-medium text-white shadow-sm transition-all hover:bg-teal-700 hover:shadow active:scale-[0.98] focus-visible:ring-teal-500/60 dark:bg-teal-500 dark:hover:bg-teal-400 dark:text-teal-950"
              onClick={onPromote}
              disabled={promoting}
              aria-label={`Promote “${candidate.question}” to the golden set`}
            >
              {promoting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
              {promoting ? "Verifying…" : "Promote to golden set"}
            </Button>
          </TooltipTrigger>
          <TooltipContent side="top" className="max-w-64">
            <p className="text-[10px] leading-relaxed">
              promote this Q&amp;A as a user golden case — the backend immediately re-runs it
              through the agent graph and returns a verification verdict
            </p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    </motion.div>
  );
}

// Compact golden-cases list — source badges, keywords, delete for user cases.
function GoldenCasesList({
  data,
  loading,
  onDelete,
  deletingId,
}: {
  data: CasesResponse | null;
  loading: boolean;
  onDelete: (caseId: string) => void;
  deletingId: string | null;
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex flex-wrap items-center justify-between gap-1.5">
        <p className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
          <ClipboardCheck className="h-3 w-3 text-teal-600 dark:text-teal-500" aria-hidden />
          golden cases
        </p>
        {data && (
          <TooltipProvider delayDuration={150}>
            <Tooltip>
              <TooltipTrigger asChild>
                <span
                  className="cursor-help rounded bg-teal-500/10 px-1.5 py-px font-mono text-[9px] normal-case text-teal-700 dark:text-teal-400"
                  aria-label={`${data.stats.total} cases: ${data.stats.base} base + ${data.stats.user} user`}
                >
                  {data.stats.total} cases · {data.stats.base} base + {data.stats.user} user
                </span>
              </TooltipTrigger>
              <TooltipContent side="top">
                <p className="max-w-56 text-[10px]">
                  POST /eval runs this merged set — promoted 👍 cases grow the suite honestly.
                </p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        )}
      </div>
      {loading && !data ? (
        <div className="space-y-1.5" aria-label="Loading golden cases">
          {[0, 1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-8 w-full" />
          ))}
        </div>
      ) : data && data.cases.length > 0 ? (
        <div className="thin-scrollbar max-h-64 overflow-y-auto rounded-md border">
          <div className="divide-y divide-border/60">
            {data.cases.map((c) => (
              <div
                key={c.id}
                className="hover-lift flex items-center gap-2 px-2.5 py-2 transition-colors hover:bg-muted/50"
              >
                <div className="min-w-0 flex-1">
                  <p className="truncate text-[11px] font-medium">{c.question}</p>
                  <div className="mt-0.5 flex flex-wrap items-center gap-1">
                    {(c.keywords ?? []).slice(0, 2).map((kw) => (
                      <KeywordChip key={kw} kw={kw} />
                    ))}
                    {(c.keywords ?? []).length > 2 && (
                      <TooltipProvider delayDuration={150}>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <span
                              className="cursor-help rounded bg-muted px-1 py-px font-mono text-[9px] text-muted-foreground"
                              aria-label={`${(c.keywords ?? []).length - 2} more fact patterns`}
                            >
                              +{(c.keywords ?? []).length - 2}
                            </span>
                          </TooltipTrigger>
                          <TooltipContent side="top" className="max-w-64">
                            {(c.keywords ?? []).slice(2).map((kw) => (
                              <p key={kw} className="break-all font-mono text-[10px] text-muted-foreground">
                                {kw}
                              </p>
                            ))}
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    )}
                  </div>
                </div>
                <div className="flex shrink-0 flex-col items-end gap-1">
                  <div className="flex items-center gap-1">
                    <Badge variant="outline" className={cn("px-1.5 text-[8.5px]", CATEGORY_TONES[c.category] ?? "")}>
                      {c.category}
                    </Badge>
                    <Badge
                      variant="outline"
                      className={cn(
                        "px-1.5 text-[8.5px]",
                        c.expects === "grounded"
                          ? "border-emerald-300 text-emerald-700 dark:border-emerald-800 dark:text-emerald-400"
                          : c.expects === "fallback"
                            ? "border-amber-300 text-amber-700 dark:border-amber-800 dark:text-amber-400"
                            : "border-rose-300 text-rose-700 dark:border-rose-800 dark:text-rose-400"
                      )}
                    >
                      {c.expects}
                    </Badge>
                    <SourceBadge source={c.source} />
                  </div>
                </div>
                {c.deletable ? (
                  <TooltipProvider delayDuration={150}>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-9 w-9 shrink-0 rounded-md text-muted-foreground/70 transition-colors hover:bg-red-50 hover:text-red-600 focus-visible:ring-red-500/40 dark:hover:bg-red-950/40 dark:hover:text-red-400"
                          onClick={() => onDelete(c.id)}
                          disabled={deletingId === c.id}
                          aria-label={`Delete user golden case ${c.id}`}
                        >
                          {deletingId === c.id ? <Loader2 className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />}
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent side="left">
                        <p className="max-w-52 text-[10px]">
                          delete this user-grown case — the eval set returns to {data ? `${data.stats.base} base cases` : "base cases"}
                        </p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                ) : (
                  <span className="h-9 w-9 shrink-0" aria-hidden />
                )}
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="flex flex-col items-center gap-1.5 rounded-xl border border-dashed border-border bg-muted/20 px-3 py-6 text-center">
          <ClipboardCheck className="h-5 w-5 text-muted-foreground/50" aria-hidden />
          <p className="max-w-xs text-[11px] leading-relaxed text-muted-foreground">
            No golden cases loaded yet — refresh to fetch the merged set.
          </p>
        </div>
      )}
    </div>
  );
}

// One side of the A/B comparison — A = teal tint, B = amber tint. When a word
// diff is available (Task 18-b) the answer body renders highlighted groups
// instead of the plain AnswerText.
function CompareSideCard({ side, label, diff }: { side: CompareSide; label: "A" | "B"; diff: AnswerDiff | null }) {
  const isA = label === "A";
  return (
    <div
      className={cn(
        "space-y-2 rounded-lg border p-3 transition-colors",
        isA
          ? "border-teal-300/60 bg-teal-50/30 hover:border-teal-400/60 dark:border-teal-800/60 dark:bg-teal-950/30 dark:hover:border-teal-700/60"
          : "border-amber-300/60 bg-amber-50/30 hover:border-amber-400/60 dark:border-amber-800/60 dark:bg-amber-950/30 dark:hover:border-amber-700/60"
      )}
    >
      <div className="flex items-center gap-1.5">
        <span
          className={cn(
            "inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-md font-mono text-[10px] font-bold",
            isA
              ? "bg-teal-600 text-white dark:bg-teal-500 dark:text-teal-950"
              : "bg-amber-500 text-white dark:bg-amber-400 dark:text-amber-950"
          )}
          aria-label={`Config ${label}`}
        >
          {label}
        </span>
        <TooltipProvider delayDuration={150}>
          <Tooltip>
            <TooltipTrigger asChild>
              <p className="cursor-help truncate font-mono text-[10px] text-muted-foreground" title={side.config}>
                {side.config}
              </p>
            </TooltipTrigger>
            <TooltipContent side="top" className="max-w-64">
              <p className="text-[10px] leading-relaxed">
                one full agent-graph run under this ablation config — cache-free, so both sides
                pay the honest retrieval cost
              </p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>

      <div className="rounded-md border-l-4 border border-border bg-background/70 p-2.5 text-[11px] leading-relaxed">
        {diff && !diff.skipped ? (
          <DiffAnswerText groups={diff.groups} label={label} />
        ) : (
          <AnswerText text={side.answer} />
        )}
      </div>

      <div className="flex flex-wrap gap-1">
        {side.citations.slice(0, 4).map((c, i) => (
          <Badge key={`${c.doc}-${c.page}-${i}`} variant="secondary" className="max-w-44 truncate px-1.5 font-mono text-[9px]">
            {c.doc} · p.{c.page}
          </Badge>
        ))}
        {side.citations.length > 4 && (
          <Badge variant="outline" className="px-1.5 font-mono text-[9px] text-muted-foreground">
            +{side.citations.length - 4}
          </Badge>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-1.5 font-mono text-[9.5px] text-muted-foreground">
        <span className={cn("rounded px-1 py-px", isA ? "bg-teal-500/10 text-teal-700 dark:text-teal-400" : "bg-amber-500/10 text-amber-700 dark:text-amber-400")}>
          {side.latency_ms.toFixed(1)} ms
        </span>
        <span>{side.faithfulness && side.faithfulness.score !== null ? `faith ${(side.faithfulness.score * 100).toFixed(0)}%` : "faith —"}</span>
        <span>{side.route}</span>
        {side.rerank && (
          <TooltipProvider delayDuration={150}>
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="cursor-help rounded bg-muted px-1 py-px" aria-label={`rerank: ${side.rerank.reranked} candidates, ${side.rerank.reorders} reorders`}>
                  rerank {side.rerank.reorders}/{side.rerank.reranked}
                </span>
              </TooltipTrigger>
              <TooltipContent side="top">
                <p className="font-mono text-[10px]">
                  cross-encoder reordered {side.rerank.reorders} of {side.rerank.reranked} candidates in{" "}
                  {side.rerank.latency_ms.toFixed(1)} ms
                </p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        )}
        {side.fallback && (
          <Badge variant="outline" className="gap-0.5 border-amber-300 px-1 text-[8.5px] text-amber-700 dark:border-amber-800 dark:text-amber-400">
            <AlertTriangle className="h-2.5 w-2.5" aria-hidden />
            fallback
          </Badge>
        )}
        {side.decomposed && (
          <Badge variant="outline" className="gap-0.5 border-teal-300 px-1 text-[8.5px] text-teal-700 dark:border-teal-800 dark:text-teal-400">
            <Split className="h-2.5 w-2.5" aria-hidden />
            decomposed
          </Badge>
        )}
      </div>
    </div>
  );
}

// Verdict strip — identical/differ, overlap, deltas, doc-only chips.
function VerdictStrip({ verdict }: { verdict: CompareVerdict }) {
  const overlapPct = Math.round(verdict.citation_overlap * 100);
  const lat = verdict.latency_delta_ms;
  const latFasterB = lat < 0;
  const faith = verdict.faithfulness_delta;
  return (
    <motion.div
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      role="status"
      aria-label="A/B comparison verdict"
      className="space-y-1.5 rounded-lg border border-border/70 bg-muted/30 p-2.5"
    >
      <div className="flex flex-wrap items-center gap-1.5">
        <TooltipProvider delayDuration={150}>
          <Tooltip>
            <TooltipTrigger asChild>
              <Badge
                variant="outline"
                className={cn(
                  "cursor-help gap-1",
                  verdict.answers_identical
                    ? "border-emerald-300 bg-emerald-50 text-emerald-700 dark:border-emerald-800 dark:bg-emerald-950 dark:text-emerald-400"
                    : "border-amber-300 bg-amber-50 text-amber-700 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-400"
                )}
              >
                {verdict.answers_identical ? <CheckCircle2 className="h-3 w-3" /> : <GitCompare className="h-3 w-3" />}
                {verdict.answers_identical ? "Identical answers" : "Answers differ"}
              </Badge>
            </TooltipTrigger>
            <TooltipContent side="top">
              <p className="max-w-56 text-[10px]">
                normalized text comparison — {verdict.answer_a_chars} chars (A) vs {verdict.answer_b_chars} chars (B)
              </p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
        <span className="font-mono text-[9px] text-muted-foreground">
          {verdict.answer_a_chars} vs {verdict.answer_b_chars} chars
        </span>
      </div>

      <div className="grid grid-cols-3 gap-1.5 text-center">
        <TooltipProvider delayDuration={150}>
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="cursor-help rounded-md bg-background/70 px-1 py-1.5 dark:bg-background/50" aria-label={`citation overlap ${overlapPct}%`}>
                <p className="font-mono text-[11px] font-semibold">{overlapPct}%</p>
                <p className="text-[8.5px] uppercase tracking-wide text-muted-foreground">citation overlap</p>
              </div>
            </TooltipTrigger>
            <TooltipContent side="top">
              <p className="max-w-56 text-[10px]">Jaccard overlap of the two citation doc sets</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
        <TooltipProvider delayDuration={150}>
          <Tooltip>
            <TooltipTrigger asChild>
              <div
                className="cursor-help rounded-md bg-background/70 px-1 py-1.5 dark:bg-background/50"
                aria-label={`latency delta ${lat.toFixed(1)} ms — ${latFasterB ? "B faster" : lat > 0 ? "B slower" : "equal"}`}
              >
                <p
                  className={cn(
                    "font-mono text-[11px] font-semibold",
                    lat === 0
                      ? "text-muted-foreground"
                      : latFasterB
                        ? "text-emerald-600 dark:text-emerald-400"
                        : "text-amber-600 dark:text-amber-400"
                  )}
                >
                  {lat > 0 ? "+" : ""}
                  {lat.toFixed(1)} ms
                </p>
                <p className="text-[8.5px] uppercase tracking-wide text-muted-foreground">latency B−A</p>
              </div>
            </TooltipTrigger>
            <TooltipContent side="top">
              <p className="max-w-56 text-[10px]">
                {lat === 0
                  ? "both sides took the same time"
                  : latFasterB
                    ? `B is ${Math.abs(lat).toFixed(1)} ms faster than A`
                    : `B is ${lat.toFixed(1)} ms slower than A`}
              </p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
        <TooltipProvider delayDuration={150}>
          <Tooltip>
            <TooltipTrigger asChild>
              <div
                className="cursor-help rounded-md bg-background/70 px-1 py-1.5 dark:bg-background/50"
                aria-label={`faithfulness delta ${faith !== null ? faith.toFixed(3) : "—"}`}
              >
                <p
                  className={cn(
                    "font-mono text-[11px] font-semibold",
                    faith === null || faith === 0
                      ? "text-muted-foreground"
                      : faith > 0
                        ? "text-emerald-600 dark:text-emerald-400"
                        : "text-amber-600 dark:text-amber-400"
                  )}
                >
                  {faith === null ? "—" : `${faith > 0 ? "+" : ""}${faith.toFixed(3)}`}
                </p>
                <p className="text-[8.5px] uppercase tracking-wide text-muted-foreground">faith Δ B−A</p>
              </div>
            </TooltipTrigger>
            <TooltipContent side="top">
              <p className="max-w-56 text-[10px]">grounding score difference — positive favours B</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>

      {(verdict.docs_only_a.length > 0 || verdict.docs_only_b.length > 0) && (
        <div className="space-y-0.5">
          {verdict.docs_only_a.length > 0 && (
            <p className="flex flex-wrap items-center gap-1 text-[9.5px] text-muted-foreground">
              <span className="shrink-0 rounded bg-teal-500/10 px-1 font-mono text-[8.5px] text-teal-700 dark:text-teal-400">only A</span>
              {verdict.docs_only_a.map((d) => (
                <span key={d} className="truncate font-mono text-[9px]">
                  {d}
                </span>
              ))}
            </p>
          )}
          {verdict.docs_only_b.length > 0 && (
            <p className="flex flex-wrap items-center gap-1 text-[9.5px] text-muted-foreground">
              <span className="shrink-0 rounded bg-amber-500/10 px-1 font-mono text-[8.5px] text-amber-700 dark:text-amber-400">only B</span>
              {verdict.docs_only_b.map((d) => (
                <span key={d} className="truncate font-mono text-[9px]">
                  {d}
                </span>
              ))}
            </p>
          )}
        </div>
      )}
    </motion.div>
  );
}

// ------------------------------------------- eval history + triage (18-b)
// Hand-rolled SVG sparkline: accuracy (teal, y fixed 0–1) + p50 latency
// (amber dashed, own scale) across the run sequence, oldest → left.
// Round 20 — brush zoom: drag across the plot (or use the window presets)
// to focus a run range; the chart re-renders zoomed, a stats row summarises
// the selection, and a context strip shows where the window sits in the
// full history. Reset returns to the full range.
function HistoryChart({ runs }: { runs: EvalRunSummary[] }) {
  const n = runs.length;
  const W = 640;
  const H = 190;
  const PADL = 42;
  const PADR = 16;
  const PADT = 12;
  const PADB = 18;
  const innerW = W - PADL - PADR;
  const innerH = H - PADT - PADB;
  const svgRef = useRef<SVGSVGElement | null>(null);

  // Selection window into `runs` (inclusive indices, oldest → newest).
  // Clamped during render (no effect) so a refetched/shorter run list can
  // never leave the window out of bounds.
  const [sel, setSel] = useState<[number, number]>([0, Math.max(0, n - 1)]);
  const [drag, setDrag] = useState<{ a: number; b: number } | null>(null);

  const maxI = Math.max(0, n - 1);
  const i0 = Math.min(Math.min(sel[0], sel[1]), maxI);
  const i1 = Math.min(Math.max(sel[0], sel[1]), maxI);
  const zoomed = i0 > 0 || i1 < n - 1;
  const slice = runs.slice(i0, i1 + 1);
  const m = slice.length;

  // --- geometry over the VISIBLE slice -------------------------------------
  const xAt = (i: number) => PADL + (m === 1 ? innerW / 2 : (i / (m - 1)) * innerW);
  const clamp01 = (v: number) => Math.min(1, Math.max(0, v));
  const yAcc = (v: number) => PADT + (1 - clamp01(v)) * innerH;
  const latMax = Math.max(1, ...slice.map((r) => r.latency_p50_ms));
  const yLat = (v: number) => PADT + (1 - Math.min(1, v / (latMax * 1.08))) * innerH;
  const linePath = (pts: Array<[number, number]>) =>
    pts.map(([px, py], i) => `${i === 0 ? "M" : "L"}${px.toFixed(1)} ${py.toFixed(1)}`).join(" ");
  const accPts = slice.map((r, i) => [xAt(i), yAcc(r.accuracy)] as [number, number]);
  const latPts = slice.map((r, i) => [xAt(i), yLat(r.latency_p50_ms)] as [number, number]);
  const accLine = linePath(accPts);
  const accArea = `${accLine} L ${xAt(m - 1).toFixed(1)} ${(PADT + innerH).toFixed(1)} L ${xAt(0).toFixed(1)} ${(PADT + innerH).toFixed(1)} Z`;
  const last = slice[m - 1];
  const runTime = (r: EvalRunSummary) =>
    Number.isFinite(r.ts) ? new Date(r.ts * 1000).toLocaleTimeString([], { hour12: false }) : "—";
  const pointTitle = (r: EvalRunSummary) =>
    `${r.ablation ? "ablation" : "golden"} run ${runTime(r)} — ${r.label}\n` +
    `${r.passed}/${r.total_cases} passed · accuracy ${(r.accuracy * 100).toFixed(0)}%\n` +
    `faith ${r.faithfulness_avg !== null ? `${(r.faithfulness_avg * 100).toFixed(0)}%` : "—"} · p50 ${r.latency_p50_ms.toFixed(0)} ms · p95 ${r.latency_p95_ms.toFixed(0)} ms`;
  const latTitle = (r: EvalRunSummary) => `p50 latency ${r.latency_p50_ms.toFixed(1)} ms (p95 ${r.latency_p95_ms.toFixed(1)} ms) — ${r.label}`;

  // --- window statistics -----------------------------------------------------
  const faithVals = slice.map((r) => r.faithfulness_avg).filter((v): v is number => v !== null);
  const avgAcc = m > 0 ? slice.reduce((s, r) => s + r.accuracy, 0) / m : 0;
  const minAcc = m > 0 ? Math.min(...slice.map((r) => r.accuracy)) : 0;
  const avgFaith = faithVals.length > 0 ? faithVals.reduce((s, v) => s + v, 0) / faithVals.length : null;
  const avgP50 = m > 0 ? slice.reduce((s, r) => s + r.latency_p50_ms, 0) / m : 0;
  const accDelta = m > 1 ? slice[m - 1].accuracy - slice[0].accuracy : 0;

  // --- mouse → run index (viewBox units, local slice index → global) --------
  const localToGlobal = (local: number) => i0 + local;
  const clientXToLocal = (clientX: number): number | null => {
    const svg = svgRef.current;
    if (!svg) return null;
    const rect = svg.getBoundingClientRect();
    if (rect.width <= 0) return null;
    const vx = (clientX - rect.left) * (W / rect.width);
    const clamped = Math.min(W - PADR, Math.max(PADL, vx));
    const frac = (clamped - PADL) / innerW;
    return Math.round(frac * (m - 1));
  };
  const applyPreset = (count: number | "all") => {
    if (count === "all") setSel([0, Math.max(0, n - 1)]);
    else setSel([Math.max(0, n - count), Math.max(0, n - 1)]);
  };
  const finalizeDrag = () => {
    if (drag) {
      const a = localToGlobal(Math.min(drag.a, drag.b));
      const b = localToGlobal(Math.max(drag.a, drag.b));
      if (Math.abs(drag.a - drag.b) >= 1) setSel([a, b]);
    }
    setDrag(null);
  };

  // Live drag rect in plot coordinates (local indices).
  const dragRect =
    drag && Math.abs(drag.a - drag.b) >= 1
      ? {
          x: xAt(Math.min(drag.a, drag.b)),
          w: Math.max(4, xAt(Math.max(drag.a, drag.b)) - xAt(Math.min(drag.a, drag.b))),
        }
      : null;

  // Context strip (full range, only while zoomed): shows where the window is.
  const ctxH = 22;
  const ctxY = (v: number) => 3 + (1 - clamp01(v)) * (ctxH - 6);
  const ctxX = (i: number) => (n === 1 ? 0.5 : i / (n - 1));
  const ctxPath = linePath(runs.map((r, i) => [ctxX(i), ctxY(r.accuracy)] as [number, number]));

  const presets: Array<{ label: string; apply: () => void; active: boolean }> = [
    { label: "last 5", apply: () => applyPreset(5), active: i1 - i0 + 1 === 5 && i1 === n - 1 && n > 5 },
    { label: "last 10", apply: () => applyPreset(10), active: i1 - i0 + 1 === 10 && i1 === n - 1 && n > 10 },
    { label: "all", apply: () => applyPreset("all"), active: !zoomed },
  ];

  return (
    <div
      className="rounded-lg border border-border/70 bg-background/60 transition-colors focus-within:ring-2 focus-within:ring-teal-500/40"
      title={`accuracy (teal, 0–1) and p50 latency (amber, own scale) across ${n} run${n !== 1 ? "s" : ""} — hover the points for per-run detail, drag on the plot to zoom a window`}
    >
      {zoomed && (
        <svg
          viewBox={`0 0 1 ${ctxH}`}
          preserveAspectRatio="none"
          className="block h-5 w-full border-b border-border/40 bg-muted/20"
          aria-hidden="true"
        >
          <path d={ctxPath} fill="none" strokeWidth={0.02} className="stroke-teal-600/50 dark:stroke-teal-400/50" vectorEffect="non-scaling-stroke" />
          <rect x={ctxX(i0)} y={0} width={Math.max(0.004, ctxX(i1) - ctxX(i0))} height={ctxH} className="fill-teal-500/20 dark:fill-teal-400/20" />
        </svg>
      )}
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        className="block h-auto w-full touch-none select-none"
        role="img"
        aria-label={`Accuracy trend, zoomed to ${m} of ${n} eval runs, latest ${(last.accuracy * 100).toFixed(0)}% — drag on the plot to select another window`}
        onMouseDown={(e) => {
          const local = clientXToLocal(e.clientX);
          if (local !== null) setDrag({ a: local, b: local });
        }}
        onMouseMove={(e) => {
          if (!drag) return;
          const local = clientXToLocal(e.clientX);
          if (local !== null) setDrag((d) => (d ? { ...d, b: local } : d));
        }}
        onMouseUp={finalizeDrag}
        onMouseLeave={finalizeDrag}
        style={{ cursor: drag ? "col-resize" : "crosshair" }}
      >
        {[1, 0.5, 0].map((g) => (
          <line
            key={g}
            x1={PADL}
            x2={W - PADR}
            y1={yAcc(g)}
            y2={yAcc(g)}
            strokeDasharray={g === 0 ? undefined : "3 4"}
            className={g === 0 ? "stroke-border" : "stroke-border/70"}
            strokeWidth={1}
          />
        ))}
        <text x={PADL - 6} y={yAcc(1) + 3} textAnchor="end" className="fill-muted-foreground text-[9px]">
          100%
        </text>
        <text x={PADL - 6} y={yAcc(0.5) + 3} textAnchor="end" className="fill-muted-foreground/70 text-[9px]">
          50%
        </text>
        <text x={PADL - 6} y={yAcc(0) + 3} textAnchor="end" className="fill-muted-foreground text-[9px]">
          0%
        </text>
        <text x={W - PADR} y={yLat(latMax) - 5} textAnchor="end" className="fill-amber-600/80 dark:fill-amber-400/80 text-[9px]">
          {latMax.toFixed(0)}ms
        </text>
        <path d={accArea} className="fill-teal-500/10 dark:fill-teal-400/10" />
        <path
          d={accLine}
          fill="none"
          strokeWidth={2}
          strokeLinejoin="round"
          strokeLinecap="round"
          className="stroke-teal-600 dark:stroke-teal-400"
        />
        <path
          d={linePath(latPts)}
          fill="none"
          strokeWidth={1.5}
          strokeDasharray="5 4"
          strokeLinejoin="round"
          strokeLinecap="round"
          className="stroke-amber-500/80 dark:stroke-amber-400/80"
        />
        {slice.map((r, i) => {
          const isLast = i === m - 1;
          return (
            <g key={`${r.ts}-${i}`}>
              <circle
                cx={xAt(i)}
                cy={yAcc(r.accuracy)}
                r={isLast ? 4.5 : 3.5}
                strokeWidth={1.5}
                className={cn(
                  "cursor-help transition-all",
                  r.accuracy >= 1
                    ? "fill-teal-600 dark:fill-teal-400 stroke-teal-600 dark:stroke-teal-400"
                    : "fill-background stroke-teal-600 dark:stroke-teal-400"
                )}
              >
                <title>{pointTitle(r)}</title>
              </circle>
              <circle cx={xAt(i)} cy={yLat(r.latency_p50_ms)} r={isLast ? 3.5 : 3} className="cursor-help fill-amber-500/70 dark:fill-amber-400/70">
                <title>{latTitle(r)}</title>
              </circle>
            </g>
          );
        })}
        {dragRect && (
          <g>
            <rect
              x={dragRect.x}
              y={PADT}
              width={dragRect.w}
              height={innerH}
              className="fill-teal-500/15 stroke-teal-500/70 dark:fill-teal-400/15 dark:stroke-teal-400/70"
              strokeWidth={1}
              strokeDasharray="4 3"
              rx={2}
            />
            <line x1={dragRect.x} x2={dragRect.x} y1={PADT - 4} y2={PADT + innerH} className="stroke-teal-500/70 dark:stroke-teal-400/70" strokeWidth={1} />
            <line x1={dragRect.x + dragRect.w} x2={dragRect.x + dragRect.w} y1={PADT - 4} y2={PADT + innerH} className="stroke-teal-500/70 dark:stroke-teal-400/70" strokeWidth={1} />
          </g>
        )}
      </svg>
      <div className="space-y-1.5 border-t border-border/60 px-2.5 py-1.5 text-[9px] text-muted-foreground">
        <p className="flex flex-wrap items-center gap-x-3 gap-y-1">
          <span className="flex items-center gap-1">
            <span className="h-0.5 w-4 rounded bg-teal-600 dark:bg-teal-400" aria-hidden />
            accuracy (0–1)
          </span>
          <span className="flex items-center gap-1">
            <span className="h-0 w-4 border-t-2 border-dashed border-amber-500/80 dark:border-amber-400/80" aria-hidden />
            p50 latency (own scale)
          </span>
          <span
            className="ml-auto font-mono"
            role="status"
            aria-label={`Selected run window: runs ${i0 + 1} to ${i1 + 1} of ${n}, ${m} run${m !== 1 ? "s" : ""}`}
          >
            {zoomed ? `runs ${i0 + 1}–${i1 + 1} of ${n}` : `oldest → newest · ${n} run${n !== 1 ? "s" : ""}`}
          </span>
        </p>
        <p
          className="flex flex-wrap items-center gap-x-2 gap-y-1 font-mono text-[9px]"
          role="status"
          aria-label={`Window statistics over ${m} runs: average accuracy ${(avgAcc * 100).toFixed(0)}%, minimum ${(minAcc * 100).toFixed(0)}%, average faithfulness ${avgFaith !== null ? `${(avgFaith * 100).toFixed(0)}%` : "—"}, average p50 ${avgP50.toFixed(0)} milliseconds, accuracy change ${(accDelta >= 0 ? "+" : "") + (accDelta * 100).toFixed(0)} points`}
        >
          <span className="flex items-center gap-1 whitespace-nowrap">
            <ZoomIn className="h-2.5 w-2.5 text-teal-600 dark:text-teal-500" aria-hidden />
            {m} run{m !== 1 ? "s" : ""}
          </span>
          <span className="whitespace-nowrap rounded bg-teal-500/10 px-1 py-px text-teal-700 dark:text-teal-400" title={`mean accuracy over the selected window (min ${(minAcc * 100).toFixed(1)}%)`}>
            avg {(avgAcc * 100).toFixed(0)}% · min {(minAcc * 100).toFixed(0)}%
          </span>
          <span className="whitespace-nowrap rounded bg-emerald-500/10 px-1 py-px text-emerald-700 dark:text-emerald-400" title="mean faithfulness over the window (judge-off ablation runs excluded)">
            faith {avgFaith !== null ? `${(avgFaith * 100).toFixed(0)}%` : "—"}
          </span>
          <span className="whitespace-nowrap rounded bg-amber-500/10 px-1 py-px text-amber-700 dark:text-amber-400" title="mean p50 latency over the selected window">
            p50 {avgP50.toFixed(0)}ms
          </span>
          <span
            className={cn(
              "whitespace-nowrap rounded px-1 py-px",
              accDelta > 0
                ? "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400"
                : accDelta < 0
                  ? "bg-rose-500/10 text-rose-700 dark:text-rose-400"
                  : "bg-muted text-muted-foreground"
            )}
            title="accuracy change across the window (last run − first run)"
          >
            Δ {(accDelta >= 0 ? "+" : "") + (accDelta * 100).toFixed(0)} pts
          </span>
        </p>
        <div className="flex flex-wrap items-center gap-1">
          <span className="sr-only">Chart zoom presets</span>
          {presets.map((p) => (
            <button
              key={p.label}
              type="button"
              onClick={p.apply}
              aria-pressed={p.active}
              className={cn(
                "h-6 whitespace-nowrap rounded-md border px-2 font-mono text-[9px] transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500/60",
                p.active
                  ? "border-teal-300 bg-teal-500/15 text-teal-700 dark:border-teal-700 dark:bg-teal-500/20 dark:text-teal-400"
                  : "border-border/60 bg-muted/30 text-muted-foreground hover:border-teal-300/60 hover:text-foreground dark:hover:border-teal-700/60"
              )}
            >
              {p.label}
            </button>
          ))}
          {zoomed && (
            <button
              type="button"
              onClick={() => applyPreset("all")}
              className="inline-flex h-6 items-center gap-1 whitespace-nowrap rounded-md border border-amber-300/70 bg-amber-500/10 px-2 font-mono text-[9px] text-amber-700 transition-colors hover:bg-amber-500/20 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-500/60 dark:border-amber-700/70 dark:text-amber-400"
              aria-label="Reset chart zoom to the full run range"
            >
              <RotateCcw className="h-2.5 w-2.5" aria-hidden />
              reset zoom
            </button>
          )}
          <span className="ml-auto hidden items-center gap-1 text-[8.5px] text-muted-foreground/80 sm:flex" title="drag horizontally on the plot to select a run window">
            <MousePointerClick className="h-2.5 w-2.5" aria-hidden />
            drag to zoom
          </span>
        </div>
      </div>
    </div>
  );
}

// Compact newest-first run rows — time, label chip, passed/total, accuracy,
// faithfulness ("—" for judge-off ablation runs), p50.
function HistoryRunsTable({ runs }: { runs: EvalRunSummary[] }) {
  const runTime = (r: EvalRunSummary) =>
    Number.isFinite(r.ts) ? new Date(r.ts * 1000).toLocaleTimeString([], { hour12: false }) : "—";
  return (
    <div className="thin-scrollbar max-h-56 overflow-y-auto rounded-md border" aria-label="Eval run history rows, newest first">
      <div className="divide-y divide-border/60">
        {runs.map((r, i) => {
          const ablationName = r.label.startsWith("ablation:") ? r.label.slice("ablation:".length) : null;
          const fullTitle =
            `${r.label} · ${r.passed}/${r.total_cases} · accuracy ${(r.accuracy * 100).toFixed(0)}% · ` +
            `faith ${r.faithfulness_avg !== null ? `${(r.faithfulness_avg * 100).toFixed(0)}%` : "—"} · ` +
            `p50 ${r.latency_p50_ms.toFixed(0)}ms · p95 ${r.latency_p95_ms.toFixed(0)}ms · ` +
            `retrieval ${r.retrieval_hit_rate !== null ? `${(r.retrieval_hit_rate * 100).toFixed(0)}%` : "—"} · ` +
            `fact support ${(r.fact_support_rate * 100).toFixed(0)}% · ` +
            (Number.isFinite(r.ts) ? new Date(r.ts * 1000).toLocaleString() : "—");
          return (
            <div
              key={`${r.ts}-${i}`}
              title={fullTitle}
              className={cn(
                "hover-lift flex items-center gap-2 px-2.5 py-1.5 transition-colors hover:bg-muted/50",
                i === 0 && "border-l-2 border-l-teal-500 bg-teal-50/40 dark:bg-teal-950/20"
              )}
            >
              <span className="w-16 shrink-0 font-mono text-[9.5px] text-muted-foreground">{runTime(r)}</span>
              <div className="min-w-0 flex-1">
                {r.ablation ? (
                  <span className="flex min-w-0 items-center gap-1.5">
                    <Badge
                      variant="outline"
                      className="shrink-0 gap-0.5 border-amber-300 bg-amber-50 px-1.5 text-[8.5px] uppercase tracking-wide text-amber-700 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-400"
                    >
                      ablation
                    </Badge>
                    <span className="truncate font-mono text-[9.5px] text-muted-foreground">{ablationName ?? r.label}</span>
                  </span>
                ) : (
                  <Badge
                    variant="outline"
                    className="gap-0.5 border-teal-300 bg-teal-50 px-1.5 text-[8.5px] uppercase tracking-wide text-teal-700 dark:border-teal-800 dark:bg-teal-950 dark:text-teal-400"
                  >
                    golden run
                  </Badge>
                )}
              </div>
              <div className="shrink-0 text-right font-mono text-[9.5px] leading-tight">
                <p className={cn("font-semibold", r.accuracy >= 1 ? "text-emerald-600 dark:text-emerald-400" : "text-amber-600 dark:text-amber-400")}>
                  {r.passed}/{r.total_cases} · {(r.accuracy * 100).toFixed(0)}%
                </p>
                <p className="text-muted-foreground">
                  faith {r.faithfulness_avg !== null ? `${(r.faithfulness_avg * 100).toFixed(0)}%` : "—"} · {r.latency_p50_ms.toFixed(0)}ms
                </p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// One 👎 triage item — question, reason comment, preview, badges, meta.
// Task 19-b: open items carry the resolve workflow (inline note + confirm);
// resolved items render as the emerald “fixed” variant with a reopen action.
function TriageItemCard({
  item,
  onResolve,
  onReopen,
  resolvingId,
}: {
  item: TriageItem;
  onResolve: (requestId: string, note: string) => Promise<boolean>;
  onReopen: (requestId: string) => Promise<void>;
  resolvingId: string | null;
}) {
  const rm = useReducedMotion();
  const [formOpen, setFormOpen] = useState(false);
  const [note, setNote] = useState("");
  const inFlight = resolvingId === item.request_id;
  const ratedFull = Number.isFinite(item.rated_at) ? new Date(item.rated_at * 1000).toLocaleString() : "—";
  const resolvedFull =
    item.resolved_at !== null && Number.isFinite(item.resolved_at)
      ? new Date(item.resolved_at * 1000).toLocaleString()
      : "—";

  const confirmFix = async () => {
    const ok = await onResolve(item.request_id, note.trim());
    if (ok) {
      setFormOpen(false);
      setNote("");
    }
  };

  return (
    <motion.div
      initial={rm ? undefined : { opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      className={cn(
        "hover-lift space-y-2 rounded-lg border p-3 transition-colors",
        item.resolved
          ? "border-emerald-200/60 bg-emerald-50/30 hover:border-emerald-300/70 dark:border-emerald-800/50 dark:bg-emerald-950/20 dark:hover:border-emerald-700/60"
          : "border-border/70 bg-muted/20 hover:border-amber-300/60 dark:hover:border-amber-800/60"
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <p className="min-w-0 flex-1 text-xs font-medium leading-snug">{item.question}</p>
        <span className="flex shrink-0 items-center gap-1.5">
          {!item.resolved && item.age_hours !== null && item.age_hours !== undefined && (
            <TooltipProvider delayDuration={150}>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Badge
                    variant="outline"
                    className={cn(
                      "cursor-help gap-0.5 px-1.5 font-mono text-[8.5px] normal-case tracking-wide",
                      AGE_TONE_CLASSES[ageTone(item.age_hours)]
                    )}
                    aria-label={`Open for ${formatAgeHours(item.age_hours)} (fix-list SLA)`}
                  >
                    <Clock className="h-2.5 w-2.5" aria-hidden />
                    {formatAgeHours(item.age_hours)}
                  </Badge>
                </TooltipTrigger>
                <TooltipContent side="top" className="max-w-56">
                  <p className="text-[10px] leading-relaxed">
                    this 👎 has been open for {formatAgeHours(item.age_hours)} — the SLA escalates to amber at 6h and
                    rose at 48h ({ageTone(item.age_hours) === "rose" ? "breached" : ageTone(item.age_hours) === "amber" ? "warning" : "fresh"})
                  </p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}
          {item.resolved && (
            <Badge
              variant="outline"
              className="shrink-0 gap-0.5 border-emerald-300 bg-emerald-50 px-1.5 text-[8.5px] uppercase tracking-wide text-emerald-700 dark:border-emerald-800 dark:bg-emerald-950 dark:text-emerald-400"
            >
              <CheckCheck className="h-2.5 w-2.5" aria-hidden />
              resolved ✓
            </Badge>
          )}
        </span>
      </div>

      {item.reason && (
        <blockquote
          className="rounded-md border-l-2 border-amber-400/70 bg-amber-50/60 px-2 py-1.5 text-[10.5px] italic leading-snug text-amber-700 dark:border-amber-500/60 dark:bg-amber-950/40 dark:text-amber-400"
          aria-label={`Downvote reason: ${item.reason}`}
        >
          reason: “{item.reason}”
        </blockquote>
      )}

      {item.resolved && item.resolution_note && (
        <blockquote
          className="rounded-md border-l-2 border-emerald-400/70 bg-emerald-50/60 px-2 py-1.5 text-[10.5px] italic leading-snug text-emerald-700 dark:border-emerald-500/60 dark:bg-emerald-950/40 dark:text-emerald-400"
          aria-label={`Resolution note: ${item.resolution_note}`}
        >
          resolution note: “{item.resolution_note}”
        </blockquote>
      )}

      <p className="line-clamp-2 text-[11px] leading-relaxed text-muted-foreground" title={item.answer_preview}>
        {item.answer_preview}
      </p>

      <div className="flex flex-wrap items-center gap-1.5">
        {item.fallback && (
          <Badge variant="outline" className="gap-0.5 border-red-300 px-1.5 text-[8.5px] uppercase tracking-wide text-red-600 dark:border-red-800 dark:text-red-400">
            <AlertTriangle className="h-2.5 w-2.5" aria-hidden />
            fallback
          </Badge>
        )}
        {item.citations.length === 0 && (
          <TooltipProvider delayDuration={150}>
            <Tooltip>
              <TooltipTrigger asChild>
                <Badge
                  variant="outline"
                  className="cursor-help gap-0.5 border-amber-300 px-1.5 text-[8.5px] uppercase tracking-wide text-amber-700 dark:border-amber-800 dark:text-amber-400"
                  aria-label="Answer had no citations"
                >
                  <Link2 className="h-2.5 w-2.5" aria-hidden />
                  no citations
                </Badge>
              </TooltipTrigger>
              <TooltipContent side="top" className="max-w-56">
                <p className="text-[10px]">the rated answer carried no grounded citations — a grounding red flag</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        )}
        {item.citations.slice(0, 3).map((c, i) => (
          <Badge key={`${c.doc}-${c.page}-${i}`} variant="secondary" className="max-w-40 truncate px-1.5 font-mono text-[9px]">
            {c.doc} · p.{c.page}
          </Badge>
        ))}
        {item.citations.length > 3 && (
          <Badge variant="outline" className="px-1.5 font-mono text-[9px] text-muted-foreground">
            +{item.citations.length - 3}
          </Badge>
        )}
      </div>

      <p
        className="font-mono text-[9.5px] text-muted-foreground"
        title={`rated ${ratedFull} · ${item.route}${item.cache_hit ? " · served from cache" : ""}${
          item.resolved ? ` · resolved ${resolvedFull}` : ""
        } · request ${item.request_id}`}
      >
        {item.route} · {item.latency_ms.toFixed(0)} ms · rated {formatRatedAt(item.rated_at)}
        {item.resolved && item.resolved_at !== null && ` · resolved ${formatRatedAt(item.resolved_at)}`}
      </p>

      {item.resolved ? (
        <TooltipProvider delayDuration={200}>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="outline"
                size="sm"
                className="h-11 w-full gap-1.5 border-emerald-300/60 text-xs font-medium text-emerald-700 transition-all hover:border-emerald-400 hover:bg-emerald-50 focus-visible:ring-emerald-500/40 active:scale-[0.98] dark:border-emerald-800/60 dark:text-emerald-400 dark:hover:border-emerald-600 dark:hover:bg-emerald-950/40"
                onClick={() => void onReopen(item.request_id)}
                disabled={inFlight}
                aria-label={`Reopen “${item.question}” on the open fix-list`}
              >
                {inFlight ? <Loader2 className="h-4 w-4 animate-spin" /> : <Undo2 className="h-4 w-4" />}
                Reopen
              </Button>
            </TooltipTrigger>
            <TooltipContent side="top" className="max-w-64">
              <p className="text-[10px] leading-relaxed">put this back on the open fix-list</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      ) : (
        <div className="space-y-2">
          <TooltipProvider delayDuration={200}>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="outline"
                  size="sm"
                  className="h-11 w-full gap-1.5 border-emerald-300/60 text-xs font-medium text-emerald-700 transition-all hover:border-emerald-400 hover:bg-emerald-50 focus-visible:ring-emerald-500/40 active:scale-[0.98] dark:border-emerald-800/60 dark:text-emerald-400 dark:hover:border-emerald-600 dark:hover:bg-emerald-950/40"
                  onClick={() => setFormOpen((v) => !v)}
                  disabled={inFlight}
                  aria-label={`Mark “${item.question}” as fixed`}
                >
                  {inFlight ? <Loader2 className="h-4 w-4 animate-spin" /> : <CheckCheck className="h-4 w-4" />}
                  Mark fixed
                </Button>
              </TooltipTrigger>
              <TooltipContent side="top" className="max-w-64">
                <p className="text-[10px] leading-relaxed">
                  mark this 👎 as fixed — it moves to the resolved list with your note
                </p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>

          <AnimatePresence initial={false}>
            {formOpen && (
              <motion.div
                key="resolve-form"
                initial={rm ? undefined : { height: 0, opacity: 0 }}
                animate={rm ? { opacity: 1 } : { height: "auto", opacity: 1 }}
                exit={rm ? { opacity: 0 } : { height: 0, opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="overflow-hidden"
              >
                <div className="space-y-1.5 rounded-md border border-emerald-200/60 bg-emerald-50/40 p-2 dark:border-emerald-800/50 dark:bg-emerald-950/30">
                  <label
                    htmlFor={`fix-note-${item.request_id}`}
                    className="block text-[10px] uppercase tracking-wide text-muted-foreground"
                  >
                    note <span className="font-mono normal-case opacity-70">(optional)</span>
                  </label>
                  <Input
                    id={`fix-note-${item.request_id}`}
                    value={note}
                    onChange={(e) => setNote(e.target.value)}
                    maxLength={300}
                    placeholder="what fixed it? (optional)"
                    disabled={inFlight}
                    className="h-9 border-emerald-200/60 text-[11px] focus-visible:ring-emerald-500/40 dark:border-emerald-900/60"
                    aria-label={`Optional fix note for: ${item.question}`}
                  />
                  <div className="flex items-center gap-2">
                    <Button
                      size="sm"
                      className="h-11 flex-1 gap-1.5 bg-emerald-600 text-xs font-medium text-white shadow-sm transition-all hover:bg-emerald-700 hover:shadow active:scale-[0.98] focus-visible:ring-emerald-500/60 dark:bg-emerald-500 dark:hover:bg-emerald-400 dark:text-emerald-950"
                      onClick={() => void confirmFix()}
                      disabled={inFlight}
                    >
                      {inFlight ? <Loader2 className="h-4 w-4 animate-spin" /> : <CheckCheck className="h-4 w-4" />}
                      Confirm fix
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-11 px-3 text-xs text-muted-foreground"
                      onClick={() => setFormOpen(false)}
                      disabled={inFlight}
                    >
                      Cancel
                    </Button>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}
    </motion.div>
  );
}

// Pulse tab growth card — mirrors the cross-encoder stage card pattern.
// Task 19-b: gains triage chips (open fix-list / resolved counts from /stats).
function GoldenGrowthCard({
  golden,
  evalRuns,
  triage,
}: {
  golden: NonNullable<StatsResponse["golden"]>;
  evalRuns?: StatsResponse["eval_runs"];
  triage?: StatsResponse["triage"];
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2, delay: 0.05 }}
      className="space-y-1.5 rounded-lg border border-teal-200/50 bg-teal-50/20 p-2.5 dark:border-teal-900/50 dark:bg-teal-950/20"
      aria-label="Golden-set growth stats"
    >
      <p className="flex flex-wrap items-center justify-between gap-1.5 text-[10px] uppercase tracking-wide text-muted-foreground">
        <span className="flex items-center gap-1">
          <Sprout className="h-3 w-3 text-teal-600 dark:text-teal-500" aria-hidden />
          golden-set growth
        </span>
        <span className="flex items-center gap-1.5">
          {evalRuns && (
            <TooltipProvider delayDuration={150}>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span
                    className="cursor-help rounded bg-teal-500/10 px-1 py-px font-mono text-[9px] normal-case text-teal-700 dark:text-teal-400"
                    aria-label={`${evalRuns.total} eval runs recorded`}
                  >
                    {evalRuns.total} runs
                  </span>
                </TooltipTrigger>
                <TooltipContent side="top">
                  <p className="max-w-56 text-[10px]">
                    eval runs recorded ({evalRuns.golden} golden + {evalRuns.ablation} ablation) — see the trend in
                    the Eval tab
                  </p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}
          {(golden.downvoted ?? 0) > 0 && (
            <TooltipProvider delayDuration={150}>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span
                    className="cursor-help rounded bg-amber-500/10 px-1 py-px font-mono text-[9px] normal-case text-amber-700 dark:text-amber-400"
                    aria-label={`${golden.downvoted ?? 0} downvoted answers`}
                  >
                    👎 {golden.downvoted}
                  </span>
                </TooltipTrigger>
                <TooltipContent side="top">
                  <p className="max-w-56 text-[10px]">
                    downvoted answers — see the quality radar in the Eval tab
                  </p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}
          {triage && triage.open > 0 && (
            <TooltipProvider delayDuration={150}>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span
                    className={cn(
                      "inline-flex cursor-help items-center gap-0.5 rounded px-1 py-px font-mono text-[9px] normal-case",
                      triage.oldest_open_hours != null && ageTone(triage.oldest_open_hours) === "rose"
                        ? "bg-rose-500/10 text-rose-700 dark:text-rose-400"
                        : "bg-amber-500/10 text-amber-700 dark:text-amber-400"
                    )}
                    aria-label={`${triage.open} unresolved downvote items on the fix-list`}
                  >
                    <ThumbsDown className="h-2.5 w-2.5" aria-hidden />
                    {triage.open} open
                    {triage.oldest_open_hours != null && (
                      <span className="inline-flex items-center gap-0.5 opacity-80">
                        <Clock className="h-2.5 w-2.5" aria-hidden />
                        {formatAgeHours(triage.oldest_open_hours)}
                      </span>
                    )}
                  </span>
                </TooltipTrigger>
                <TooltipContent side="top">
                  <p className="max-w-64 text-[10px] leading-relaxed">
                    unresolved 👎 items on the fix-list{triage.oldest_open_hours != null ? ` — the oldest has waited ${formatAgeHours(triage.oldest_open_hours)} (SLA: amber 6h · rose 48h). See the quality radar in the Eval tab` : " — see the quality radar in the Eval tab"}
                  </p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}
          {triage && triage.resolved > 0 && (
            <TooltipProvider delayDuration={150}>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span
                    className="inline-flex cursor-help items-center gap-0.5 rounded bg-emerald-500/10 px-1 py-px font-mono text-[9px] normal-case text-emerald-700 dark:text-emerald-400"
                    aria-label={`${triage.resolved} downvoted answers marked fixed`}
                  >
                    <ShieldCheck className="h-2.5 w-2.5" aria-hidden />✓ {triage.resolved} resolved
                  </span>
                </TooltipTrigger>
                <TooltipContent side="top">
                  <p className="max-w-56 text-[10px]">downvoted answers marked fixed</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}
          <TooltipProvider delayDuration={150}>
            <Tooltip>
              <TooltipTrigger asChild>
                <span
                  className="cursor-help rounded bg-teal-500/10 px-1 py-px font-mono text-[9px] normal-case text-teal-700 dark:text-teal-400"
                  aria-label={`${golden.upvoted} upvotes recorded`}
                >
                  👍 {golden.upvoted}
                </span>
              </TooltipTrigger>
              <TooltipContent side="top">
                <p className="max-w-56 text-[10px]">
                  total 👍 ratings recorded — each grounded one becomes a promotion candidate
                </p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </span>
      </p>
      <div className="grid grid-cols-4 gap-1.5 text-center">
        {[
          { label: "base", value: golden.base.toLocaleString(), teal: false },
          { label: "user", value: golden.user.toLocaleString(), teal: golden.user > 0 },
          { label: "total", value: golden.total.toLocaleString(), teal: golden.user > 0 },
          { label: "ledger", value: golden.ledger_entries.toLocaleString(), teal: false },
        ].map(({ label, value, teal }) => (
          <div key={label} className="rounded-md bg-background/70 px-1 py-1.5 dark:bg-background/50">
            <p className={cn("font-mono text-[11px] font-semibold", teal && "text-teal-600 dark:text-teal-400")}>
              <AnimatedStat value={value} />
            </p>
            <p className="text-[8.5px] uppercase tracking-wide text-muted-foreground">{label}</p>
          </div>
        ))}
      </div>
      <p className="text-[9px] leading-snug text-muted-foreground/80">
        👍 upvotes → candidates → promoted cases; the answer ledger joins ratings to answers.
      </p>
    </motion.div>
  );
}


// ------------------------------------------------------ question analytics
// Round 20 — "most asked questions" card for the Memory tab: frequency
// leaderboard mined from the answer ledger (GET /analytics/questions), with
// per-question quality aggregates (citation rate, fallbacks, votes, latency).
// The input signal for golden-set growth: what users actually ask.
function TopQuestionsCard({
  data,
  loading,
  error,
  onRefresh,
}: {
  data: AnalyticsResponse | null;
  loading: boolean;
  error: string | null;
  onRefresh: () => void;
}) {
  const rm = useReducedMotion();
  const maxCount = data ? Math.max(1, ...data.top.map((q) => q.count)) : 1;
  return (
    <Card className="card-hairline">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between gap-2">
          <CardTitle className="flex items-center gap-2 text-sm">
            <MessageSquareQuote className="h-4 w-4 text-teal-600" />
            Most Asked Questions
          </CardTitle>
          <Button
            variant="ghost"
            size="sm"
            className="h-8 shrink-0 gap-1 px-2 text-[10px] text-muted-foreground"
            onClick={onRefresh}
            disabled={loading}
            aria-label="Refresh question analytics"
            title="Re-fetch the question frequency leaderboard"
          >
            <RefreshCw className={cn("h-3 w-3", loading && "animate-spin")} />
            Refresh
          </Button>
        </div>
        <CardDescription className="text-[11px] leading-relaxed">
          Frequency leaderboard from the answer ledger — the usage lens for corpus coverage
          and golden-set growth.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-2.5">
        {data && (
          <p
            className="flex flex-wrap items-center gap-x-2 gap-y-1 font-mono text-[9.5px] text-muted-foreground"
            role="status"
            aria-label={`Question analytics totals: ${data.totals.distinct_questions} distinct questions over ${data.totals.answered_asks} answered asks`}
          >
            <span className="flex items-center gap-0.5">
              <BarChart3 className="h-2.5 w-2.5 text-teal-600 dark:text-teal-500" aria-hidden />
              {data.totals.distinct_questions} distinct
            </span>
            <span className="rounded bg-teal-500/10 px-1 py-px text-teal-700 dark:text-teal-400">
              {data.totals.answered_asks} asks
            </span>
            <span
              className="rounded bg-muted px-1 py-px"
              title={`the ledger keeps the most recent ${data.totals.ledger_capacity} answered asks (LRU eviction) — frequencies describe that window`}
            >
              window {data.totals.ledger_capacity}
            </span>
          </p>
        )}

        {error ? (
          <p className="flex items-center gap-1.5 rounded-md border border-amber-200/60 bg-amber-50/40 px-2 py-1.5 text-[10.5px] text-amber-700 dark:border-amber-900/60 dark:bg-amber-950/30 dark:text-amber-400">
            <CircleAlert className="h-3 w-3 shrink-0" aria-hidden />
            analytics unavailable ({error})
          </p>
        ) : loading && !data ? (
          <div className="space-y-2" aria-label="Loading question analytics">
            {[0, 1, 2, 3].map((i) => (
              <Skeleton key={i} className="h-10 w-full" />
            ))}
          </div>
        ) : data && data.top.length > 0 ? (
          <ol className="thin-scrollbar max-h-96 space-y-1.5 overflow-y-auto pr-1" aria-label="Most asked questions, by ask count">
            {data.top.map((q, i) => (
              <motion.li
                key={`${q.question}-${i}`}
                initial={rm ? undefined : { opacity: 0, x: -6 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.18, delay: i * 0.03 }}
                className="hover-lift group relative overflow-hidden rounded-lg border border-border/70 bg-muted/20 p-2 transition-colors hover:border-teal-300/60 dark:hover:border-teal-800/60"
              >
                {/* frequency bar — relative to the leaderboard max */}
                <span
                  className="pointer-events-none absolute inset-y-0 left-0 bg-teal-500/10 transition-all dark:bg-teal-400/10"
                  style={{ width: `${(q.count / maxCount) * 100}%` }}
                  aria-hidden
                />
                <div className="relative space-y-1">
                  <p className="flex items-start gap-1.5">
                    <span
                      className="mt-px shrink-0 rounded bg-teal-500/15 px-1 font-mono text-[9px] font-semibold text-teal-700 dark:bg-teal-500/20 dark:text-teal-400"
                      title={`asked ${q.count} time${q.count !== 1 ? "s" : ""}`}
                    >
                      ×{q.count}
                    </span>
                    <span className="min-w-0 flex-1 text-[11px] font-medium leading-snug" title={q.question}>
                      {q.question}
                    </span>
                  </p>
                  <p className="flex flex-wrap items-center gap-x-1.5 gap-y-0.5 font-mono text-[9px] text-muted-foreground">
                    <span className="flex items-center gap-0.5" title={`citation rate ${(q.citation_rate * 100).toFixed(0)}% — grounded answers / asks`}>
                      <Link2 className="h-2.5 w-2.5" aria-hidden />
                      {(q.citation_rate * 100).toFixed(0)}%
                    </span>
                    {q.fallback_rate > 0 && (
                      <span className="rounded bg-rose-500/10 px-1 py-px text-rose-700 dark:text-rose-400" title={`fallback rate ${(q.fallback_rate * 100).toFixed(0)}%`}>
                        fallback {(q.fallback_rate * 100).toFixed(0)}%
                      </span>
                    )}
                    <span title={`mean p50 latency ${q.avg_latency_ms.toFixed(0)} ms`}>
                      {q.avg_latency_ms.toFixed(0)}ms
                    </span>
                    {q.cache_rate > 0 && (
                      <span className="rounded bg-amber-500/10 px-1 py-px text-amber-700 dark:text-amber-400" title={`cache hit rate ${(q.cache_rate * 100).toFixed(0)}%`}>
                        cache {(q.cache_rate * 100).toFixed(0)}%
                      </span>
                    )}
                    {q.ups > 0 && (
                      <span className="rounded bg-emerald-500/10 px-1 py-px text-emerald-700 dark:text-emerald-400" title={`${q.ups} 👍 on this question's answers`}>
                        👍 {q.ups}
                      </span>
                    )}
                    {q.downs > 0 && (
                      <span className="rounded bg-rose-500/10 px-1 py-px text-rose-700 dark:text-rose-400" title={`${q.downs} 👎 on this question's answers — see the quality radar`}>
                        👎 {q.downs}
                      </span>
                    )}
                    <span className="ml-auto" title={`routes: ${q.routes.join(", ")}`}>
                      {q.routes.join("/")}
                    </span>
                    <span title={`last asked ${formatRatedAt(q.last_ts)}`}>· {formatRatedAt(q.last_ts)}</span>
                  </p>
                </div>
              </motion.li>
            ))}
          </ol>
        ) : (
          <div className="flex flex-col items-center gap-1.5 rounded-xl border border-dashed border-border bg-muted/20 px-3 py-8 text-center">
            <MessageSquareQuote className="h-5 w-5 text-muted-foreground/50" aria-hidden />
            <p className="text-[11px] font-medium">No answered questions recorded yet</p>
            <p className="max-w-xs text-[10.5px] leading-relaxed text-muted-foreground">
              ask something in the chat — the answer ledger feeds this leaderboard.
            </p>
          </div>
        )}
        <p className="text-[9px] leading-snug text-muted-foreground/70">
          questions are normalised (lower-cased, whitespace-collapsed); 👍/👎 join via the feedback store — frequent
          grounded questions are promotion candidates, frequent fallbacks are corpus-coverage gaps.
        </p>
      </CardContent>
    </Card>
  );
}


// ------------------------------------------------- prometheus exporter card
// Task 2: observability card for the Pulse tab — parses a live scrape of
// GET /metrics (text/plain; version=0.0.4) client-side.
function PrometheusCard({
  text,
  loading,
  error,
  onRefresh,
  onGrafana,
}: {
  text: string | null;
  loading: boolean;
  error: string | null;
  onRefresh: () => void;
  onGrafana: () => void;
}) {
  const { toast } = useToast();
  const [rawOpen, setRawOpen] = useState(false);
  const [copiedEndpoint, setCopiedEndpoint] = useState(false);
  const [copiedRaw, setCopiedRaw] = useState(false);
  const headline = useMemo(() => (text ? parsePromHeadline(text) : null), [text]);

  const copy = async (value: string, setCopied: (v: boolean) => void): Promise<boolean> => {
    try {
      await navigator.clipboard?.writeText(value);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
      return true;
    } catch {
      return false;
    }
  };

  const tiles: Array<{ label: string; value: string }> = headline
    ? [
        { label: "asks", value: promInt(headline.asks) },
        { label: "lat obs", value: promInt(headline.latencyObs) },
        { label: "http reqs", value: promInt(headline.httpReqs) },
        {
          label: "cache hit",
          value: headline.cacheHitRatio !== null ? `${(headline.cacheHitRatio * 100).toFixed(0)}%` : "—",
        },
      ]
    : [];

  return (
    <motion.div
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2, delay: 0.05 }}
      className="space-y-2 rounded-lg border border-teal-200/50 bg-teal-50/20 p-2.5 dark:border-teal-900/50 dark:bg-teal-950/20"
    >
      <div className="flex items-center justify-between gap-2">
        <p className="flex items-center gap-1 text-[10px] uppercase tracking-wide text-muted-foreground">
          <Gauge className="h-3 w-3 text-teal-600 dark:text-teal-500" />
          prometheus exporter
        </p>
        <span className="flex items-center gap-1.5 font-mono text-[9px] text-muted-foreground">
          <span className="relative flex h-1.5 w-1.5" aria-hidden>
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-teal-400 opacity-60" />
            <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-teal-500" />
          </span>
          live
          <TooltipProvider delayDuration={200}>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-5 w-5 rounded-full text-muted-foreground/70 hover:text-foreground"
                  onClick={onRefresh}
                  disabled={loading}
                  aria-label="Refresh Prometheus metrics"
                  title="Re-scrape /metrics"
                >
                  <RefreshCw className={cn("h-3 w-3", loading && "animate-spin")} />
                </Button>
              </TooltipTrigger>
              <TooltipContent side="left">
                <p className="text-[10px]">re-scrape the exporter now</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </span>
      </div>

      <p className="text-[11px] leading-relaxed text-muted-foreground">
        Standard text exposition at GET /metrics — scrape with Prometheus or view raw.
      </p>

      {error ? (
        <p className="flex items-center gap-1.5 rounded-md border border-amber-200/60 bg-amber-50/40 px-2 py-1.5 text-[10.5px] text-amber-700 dark:border-amber-900/60 dark:bg-amber-950/30 dark:text-amber-400">
          <CircleAlert className="h-3 w-3 shrink-0" />
          exporter unreachable ({error})
        </p>
      ) : tiles.length > 0 ? (
        <div className="grid grid-cols-2 gap-1.5 sm:grid-cols-4">
          {tiles.map(({ label, value }) => (
            <div key={label} className="rounded-md bg-background/70 px-1.5 py-1.5 text-center dark:bg-background/50">
              <p className="font-mono text-[11px] font-semibold tabular-nums">{value}</p>
              <p className="text-[8.5px] uppercase tracking-wide text-muted-foreground">{label}</p>
            </div>
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-4 gap-1.5" aria-label="Loading metrics">
          {[0, 1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-8 rounded-md" />
          ))}
        </div>
      )}

      <div className="flex flex-wrap items-center gap-1.5">
        <Button
          variant="outline"
          size="sm"
          className="h-7 gap-1 px-2.5 text-[11px]"
          onClick={() => setRawOpen(true)}
          disabled={!text}
        >
          <FileText className="h-3 w-3" />
          View raw
        </Button>
        <Button
          variant="outline"
          size="sm"
          className="h-7 gap-1 px-2.5 text-[11px]"
          onClick={() => {
            void copy("/metrics?XTransformPort=3003", setCopiedEndpoint).then((ok) => {
              if (ok) toast({ title: "Endpoint copied", description: "/metrics?XTransformPort=3003" });
            });
          }}
        >
          {copiedEndpoint ? <CheckCircle2 className="h-3 w-3 text-emerald-600" /> : <Link2 className="h-3 w-3" />}
          {copiedEndpoint ? "Copied" : "Copy endpoint"}
        </Button>
        <TooltipProvider delayDuration={200}>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="outline"
                size="sm"
                className="h-7 gap-1 px-2.5 text-[11px] transition-all hover:border-teal-400 hover:bg-teal-50/60 dark:hover:border-teal-700 dark:hover:bg-teal-950/40"
                onClick={onGrafana}
              >
                <LayoutDashboard className="h-3 w-3 text-teal-600 dark:text-teal-400" />
                Grafana JSON
              </Button>
            </TooltipTrigger>
            <TooltipContent side="top" className="max-w-64">
              <p className="text-[10px]">
                download the 8-panel importable dashboard (GET /ops/grafana) — import via Grafana →
                Dashboards → Import
              </p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>

      <p className="text-[9px] leading-snug text-muted-foreground/80">
        http_requests_total counts by route pattern — label cardinality is bounded.
      </p>

      <Dialog open={rawOpen} onOpenChange={setRawOpen}>
        <DialogContent className="sm:max-w-2xl">
          <DialogHeader>
            <DialogTitle className="flex flex-wrap items-center gap-2">
              <Gauge className="h-4 w-4 text-teal-600" />
              <span className="font-mono text-sm">GET /metrics</span>
              <Badge variant="outline" className="font-mono text-[9px] text-muted-foreground">
                text/plain; version=0.0.4
              </Badge>
            </DialogTitle>
            <DialogDescription>
              Raw Prometheus exposition as served by the RAG service — the exact payload a
              Prometheus or Grafana agent scraper would ingest.
            </DialogDescription>
          </DialogHeader>
          <div className="thin-scrollbar max-h-[60vh] overflow-y-auto rounded-lg border border-zinc-800 bg-zinc-950 p-3 dark:border-zinc-800">
            <div className="font-mono text-[10.5px] leading-relaxed">
              {(text ?? "")
                .split("\n")
                .map((line, i) => (
                  <div
                    key={i}
                    className={cn(
                      "whitespace-pre-wrap break-all",
                      line.startsWith("#") ? "text-emerald-500/80" : "text-zinc-300"
                    )}
                  >
                    {line || "\u00A0"}
                  </div>
                ))}
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setRawOpen(false)}>
              Close
            </Button>
            <Button
              variant="outline"
              className="gap-1"
              onClick={() => void copy(text ?? "", setCopiedRaw)}
            >
              {copiedRaw ? <CheckCircle2 className="h-4 w-4 text-emerald-600" /> : <Copy className="h-4 w-4" />}
              {copiedRaw ? "Copied" : "Copy raw"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </motion.div>
  );
}

// ---------------------------------------------------------- export helpers
// Task 2: client-side conversation/session export (Markdown + JSON) via
// Blob + object URL + transient anchor click.
function downloadBlob(content: string, filename: string, mime: string): void {
  const blob = new Blob([content], { type: `${mime};charset=utf-8` });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function exportTimestamp(): string {
  return new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
}

// ------------------------------------------------------ eval report exports
// Round 20 — shareable quality report (Markdown) + golden-set CSV, built
// client-side from the data the Eval tab already holds. The markdown joins
// the last eval run, the merged golden set, the persisted run history and
// the triage counts into one reviewable artifact.

function mdEscapeCell(text: string): string {
  return text.replace(/\|/g, "\\|").replace(/\n/g, " ").trim();
}

function buildEvalReportMarkdown(args: {
  mode: string;
  evalReport: EvalReport | null;
  casesData: CasesResponse | null;
  historyData: EvalHistoryResponse | null;
  triageData: TriageResponse | null;
}): string {
  const { mode, evalReport, casesData, historyData, triageData } = args;
  const lines: string[] = [];
  lines.push("# RAG Agentic Q&A — evaluation report");
  lines.push("");
  lines.push(`Generated: ${new Date().toLocaleString()} · serving mode: \`${mode}\``);
  lines.push("");

  if (evalReport) {
    lines.push("## Last golden-set run");
    lines.push("");
    lines.push(`- ran at: ${evalReport.ran_at}`);
    lines.push(`- cases: ${evalReport.total_cases} · passed ${evalReport.passed} · failed ${evalReport.failed}`);
    lines.push(`- accuracy: ${(evalReport.accuracy * 100).toFixed(1)}%`);
    lines.push(`- retrieval hit rate: ${evalReport.retrieval_hit_rate !== null ? `${(evalReport.retrieval_hit_rate * 100).toFixed(1)}%` : "—"}`);
    lines.push(`- fact support: ${(evalReport.fact_support_rate * 100).toFixed(1)}%`);
    lines.push(`- faithfulness: ${evalReport.faithfulness_avg !== null ? `${(evalReport.faithfulness_avg * 100).toFixed(1)}%` : "—"} (judge: ${evalReport.faithfulness_judge ?? "—"})`);
    lines.push(`- latency: p50 ${evalReport.latency_p50_ms.toFixed(0)} ms · p95 ${evalReport.latency_p95_ms.toFixed(0)} ms`);
    if (Object.keys(evalReport.by_category).length > 0) {
      lines.push("");
      lines.push("| category | passed | total |");
      lines.push("|---|---|---|");
      for (const [cat, s] of Object.entries(evalReport.by_category)) {
        lines.push(`| ${mdEscapeCell(cat)} | ${s.passed} | ${s.total} |`);
      }
    }
    lines.push("");
  } else {
    lines.push("_No golden-set run recorded in this session yet — press “Run evaluation” in the Eval tab._");
    lines.push("");
  }

  if (casesData && casesData.cases.length > 0) {
    const resultById = new Map((evalReport?.results ?? []).map((r) => [r.id, r]));
    lines.push("## Golden set");
    lines.push("");
    lines.push(`${casesData.stats.total} cases (${casesData.stats.base} base + ${casesData.stats.user} user-grown):`);
    lines.push("");
    lines.push("| id | category | source | result | question |");
    lines.push("|---|---|---|---|---|");
    for (const c of casesData.cases) {
      const r = resultById.get(c.id);
      const result = r ? (r.passed ? "✅ pass" : "❌ fail") : "—";
      lines.push(`| ${c.id} | ${mdEscapeCell(c.category)} | ${c.source} | ${result} | ${mdEscapeCell(c.question)} |`);
    }
    lines.push("");
  }

  if (historyData && historyData.runs.length > 0) {
    lines.push("## Run history (newest first)");
    lines.push("");
    lines.push("| time | label | passed | accuracy | faith | p50 ms |");
    lines.push("|---|---|---|---|---|---|");
    for (const r of historyData.runs.slice(0, 20)) {
      const time = Number.isFinite(r.ts) ? new Date(r.ts * 1000).toLocaleString() : "—";
      lines.push(
        `| ${time} | ${mdEscapeCell(r.label)} | ${r.passed}/${r.total_cases} | ${(r.accuracy * 100).toFixed(0)}% | ${
          r.faithfulness_avg !== null ? `${(r.faithfulness_avg * 100).toFixed(0)}%` : "—"
        } | ${r.latency_p50_ms.toFixed(0)} |`
      );
    }
    lines.push("");
  }

  if (triageData) {
    const c = triageData.counts;
    lines.push("## Triage fix-list");
    lines.push("");
    lines.push(`- open: ${c.open} · resolved: ${c.resolved}`);
    if (c.oldest_open_hours !== null && c.oldest_open_hours !== undefined) {
      lines.push(`- oldest open item: ${formatAgeHours(c.oldest_open_hours)} (SLA — amber 6h · rose 48h)`);
    }
    lines.push(`- fallback answers: ${c.fallback_answers} · uncited: ${c.uncited_answers} · with reason: ${c.with_reason}`);
    lines.push("");
  }

  lines.push("----");
  lines.push("");
  lines.push("Report exported from the Document Intelligence Assistant Eval tab.");
  return lines.join("\n");
}

function buildGoldenCasesCsv(casesData: CasesResponse): string {
  const header = ["id", "category", "source", "question", "expects", "keywords", "docs_any"];
  const rows = casesData.cases.map((c) => [
    c.id,
    c.category,
    c.source,
    c.question,
    c.expects,
    (c.keywords ?? []).join(";"),
    (c.docs_any ?? []).join(";"),
  ]);
  const esc = (v: string) => {
    const s = String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return [header, ...rows].map((r) => r.map(esc).join(",")).join("\n");
}

function buildTurnMarkdown(
  question: string,
  answer: string,
  sources: string[],
  meta: string
): string {
  const parts = [`## Q: ${question}`, "", answer];
  if (sources.length > 0) {
    parts.push("", "**Sources:**", ...sources.map((s, i) => `${i + 1}. ${s}`));
  }
  if (meta) parts.push("", `*${meta}*`);
  return parts.join("\n");
}

// ------------------------------------------------------------------- page
export default function Home() {
  const { toast } = useToast();
  const reduceMotion = useReducedMotion();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [corpus, setCorpus] = useState<CorpusResponse | null>(null);
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [bench, setBench] = useState<BenchmarkResult | null>(null);
  const [benchRunning, setBenchRunning] = useState(false);
  const [ingesting, setIngesting] = useState(false);
  const [evalReport, setEvalReport] = useState<EvalReport | null>(null);
  const [evalRunning, setEvalRunning] = useState(false);
  const [ablationReport, setAblationReport] = useState<AblationReport | null>(null);
  const [ablationRunning, setAblationRunning] = useState(false);
  // Golden-set growth (Task 17-b): candidates, keyword edits, promote results.
  const [candidatesData, setCandidatesData] = useState<CandidatesResponse | null>(null);
  const [casesData, setCasesData] = useState<CasesResponse | null>(null);
  const [growthLoading, setGrowthLoading] = useState(false);
  const [growthError, setGrowthError] = useState<string | null>(null);
  const [keywordEdits, setKeywordEdits] = useState<Record<string, string>>({});
  const [promotingId, setPromotingId] = useState<string | null>(null);
  const [promoteOutcome, setPromoteOutcome] = useState<PromoteOutcome | null>(null);
  // Batch promotion + candidate selection (Task 19-b).
  const [selectedCandidateIds, setSelectedCandidateIds] = useState<Set<string>>(new Set());
  const [batchPromoting, setBatchPromoting] = useState(false);
  const [batchOutcome, setBatchOutcome] = useState<BatchOutcome | null>(null);
  const [deletingCaseId, setDeletingCaseId] = useState<string | null>(null);
  // A/B answer-quality comparison (Task 17-b).
  const [compareQuestion, setCompareQuestion] = useState("What is the standard warranty on the X200?");
  const [compareConfigs, setCompareConfigs] = useState<string[]>(COMPARE_CONFIGS);
  const [compareConfigA, setCompareConfigA] = useState(COMPARE_CONFIGS[0]);
  const [compareConfigB, setCompareConfigB] = useState(COMPARE_CONFIGS[1]);
  const [compareRunning, setCompareRunning] = useState(false);
  const [compareError, setCompareError] = useState<string | null>(null);
  const [compareReport, setCompareReport] = useState<CompareResponse | null>(null);
  const growthFetched = useRef(false);
  // Eval run history + downvote triage (Task 18-b).
  const [historyData, setHistoryData] = useState<EvalHistoryResponse | null>(null);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [includeAblation, setIncludeAblation] = useState(false);
  const [triageData, setTriageData] = useState<TriageResponse | null>(null);
  const [triageLoading, setTriageLoading] = useState(false);
  const [triageError, setTriageError] = useState<string | null>(null);
  // Open/resolved/all radar filter + in-flight resolve/reopen marker (19-b).
  const [triageFilter, setTriageFilter] = useState<"open" | "resolved" | "all">("open");
  const [resolvingId, setResolvingId] = useState<string | null>(null);
  const historyFetched = useRef(false);
  const triageFetched = useRef(false);
  // Word-level diff highlighting for the A/B cards (Task 18-b).
  const [diffHighlight, setDiffHighlight] = useState(true);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [sessionsLoading, setSessionsLoading] = useState(true);
  const [sessionStats, setSessionStats] = useState<{ sessions: number; total_turns: number; follow_ups_resolved: number } | null>(null);
  // Round 20 — question analytics (GET /analytics/questions) for the Memory tab.
  const [analyticsData, setAnalyticsData] = useState<AnalyticsResponse | null>(null);
  const [analyticsLoading, setAnalyticsLoading] = useState(false);
  const [analyticsError, setAnalyticsError] = useState<string | null>(null);
  const analyticsFetched = useRef(false);
  const [sessionId, setSessionId] = useState<string>("");
  const [historyFor, setHistoryFor] = useState<string | null>(null);
  const [historyTurns, setHistoryTurns] = useState<SessionTurn[] | null>(null);
  const [imgDialog, setImgDialog] = useState(false);
  const [imgFile, setImgFile] = useState<File | null>(null);
  const [imgQuestion, setImgQuestion] = useState("");
  const [imgPreview, setImgPreview] = useState<string | null>(null);
  const [imgResult, setImgResult] = useState<ImageAskResponse | null>(null);
  const [imgBusy, setImgBusy] = useState(false);
  const [sourceOpen, setSourceOpen] = useState(false);
  const [sourceDetail, setSourceDetail] = useState<SourceDetail | null>(null);
  const [sourceLoading, setSourceLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchHits, setSearchHits] = useState<SearchHit[] | null>(null);
  const [searchBusy, setSearchBusy] = useState(false);
  const [searchK, setSearchK] = useState(5);
  const [wf, setWf] = useState(DEFAULT_WF);
  const [wc, setWc] = useState(DEFAULT_WC);
  const [wa, setWa] = useState(DEFAULT_WA);
  const [rrEnabled, setRrEnabled] = useState(true);
  const [searchRerank, setSearchRerank] = useState<SearchResponse["rerank"]>(null);
  const [searchApplied, setSearchApplied] = useState<{ k: number; weights: SearchWeights; rr: boolean } | null>(null);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [activeTab, setActiveTab] = useState("knowledge");
  const [metricsText, setMetricsText] = useState<string | null>(null);
  const [metricsLoading, setMetricsLoading] = useState(false);
  const [metricsError, setMetricsError] = useState<string | null>(null);
  const metricsFetched = useRef(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const refreshStatus = useCallback(async () => {
    try {
      const [h, c, s] = await Promise.all([
        fetch(api("/health")).then((r) => r.json()),
        fetch(api("/corpus")).then((r) => r.json()),
        fetch(api("/suggestions")).then((r) => r.json()),
      ]);
      setHealth(h);
      setCorpus(c);
      setSuggestions(s.suggestions ?? []);
    } catch {
      /* service warming up */
    }
  }, []);

  const refreshStats = useCallback(async () => {
    try {
      const r = await fetch(api("/stats"));
      if (!r.ok) return;
      setStats(await r.json());
    } catch {
      /* stats are best-effort */
    }
  }, []);

  // Scrape the Prometheus exporter (text/plain, NOT JSON — Task 2).
  const refreshMetrics = useCallback(async () => {
    setMetricsLoading(true);
    setMetricsError(null);
    try {
      const r = await fetch(api("/metrics"));
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setMetricsText(await r.text());
    } catch (err) {
      setMetricsError(err instanceof Error ? err.message : "metrics unavailable");
    } finally {
      setMetricsLoading(false);
    }
  }, []);

  const refreshSessions = useCallback(async () => {
    try {
      const r = await fetch(api("/sessions"));
      const d = await r.json();
      setSessions(d.sessions ?? []);
      setSessionStats(d.stats ?? null);
    } catch {
      /* ignore */
    } finally {
      setSessionsLoading(false);
    }
  }, []);

  // Round 20 — question analytics for the Memory tab (top asked questions
  // with route/fallback/citation/latency/vote aggregates from the ledger).
  const refreshAnalytics = useCallback(async () => {
    setAnalyticsLoading(true);
    setAnalyticsError(null);
    try {
      const r = await fetch(api("/analytics/questions?limit=8"));
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const d = (await r.json()) as AnalyticsResponse;
      setAnalyticsData(d);
    } catch (err) {
      setAnalyticsError(err instanceof Error ? err.message : "analytics unavailable");
    } finally {
      setAnalyticsLoading(false);
    }
  }, []);

  useEffect(() => {
    setSessionId(() => crypto.randomUUID().slice(0, 8));
    void refreshStatus();
    void refreshSessions();
    void refreshStats();
    const t = setInterval(() => void refreshStatus(), 30000);
    const t2 = setInterval(() => void refreshStats(), 15000);
    return () => {
      clearInterval(t);
      clearInterval(t2);
    };
  }, [refreshStatus, refreshSessions, refreshStats]);

  // Fetch the Prometheus exposition once, when the Pulse tab first opens.
  useEffect(() => {
    if (activeTab === "pulse" && !metricsFetched.current) {
      metricsFetched.current = true;
      void refreshMetrics();
    }
  }, [activeTab, refreshMetrics]);

  // Round 20 — fetch question analytics when the Memory tab first opens.
  useEffect(() => {
    if (activeTab === "sessions" && !analyticsFetched.current) {
      analyticsFetched.current = true;
      void refreshAnalytics();
    }
  }, [activeTab, refreshAnalytics]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  // "/" focuses the question input
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "/" && document.activeElement !== inputRef.current) {
        const tag = (document.activeElement as HTMLElement | null)?.tagName;
        if (tag !== "INPUT" && tag !== "TEXTAREA") {
          e.preventDefault();
          inputRef.current?.focus();
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // ------------------------------------------------------------- ask (SSE)
  const ask = useCallback(
    async (question: string) => {
      const q = question.trim();
      if (!q || busy) return;
      setBusy(true);
      setInput("");
      const userMsg: ChatMessage = { id: crypto.randomUUID(), role: "user", text: q };
      const botId = crypto.randomUUID();
      const botMsg: ChatMessage = {
        id: botId,
        role: "assistant",
        question: q,
        streaming: true,
        path: [],
      };
      setMessages((prev) => [...prev, userMsg, botMsg]);

      try {
        const resp = await fetch(api("/ask/stream"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: q, session_id: sessionId || undefined }),
        });
        if (!resp.ok || !resp.body) throw new Error(`service returned ${resp.status}`);
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const frames = buffer.split("\n\n");
          buffer = frames.pop() ?? "";
          for (const frame of frames) {
            const evLine = frame.split("\n").find((l) => l.startsWith("event: "));
            const dataLine = frame.split("\n").find((l) => l.startsWith("data: "));
            if (!evLine || !dataLine) continue;
            const event = evLine.slice(7).trim();
            let payload: Record<string, unknown> = {};
            try {
              payload = JSON.parse(dataLine.slice(6));
            } catch {
              continue;
            }
            if (event === "node") {
              const node = String(payload.node ?? "");
              if (node) {
                // Progressive path display: each node event appends a live chip.
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === botId && m.role === "assistant" && !m.path.includes(node)
                      ? { ...m, path: [...m.path, node] }
                      : m
                  )
                );
                // The generator node reports its own TTFT — a fallback source
                // if the first answer_delta did not carry one.
                if (node === "generator") {
                  const detail = payload.detail as { ttft_ms?: unknown } | undefined;
                  const genTtft =
                    detail && typeof detail.ttft_ms === "number" ? detail.ttft_ms : undefined;
                  if (genTtft !== undefined) {
                    setMessages((prev) =>
                      prev.map((m) =>
                        m.id === botId && m.role === "assistant" && m.ttftMs === undefined
                          ? { ...m, ttftMs: genTtft }
                          : m
                      )
                    );
                  }
                }
              }
            } else if (event === "answer_delta") {
              const delta = String(payload.delta ?? "");
              // ttft_ms rides only on the FIRST delta of a generation.
              const ttft = typeof payload.ttft_ms === "number" ? payload.ttft_ms : undefined;
              if (delta || ttft !== undefined) {
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === botId && m.role === "assistant"
                      ? {
                          ...m,
                          streamedAnswer: delta ? (m.streamedAnswer ?? "") + delta : m.streamedAnswer,
                          ttftMs: m.ttftMs === undefined ? ttft : m.ttftMs,
                        }
                      : m
                  )
                );
              }
            } else if (event === "final") {
              const finalData = payload.payload as unknown as AskResponse;
              setMessages((prev) =>
                prev.map((m) => (m.id === botId ? { ...m, streaming: false, data: finalData } : m))
              );
              if (finalData.follow_up_applied) {
                void refreshSessions();
              }
            } else if (event === "error") {
              throw new Error(String(payload.error ?? "stream error"));
            }
          }
        }
        void refreshSessions();
        void refreshStats();
      } catch (err) {
        const message = err instanceof Error ? err.message : "request failed";
        setMessages((prev) =>
          prev.map((m) =>
            m.id === botId
              ? {
                  ...m,
                  streaming: false,
                  data: {
                    question: q,
                    answer: `The assistant service could not be reached (${message}). Verify the Python service is running on port ${API_PORT}.`,
                    citations: [],
                    route: "out_of_scope",
                    agent_path: ["router", "fallback"],
                    trace: [],
                    retries: 0,
                    fallback: true,
                    latency_ms: 0,
                    serving_mode: "simulation",
                    request_id: "",
                    session_id: "",
                    effective_question: q,
                    follow_up_applied: false,
                    inherited_entities: [],
                    turn_index: 0,
                  },
                }
              : m
          )
        );
        toast({ title: "Request failed", description: message, variant: "destructive" });
      } finally {
        setBusy(false);
        void refreshStats();
      }
    },
    [busy, toast, sessionId, refreshSessions, refreshStats]
  );

  const newConversation = useCallback(() => {
    setSessionId(() => crypto.randomUUID().slice(0, 8));
    setMessages([]);
    setHistoryFor(null);
    setHistoryTurns(null);
    toast({ title: "New conversation started", description: "Follow-up context was reset." });
  }, [toast]);

  // ------------------------------------------------------------ side panels
  const runBenchmark = useCallback(async () => {
    setBenchRunning(true);
    try {
      const resp = await fetch(api("/benchmark"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ concurrency: 8, requests: 24 }),
      });
      setBench(await resp.json());
      toast({ title: "Benchmark complete", description: "24 concurrent requests finished." });
    } catch {
      toast({ title: "Benchmark failed", variant: "destructive" });
    } finally {
      setBenchRunning(false);
    }
  }, [toast]);

  // ------------------------------ eval run history + quality triage (18-b)
  // NOTE: both callbacks are declared before runEval/runAblation/submitFeedback
  // (they reference them in their dependency arrays — keep this order).
  const refreshHistory = useCallback(
    async (include?: boolean) => {
      const flag = include ?? includeAblation;
      setHistoryLoading(true);
      setHistoryError(null);
      try {
        const r = await fetch(api(`/eval/history?include_ablation=${flag}`));
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        setHistoryData((await r.json()) as EvalHistoryResponse);
      } catch (err) {
        setHistoryError(err instanceof Error ? err.message : "run history unavailable");
      } finally {
        setHistoryLoading(false);
      }
    },
    [includeAblation]
  );

  const refreshTriage = useCallback(async () => {
    setTriageLoading(true);
    setTriageError(null);
    try {
      const r = await fetch(api("/quality/triage"));
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setTriageData((await r.json()) as TriageResponse);
    } catch (err) {
      setTriageError(err instanceof Error ? err.message : "triage unavailable");
    } finally {
      setTriageLoading(false);
    }
  }, []);

  const runEval = useCallback(async () => {
    setEvalRunning(true);
    try {
      const resp = await fetch(api("/eval"), { method: "POST" });
      if (!resp.ok) throw new Error("evaluation failed");
      const report: EvalReport = await resp.json();
      setEvalReport(report);
      toast({
        title: `Evaluation complete — ${report.passed}/${report.total_cases} passed`,
        description: `accuracy ${(report.accuracy * 100).toFixed(0)}% · retrieval hit ${report.retrieval_hit_rate !== null ? `${(report.retrieval_hit_rate * 100).toFixed(0)}%` : "—"}`,
      });
      // Every POST /eval appends a run — refresh the trend (Task 18-b).
      void refreshHistory();
    } catch (err) {
      toast({
        title: "Evaluation failed",
        description: err instanceof Error ? err.message : "unknown error",
        variant: "destructive",
      });
    } finally {
      setEvalRunning(false);
    }
  }, [toast, refreshHistory]);

  const runAblation = useCallback(async () => {
    setAblationRunning(true);
    try {
      const resp = await fetch(api("/eval/ablation"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      if (!resp.ok) throw new Error("ablation study failed");
      const report: AblationReport = await resp.json();
      setAblationReport(report);
      toast({
        title: "Ablation study complete",
        description: `winner: ${report.winner} · ${report.configs.length} configs in ${(report.duration_ms / 1000).toFixed(1)}s`,
      });
      // Each ablation config appends an ablation-tagged run — refresh history.
      void refreshHistory();
    } catch (err) {
      toast({
        title: "Ablation study failed",
        description: err instanceof Error ? err.message : "unknown error",
        variant: "destructive",
      });
    } finally {
      setAblationRunning(false);
    }
  }, [toast, refreshHistory]);

  // ------------------------------------------------- golden-set growth
  // Task 17-b: fetch candidates + merged cases together; keyword edits are
  // seeded from the backend's suggested patterns and stay editable.
  const refreshGrowth = useCallback(async () => {
    setGrowthLoading(true);
    setGrowthError(null);
    try {
      const [cand, cases] = await Promise.all([
        fetch(api("/eval/candidates")).then((r) => {
          if (!r.ok) throw new Error(`candidates HTTP ${r.status}`);
          return r.json() as Promise<CandidatesResponse>;
        }),
        fetch(api("/eval/cases")).then((r) => {
          if (!r.ok) throw new Error(`cases HTTP ${r.status}`);
          return r.json() as Promise<CasesResponse>;
        }),
      ]);
      setCandidatesData(cand);
      setCasesData(cases);
      // Prune stale selections/keyword edits against the live candidate list
      // (promoted or otherwise vanished candidates drop out of both).
      setSelectedCandidateIds((prev) => {
        const live = new Set((cand.candidates ?? []).map((c) => c.request_id));
        const next = new Set<string>();
        for (const id of prev) if (live.has(id)) next.add(id);
        return next;
      });
      setKeywordEdits((prev) => {
        const next: Record<string, string> = {};
        for (const c of cand.candidates ?? []) {
          next[c.request_id] = prev[c.request_id] ?? (c.suggested_keywords ?? []).join(", ");
        }
        return next;
      });
    } catch (err) {
      setGrowthError(err instanceof Error ? err.message : "growth data unavailable");
    } finally {
      setGrowthLoading(false);
    }
  }, []);

  // Fetch candidates + golden cases once, when the Eval tab first opens.
  useEffect(() => {
    if (activeTab === "eval" && !growthFetched.current) {
      growthFetched.current = true;
      void refreshGrowth();
    }
  }, [activeTab, refreshGrowth]);

  // Fetch the run history + downvote triage once, when the Eval tab opens
  // (Task 18-b) — same fetch-once rhythm as the growth data.
  useEffect(() => {
    if (activeTab === "eval") {
      if (!historyFetched.current) {
        historyFetched.current = true;
        void refreshHistory();
      }
      if (!triageFetched.current) {
        triageFetched.current = true;
        void refreshTriage();
      }
    }
  }, [activeTab, refreshHistory, refreshTriage]);

  const promoteCandidate = useCallback(
    async (requestId: string) => {
      setPromotingId(requestId);
      try {
        // Edited patterns (comma-joined) override the backend suggestions;
        // an empty field falls back to the server-side miner defaults.
        const raw = keywordEdits[requestId] ?? "";
        const keywords = raw.split(",").map((k) => k.trim()).filter((k) => k.length > 0);
        const body: Record<string, unknown> = { request_id: requestId };
        if (keywords.length > 0) body.keywords = keywords;
        const r = await fetch(api("/eval/candidates/promote"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (!r.ok) {
          const d = (await r.json().catch(() => null)) as { detail?: unknown } | null;
          const detail = Array.isArray(d?.detail)
            ? (d?.detail as { msg?: string }[]).map((x) => x?.msg ?? "").filter(Boolean).join("; ")
            : (d?.detail as string | undefined);
          throw new Error(detail ?? `promote failed (HTTP ${r.status})`);
        }
        const d: PromoteResponse = await r.json();
        const v = d.verification;
        const passed = Boolean(v?.passed);
        setPromoteOutcome({
          requestId,
          passed,
          matched: v?.matched ?? null,
          goldenSet: d.golden_set,
          caseId: d.case.id,
        });
        toast({
          title: passed ? "Candidate promoted" : "Promoted — verification failed",
          description: passed
            ? `${d.case.id} verified${v?.matched ? ` (matched ${v.matched})` : ""} · golden set ${d.golden_set.base} base + ${d.golden_set.user} user = ${d.golden_set.total}`
            : `${d.case.id} did not pass verification — review or delete it in the cases list (golden set ${d.golden_set.total}).`,
          variant: passed ? "default" : "destructive",
        });
        await refreshGrowth();
      } catch (err) {
        toast({
          title: "Promotion failed",
          description: err instanceof Error ? err.message : "unknown error",
          variant: "destructive",
        });
      } finally {
        setPromotingId(null);
      }
    },
    [keywordEdits, refreshGrowth, toast]
  );

  // ------------------------------ batch promote + triage lifecycle (19-b)
  // NOTE: these callbacks sit AFTER refreshGrowth/refreshTriage in source
  // order (they reference them — keep this order, TDZ applies at definition).
  const toggleCandidateSelect = useCallback((requestId: string) => {
    setSelectedCandidateIds((prev) => {
      const next = new Set(prev);
      if (next.has(requestId)) next.delete(requestId);
      else next.add(requestId);
      return next;
    });
  }, []);

  const toggleSelectAllCandidates = useCallback(() => {
    setSelectedCandidateIds((prev) => {
      const list = candidatesData?.candidates ?? [];
      const all = list.length > 0 && list.every((c) => prev.has(c.request_id));
      if (all) return new Set<string>();
      return new Set(list.map((c) => c.request_id));
    });
  }, [candidatesData]);

  // Promote every selected candidate in one batch (1–12): edited keyword
  // patterns ride along as keywords_map — per-item failures never abort it.
  const promoteSelected = useCallback(async () => {
    const ids = Array.from(selectedCandidateIds);
    if (ids.length === 0) return;
    setBatchPromoting(true);
    try {
      const keywordsMap: Record<string, string[]> = {};
      for (const id of ids) {
        const raw = keywordEdits[id] ?? "";
        const kws = raw.split(",").map((k) => k.trim()).filter((k) => k.length > 0);
        if (kws.length > 0) keywordsMap[id] = kws;
      }
      const body: Record<string, unknown> = { request_ids: ids };
      if (Object.keys(keywordsMap).length > 0) body.keywords_map = keywordsMap;
      const r = await fetch(api("/eval/candidates/promote-batch"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) {
        const d = (await r.json().catch(() => null)) as { detail?: unknown } | null;
        const detail = Array.isArray(d?.detail)
          ? (d?.detail as { msg?: string }[]).map((x) => x?.msg ?? "").filter(Boolean).join("; ")
          : (d?.detail as string | undefined);
        throw new Error(detail ?? `batch promote failed (HTTP ${r.status})`);
      }
      const d: BatchPromoteResponse = await r.json();
      setBatchOutcome({ promoted: d.promoted, failed: d.failed, results: d.results, goldenSet: d.golden_set });
      toast({
        title: "Batch promotion complete",
        description: `${d.promoted} promoted · ${d.failed} failed · golden set ${d.golden_set.base}+${d.golden_set.user}=${d.golden_set.total}`,
        variant: d.failed > 0 && d.promoted === 0 ? "destructive" : "default",
      });
      setSelectedCandidateIds(new Set<string>());
      await refreshGrowth();
    } catch (err) {
      toast({
        title: "Batch promotion failed",
        description: err instanceof Error ? err.message : "unknown error",
        variant: "destructive",
      });
    } finally {
      setBatchPromoting(false);
    }
  }, [selectedCandidateIds, keywordEdits, refreshGrowth, toast]);

  // Mark a 👎 triage item as fixed (note ≤300 chars) — moves it to the
  // resolved list. Returns true so the card can collapse its inline form.
  const resolveTriageItem = useCallback(
    async (requestId: string, note: string): Promise<boolean> => {
      setResolvingId(requestId);
      try {
        const r = await fetch(api("/quality/triage/resolve"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ request_id: requestId, note }),
        });
        if (!r.ok) {
          const d = (await r.json().catch(() => null)) as { detail?: string } | null;
          throw new Error(d?.detail ?? `HTTP ${r.status}`);
        }
        const d = (await r.json()) as { status: string; request_id: string; note: string | null; resolved_at: number; counts: { open: number; resolved: number } };
        toast({
          title: "Marked fixed",
          description: d.note
            ? `“${d.note}” · ${d.counts.open} open · ${d.counts.resolved} resolved`
            : `${d.counts.open} open · ${d.counts.resolved} resolved on the fix-list`,
        });
        await refreshTriage();
        return true;
      } catch (err) {
        toast({
          title: "Resolve failed",
          description: err instanceof Error ? err.message : "unknown error",
          variant: "destructive",
        });
        return false;
      } finally {
        setResolvingId(null);
      }
    },
    [refreshTriage, toast]
  );

  const reopenTriageItem = useCallback(
    async (requestId: string) => {
      setResolvingId(requestId);
      try {
        const r = await fetch(api("/quality/triage/reopen"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ request_id: requestId }),
        });
        if (!r.ok) {
          const d = (await r.json().catch(() => null)) as { detail?: string } | null;
          throw new Error(d?.detail ?? `HTTP ${r.status}`);
        }
        const d = (await r.json()) as { status: string; request_id: string; counts: { open: number; resolved: number } };
        toast({
          title: "Reopened",
          description: `${d.counts.open} open · ${d.counts.resolved} resolved — back on the fix-list`,
        });
        await refreshTriage();
      } catch (err) {
        toast({
          title: "Reopen failed",
          description: err instanceof Error ? err.message : "unknown error",
          variant: "destructive",
        });
      } finally {
        setResolvingId(null);
      }
    },
    [refreshTriage, toast]
  );

  const deleteGoldenCase = useCallback(
    async (caseId: string) => {
      setDeletingCaseId(caseId);
      try {
        const r = await fetch(api(`/eval/cases/${caseId}`), { method: "DELETE" });
        if (!r.ok) {
          const d = (await r.json().catch(() => null)) as { detail?: string } | null;
          throw new Error(d?.detail ?? `HTTP ${r.status}`);
        }
        const d = (await r.json()) as { status: string; case_id: string; golden_set: GoldenSetCounts };
        toast({
          title: "User case deleted",
          description: `${d.case_id} removed · golden set ${d.golden_set.base} base + ${d.golden_set.user} user = ${d.golden_set.total}`,
        });
        await refreshGrowth();
      } catch (err) {
        toast({
          title: "Delete failed",
          description: err instanceof Error ? err.message : "unknown error",
          variant: "destructive",
        });
      } finally {
        setDeletingCaseId(null);
      }
    },
    [refreshGrowth, toast]
  );

  // ------------------------------------------------------ A/B comparison
  // One question through the full graph under two named ablation configs.
  const runCompare = useCallback(async () => {
    const q = compareQuestion.trim();
    if (q.length < 2) {
      setCompareError("Enter a question (at least 2 characters).");
      return;
    }
    if (compareConfigA === compareConfigB) {
      setCompareError("Pick two different configs — A and B must differ.");
      return;
    }
    setCompareRunning(true);
    setCompareError(null);
    try {
      const r = await fetch(api("/eval/compare"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, config_a: compareConfigA, config_b: compareConfigB }),
      });
      if (!r.ok) {
        const d = (await r.json().catch(() => null)) as { detail?: unknown } | null;
        const detail = Array.isArray(d?.detail)
          ? (d?.detail as { msg?: string }[]).map((x) => x?.msg ?? "").filter(Boolean).join("; ")
          : (d?.detail as string | undefined);
        throw new Error(detail ?? `compare failed (HTTP ${r.status})`);
      }
      const d: CompareResponse = await r.json();
      setCompareReport(d);
      if (d.available_configs?.length) setCompareConfigs(d.available_configs);
      toast({
        title: "A/B comparison complete",
        description: d.verdict.answers_identical
          ? "Both configs produced identical answers."
          : `Answers differ · citation overlap ${(d.verdict.citation_overlap * 100).toFixed(0)}% · latency Δ ${d.verdict.latency_delta_ms.toFixed(1)} ms`,
      });
    } catch (err) {
      setCompareError(err instanceof Error ? err.message : "compare failed");
    } finally {
      setCompareRunning(false);
    }
  }, [compareQuestion, compareConfigA, compareConfigB, toast]);

  // Word-level diff for the A/B cards (Task 18-b): computed once per report;
  // null when identical, when the toggle is off, or (skipped=true) when either
  // answer exceeds the 400-word LCS cap.
  const compareDiff = useMemo(() => {
    if (!compareReport || compareReport.verdict.answers_identical || !diffHighlight) return null;
    return buildAnswerDiff(compareReport.side_a.answer, compareReport.side_b.answer);
  }, [compareReport, diffHighlight]);

  const ingestFiles = useCallback(
    async (files: FileList | null) => {
      if (!files || !files.length) return;
      setIngesting(true);
      try {
        const form = new FormData();
        Array.from(files).forEach((f) => form.append("files", f));
        const resp = await fetch(api("/ingest"), { method: "POST", body: form });
        if (!resp.ok) throw new Error((await resp.json()).detail ?? "ingest failed");
        const data = await resp.json();
        toast({
          title: "Documents indexed",
          description: `${data.ingested.length} file(s) → ${data.total_chunks} total chunks.`,
        });
        await refreshStatus();
      } catch (err) {
        toast({
          title: "Ingestion failed",
          description: err instanceof Error ? err.message : "unknown error",
          variant: "destructive",
        });
      } finally {
        setIngesting(false);
        if (fileInputRef.current) fileInputRef.current.value = "";
      }
    },
    [refreshStatus, toast]
  );

  const reingestSample = useCallback(async () => {
    setIngesting(true);
    try {
      const resp = await fetch(api("/ingest/sample"), { method: "POST" });
      const data = await resp.json();
      toast({ title: "Demo corpus re-indexed", description: `${data.total_documents} documents, ${data.total_chunks} chunks.` });
      await refreshStatus();
    } finally {
      setIngesting(false);
    }
  }, [refreshStatus, toast]);

  const loadHistory = useCallback(async (sid: string) => {
    if (historyFor === sid) {
      setHistoryFor(null);
      setHistoryTurns(null);
      return;
    }
    setHistoryFor(sid);
    setHistoryTurns(null);
    try {
      const r = await fetch(api(`/sessions/${sid}`));
      if (!r.ok) throw new Error("session not found");
      const d = await r.json();
      setHistoryTurns(d.turns ?? []);
    } catch {
      setHistoryTurns([]);
    }
  }, [historyFor]);

  const deleteSession = useCallback(
    async (sid: string) => {
      try {
        await fetch(api(`/sessions/${sid}`), { method: "DELETE" });
        if (historyFor === sid) {
          setHistoryFor(null);
          setHistoryTurns(null);
        }
        await refreshSessions();
        toast({ title: "Session deleted", description: sid });
      } catch {
        toast({ title: "Delete failed", variant: "destructive" });
      }
    },
    [historyFor, refreshSessions, toast]
  );

  // -------------------------------------------------- source viewer + search
  const openSource = useCallback(
    async (chunkId: string) => {
      setSourceOpen(true);
      setSourceLoading(true);
      setSourceDetail(null);
      try {
        const r = await fetch(api(`/source/${chunkId}`));
        if (!r.ok) throw new Error("source chunk not found");
        setSourceDetail(await r.json());
      } catch (err) {
        setSourceOpen(false);
        toast({
          title: "Could not load source",
          description: err instanceof Error ? err.message : "unknown error",
          variant: "destructive",
        });
      } finally {
        setSourceLoading(false);
      }
    },
    [toast]
  );

  const runSearch = useCallback(
    async (q: string) => {
      const query = q.trim();
      if (query.length < 2) return;
      setSearchBusy(true);
      try {
        // Playground params: k and the cross-encoder rr flag always; weight
        // overrides only when changed from the defaults (omitted params fall
        // back to defaults server-side).
        const params = new URLSearchParams({ q: query, k: String(searchK), rr: String(rrEnabled) });
        if (wf !== DEFAULT_WF) params.set("wf", String(wf));
        if (wc !== DEFAULT_WC) params.set("wc", String(wc));
        if (wa !== DEFAULT_WA) params.set("wa", String(wa));
        const r = await fetch(api(`/search?${params.toString()}`));
        if (!r.ok) throw new Error("search failed");
        const d: SearchResponse = await r.json();
        setSearchHits(d.results ?? []);
        setSearchRerank(d.rerank ?? null);
        // Effective weights the search ran with (API restores defaults after
        // each request, so capture the applied profile client-side).
        setSearchApplied({ k: searchK, weights: { w_fused: wf, w_coverage: wc, w_agreement: wa }, rr: rrEnabled });
      } catch {
        setSearchHits([]);
        setSearchRerank(null);
      } finally {
        setSearchBusy(false);
      }
    },
    [searchK, wf, wc, wa, rrEnabled]
  );

  const resetPlayground = useCallback(() => {
    setSearchK(5);
    setWf(DEFAULT_WF);
    setWc(DEFAULT_WC);
    setWa(DEFAULT_WA);
    setRrEnabled(true);
  }, []);

  const submitFeedback = useCallback(
    async (requestId: string, rating: "up" | "down") => {
      try {
        const r = await fetch(api("/feedback"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ request_id: requestId, rating }),
        });
        if (!r.ok) throw new Error("feedback rejected");
        toast({
          title: rating === "up" ? "Thanks for the feedback" : "Feedback recorded",
          description: rating === "up" ? "Glad the answer helped." : "This helps improve answer quality.",
        });
        // A 👎 lands in the quality radar (Task 18-b) — keep it fresh.
        if (rating === "down") void refreshTriage();
      } catch {
        toast({ title: "Feedback failed", variant: "destructive" });
      }
    },
    [toast, refreshTriage]
  );

  // ------------------------------------------------- conversation export
  // Task 2: Markdown + JSON export of the live conversation, plus per-session
  // export in the Memory tab (turns fetched from GET /sessions/{id}).
  const exportConversation = useCallback(
    (format: "md" | "json") => {
      if (messages.length === 0) {
        toast({ title: "Nothing to export", description: "Ask a question first — the conversation is empty." });
        return;
      }
      const sid = sessionId || "session";
      const stamp = exportTimestamp();
      const mode = health?.serving.mode ?? "simulation";
      const turns = messages.filter((m): m is Extract<ChatMessage, { role: "assistant" }> => m.role === "assistant");
      if (format === "md") {
        const blocks = turns.map((m) => {
          const d = m.data;
          const answer = d?.answer ?? m.streamedAnswer ?? "(no answer)";
          const sources = (d?.citations ?? []).map((c) => `${c.doc} · page ${c.page}`);
          const meta = d
            ? [
                `route: ${d.route}`,
                `latency: ${d.latency_ms.toFixed(0)} ms`,
                typeof m.ttftMs === "number" ? `ttft: ${m.ttftMs.toFixed(1)} ms` : null,
                d.cache_hit ? "cache hit" : null,
                d.fallback ? "fallback" : null,
                d.decomposed ? `decomposed ×${d.sub_queries?.length ?? 0}` : null,
              ]
                .filter(Boolean)
                .join(" · ")
            : "";
          return buildTurnMarkdown(m.question, answer, sources, meta);
        });
        const md = [
          `# Conversation ${sid}`,
          "",
          `*exported ${new Date().toLocaleString()} · ${turns.length} turn${turns.length !== 1 ? "s" : ""} · serving mode: ${mode}*`,
          "",
          "---",
          "",
          blocks.join("\n\n---\n\n"),
          "",
        ].join("\n");
        downloadBlob(md, `conversation-${sid}-${stamp}.md`, "text/markdown");
      } else {
        const payload = {
          title: `Conversation ${sid}`,
          session_id: sid,
          exported_at: new Date().toISOString(),
          serving_mode: mode,
          turn_count: turns.length,
          messages: messages.map((m) =>
            m.role === "user"
              ? { role: "user", text: m.text }
              : {
                  role: "assistant",
                  question: m.question,
                  answer: m.data?.answer ?? m.streamedAnswer ?? "",
                  route: m.data?.route ?? null,
                  latency_ms: m.data?.latency_ms ?? null,
                  ttft_ms: typeof m.ttftMs === "number" ? m.ttftMs : null,
                  request_id: m.data?.request_id ?? null,
                  turn_index: m.data?.turn_index ?? null,
                  fallback: m.data?.fallback ?? false,
                  cache_hit: m.data?.cache_hit ?? false,
                  decomposed: m.data?.decomposed ?? false,
                  sub_queries: m.data?.sub_queries ?? [],
                  agent_path: m.data?.agent_path ?? m.path,
                  citations: (m.data?.citations ?? []).map((c) => ({ doc: c.doc, page: c.page, score: c.score, chunk_id: c.chunk_id })),
                  trace: (m.data?.trace ?? []).map((t) => ({ node: t.node, status: t.status, latency_ms: t.latency_ms })),
                }
          ),
        };
        downloadBlob(JSON.stringify(payload, null, 2), `conversation-${sid}-${stamp}.json`, "application/json");
      }
      toast({
        title: "Conversation exported",
        description: `${format === "md" ? "Markdown" : "JSON"} file downloaded (${turns.length} turn${turns.length !== 1 ? "s" : ""}).`,
      });
    },
    [messages, sessionId, health, toast]
  );

  const exportSession = useCallback(
    async (sid: string, format: "md" | "json") => {
      let turns: SessionTurn[] | null = historyFor === sid ? historyTurns : null;
      if (!turns) {
        try {
          const r = await fetch(api(`/sessions/${sid}`));
          if (!r.ok) throw new Error("session not found");
          const d = await r.json();
          turns = (d.turns ?? []) as SessionTurn[];
        } catch (err) {
          toast({
            title: "Export failed",
            description: err instanceof Error ? err.message : "could not load session turns",
            variant: "destructive",
          });
          return;
        }
      }
      if (!turns || turns.length === 0) {
        toast({ title: "Nothing to export", description: `Session ${sid} has no turns.` });
        return;
      }
      const stamp = exportTimestamp();
      if (format === "md") {
        const blocks = turns.map((t) =>
          buildTurnMarkdown(
            t.question,
            t.answer,
            (t.citations ?? []).map((c) => `${c.doc} · page ${c.page}`),
            [t.route ? `route: ${t.route}` : null, t.ts ? new Date(t.ts * 1000).toLocaleString() : null, t.follow_up_applied ? "follow-up resolved" : null]
              .filter(Boolean)
              .join(" · ")
          )
        );
        const md = [
          `# Session ${sid}`,
          "",
          `*exported ${new Date().toLocaleString()} · ${turns.length} turn${turns.length !== 1 ? "s" : ""} · conversation memory*`,
          "",
          "---",
          "",
          blocks.join("\n\n---\n\n"),
          "",
        ].join("\n");
        downloadBlob(md, `session-${sid}-${stamp}.md`, "text/markdown");
      } else {
        const payload = {
          title: `Session ${sid}`,
          session_id: sid,
          exported_at: new Date().toISOString(),
          turn_count: turns.length,
          turns: turns.map((t) => ({
            question: t.question,
            effective_question: t.effective_question,
            answer: t.answer,
            route: t.route,
            agent_path: t.agent_path,
            follow_up_applied: t.follow_up_applied,
            ts: t.ts ? new Date(t.ts * 1000).toISOString() : null,
            citations: (t.citations ?? []).map((c) => ({ doc: c.doc, page: c.page, score: c.score })),
          })),
        };
        downloadBlob(JSON.stringify(payload, null, 2), `session-${sid}-${stamp}.json`, "application/json");
      }
      toast({
        title: "Session exported",
        description: `${sid} — ${format === "md" ? "Markdown" : "JSON"} file downloaded (${turns.length} turn${turns.length !== 1 ? "s" : ""}).`,
      });
    },
    [historyFor, historyTurns, toast]
  );

  // Task 16 bonus (orchestrator plan): download the importable Grafana dashboard.
  const downloadGrafana = useCallback(async () => {
    try {
      const r = await fetch(api("/ops/grafana"));
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const dash = await r.text();
      downloadBlob(dash, "rag-grafana-dashboard.json", "application/json");
      toast({
        title: "Grafana dashboard downloaded",
        description: "Import rag-grafana-dashboard.json via Dashboards → Import (10 panels, 30s refresh).",
      });
    } catch (err) {
      toast({
        title: "Grafana download failed",
        description: err instanceof Error ? err.message : "unknown error",
        variant: "destructive",
      });
    }
  }, [toast]);

  // Round 20 — eval report exports (Eval tab). Both are built client-side
  // from data already fetched for the tab; the report can be shared as-is,
  // the CSV feeds spreadsheets / CI diffing of the grown golden set.
  const exportEvalReport = useCallback(() => {
    const hasAny = evalReport || (casesData && casesData.cases.length > 0) || historyData || triageData;
    if (!hasAny) {
      toast({ title: "Nothing to export yet", description: "Run the evaluation (or open the Eval tab) so there is a report to build." });
      return;
    }
    const md = buildEvalReportMarkdown({
      mode: health?.serving.mode ?? "simulation",
      evalReport,
      casesData,
      historyData,
      triageData,
    });
    downloadBlob(md, `eval-report-${exportTimestamp()}.md`, "text/markdown");
    toast({
      title: "Evaluation report exported",
      description: `Markdown file downloaded${
        evalReport ? ` — last run ${evalReport.passed}/${evalReport.total_cases} (${(evalReport.accuracy * 100).toFixed(0)}%)` : ""
      }.`,
    });
  }, [evalReport, casesData, historyData, triageData, health, toast]);

  const exportGoldenCsv = useCallback(() => {
    if (!casesData || casesData.cases.length === 0) {
      toast({ title: "No golden cases loaded", description: "Open the golden-set list in the Eval tab first." });
      return;
    }
    const csv = buildGoldenCasesCsv(casesData);
    downloadBlob(csv, `golden-cases-${exportTimestamp()}.csv`, "text/csv");
    toast({
      title: "Golden set exported",
      description: `CSV file downloaded — ${casesData.stats.total} cases (${casesData.stats.base} base + ${casesData.stats.user} user).`,
    });
  }, [casesData, toast]);

  // ------------------------------------------------------------- image QA
  const onPickImage = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setImgFile(f);
    setImgResult(null);
    const reader = new FileReader();
    reader.onload = () => setImgPreview(String(reader.result));
    reader.readAsDataURL(f);
  };

  const submitImage = useCallback(async () => {
    if (!imgFile || !imgQuestion.trim()) return;
    setImgBusy(true);
    try {
      const form = new FormData();
      form.append("file", imgFile);
      form.append("question", imgQuestion.trim());
      const resp = await fetch(api("/ask-image"), { method: "POST", body: form });
      if (!resp.ok) throw new Error((await resp.json()).detail ?? "image QA failed");
      setImgResult(await resp.json());
    } catch (err) {
      toast({
        title: "Image Q&A failed",
        description: err instanceof Error ? err.message : "unknown error",
        variant: "destructive",
      });
    } finally {
      setImgBusy(false);
    }
  }, [imgFile, imgQuestion, toast]);

  const mode = health?.serving.mode ?? "simulation";
  const live = mode === "vllm";
  // Merged golden-set size (base + promoted user cases) for the Eval tab.
  const evalTotal = casesData?.stats.total ?? 16;

  return (
    <div className="flex min-h-screen flex-col bg-background">
      {/* ---------------------------------------------------------- header */}
      <header className="sticky top-0 z-20 border-b border-border/70 bg-background/85 shadow-[0_1px_12px_-8px_rgba(0,0,0,0.25)] backdrop-blur">
        <div className="mx-auto flex w-full max-w-6xl items-center gap-3 px-4 py-3">
          <div className="relative flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-emerald-500 to-teal-600 text-white shadow-sm ring-1 ring-emerald-500/30">
            <BookOpen className="h-5 w-5" />
            <span className="absolute -bottom-1 -right-1 flex h-3 w-3 items-center justify-center rounded-full bg-background ring-1 ring-border" aria-hidden>
              <span className={cn("h-1.5 w-1.5 rounded-full", live ? "bg-emerald-500" : "bg-amber-500")} />
            </span>
          </div>
          <div className="min-w-0">
            <h1 className="truncate bg-gradient-to-r from-emerald-700 via-teal-700 to-emerald-700 bg-clip-text text-base font-semibold leading-tight text-transparent dark:from-emerald-400 dark:via-teal-300 dark:to-emerald-400">
              Document Intelligence Assistant
            </h1>
            <p className="truncate text-[11px] text-muted-foreground">
              RAG · agentic workflow · citation-grounded · vLLM-ready
            </p>
          </div>
          <div className="ml-auto flex items-center gap-2">
            {sessionId && (
              <TooltipProvider delayDuration={200}>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Badge variant="secondary" className="hidden gap-1 font-mono text-[10px] text-muted-foreground/80 md:inline-flex">
                      <History className="h-2.5 w-2.5" />
                      {sessionId}
                    </Badge>
                  </TooltipTrigger>
                  <TooltipContent side="bottom">
                    <p className="text-[11px]">Conversation session — follow-ups inherit context.</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            )}
            {evalReport && (
              <Badge
                variant="outline"
                className={cn(
                  "hidden gap-1 font-mono text-[11px] sm:inline-flex",
                  evalReport.failed === 0
                    ? "border-emerald-300 bg-emerald-50 text-emerald-700 dark:border-emerald-800 dark:bg-emerald-950 dark:text-emerald-400"
                    : "border-amber-300 bg-amber-50 text-amber-700 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-400"
                )}
              >
                <ClipboardCheck className="h-3 w-3" />
                eval {evalReport.passed}/{evalReport.total_cases}
              </Badge>
            )}
            <ThemeToggle />
            <DropdownMenu>
              <TooltipProvider delayDuration={200}>
                <Tooltip>
                  <TooltipTrigger asChild>
                    {/* span wrapper keeps the tooltip hoverable while the
                        disabled button swallows pointer events */}
                    <span className="inline-flex">
                      <DropdownMenuTrigger asChild>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-8 w-8"
                          disabled={messages.length === 0}
                          aria-label="Export conversation"
                        >
                          <Download className="h-4 w-4" />
                        </Button>
                      </DropdownMenuTrigger>
                    </span>
                  </TooltipTrigger>
                  <TooltipContent side="bottom">
                    <p className="text-[11px]">
                      {messages.length === 0 ? "No messages yet" : "Export conversation"}
                    </p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <DropdownMenuContent align="end" className="w-48">
                <DropdownMenuLabel className="text-[10px] uppercase tracking-wide text-muted-foreground">
                  Export conversation
                </DropdownMenuLabel>
                <DropdownMenuItem
                  onClick={() => exportConversation("md")}
                  disabled={messages.length === 0}
                  className="h-11 cursor-pointer gap-2 text-xs focus-visible:outline-none"
                >
                  <FileText className="h-4 w-4 text-emerald-600" />
                  Markdown
                  <span className="ml-auto font-mono text-[9px] text-muted-foreground">.md</span>
                </DropdownMenuItem>
                <DropdownMenuItem
                  onClick={() => exportConversation("json")}
                  disabled={messages.length === 0}
                  className="h-11 cursor-pointer gap-2 text-xs"
                >
                  <FileJson className="h-4 w-4 text-teal-600" />
                  JSON
                  <span className="ml-auto font-mono text-[9px] text-muted-foreground">.json</span>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
            <TooltipProvider delayDuration={200}>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Badge
                    variant="outline"
                    className={cn(
                      "gap-1.5 font-mono text-[11px]",
                      live
                        ? "border-emerald-300 bg-emerald-50 text-emerald-700 dark:border-emerald-800 dark:bg-emerald-950 dark:text-emerald-400"
                        : "border-amber-300 bg-amber-50 text-amber-700 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-400"
                    )}
                  >
                    <span className="relative flex h-1.5 w-1.5">
                      <span className={cn("absolute inline-flex h-full w-full animate-ping rounded-full opacity-60", live ? "bg-emerald-500" : "bg-amber-500")} />
                      <span className={cn("relative inline-flex h-1.5 w-1.5 rounded-full", live ? "bg-emerald-500" : "bg-amber-500")} />
                    </span>
                    {live ? "VLLM LIVE" : "SIMULATION"}
                  </Badge>
                </TooltipTrigger>
                <TooltipContent side="bottom">
                  <p className="font-mono text-[11px]">{health ? `${health.serving.model} @ ${health.serving.vllm_base_url}` : "probing…"}</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
            {corpus && (
              <Badge variant="secondary" className="hidden gap-1 font-mono text-[11px] sm:inline-flex">
                <Database className="h-3 w-3" />
                {corpus.total_documents} docs · {corpus.total_chunks} chunks
              </Badge>
            )}
            <Button variant="ghost" size="icon" className="h-8 w-8" onClick={() => void refreshStatus()} aria-label="Refresh status">
              <RefreshCw className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </header>

      {/* ------------------------------------------------------------ body */}
      <div className="mx-auto flex w-full max-w-6xl flex-1 flex-col gap-4 px-4 py-4 lg:flex-row">
        {/* chat column */}
        <main className="flex min-w-0 flex-1 flex-col">
          <div className="card-hairline flex min-h-0 flex-1 flex-col">
            <div
              ref={scrollRef}
              className={cn(
                "thin-scrollbar flex-1 space-y-4 overflow-y-auto pb-4 pr-1",
                busy && "busy-fade-mask"
              )}
              style={{ maxHeight: "calc(100vh - 17rem)" }}
            >
            {messages.length === 0 ? (
              <div className="space-y-5 pt-6">
                <div className="relative overflow-hidden rounded-xl border border-dashed border-border bg-gradient-to-b from-muted/40 to-muted/10 p-6 text-center">
                  <div className="pointer-events-none absolute -right-16 -top-16 h-40 w-40 rounded-full bg-emerald-500/10 blur-3xl" aria-hidden />
                  <div className="pointer-events-none absolute -left-16 -bottom-16 h-40 w-40 rounded-full bg-teal-500/10 blur-3xl" aria-hidden />
                  <motion.div
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ type: "spring", stiffness: 200, damping: 16 }}
                    className="mx-auto flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-emerald-500 to-teal-600 text-white shadow-md"
                  >
                    <Bot className="h-6 w-6" />
                  </motion.div>
                  <p className="mt-3 text-sm font-semibold">Ask anything about the indexed corpus</p>
                  <p className="mx-auto mt-1 max-w-md text-xs leading-relaxed text-muted-foreground">
                    Every answer is retrieved, graded, generated and verified by the agent graph —
                    and honestly declines when the documents don&apos;t contain the answer.
                  </p>
                  <div className="mx-auto mt-3 flex flex-wrap items-center justify-center gap-1.5 font-mono text-[10px] text-muted-foreground">
                    <span className="rounded-full border border-border/60 bg-background/70 px-2 py-0.5">
                      <span className={live ? "text-emerald-600" : "text-amber-600"}>●</span> {mode}
                    </span>
                    <span className="rounded-full border border-border/60 bg-background/70 px-2 py-0.5">{corpus?.total_documents ?? "…"} docs</span>
                    <span className="rounded-full border border-border/60 bg-background/70 px-2 py-0.5">{corpus?.total_chunks ?? "…"} chunks</span>
                    {stats && stats.questions.total > 0 && (
                      <span className="rounded-full border border-border/60 bg-background/70 px-2 py-0.5">{stats.questions.total} answered</span>
                    )}
                    {stats && stats.questions.latency_p50_ms > 0 && (
                      <span className="rounded-full border border-border/60 bg-background/70 px-2 py-0.5">p50 {stats.questions.latency_p50_ms.toFixed(0)}ms</span>
                    )}
                  </div>
                  <div className="mx-auto mt-4 grid max-w-lg grid-cols-2 gap-2 sm:grid-cols-4">
                    {[
                      { icon: Database, label: "Hybrid retrieval", hint: "BM25 + dense · tunable fusion" },
                      { icon: Sparkles, label: "Agent graph", hint: "route · decompose · grade" },
                      { icon: Link2, label: "Citations", hint: "doc + page for every claim" },
                      { icon: ShieldCheck, label: "Verification", hint: "hallucination gate" },
                    ].map(({ icon: Icon, label, hint }) => (
                      <div key={label} className="rounded-lg border border-border/60 bg-card px-2 py-2 text-center transition-all hover:-translate-y-0.5 hover:border-emerald-300 hover:bg-emerald-50/40 hover:shadow-sm dark:hover:border-emerald-800 dark:hover:bg-emerald-950/40">
                        <Icon className="mx-auto h-4 w-4 text-emerald-600" />
                        <p className="mt-1 text-[11px] font-medium">{label}</p>
                        <p className="text-[9px] text-muted-foreground">{hint}</p>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="flex flex-wrap justify-center gap-2">
                  <AnimatePresence>
                    {suggestions.map((s, i) => (
                      <motion.button
                        key={s.question}
                        initial={{ opacity: 0, y: 6 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: i * 0.05 }}
                        whileTap={reduceMotion ? undefined : { scale: 0.97 }}
                        onClick={() => void ask(s.question)}
                        title={s.question}
                        className="group inline-flex items-center gap-1.5 rounded-full border border-border bg-card px-3 py-1.5 text-xs text-muted-foreground transition-all hover:-translate-y-0.5 hover:border-emerald-300 hover:bg-emerald-50 hover:text-emerald-700 hover:shadow-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500/60 focus-visible:ring-offset-1 dark:hover:border-emerald-800 dark:hover:bg-emerald-950 dark:hover:text-emerald-400"
                      >
                        {s.expects === "fallback" ? (
                          <CircleAlert className="h-3 w-3 text-amber-500" />
                        ) : s.label.toLowerCase().includes("multi-hop") ? (
                          <Split className="h-3 w-3 text-teal-500" />
                        ) : s.expects === "chitchat" ? (
                          <MessageSquare className="h-3 w-3" />
                        ) : (
                          <Zap className="h-3 w-3 text-emerald-500" />
                        )}
                        {s.label}
                      </motion.button>
                    ))}
                  </AnimatePresence>
                </div>
              </div>
            ) : (
              <AnimatePresence initial={false}>
                {messages.map((m) =>
                  m.role === "user" ? (
                    <motion.div
                      key={m.id}
                      initial={{ opacity: 0, x: 12 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.2 }}
                      className="flex justify-end gap-3"
                    >
                      <div className="max-w-[80%] rounded-xl rounded-br-sm bg-primary px-3.5 py-2.5 text-sm text-primary-foreground shadow-sm">
                        {m.text}
                      </div>
                      <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border bg-card">
                        <User className="h-4 w-4 text-muted-foreground" />
                      </div>
                    </motion.div>
                  ) : (
                    <AssistantMessage key={m.id} msg={m} onAsk={(q) => void ask(q)} onOpenSource={(cid) => void openSource(cid)} onRate={(rid, rating) => void submitFeedback(rid, rating)} />
                  )
                )}
              </AnimatePresence>
            )}
            </div>
          </div>

          {/* input bar */}
          <div className="sticky bottom-0 border-t border-border/70 bg-background/90 pt-3 backdrop-blur">
            <div className="mb-1.5 flex items-center justify-between gap-2">
              <div className="flex min-w-0 items-center gap-1.5 text-[10px] text-muted-foreground">
                <History className="h-3 w-3 shrink-0 text-violet-500" />
                <span className="truncate">
                  follow-ups like “and the warranty?” inherit context from <span className="font-mono">{sessionId || "…"}</span>
                </span>
              </div>
              <Button
                variant="ghost"
                size="sm"
                className="h-6 shrink-0 gap-1 px-2 text-[10px] text-muted-foreground"
                onClick={newConversation}
                disabled={busy}
              >
                <RefreshCw className="h-3 w-3" />
                New conversation
              </Button>
            </div>
            <form
              onSubmit={(e) => {
                e.preventDefault();
                void ask(input);
              }}
              className="flex items-end gap-2"
            >
              <Button
                type="button"
                variant="outline"
                size="icon"
                className="h-10 w-10 shrink-0"
                onClick={() => setImgDialog(true)}
                aria-label="Ask about an image"
                title="Visual Q&A"
              >
                <ImageIcon className="h-4 w-4" />
              </Button>
              <div className="relative min-w-0 flex-1">
                <Input
                  ref={inputRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Ask a grounded question…"
                  className="h-10 min-w-0 pr-16 shadow-xs transition-shadow focus-visible:shadow-md focus-visible:ring-emerald-500/40"
                  disabled={busy}
                  aria-label="Question"
                />
                <span className="pointer-events-none absolute right-3 top-1/2 flex -translate-y-1/2 items-center gap-1.5">
                  {input.length > 0 && (
                    <span className="font-mono text-[10px] text-muted-foreground/60">{input.length}</span>
                  )}
                  <kbd className="hidden rounded border border-border bg-muted px-1.5 py-0.5 font-mono text-[9px] text-muted-foreground/70 sm:inline">
                    /
                  </kbd>
                </span>
              </div>
              <Button
                type="submit"
                size="icon"
                className="h-10 w-10 shrink-0 bg-emerald-600 shadow-sm transition-all hover:bg-emerald-700 hover:shadow"
                disabled={busy || !input.trim()}
                aria-label="Send question"
              >
                {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <ArrowUp className="h-4 w-4" />}
              </Button>
            </form>
            <p className="pb-1 pt-1.5 text-center text-[10px] text-foreground/60">
              {live
                ? "Answers generated by the self-hosted vLLM model — grounded in the indexed corpus."
                : "Offline simulation mode: deterministic extractive engine (no vLLM server detected)."}
            </p>
          </div>
        </main>

        {/* side panel */}
        <aside className="w-full shrink-0 lg:w-96" aria-label="System panel">
          <Tabs defaultValue="knowledge" onValueChange={(v) => setActiveTab(v)} className="w-full">
            <TabsList className="grid h-9 w-full grid-cols-5">
              <TabsTrigger value="knowledge" className="gap-1 text-[11px]">
                <Database className="h-3.5 w-3.5" />
                <span className="hidden sm:inline">Corpus</span>
              </TabsTrigger>
              <TabsTrigger value="eval" className="gap-1 text-[11px]">
                <ClipboardCheck className="h-3.5 w-3.5" />
                <span className="hidden sm:inline">Eval</span>
              </TabsTrigger>
              <TabsTrigger value="sessions" className="gap-1 text-[11px]">
                <History className="h-3.5 w-3.5" />
                <span className="hidden sm:inline">Memory</span>
              </TabsTrigger>
              <TabsTrigger value="bench" className="gap-1 text-[11px]">
                <Gauge className="h-3.5 w-3.5" />
                <span className="hidden sm:inline">Bench</span>
              </TabsTrigger>
              <TabsTrigger value="pulse" className="gap-1 text-[11px]">
                <HeartPulse className="h-3.5 w-3.5" />
                <span className="hidden sm:inline">Pulse</span>
              </TabsTrigger>
            </TabsList>

            {/* ------------------------------------------------- knowledge */}
            <TabsContent value="knowledge" className="mt-3">
              <TabPanel>
              <Card className="card-hairline">
                <CardHeader className="pb-2">
                  <CardTitle className="flex items-center gap-2 text-sm">
                    <Database className="h-4 w-4 text-emerald-600" />
                    Knowledge Base
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="space-y-2 rounded-lg border bg-muted/30 p-2.5">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <p className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                        <SlidersHorizontal className="h-3 w-3" />
                        retrieval playground
                      </p>
                      <div className="flex items-center gap-1.5">
                        <label htmlFor="topk-input" className="font-mono text-[10px] text-muted-foreground">
                          top-k
                        </label>
                        <Input
                          id="topk-input"
                          type="number"
                          min={1}
                          max={20}
                          value={searchK}
                          onChange={(e) => {
                            const v = Number(e.target.value);
                            if (Number.isNaN(v)) return;
                            setSearchK(Math.min(20, Math.max(1, Math.round(v))));
                          }}
                          className="h-6 w-14 px-1.5 font-mono text-[11px]"
                          aria-label="Number of search results (top-k)"
                        />
                        <TooltipProvider delayDuration={200}>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Button
                                variant="ghost"
                                size="sm"
                                className="h-6 gap-1 px-1.5 text-[10px] text-muted-foreground"
                                onClick={resetPlayground}
                                aria-label="Reset retrieval playground to defaults"
                              >
                                <RotateCcw className="h-3 w-3" />
                                Reset
                              </Button>
                            </TooltipTrigger>
                            <TooltipContent side="top">
                              <p className="font-mono text-[10px]">k=5 · wf 0.55 · wc 0.35 · wa 0.10</p>
                            </TooltipContent>
                          </Tooltip>
                        </TooltipProvider>
                      </div>
                    </div>
                    {([
                      { label: "fusion", symbol: "wf", value: wf, onChange: setWf, tip: "RRF fusion of dense + BM25 ranks" },
                      { label: "coverage", symbol: "wc", value: wc, onChange: setWc, tip: "query-term coverage of the chunk" },
                      { label: "agreement", symbol: "wa", value: wa, onChange: setWa, tip: "dense/BM25 rank agreement bonus" },
                    ] as Array<{ label: string; symbol: string; value: number; onChange: (v: number) => void; tip: string }>).map(
                      ({ label, symbol, value, onChange, tip }) => (
                        <div key={symbol} className="flex items-center gap-2">
                          <TooltipProvider delayDuration={250}>
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <span
                                  className="flex w-24 shrink-0 cursor-help items-center gap-1 text-[10px] text-muted-foreground transition-colors hover:text-foreground"
                                  aria-label={`${label} weight (${symbol}) — ${tip}`}
                                >
                                  {label}
                                  <span className="font-mono opacity-70">· {symbol}</span>
                                </span>
                              </TooltipTrigger>
                              <TooltipContent side="top">
                                <p className="text-[10px]">{tip}</p>
                              </TooltipContent>
                            </Tooltip>
                          </TooltipProvider>
                          <Slider
                            value={[value]}
                            min={0}
                            max={1}
                            step={0.05}
                            onValueChange={(v) => onChange(v[0] ?? value)}
                            aria-label={`${label} weight (${symbol})`}
                            className="min-w-0 flex-1"
                          />
                          <span className="w-9 shrink-0 text-right font-mono text-[10px] tabular-nums text-muted-foreground">
                            {value.toFixed(2)}
                          </span>
                        </div>
                      )
                    )}
                    {/* Cross-encoder rerank stage toggle (Task 1b) */}
                    <div className="flex items-center justify-between gap-2 border-t border-border/60 pt-1.5">
                      <TooltipProvider delayDuration={200}>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <label
                              htmlFor="rerank-switch"
                              className="flex cursor-help items-center gap-1.5 rounded-sm text-[10px] text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500/60 focus-visible:ring-offset-1"
                            >
                              <ArrowDownUp className="h-3 w-3 text-teal-600 dark:text-teal-400" aria-hidden />
                              cross-encoder
                              <span className="font-mono opacity-70">· rerank</span>
                            </label>
                          </TooltipTrigger>
                          <TooltipContent side="top" className="max-w-64">
                            <p className="text-[10px] leading-relaxed">
                              Joint (query, chunk) scoring stage after RRF fusion — re-orders the fused
                              candidate pool before top-k is cut. Simulation mode uses the deterministic
                              lexical surrogate.
                            </p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                      <Switch
                        id="rerank-switch"
                        checked={rrEnabled}
                        onCheckedChange={(v) => setRrEnabled(v)}
                        aria-label="Toggle the cross-encoder rerank stage"
                        title="Cross-encoder rerank stage"
                        className="data-[state=checked]:bg-teal-600 data-[state=checked]:dark:bg-teal-500"
                      />
                    </div>
                  </div>
                  <form
                    onSubmit={(e) => {
                      e.preventDefault();
                      void runSearch(searchQuery);
                    }}
                    className="flex items-center gap-2"
                  >
                    <div className="relative min-w-0 flex-1">
                      <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground/60" />
                      <Input
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        placeholder="Search the corpus (hybrid BM25 + dense)…"
                        className="h-8 pl-8 text-xs"
                        aria-label="Corpus search"
                      />
                    </div>
                    <Button
                      type="submit"
                      variant="outline"
                      size="sm"
                      className="h-8 w-8 shrink-0 p-0"
                      disabled={searchBusy || searchQuery.trim().length < 2}
                      aria-label="Run corpus search"
                    >
                      {searchBusy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <TextSearch className="h-3.5 w-3.5" />}
                    </Button>
                  </form>

                  {(searchHits !== null || searchBusy) && (
                    <motion.div initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} className="space-y-1.5">
                      <p className="flex flex-wrap items-center justify-between gap-1 text-[10px] uppercase tracking-wide text-muted-foreground">
                        <span className="flex min-w-0 flex-wrap items-center gap-1">
                          <span className="shrink-0">retrieval results</span>
                          {searchApplied && (
                            <span className="flex flex-wrap items-center gap-1 normal-case" aria-label="Weights applied to this search">
                              <span className="rounded bg-teal-500/10 px-1 py-px font-mono text-[9px] text-teal-700 dark:text-teal-400">k={searchApplied.k}</span>
                              <span className="rounded bg-teal-500/10 px-1 py-px font-mono text-[9px] text-teal-700 dark:text-teal-400">wf {searchApplied.weights.w_fused.toFixed(2)}</span>
                              <span className="rounded bg-amber-500/10 px-1 py-px font-mono text-[9px] text-amber-700 dark:text-amber-400">wc {searchApplied.weights.w_coverage.toFixed(2)}</span>
                              <span className="rounded bg-emerald-500/10 px-1 py-px font-mono text-[9px] text-emerald-700 dark:text-emerald-400">wa {searchApplied.weights.w_agreement.toFixed(2)}</span>
                              {!searchApplied.rr && (
                                <span className="rounded bg-amber-500/10 px-1 py-px font-mono text-[9px] text-amber-700 dark:text-amber-400">ce off</span>
                              )}
                            </span>
                          )}
                        </span>
                        <button
                          className="shrink-0 rounded px-1 text-muted-foreground/70 transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500/40"
                          onClick={() => {
                            setSearchHits(null);
                            setSearchApplied(null);
                            setSearchRerank(null);
                          }}
                          aria-label="Clear search results"
                        >
                          clear
                        </button>
                      </p>
                      {searchHits !== null && !searchBusy && (searchRerank || searchApplied?.rr === false) && (
                        <motion.p
                          initial={{ opacity: 0, y: 2 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ duration: 0.2 }}
                          role="status"
                          className={cn(
                            "flex items-center gap-1.5 rounded-md border px-2 py-1 font-mono text-[10px]",
                            searchRerank
                              ? "border-teal-200/60 bg-teal-50/40 text-teal-700 dark:border-teal-900/60 dark:bg-teal-950/30 dark:text-teal-400"
                              : "border-border/60 bg-muted/30 text-muted-foreground"
                          )}
                          aria-label="Cross-encoder rerank telemetry"
                        >
                          <ArrowDownUp className="h-3 w-3 shrink-0" aria-hidden />
                          {searchRerank ? (
                            <>
                              {searchRerank.stage} · {searchRerank.reranked} candidates · {searchRerank.reorders} reorders ·{" "}
                              {searchRerank.latency_ms.toFixed(1)} ms
                              <span className="opacity-70">({searchRerank.mode})</span>
                            </>
                          ) : (
                            <>cross-encoder off — pure fusion order</>
                          )}
                        </motion.p>
                      )}
                      {searchBusy ? (
                        <div className="space-y-1.5" aria-label="Searching the corpus">
                          {[0, 1, 2].map((i) => (
                            <div key={i} className="flex items-center gap-2 rounded-md border border-border/60 bg-muted/30 px-2.5 py-1.5">
                              <Skeleton className="h-4 w-4 shrink-0 rounded" />
                              <div className="min-w-0 flex-1 space-y-1">
                                <Skeleton className="h-3 w-2/3" />
                                <Skeleton className="h-2.5 w-11/12" />
                              </div>
                              <Skeleton className="h-4 w-9 shrink-0" />
                            </div>
                          ))}
                        </div>
                      ) : searchHits.length === 0 ? (
                        <div className="flex flex-col items-center gap-1.5 rounded-xl border border-dashed border-border bg-muted/20 px-3 py-8 text-center">
                          <TextSearch className="h-5 w-5 text-muted-foreground/50" aria-hidden />
                          <p className="max-w-xs text-[11px] leading-relaxed text-muted-foreground">
                            No chunk passed the hybrid retriever — try different terms or adjust
                            the playground weights.
                          </p>
                        </div>
                      ) : (
                        searchHits.map((h, i) => (
                          <button
                            key={h.chunk_id}
                            onClick={() => void openSource(h.chunk_id)}
                            className="hover-lift flex w-full items-start gap-2 rounded-md border border-border/60 bg-muted/30 px-2.5 py-1.5 text-left hover:border-emerald-300 hover:bg-emerald-50/40 hover:shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500/60 focus-visible:ring-offset-1 dark:hover:border-emerald-800 dark:hover:bg-emerald-950/40"
                            aria-label={`Open source: ${h.doc} page ${h.page}`}
                          >
                            <span className="mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded bg-emerald-600/10 font-mono text-[9px] font-bold text-emerald-700 dark:text-emerald-400">
                              {i + 1}
                            </span>
                            <div className="min-w-0 flex-1">
                              <p className="truncate font-mono text-[11px] font-medium">
                                {h.doc}
                                <span className="text-muted-foreground"> · p.{h.page}</span>
                              </p>
                              <p className="line-clamp-2 text-[10px] leading-snug text-muted-foreground">{h.preview}</p>
                              <p className="mt-0.5 flex flex-wrap items-center gap-1 font-mono text-[9px] text-muted-foreground">
                                <span className="rounded bg-teal-500/10 px-1 py-px text-teal-700 dark:text-teal-400">dense #{h.dense_rank || "—"}</span>
                                <span className="rounded bg-amber-500/10 px-1 py-px text-amber-700 dark:text-amber-400">bm25 #{h.bm25_rank || "—"}</span>
                                <span>cov {(h.coverage * 100).toFixed(0)}%</span>
                                {typeof h.rerank_score === "number" && (
                                  <TooltipProvider delayDuration={150}>
                                    <Tooltip>
                                      <TooltipTrigger asChild>
                                        <span
                                          className="rounded bg-teal-500/10 px-1 py-px text-teal-700 dark:text-teal-400"
                                          aria-label="Cross-encoder rerank score"
                                        >
                                          ce {h.rerank_score.toFixed(2)}
                                        </span>
                                      </TooltipTrigger>
                                      <TooltipContent side="top">
                                        <p className="max-w-56 text-[10px]">
                                          cross-encoder (query, chunk) pair score — re-sorted the fused pool
                                          (lexical surrogate in simulation mode)
                                        </p>
                                      </TooltipContent>
                                    </Tooltip>
                                  </TooltipProvider>
                                )}
                                {typeof h.fusion_rank === "number" && <RankShiftChip from={h.fusion_rank} to={i + 1} />}
                              </p>
                            </div>
                            <Badge variant="secondary" className="shrink-0 font-mono text-[10px]">
                              {h.score.toFixed(2)}
                            </Badge>
                          </button>
                        ))
                      )}
                    </motion.div>
                  )}

                  {corpus ? (
                    <>
                      <div className="grid grid-cols-3 gap-2 text-center">
                        {[
                          ["docs", corpus.total_documents],
                          ["chunks", corpus.total_chunks],
                          ["dims", corpus.embedding_dim],
                        ].map(([label, value]) => (
                          <div key={String(label)} className="rounded-lg border bg-muted/40 px-2 py-1.5">
                            <p className="font-mono text-sm font-semibold">{String(value)}</p>
                            <p className="text-[10px] uppercase tracking-wide text-muted-foreground">{String(label)}</p>
                          </div>
                        ))}
                      </div>
                      <div className="thin-scrollbar max-h-56 overflow-y-auto rounded-md border">
                        <div className="divide-y divide-border/60">
                          {corpus.documents.map((d) => (
                            <TooltipProvider key={d.doc} delayDuration={150}>
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <div className="hover-lift flex items-center gap-2 px-2.5 py-1.5 hover:bg-muted/50">
                                    <FileText className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                                    <span className="min-w-0 flex-1 truncate font-mono text-[11px]">{d.doc}</span>
                                    <span className="shrink-0 text-[10px] text-muted-foreground">
                                      {d.pages}p · {d.chunks}c{d.figures > 0 ? ` · ${d.figures}f` : ""}
                                    </span>
                                  </div>
                                </TooltipTrigger>
                                <TooltipContent side="left">
                                  <p className="font-mono text-[10px]">{d.kind} · {d.words} words · {d.chunks} chunks</p>
                                </TooltipContent>
                              </Tooltip>
                            </TooltipProvider>
                          ))}
                        </div>
                      </div>
                    </>
                  ) : (
                    <div className="space-y-2">
                      <Skeleton className="h-9 w-full" />
                      <Skeleton className="h-32 w-full" />
                    </div>
                  )}
                  <div className="flex gap-2">
                    <input
                      ref={fileInputRef}
                      type="file"
                      multiple
                      accept=".pdf,.md,.txt,.markdown"
                      className="hidden"
                      onChange={(e) => void ingestFiles(e.target.files)}
                      aria-label="Upload documents"
                    />
                    <Button
                      variant="outline"
                      size="sm"
                      className="h-8 flex-1 gap-1.5 text-xs"
                      onClick={() => fileInputRef.current?.click()}
                      disabled={ingesting}
                    >
                      {ingesting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Upload className="h-3.5 w-3.5" />}
                      Upload docs
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      className="h-8 flex-1 gap-1.5 text-xs"
                      onClick={() => void reingestSample()}
                      disabled={ingesting}
                    >
                      <RefreshCw className="h-3.5 w-3.5" />
                      Re-index demo
                    </Button>
                  </div>
                </CardContent>
              </Card>
              </TabPanel>
            </TabsContent>

            {/* ------------------------------------------------------ eval */}
            <TabsContent value="eval" className="mt-3">
              <TabPanel>
              <Card className="card-hairline">
                <CardHeader className="pb-2">
                  <CardTitle className="flex items-center gap-2 text-sm">
                    <ClipboardCheck className="h-4 w-4 text-emerald-600" />
                    Golden-Set Evaluation
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <p className="text-[11px] leading-relaxed text-muted-foreground">
                    {evalTotal} golden Q&amp;A cases{casesData ? ` (${casesData.stats.base} base + ${casesData.stats.user} 👍-grown)` : ""} — incl. multi-hop
                    comparisons — run through the full agent pipeline, scoring route correctness,
                    retrieval hit@k, answer fact-support and citation faithfulness (grounding judge).
                  </p>
                  <Button
                    size="sm"
                    className="h-8 w-full gap-1.5 text-xs"
                    onClick={() => void runEval()}
                    disabled={evalRunning}
                  >
                    {evalRunning ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <ClipboardCheck className="h-3.5 w-3.5" />}
                    {evalRunning ? `Running ${evalTotal} cases…` : "Run evaluation"}
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-8 w-full gap-1.5 text-xs"
                    onClick={() => void runAblation()}
                    disabled={ablationRunning || evalRunning}
                  >
                    {ablationRunning ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <FlaskConical className="h-3.5 w-3.5" />}
                    {ablationRunning ? "Running 7 configs…" : "Run ablation study"}
                  </Button>

                  {ablationRunning && !ablationReport && (
                    <div className="space-y-1.5 rounded-md border bg-muted/30 p-2.5" aria-label="Ablation study loading">
                      <Skeleton className="h-6 w-full" />
                      <Skeleton className="h-6 w-full" />
                      <Skeleton className="h-6 w-2/3" />
                    </div>
                  )}

                  {ablationReport && <AblationTable report={ablationReport} />}

                  {ablationReport && <Separator className="bg-border/60" />}

                  {evalReport && (
                    <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} className="space-y-3">
                      <div className="flex items-center gap-3 rounded-lg border bg-muted/30 p-3">
                        <ScoreRing value={evalReport.accuracy} />
                        <div className="min-w-0 flex-1 space-y-1.5 font-mono text-[11px]">
                          <div className="flex justify-between gap-2">
                            <span className="text-muted-foreground">retrieval hit</span>
                            <span className="font-semibold text-emerald-600">
                              {evalReport.retrieval_hit_rate !== null ? `${(evalReport.retrieval_hit_rate * 100).toFixed(0)}%` : "—"}
                            </span>
                          </div>
                          <div className="flex justify-between gap-2">
                            <span className="text-muted-foreground">fact support</span>
                            <span className="font-semibold text-emerald-600">
                              {(evalReport.fact_support_rate * 100).toFixed(0)}%
                            </span>
                          </div>
                          <div className="flex items-center justify-between gap-2">
                            <span className="text-muted-foreground">faithfulness</span>
                            <TooltipProvider delayDuration={150}>
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <span className="cursor-help font-semibold text-emerald-600 underline decoration-dotted underline-offset-2">
                                    {evalReport.faithfulness_avg !== null ? `${(evalReport.faithfulness_avg * 100).toFixed(0)}%` : "—"}
                                  </span>
                                </TooltipTrigger>
                                <TooltipContent side="left" className="max-w-64">
                                  <p className="text-[11px]">RAGAS-style grounding score: how well each answer is supported by its own citations.</p>
                                  <p className="mt-0.5 font-mono text-[10px] text-muted-foreground">
                                    judge: {evalReport.faithfulness_judge ?? "—"}
                                    {evalReport.faithfulness_judge === "llm" ? " (LLM-as-judge via served model)" : " (deterministic lexical judge)"}
                                  </p>
                                </TooltipContent>
                              </Tooltip>
                            </TooltipProvider>
                          </div>
                          <div className="flex justify-between gap-2">
                            <span className="text-muted-foreground">lat p50 / p95</span>
                            <span>
                              {evalReport.latency_p50_ms.toFixed(0)} / {evalReport.latency_p95_ms.toFixed(0)} ms
                            </span>
                          </div>
                          <p className="pt-0.5 text-[9px] text-muted-foreground">
                            ran {new Date(evalReport.ran_at).toLocaleTimeString()}
                          </p>
                        </div>
                      </div>

                      {Object.keys(evalReport.by_category).length > 0 && (
                        <div className="space-y-1">
                          {Object.entries(evalReport.by_category).map(([cat, s]) => (
                            <div key={cat} className="flex items-center gap-2 text-[11px]">
                              <span className="w-20 shrink-0 capitalize text-muted-foreground">{cat}</span>
                              <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
                                <motion.div
                                  initial={{ width: 0 }}
                                  animate={{ width: `${(s.passed / Math.max(1, s.total)) * 100}%` }}
                                  transition={{ duration: 0.6, ease: "easeOut" }}
                                  className={cn(
                                    "h-full rounded-full",
                                    s.passed === s.total ? "bg-emerald-500" : "bg-amber-500"
                                  )}
                                />
                              </div>
                              <span className="shrink-0 font-mono text-muted-foreground">
                                {s.passed}/{s.total}
                              </span>
                            </div>
                          ))}
                        </div>
                      )}

                      <div className="thin-scrollbar max-h-80 overflow-y-auto rounded-md border">
                        <div className="divide-y divide-border/60">
                          {evalReport.results.map((r) => (
                            <TooltipProvider key={r.id} delayDuration={150}>
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <div className="flex items-start gap-2 px-2.5 py-2 transition-colors hover:bg-muted/50">
                                    {r.passed ? (
                                      <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 shrink-0 text-emerald-600" />
                                    ) : (
                                      <XCircle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-red-500" />
                                    )}
                                    <div className="min-w-0 flex-1">
                                      <p className="truncate text-[11px] font-medium">{r.question}</p>
                                      <p className="truncate font-mono text-[10px] text-muted-foreground">
                                        {r.route} · {r.latency_ms.toFixed(0)}ms
                                        {r.retrieved_doc ? ` · ${r.retrieved_doc}` : ""}
                                      </p>
                                    </div>
                                    <div className="flex shrink-0 flex-col items-end gap-1">
                                      <Badge
                                        variant="outline"
                                        className={cn("px-1.5 text-[9px]", CATEGORY_TONES[r.category] ?? "")}
                                      >
                                        {r.category}
                                      </Badge>
                                      <FaithChip f={r.faithfulness} />
                                    </div>
                                  </div>
                                </TooltipTrigger>
                                <TooltipContent side="left" className="max-w-80">
                                  <p className="text-[11px] leading-relaxed">{r.answer_preview}</p>
                                  {r.note && <p className="mt-1 text-[10px] italic text-muted-foreground">{r.note}</p>}
                                  {r.faithfulness && r.faithfulness.score !== null && (
                                    <p className="mt-1 font-mono text-[10px] text-muted-foreground">
                                      faithfulness {(r.faithfulness.score * 100).toFixed(0)}% · {r.faithfulness.supported_claims}/{r.faithfulness.total_claims} claims
                                    </p>
                                  )}
                                </TooltipContent>
                              </Tooltip>
                            </TooltipProvider>
                          ))}
                        </div>
                      </div>
                    </motion.div>
                  )}

                  {/* ------------------------------------ eval run history */}
                  {/* Task 18-b: persisted run ring → accuracy/latency trend. */}
                  <Separator className="bg-border/60" />
                  <motion.div
                    initial={reduceMotion ? undefined : { opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.25 }}
                    className="space-y-2.5"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <TooltipProvider delayDuration={200}>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <p className="flex cursor-help items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                              <History className="h-3.5 w-3.5 text-teal-600 dark:text-teal-500" aria-hidden />
                              eval run history
                            </p>
                          </TooltipTrigger>
                          <TooltipContent side="top" className="max-w-64">
                            <p className="text-[10px] leading-relaxed">
                              every POST /eval and every ablation config appends a persisted run —
                              accuracy and p50 latency over time show whether the golden set and
                              retrieval changes help or hurt.
                            </p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                      <div className="flex shrink-0 items-center gap-1">
                        <TooltipProvider delayDuration={200}>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Button
                                variant="ghost"
                                size="sm"
                                className="h-8 gap-1 px-2 text-[10px] text-muted-foreground transition-colors hover:text-teal-700 dark:hover:text-teal-400"
                                onClick={exportEvalReport}
                                aria-label="Export the evaluation report as Markdown"
                                title="Download a Markdown report: last run, golden set, run history, triage"
                              >
                                <FileDown className="h-3 w-3" />
                                <span className="hidden sm:inline">Report</span>
                              </Button>
                            </TooltipTrigger>
                            <TooltipContent side="top" className="max-w-64">
                              <p className="text-[10px] leading-relaxed">
                                download a shareable Markdown report — last golden-set run with per-category table,
                                the merged golden set with pass/fail, the persisted run history and the triage counts
                              </p>
                            </TooltipContent>
                          </Tooltip>
                        </TooltipProvider>
                        <TooltipProvider delayDuration={200}>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Button
                                variant="ghost"
                                size="sm"
                                className="h-8 gap-1 px-2 text-[10px] text-muted-foreground transition-colors hover:text-teal-700 dark:hover:text-teal-400"
                                onClick={exportGoldenCsv}
                                disabled={!casesData || casesData.cases.length === 0}
                                aria-label="Export the golden set as CSV"
                                title="Download the golden-set cases as CSV (id, question, keywords, docs)"
                              >
                                <Download className="h-3 w-3" />
                                <span className="hidden sm:inline">CSV</span>
                              </Button>
                            </TooltipTrigger>
                            <TooltipContent side="top" className="max-w-64">
                              <p className="text-[10px] leading-relaxed">
                                golden-set cases as CSV — id, category, source, question, expectations, keywords,
                                target docs — for spreadsheets or CI diffing of the grown set
                              </p>
                            </TooltipContent>
                          </Tooltip>
                        </TooltipProvider>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-8 shrink-0 gap-1 px-2 text-[10px] text-muted-foreground"
                          onClick={() => void refreshHistory()}
                          disabled={historyLoading}
                          aria-label="Refresh eval run history"
                          title="Re-fetch the run history"
                        >
                          <RefreshCw className={cn("h-3 w-3", historyLoading && "animate-spin")} />
                          <span className="hidden sm:inline">Refresh</span>
                        </Button>
                      </div>
                    </div>

                    {historyData && (
                      <p
                        className="flex flex-wrap items-center gap-1.5 font-mono text-[9.5px] text-muted-foreground"
                        role="status"
                        aria-label="Eval run trend summary"
                      >
                        <span>
                          {historyData.golden_runs} golden run{historyData.golden_runs !== 1 ? "s" : ""} ·{" "}
                          {historyData.ablation_runs} ablation · best{" "}
                          {(historyData.best_accuracy * 100).toFixed(0)}%
                        </span>
                        <TooltipProvider delayDuration={150}>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <span
                                className={cn(
                                  "inline-flex cursor-help items-center gap-0.5 rounded px-1 py-px",
                                  historyData.accuracy_trend === "flat"
                                    ? "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400"
                                    : "bg-amber-500/10 text-amber-700 dark:text-amber-400"
                                )}
                                aria-label={`Accuracy trend: ${historyData.accuracy_trend}`}
                              >
                                {historyData.accuracy_trend === "flat" ? (
                                  <MoveHorizontal className="h-3 w-3" aria-hidden />
                                ) : (
                                  <TrendingUp className="h-3 w-3" aria-hidden />
                                )}
                                {historyData.accuracy_trend}
                              </span>
                            </TooltipTrigger>
                            <TooltipContent side="top" className="max-w-56">
                              <p className="text-[10px]">
                                {historyData.accuracy_trend === "flat"
                                  ? "accuracy has not changed across the recorded runs"
                                  : "accuracy changed across the recorded runs — check the failing runs below"}
                              </p>
                            </TooltipContent>
                          </Tooltip>
                        </TooltipProvider>
                      </p>
                    )}

                    <div className="flex items-center justify-between gap-2">
                      <TooltipProvider delayDuration={200}>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <label
                              htmlFor="include-ablation-switch"
                              className="flex min-h-11 cursor-help items-center gap-2 text-[10.5px] text-muted-foreground transition-colors hover:text-foreground"
                              title="ablation runs execute with the faithfulness judge off"
                            >
                              include ablation runs
                              <span className="font-mono text-[9px] opacity-70">(judge off)</span>
                            </label>
                          </TooltipTrigger>
                          <TooltipContent side="top" className="max-w-64">
                            <p className="text-[10px] leading-relaxed">
                              ablation runs record accuracy/latency per retrieval config but skip the
                              faithfulness judge — include them to see the full run sequence.
                            </p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                      <Switch
                        id="include-ablation-switch"
                        checked={includeAblation}
                        onCheckedChange={(v) => {
                          setIncludeAblation(v);
                          void refreshHistory(v);
                        }}
                        aria-label="Include ablation runs in the history"
                        title="Include ablation runs"
                        className="data-[state=checked]:bg-amber-500 data-[state=checked]:dark:bg-amber-400"
                      />
                    </div>

                    {historyError ? (
                      <p className="flex items-center gap-1.5 rounded-md border border-amber-200/60 bg-amber-50/40 px-2 py-1.5 text-[10.5px] text-amber-700 dark:border-amber-900/60 dark:bg-amber-950/30 dark:text-amber-400">
                        <CircleAlert className="h-3 w-3 shrink-0" aria-hidden />
                        run history unavailable ({historyError})
                      </p>
                    ) : historyLoading && !historyData ? (
                      <div className="space-y-2" aria-label="Loading eval run history">
                        <Skeleton className="h-28 w-full" />
                        <Skeleton className="h-8 w-full" />
                        <Skeleton className="h-8 w-full" />
                      </div>
                    ) : historyData ? (
                      <>
                        {historyData.runs.length > 1 ? (
                          <HistoryChart runs={[...historyData.runs].reverse()} />
                        ) : (
                          <div className="flex flex-col items-center gap-1.5 rounded-xl border border-dashed border-border bg-muted/20 px-3 py-5 text-center">
                            <MoveHorizontal className="h-5 w-5 text-muted-foreground/50" aria-hidden />
                            <p className="text-[11px] font-medium">Run the evaluation to start the trend</p>
                            <p className="max-w-xs text-[10.5px] leading-relaxed text-muted-foreground">
                              one run recorded so far — each “Run evaluation” or ablation config adds a
                              point to the accuracy/latency chart.
                            </p>
                          </div>
                        )}
                        {historyData.runs.length > 0 && <HistoryRunsTable runs={historyData.runs} />}
                      </>
                    ) : null}
                  </motion.div>

                  {/* ------------------------------------- golden-set growth */}
                  {/* Task 17-b: 👍 candidates review + promote + merged cases. */}
                  <Separator className="bg-border/60" />
                  <div className="space-y-2.5">
                    <div className="flex items-center justify-between gap-2">
                      <TooltipProvider delayDuration={200}>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <p className="flex cursor-help items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                              <Sprout className="h-3.5 w-3.5 text-teal-600 dark:text-teal-500" aria-hidden />
                              golden-set growth
                            </p>
                          </TooltipTrigger>
                          <TooltipContent side="top" className="max-w-64">
                            <p className="text-[10px] leading-relaxed">
                              feedback grows the eval set — every grounded answer is recorded in a
                              ledger, 👍 upvotes become candidates, and promoting one adds a
                              verified user case to the merged golden set.
                            </p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-8 shrink-0 gap-1 px-2 text-[10px] text-muted-foreground"
                        onClick={() => void refreshGrowth()}
                        disabled={growthLoading}
                        aria-label="Refresh golden-set growth data"
                        title="Re-fetch candidates and golden cases"
                      >
                        <RefreshCw className={cn("h-3 w-3", growthLoading && "animate-spin")} />
                        Refresh
                      </Button>
                    </div>

                    {candidatesData && (
                      <TooltipProvider delayDuration={200}>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <p
                              className="flex w-fit cursor-help items-center gap-1 font-mono text-[9.5px] text-muted-foreground"
                              aria-label="Candidate and ledger counts"
                            >
                              {candidatesData.counts.shown} candidate{candidatesData.counts.shown !== 1 ? "s" : ""} ·{" "}
                              {stats?.golden?.upvoted ?? "—"} upvote{(stats?.golden?.upvoted ?? 0) !== 1 ? "s" : ""} ·{" "}
                              {candidatesData.ledger.entries}/{candidatesData.ledger.capacity} ledger entr
                              {candidatesData.ledger.entries !== 1 ? "ies" : "y"}
                            </p>
                          </TooltipTrigger>
                          <TooltipContent side="bottom" className="max-w-64">
                            <p className="font-mono text-[10px] leading-relaxed text-muted-foreground">
                              shown: {candidatesData.counts.shown}
                              <br />
                              unlinkable ratings: {candidatesData.counts.unlinkable_ratings}
                              <br />
                              skipped (route): {candidatesData.counts.skipped_route}
                              <br />
                              skipped (duplicate): {candidatesData.counts.skipped_duplicate}
                              <br />
                              ledger grounded: {candidatesData.ledger.grounded} · cache-served: {candidatesData.ledger.cache_served}
                            </p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    )}

                    {promoteOutcome && (
                      <motion.div
                        initial={reduceMotion ? undefined : { opacity: 0, y: 4 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.2 }}
                        role="status"
                        aria-label="Promotion verification result"
                        className={cn(
                          "flex items-start gap-1.5 rounded-md border px-2.5 py-2 text-[10.5px] leading-snug",
                          promoteOutcome.passed
                            ? "border-emerald-300/70 bg-emerald-50/60 text-emerald-700 dark:border-emerald-800/70 dark:bg-emerald-950/40 dark:text-emerald-400"
                            : "border-amber-300/70 bg-amber-50/60 text-amber-700 dark:border-amber-800/70 dark:bg-amber-950/40 dark:text-amber-400"
                        )}
                      >
                        {promoteOutcome.passed ? (
                          <CheckCircle2 className="mt-px h-3.5 w-3.5 shrink-0" aria-hidden />
                        ) : (
                          <AlertTriangle className="mt-px h-3.5 w-3.5 shrink-0" aria-hidden />
                        )}
                        <span className="min-w-0 flex-1">
                          {promoteOutcome.passed ? (
                            <>
                              Promoted — verification ✓
                              {promoteOutcome.matched && (
                                <span className="ml-1 break-all font-mono text-[9.5px] opacity-80">(matched {promoteOutcome.matched})</span>
                              )}
                            </>
                          ) : (
                            <>Promoted but verification failed — review or delete</>
                          )}
                        </span>
                        <span className="shrink-0 font-mono text-[9px] opacity-70">
                          {promoteOutcome.goldenSet.base}+{promoteOutcome.goldenSet.user}={promoteOutcome.goldenSet.total}
                        </span>
                      </motion.div>
                    )}

                    {growthError ? (
                      <p className="flex items-center gap-1.5 rounded-md border border-amber-200/60 bg-amber-50/40 px-2 py-1.5 text-[10.5px] text-amber-700 dark:border-amber-900/60 dark:bg-amber-950/30 dark:text-amber-400">
                        <CircleAlert className="h-3 w-3 shrink-0" aria-hidden />
                        growth data unavailable ({growthError})
                      </p>
                    ) : growthLoading && !candidatesData ? (
                      <div className="space-y-2" aria-label="Loading promotion candidates">
                        {[0, 1].map((i) => (
                          <div key={i} className="space-y-1.5 rounded-lg border border-border/70 bg-muted/20 p-3">
                            <Skeleton className="h-3.5 w-2/3" />
                            <Skeleton className="h-3 w-full" />
                            <Skeleton className="h-3 w-11/12" />
                            <Skeleton className="h-8 w-full" />
                            <Skeleton className="h-11 w-full" />
                          </div>
                        ))}
                      </div>
                    ) : (candidatesData?.candidates ?? []).length > 0 ? (
                      <>
                        {/* Batch selection + promote action row (Task 19-b). */}
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-8 gap-1 px-2 text-[10px] text-muted-foreground"
                            onClick={toggleSelectAllCandidates}
                            disabled={batchPromoting}
                            aria-label={
                              (candidatesData?.candidates ?? []).length > 0 &&
                              (candidatesData?.candidates ?? []).every((c) => selectedCandidateIds.has(c.request_id))
                                ? "Clear the batch selection"
                                : "Select all candidates for batch promotion"
                            }
                          >
                            {(candidatesData?.candidates ?? []).length > 0 &&
                            (candidatesData?.candidates ?? []).every((c) => selectedCandidateIds.has(c.request_id))
                              ? "clear selection"
                              : "select all"}
                          </Button>
                          <TooltipProvider delayDuration={200}>
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <Button
                                  size="sm"
                                  className="h-11 gap-1.5 bg-teal-600 text-xs font-medium text-white shadow-sm transition-all hover:bg-teal-700 hover:shadow active:scale-[0.98] focus-visible:ring-teal-500/60 disabled:opacity-60 dark:bg-teal-500 dark:hover:bg-teal-400 dark:text-teal-950"
                                  onClick={() => void promoteSelected()}
                                  disabled={
                                    selectedCandidateIds.size === 0 || batchPromoting || selectedCandidateIds.size > 12
                                  }
                                  aria-label={`Promote ${selectedCandidateIds.size} selected candidate${
                                    selectedCandidateIds.size !== 1 ? "s" : ""
                                  } to the golden set`}
                                >
                                  {batchPromoting ? (
                                    <Loader2 className="h-4 w-4 animate-spin" />
                                  ) : (
                                    <ListChecks className="h-4 w-4" />
                                  )}
                                  {batchPromoting
                                    ? `Promoting ${selectedCandidateIds.size}…`
                                    : `Promote ${selectedCandidateIds.size} selected`}
                                </Button>
                              </TooltipTrigger>
                              <TooltipContent side="top" className="max-w-64">
                                <p className="text-[10px] leading-relaxed">
                                  {selectedCandidateIds.size > 12
                                    ? "batch promotion is capped at 12 candidates per run (backend limit 422)"
                                    : `promote ${selectedCandidateIds.size} candidate${
                                        selectedCandidateIds.size !== 1 ? "s" : ""
                                      } as user golden cases — the backend verifies each (~1 s per case); per-item failures are reported, never abort the batch`}
                                </p>
                              </TooltipContent>
                            </Tooltip>
                          </TooltipProvider>
                        </div>

                        {/* Batch outcome banner (Task 19-b): promoted candidates
                            vanish from the list after refresh — this is the
                            record of what happened. */}
                        {batchOutcome && (
                          <motion.div
                            initial={reduceMotion ? undefined : { opacity: 0, y: 4 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.2 }}
                            role="status"
                            aria-label="Batch promotion outcome"
                            className={cn(
                              "space-y-1.5 rounded-md border px-2.5 py-2 text-[10.5px] leading-snug",
                              batchOutcome.failed > 0
                                ? "border-amber-300/70 bg-amber-50/60 text-amber-700 dark:border-amber-800/70 dark:bg-amber-950/40 dark:text-amber-400"
                                : "border-emerald-300/70 bg-emerald-50/60 text-emerald-700 dark:border-emerald-800/70 dark:bg-emerald-950/40 dark:text-emerald-400"
                            )}
                          >
                            <div className="flex items-start gap-1.5">
                              {batchOutcome.failed > 0 ? (
                                <AlertTriangle className="mt-px h-3.5 w-3.5 shrink-0" aria-hidden />
                              ) : (
                                <CheckCircle2 className="mt-px h-3.5 w-3.5 shrink-0" aria-hidden />
                              )}
                              <span className="min-w-0 flex-1">
                                Batch promotion — {batchOutcome.promoted} promoted · {batchOutcome.failed} failed
                              </span>
                              <span className="shrink-0 font-mono text-[9px] opacity-70">
                                {batchOutcome.goldenSet.base}+{batchOutcome.goldenSet.user}={batchOutcome.goldenSet.total}
                              </span>
                              <button
                                type="button"
                                onClick={() => setBatchOutcome(null)}
                                className="-m-0.5 shrink-0 rounded p-1 opacity-60 transition-opacity hover:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/60"
                                aria-label="Dismiss the batch promotion outcome"
                              >
                                <X className="h-3 w-3" aria-hidden />
                              </button>
                            </div>
                            <div className="flex flex-wrap items-center gap-1" aria-label="Per-candidate promotion results">
                              {batchOutcome.results.map((res) => {
                                const verified = res.status === "promoted" && res.passed !== false;
                                const tone = verified
                                  ? "border-emerald-300/70 bg-emerald-50 text-emerald-700 dark:border-emerald-800/70 dark:bg-emerald-950/60 dark:text-emerald-400"
                                  : res.status === "promoted"
                                    ? "border-amber-300/70 bg-amber-50 text-amber-700 dark:border-amber-800/70 dark:bg-amber-950/60 dark:text-amber-400"
                                    : "border-rose-300/70 bg-rose-50 text-rose-700 dark:border-rose-800/70 dark:bg-rose-950/60 dark:text-rose-400";
                                const tip =
                                  `${res.question ?? res.request_id} — ` +
                                  (res.status === "promoted"
                                    ? res.passed !== false
                                      ? `verified ✓${res.matched ? ` (matched ${res.matched})` : ""}`
                                      : "verification failed"
                                    : `error: ${res.error ?? "unknown"}`);
                                return (
                                  <span
                                    key={res.request_id}
                                    title={tip}
                                    className={cn(
                                      "inline-flex max-w-44 items-center gap-0.5 rounded border px-1.5 py-px font-mono text-[9px]",
                                      tone
                                    )}
                                  >
                                    {verified ? (
                                      <CheckCheck className="h-2.5 w-2.5 shrink-0" aria-hidden />
                                    ) : res.status === "failed" ? (
                                      <XCircle className="h-2.5 w-2.5 shrink-0" aria-hidden />
                                    ) : (
                                      <AlertTriangle className="h-2.5 w-2.5 shrink-0" aria-hidden />
                                    )}
                                    <span className="truncate">{res.case_id ?? res.request_id}</span>
                                  </span>
                                );
                              })}
                            </div>
                          </motion.div>
                        )}

                        <div className="thin-scrollbar max-h-96 space-y-2.5 overflow-y-auto pr-1" aria-label="Promotion candidates">
                          {(candidatesData?.candidates ?? []).map((c) => (
                            <CandidateCard
                              key={c.request_id}
                              candidate={c}
                              keywords={keywordEdits[c.request_id] ?? ""}
                              onKeywordsChange={(v) => setKeywordEdits((prev) => ({ ...prev, [c.request_id]: v }))}
                              onPromote={() => void promoteCandidate(c.request_id)}
                              promoting={promotingId === c.request_id}
                              selected={selectedCandidateIds.has(c.request_id)}
                              onToggleSelect={() => toggleCandidateSelect(c.request_id)}
                              selectDisabled={batchPromoting}
                            />
                          ))}
                        </div>
                      </>
                    ) : (
                      <div className="flex flex-col items-center gap-1.5 rounded-xl border border-dashed border-border bg-muted/20 px-3 py-6 text-center">
                        <ThumbsUp className="h-5 w-5 text-muted-foreground/50" aria-hidden />
                        <p className="text-[11px] font-medium">No upvoted answers yet</p>
                        <p className="max-w-xs text-[10.5px] leading-relaxed text-muted-foreground">
                          👍 an answer in chat to grow the eval set — ask a question, rate the
                          answer, then promote the candidate here.
                        </p>
                        <p className="flex items-center gap-1 pt-0.5 font-mono text-[9px] uppercase tracking-wide text-muted-foreground/70">
                          <span>ask</span>
                          <ChevronRight className="h-2.5 w-2.5" aria-hidden />
                          <span>rate</span>
                          <ChevronRight className="h-2.5 w-2.5" aria-hidden />
                          <span>promote</span>
                        </p>
                      </div>
                    )}

                    <GoldenCasesList
                      data={casesData}
                      loading={growthLoading && !casesData}
                      onDelete={(caseId) => void deleteGoldenCase(caseId)}
                      deletingId={deletingCaseId}
                    />
                  </div>

                  {/* --------------------------- quality radar (downvote triage) */}
                  {/* Task 18-b: 👎 answers joined against the answer ledger. */}
                  <Separator className="bg-border/60" />
                  <motion.div
                    initial={reduceMotion ? undefined : { opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.25 }}
                    className="space-y-2.5"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <TooltipProvider delayDuration={200}>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <p className="flex cursor-help items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                              <ThumbsDown className="h-3.5 w-3.5 text-amber-600 dark:text-amber-500" aria-hidden />
                              quality radar (downvote triage)
                            </p>
                          </TooltipTrigger>
                          <TooltipContent side="top" className="max-w-64">
                            <p className="text-[10px] leading-relaxed">
                              👎-rated answers joined against the answer ledger — question, preview,
                              citations and the reviewer&apos;s reason, so weaknesses become a fix-list.
                            </p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-8 shrink-0 gap-1 px-2 text-[10px] text-muted-foreground"
                        onClick={() => void refreshTriage()}
                        disabled={triageLoading}
                        aria-label="Refresh the downvote triage list"
                        title="Re-fetch the downvote triage list"
                      >
                        <RefreshCw className={cn("h-3 w-3", triageLoading && "animate-spin")} />
                        Refresh
                      </Button>
                    </div>

                    <p className="text-[10px] leading-relaxed text-muted-foreground">
                      <span className="text-emerald-600 dark:text-emerald-400">👍 grows the golden set above</span>{" "}
                      · <span className="text-amber-600 dark:text-amber-400">👎 lands here as the fix-list</span>
                    </p>

                    {triageError ? (
                      <p className="flex items-center gap-1.5 rounded-md border border-amber-200/60 bg-amber-50/40 px-2 py-1.5 text-[10.5px] text-amber-700 dark:border-amber-900/60 dark:bg-amber-950/30 dark:text-amber-400">
                        <CircleAlert className="h-3 w-3 shrink-0" aria-hidden />
                        triage unavailable ({triageError})
                      </p>
                    ) : triageLoading && !triageData ? (
                      <div className="space-y-2" aria-label="Loading downvote triage">
                        <Skeleton className="h-20 w-full" />
                        <Skeleton className="h-20 w-full" />
                      </div>
                    ) : triageData ? (
                      <>
                        {/* Open/resolved/all segmented filter + counts (Task 19-b). */}
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <div
                            role="radiogroup"
                            aria-label="Quality radar list filter"
                            className="inline-flex items-center rounded-lg border border-border/70 bg-muted/30 p-0.5"
                          >
                            {(
                              [
                                {
                                  key: "open",
                                  label: `open (${triageData.counts.open})`,
                                  activeClass: "bg-amber-500/15 text-amber-700 dark:bg-amber-500/20 dark:text-amber-400",
                                },
                                {
                                  key: "resolved",
                                  label: `resolved (${triageData.counts.resolved})`,
                                  activeClass:
                                    "bg-emerald-500/15 text-emerald-700 dark:bg-emerald-500/20 dark:text-emerald-400",
                                },
                                {
                                  key: "all",
                                  label: `all (${triageData.counts.open + triageData.counts.resolved})`,
                                  activeClass: "bg-background text-foreground shadow-sm",
                                },
                              ] as const
                            ).map((f) => (
                              <button
                                key={f.key}
                                type="button"
                                role="radio"
                                aria-checked={triageFilter === f.key}
                                onClick={() => setTriageFilter(f.key)}
                                className={cn(
                                  "h-7 rounded-md px-2.5 text-[10.5px] font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/60",
                                  triageFilter === f.key
                                    ? f.activeClass
                                    : "text-muted-foreground hover:bg-muted/60 hover:text-foreground"
                                )}
                              >
                                {f.label}
                              </button>
                            ))}
                          </div>
                          <p
                            className="flex flex-wrap items-center gap-x-1 font-mono text-[9.5px] text-muted-foreground"
                            role="status"
                            aria-label="Downvote triage counts"
                          >
                            <span>
                              {triageData.counts.open} open · {triageData.counts.resolved} resolved ·{" "}
                              {triageData.counts.fallback_answers} fallback
                              {triageData.counts.fallback_answers !== 1 ? "s" : ""} · {triageData.counts.uncited_answers}{" "}
                              uncited · {triageData.counts.with_reason} with reason
                              {triageData.counts.with_reason !== 1 ? "s" : ""}
                            </span>
                            {triageData.counts.oldest_open_hours != null && triageData.counts.open > 0 && (
                              <TooltipProvider delayDuration={150}>
                                <Tooltip>
                                  <TooltipTrigger asChild>
                                    <span
                                      className={cn(
                                        "inline-flex cursor-help items-center gap-1 whitespace-nowrap rounded px-1 py-px font-mono text-[9.5px]",
                                        ageTone(triageData.counts.oldest_open_hours) === "rose"
                                          ? "bg-rose-500/10 text-rose-700 dark:text-rose-400"
                                          : ageTone(triageData.counts.oldest_open_hours) === "amber"
                                            ? "bg-amber-500/10 text-amber-700 dark:text-amber-400"
                                            : "bg-muted text-muted-foreground"
                                      )}
                                      aria-label={`Oldest open fix-list item: ${formatAgeHours(triageData.counts.oldest_open_hours)}`}
                                    >
                                      <Clock className="h-2.5 w-2.5" aria-hidden />
                                      oldest {formatAgeHours(triageData.counts.oldest_open_hours)}
                                      {triageData.counts.mean_open_hours != null && triageData.counts.open > 1 && (
                                        <span className="opacity-70"> · avg {formatAgeHours(triageData.counts.mean_open_hours)}</span>
                                      )}
                                    </span>
                                  </TooltipTrigger>
                                  <TooltipContent side="top" className="max-w-64">
                                    <p className="text-[10px] leading-relaxed">
                                      fix-list SLA — the oldest open 👎 has waited{" "}
                                      {formatAgeHours(triageData.counts.oldest_open_hours)} (amber at 6h, rose/breach at 48h).
                                      Promoted to /metrics as rag_triage_oldest_open_hours.
                                    </p>
                                  </TooltipContent>
                                </Tooltip>
                              </TooltipProvider>
                            )}
                          </p>
                        </div>
                        {(() => {
                          const mergedItems =
                            triageFilter === "open"
                              ? triageData.items
                              : triageFilter === "resolved"
                                ? triageData.resolved_items
                                : [...triageData.items, ...triageData.resolved_items];
                          return mergedItems.length > 0 ? (
                            <div className="thin-scrollbar max-h-96 space-y-2.5 overflow-y-auto pr-1" aria-label="Downvoted answers triage list">
                              {mergedItems.map((item) => (
                                <TriageItemCard
                                  key={item.request_id}
                                  item={item}
                                  onResolve={resolveTriageItem}
                                  onReopen={reopenTriageItem}
                                  resolvingId={resolvingId}
                                />
                              ))}
                            </div>
                          ) : triageFilter === "resolved" ? (
                            <div className="flex flex-col items-center gap-1.5 rounded-xl border border-dashed border-border bg-muted/20 px-3 py-6 text-center">
                              <Undo2 className="h-5 w-5 text-muted-foreground/50" aria-hidden />
                              <p className="text-[11px] font-medium">Nothing resolved yet</p>
                              <p className="max-w-xs text-[10.5px] leading-relaxed text-muted-foreground">
                                mark a fixed item from the open list — resolved 👎 answers collect here
                                with their fix notes.
                              </p>
                            </div>
                          ) : triageData.counts.resolved > 0 ? (
                            <div className="flex flex-col items-center gap-1.5 rounded-xl border border-emerald-200/60 bg-emerald-50/40 px-3 py-6 text-center dark:border-emerald-900/60 dark:bg-emerald-950/30">
                              <ShieldCheck className="h-5 w-5 text-emerald-600 dark:text-emerald-400" aria-hidden />
                              <p className="text-[11px] font-medium text-emerald-700 dark:text-emerald-400">
                                All 👎 resolved — quality radar clean ✓
                              </p>
                              <p className="max-w-xs text-[10.5px] leading-relaxed text-muted-foreground">
                                {triageData.counts.resolved} fixed item
                                {triageData.counts.resolved !== 1 ? "s" : ""} below — switch the filter to
                                “resolved” to review the fix notes.
                              </p>
                            </div>
                          ) : (
                            <div className="flex flex-col items-center gap-1.5 rounded-xl border border-emerald-200/60 bg-emerald-50/40 px-3 py-6 text-center dark:border-emerald-900/60 dark:bg-emerald-950/30">
                              <ShieldCheck className="h-5 w-5 text-emerald-600 dark:text-emerald-400" aria-hidden />
                              <p className="text-[11px] font-medium text-emerald-700 dark:text-emerald-400">
                                No downvoted answers — quality radar clean ✓
                              </p>
                              <p className="max-w-xs text-[10.5px] leading-relaxed text-muted-foreground">
                                👎 an answer in chat and it appears here with its reason — the fix-list
                                for answer quality.
                              </p>
                            </div>
                          );
                        })()}
                      </>
                    ) : null}
                  </motion.div>

                  {/* ------------------------------------ A/B answer comparison */}
                  {/* Task 17-b: one question, two configs, side-by-side answers. */}
                  <Separator className="bg-border/60" />
                  <div className="@container space-y-2.5">
                    <TooltipProvider delayDuration={200}>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <p className="flex w-fit cursor-help items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                            <GitCompare className="h-3.5 w-3.5 text-teal-600 dark:text-teal-500" aria-hidden />
                            A/B answer comparison
                          </p>
                        </TooltipTrigger>
                        <TooltipContent side="top" className="max-w-64">
                          <p className="text-[10px] leading-relaxed">
                            run one question through the full agent graph under two retriever
                            configs and read the actual answers side by side — the answer-quality
                            view the ablation table can&apos;t show.
                          </p>
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>

                    <Input
                      value={compareQuestion}
                      onChange={(e) => setCompareQuestion(e.target.value)}
                      disabled={compareRunning}
                      placeholder="Question to compare…"
                      className="h-9 text-xs"
                      aria-label="A/B comparison question"
                    />

                    <div className="grid grid-cols-1 gap-2 @xl:grid-cols-2">
                      <div className="min-w-0 space-y-1">
                        <p className="flex items-center gap-1 text-[10px] uppercase tracking-wide text-muted-foreground">
                          <span className="inline-flex h-3.5 w-3.5 items-center justify-center rounded bg-teal-600 font-mono text-[8px] font-bold text-white dark:bg-teal-500 dark:text-teal-950" aria-hidden>
                            A
                          </span>
                          config A
                        </p>
                        <Select value={compareConfigA} onValueChange={setCompareConfigA}>
                          <SelectTrigger
                            className="h-9 w-full min-w-0 border-teal-200/60 font-mono text-[10.5px] focus-visible:ring-teal-500/40 dark:border-teal-900/60"
                            aria-label="A/B comparison config A"
                            title={compareConfigA}
                          >
                            <SelectValue placeholder="pick config A" />
                          </SelectTrigger>
                          <SelectContent className="max-h-72">
                            {compareConfigs.map((c) => (
                              <SelectItem key={c} value={c} className="font-mono text-[10.5px]">
                                {c}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="min-w-0 space-y-1">
                        <p className="flex items-center gap-1 text-[10px] uppercase tracking-wide text-muted-foreground">
                          <span className="inline-flex h-3.5 w-3.5 items-center justify-center rounded bg-amber-500 font-mono text-[8px] font-bold text-white dark:bg-amber-400 dark:text-amber-950" aria-hidden>
                            B
                          </span>
                          config B
                        </p>
                        <Select value={compareConfigB} onValueChange={setCompareConfigB}>
                          <SelectTrigger
                            className="h-9 w-full min-w-0 border-amber-200/60 font-mono text-[10.5px] focus-visible:ring-amber-500/40 dark:border-amber-900/60"
                            aria-label="A/B comparison config B"
                            title={compareConfigB}
                          >
                            <SelectValue placeholder="pick config B" />
                          </SelectTrigger>
                          <SelectContent className="max-h-72">
                            {compareConfigs.map((c) => (
                              <SelectItem key={c} value={c} className="font-mono text-[10.5px]">
                                {c}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                    </div>

                    <Button
                      size="sm"
                      className="h-11 w-full gap-1.5 bg-teal-600 text-xs font-medium text-white shadow-sm transition-all hover:bg-teal-700 hover:shadow active:scale-[0.98] focus-visible:ring-teal-500/60 dark:bg-teal-500 dark:hover:bg-teal-400 dark:text-teal-950"
                      onClick={() => void runCompare()}
                      disabled={compareRunning || compareConfigA === compareConfigB}
                      aria-label="Run the A/B answer comparison"
                    >
                      {compareRunning ? <Loader2 className="h-4 w-4 animate-spin" /> : <GitCompare className="h-4 w-4" />}
                      {compareRunning ? "Running both sides…" : "Run A/B comparison"}
                    </Button>
                    {compareConfigA === compareConfigB && (
                      <p className="text-[10px] text-amber-600 dark:text-amber-400" role="note">
                        A and B must differ — pick two distinct configs.
                      </p>
                    )}

                    {compareError && (
                      <p className="flex items-start gap-1.5 rounded-md border border-red-200/60 bg-red-50/40 px-2 py-1.5 text-[10.5px] leading-snug text-red-600 dark:border-red-900/60 dark:bg-red-950/30 dark:text-red-400" role="alert">
                        <CircleAlert className="mt-px h-3 w-3 shrink-0" aria-hidden />
                        {compareError}
                      </p>
                    )}

                    {compareRunning ? (
                      <div className="grid grid-cols-1 gap-3 @xl:grid-cols-2" aria-label="A/B comparison loading">
                        {[0, 1].map((i) => (
                          <div key={i} className="min-w-0 space-y-2 rounded-lg border p-3">
                            <Skeleton className="h-4 w-1/2" />
                            <Skeleton className="h-16 w-full" />
                            <Skeleton className="h-3 w-2/3" />
                            <Skeleton className="h-3 w-1/3" />
                          </div>
                        ))}
                      </div>
                    ) : compareReport ? (
                      <motion.div
                        initial={reduceMotion ? undefined : { opacity: 0, y: 6 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.25 }}
                        className="space-y-3"
                      >
                        <p className="flex flex-wrap items-center gap-1.5 text-[10px] text-muted-foreground">
                          <span className="shrink-0 font-semibold text-foreground/70">Q:</span>
                          <span className="truncate italic">{compareReport.question}</span>
                          <span className="ml-auto shrink-0 font-mono text-[9px] opacity-70">
                            ran {new Date(compareReport.ran_at).toLocaleTimeString()}
                          </span>
                        </p>

                        {/* Word-diff legend + highlighting toggle (Task 18-b). */}
                        {!compareReport.verdict.answers_identical && (
                          <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1.5 rounded-md border border-border/60 bg-muted/20 px-2.5 py-1.5">
                            <div className="flex flex-wrap items-center gap-x-3 gap-y-1" role="status" aria-label="Word-diff legend">
                              {compareDiff && !compareDiff.skipped ? (
                                <>
                                  <span className="flex items-center gap-1 text-[9.5px] text-muted-foreground">
                                    <span className="h-2.5 w-2.5 rounded-sm bg-emerald-500/30 dark:bg-emerald-500/40" aria-hidden />
                                    only in A
                                    <span className="font-mono text-emerald-600 dark:text-emerald-400">({compareDiff.onlyAWords})</span>
                                  </span>
                                  <span className="flex items-center gap-1 text-[9.5px] text-muted-foreground">
                                    <span className="h-2.5 w-2.5 rounded-sm bg-muted-foreground/15" aria-hidden />
                                    shared
                                  </span>
                                  <span className="flex items-center gap-1 text-[9.5px] text-muted-foreground">
                                    <span className="h-2.5 w-2.5 rounded-sm bg-amber-500/30 dark:bg-amber-500/40" aria-hidden />
                                    only in B
                                    <span className="font-mono text-amber-600 dark:text-amber-400">({compareDiff.onlyBWords})</span>
                                  </span>
                                </>
                              ) : (
                                <TooltipProvider delayDuration={150}>
                                  <Tooltip>
                                    <TooltipTrigger asChild>
                                      <span className="cursor-help text-[9.5px] text-muted-foreground" aria-label="Diff highlighting unavailable">
                                        word diff disabled
                                      </span>
                                    </TooltipTrigger>
                                    <TooltipContent side="top" className="max-w-64">
                                      <p className="text-[10px] leading-relaxed">
                                        {compareDiff && compareDiff.skipped
                                          ? "one answer exceeds the 400-word cap — the O(n·m) LCS diff is skipped and both answers render plain"
                                          : "toggle diff highlighting on to highlight the wording differences"}
                                      </p>
                                    </TooltipContent>
                                  </Tooltip>
                                </TooltipProvider>
                              )}
                            </div>
                            <label
                              htmlFor="diff-highlight-switch"
                              className="flex min-h-11 cursor-pointer items-center gap-2 text-[10px] text-muted-foreground transition-colors hover:text-foreground"
                              title="Toggle word-level diff highlighting"
                            >
                              diff highlighting
                              <Switch
                                id="diff-highlight-switch"
                                checked={diffHighlight}
                                onCheckedChange={setDiffHighlight}
                                aria-label="Toggle A/B word-level diff highlighting"
                                className="data-[state=checked]:bg-emerald-600 data-[state=checked]:dark:bg-emerald-500"
                              />
                            </label>
                          </div>
                        )}

                        <div className="grid grid-cols-1 gap-3 @xl:grid-cols-2">
                          <div className="min-w-0">
                            <CompareSideCard side={compareReport.side_a} label="A" diff={compareDiff} />
                          </div>
                          <div className="min-w-0">
                            <CompareSideCard side={compareReport.side_b} label="B" diff={compareDiff} />
                          </div>
                        </div>
                        <VerdictStrip verdict={compareReport.verdict} />
                      </motion.div>
                    ) : (
                      <p className="text-[10px] leading-relaxed text-muted-foreground/70">
                        Run a comparison to see both answers, citations, latency and faithfulness
                        side by side with a diff verdict.
                      </p>
                    )}
                  </div>
                </CardContent>
              </Card>
              </TabPanel>
            </TabsContent>

            {/* -------------------------------------------------- sessions */}
            <TabsContent value="sessions" className="mt-3">
              <TabPanel>
              <Card className="card-hairline">
                <CardHeader className="pb-2">
                  <CardTitle className="flex items-center gap-2 text-sm">
                    <History className="h-4 w-4 text-violet-600" />
                    Conversation Memory
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <p className="text-[11px] leading-relaxed text-muted-foreground">
                    Multi-turn sessions with follow-up resolution: short follow-ups
                    (“and the warranty?”) inherit entities from the previous turn.
                  </p>
                  {sessionStats && (
                    <div className="grid grid-cols-3 gap-2 text-center">
                      {[
                        ["sessions", sessionStats.sessions],
                        ["turns", sessionStats.total_turns],
                        ["follow-ups", sessionStats.follow_ups_resolved],
                      ].map(([label, value]) => (
                        <div key={String(label)} className="rounded-lg border bg-muted/40 px-2 py-1.5">
                          <p className="font-mono text-sm font-semibold">{String(value)}</p>
                          <p className="text-[10px] uppercase tracking-wide text-muted-foreground">{String(label)}</p>
                        </div>
                      ))}
                    </div>
                  )}
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-8 w-full gap-1.5 text-xs"
                    onClick={newConversation}
                  >
                    <RefreshCw className="h-3.5 w-3.5" />
                    New conversation
                  </Button>
                  <div className="thin-scrollbar max-h-96 overflow-y-auto rounded-md border">
                    <div className="divide-y divide-border/60">
                      {sessionsLoading && sessions.length === 0 && (
                        <div className="space-y-2 p-2.5" aria-label="Loading sessions">
                          {[0, 1, 2].map((i) => (
                            <div key={i} className="space-y-1.5">
                              <Skeleton className="h-3.5 w-2/3" />
                              <Skeleton className="h-2.5 w-1/3" />
                            </div>
                          ))}
                        </div>
                      )}
                      {!sessionsLoading && sessions.length === 0 && (
                        <div className="flex flex-col items-center gap-1.5 rounded-xl border border-dashed border-border bg-muted/20 px-3 py-8 text-center">
                          <MessageSquare className="h-5 w-5 text-muted-foreground/50" aria-hidden />
                          <p className="text-[11px] text-muted-foreground">
                            No sessions yet — ask a question to start one.
                          </p>
                        </div>
                      )}
                      {sessions.map((s) => (
                        <div key={s.id} className="px-2.5 py-2">
                          <div className="flex items-center gap-2">
                            <button
                              className="min-w-0 flex-1 rounded-sm text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500/40"
                              onClick={() => void loadHistory(s.id)}
                            >
                              <p className="flex items-center gap-1.5 truncate text-[11px] font-medium">
                                {s.id === sessionId && <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-emerald-500" aria-label="current session" />}
                                <span className="truncate">{s.preview.split(" · ").slice(1).join(" · ") || s.id}</span>
                              </p>
                              <p className="truncate font-mono text-[10px] text-muted-foreground">{s.id} · {s.turns} turn{s.turns !== 1 ? "s" : ""}</p>
                              {s.entities.length > 0 && (
                                <p className="mt-0.5 flex flex-wrap gap-1">
                                  {s.entities.slice(0, 4).map((e) => (
                                    <Badge key={e} variant="secondary" className="px-1 font-mono text-[9px] lowercase">
                                      {e}
                                    </Badge>
                                  ))}
                                </p>
                              )}
                            </button>
                            <DropdownMenu>
                              <TooltipProvider delayDuration={200}>
                                <Tooltip>
                                  <TooltipTrigger asChild>
                                    <span className="inline-flex">
                                      <DropdownMenuTrigger asChild>
                                        <Button
                                          variant="ghost"
                                          size="icon"
                                          className="h-7 w-7 shrink-0 text-muted-foreground/70 hover:text-emerald-600 dark:hover:text-emerald-400"
                                          disabled={s.turns === 0}
                                          aria-label={`Export session ${s.id}`}
                                        >
                                          <Download className="h-3 w-3" />
                                        </Button>
                                      </DropdownMenuTrigger>
                                    </span>
                                  </TooltipTrigger>
                                  <TooltipContent side="top">
                                    <p className="text-[10px]">
                                      {s.turns === 0 ? "No turns to export" : `Export session ${s.id} (${s.turns} turn${s.turns !== 1 ? "s" : ""})`}
                                    </p>
                                  </TooltipContent>
                                </Tooltip>
                              </TooltipProvider>
                              <DropdownMenuContent align="end" className="w-44">
                                <DropdownMenuLabel className="text-[10px] uppercase tracking-wide text-muted-foreground">
                                  Export {s.id}
                                </DropdownMenuLabel>
                                <DropdownMenuItem
                                  onClick={() => void exportSession(s.id, "md")}
                                  className="h-11 cursor-pointer gap-2 text-xs"
                                >
                                  <FileText className="h-4 w-4 text-emerald-600" />
                                  Markdown
                                  <span className="ml-auto font-mono text-[9px] text-muted-foreground">.md</span>
                                </DropdownMenuItem>
                                <DropdownMenuItem
                                  onClick={() => void exportSession(s.id, "json")}
                                  className="h-11 cursor-pointer gap-2 text-xs"
                                >
                                  <FileJson className="h-4 w-4 text-teal-600" />
                                  JSON
                                  <span className="ml-auto font-mono text-[9px] text-muted-foreground">.json</span>
                                </DropdownMenuItem>
                              </DropdownMenuContent>
                            </DropdownMenu>
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-7 w-7 shrink-0 text-muted-foreground hover:text-red-500"
                              onClick={() => void deleteSession(s.id)}
                              aria-label={`Delete session ${s.id}`}
                            >
                              <Trash2 className="h-3 w-3" />
                            </Button>
                            <ChevronRight
                              className={cn(
                                "h-3.5 w-3.5 shrink-0 text-muted-foreground/50 transition-transform",
                                historyFor === s.id && "rotate-90"
                              )}
                            />
                          </div>
                          {historyFor === s.id && (
                            <div className="mt-1.5 space-y-1.5 border-l-2 border-border pl-2.5">
                              {historyTurns === null ? (
                                <Skeleton className="h-16 w-full" />
                              ) : historyTurns.length === 0 ? (
                                <p className="py-1 text-[10px] text-muted-foreground">(empty)</p>
                              ) : (
                                historyTurns.map((t, i) => (
                                  <div key={i} className="rounded bg-muted/40 p-1.5">
                                    <p className="flex items-center gap-1 text-[10px] font-medium">
                                      {t.follow_up_applied && (
                                        <Badge variant="outline" className="h-3.5 border-violet-300 px-1 font-mono text-[8px] text-violet-600 dark:border-violet-800 dark:text-violet-400">
                                          follow-up
                                        </Badge>
                                      )}
                                      <span className="truncate">{t.question}</span>
                                    </p>
                                    <p className="line-clamp-2 text-[10px] leading-snug text-muted-foreground">{t.answer}</p>
                                  </div>
                                ))
                              )}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* ---------------------- question analytics (Round 20) */}
              <div className="mt-3">
                <TopQuestionsCard
                  data={analyticsData}
                  loading={analyticsLoading}
                  error={analyticsError}
                  onRefresh={() => void refreshAnalytics()}
                />
              </div>
              </TabPanel>
            </TabsContent>

            {/* ------------------------------------------------------ bench */}
            <TabsContent value="bench" className="mt-3">
              <TabPanel>
              <Card className="card-hairline">
                <CardHeader className="pb-2">
                  <CardTitle className="flex items-center gap-2 text-sm">
                    <Gauge className="h-4 w-4 text-emerald-600" />
                    Serving Benchmark
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <p className="text-[11px] leading-relaxed text-muted-foreground">
                    24 concurrent streaming requests (8 workers) measuring TTFT, P95 latency and
                    throughput against the active backend.
                  </p>
                  <Button
                    size="sm"
                    className="h-8 w-full gap-1.5 text-xs"
                    onClick={() => void runBenchmark()}
                    disabled={benchRunning}
                  >
                    {benchRunning ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Activity className="h-3.5 w-3.5" />}
                    {benchRunning ? "Running…" : "Run benchmark"}
                  </Button>
                  {bench && (
                    <motion.div
                      initial={{ opacity: 0, y: 6 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="space-y-1.5 rounded-md border bg-muted/40 p-2.5 font-mono text-[11px]"
                    >
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">mode</span>
                        <span>{bench.mode}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">tokens/sec</span>
                        <span className="font-semibold text-emerald-600">{bench.tokens_per_sec.toFixed(0)}</span>
                      </div>
                      <Separator className="my-1" />
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">TTFT p50 / p95</span>
                        <span>
                          {bench.ttft_p50_ms.toFixed(0)} / {bench.ttft_p95_ms.toFixed(0)} ms
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">lat p50 / p95</span>
                        <span>
                          {bench.latency_p50_ms.toFixed(0)} / {bench.latency_p95_ms.toFixed(0)} ms
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">ok / total</span>
                        <span>
                          {bench.successful_requests} / {bench.total_requests}
                        </span>
                      </div>
                    </motion.div>
                  )}
                </CardContent>
              </Card>
              </TabPanel>
            </TabsContent>

            {/* ----------------------------------------------------- pulse */}
            <TabsContent value="pulse" className="mt-3">
              <TabPanel>
              <Card className="card-hairline">
                <CardHeader className="pb-2">
                  <CardTitle className="flex items-center gap-2 text-sm">
                    <HeartPulse className="h-4 w-4 text-rose-600" />
                    Service Pulse
                    {stats && (
                      <span className="ml-auto flex items-center gap-1 font-mono text-[10px] font-normal text-muted-foreground">
                        <span className="relative flex h-1.5 w-1.5">
                          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-rose-400 opacity-60" />
                          <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-rose-500" />
                        </span>
                        {stats.service.uptime_human}
                      </span>
                    )}
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <p className="text-[11px] leading-relaxed text-muted-foreground">
                    Live usage telemetry: questions by route, latency percentiles, follow-ups,
                    feedback approval and the last eval / benchmark snapshots.
                  </p>

                  {stats ? (
                    <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} className="space-y-3">
                      <div className="grid grid-cols-2 gap-2">
                        {[
                          { label: "questions", value: String(stats.questions.total), icon: MessageSquare, tone: "text-emerald-600" },
                          { label: "p50 / p95", value: `${stats.questions.latency_p50_ms.toFixed(0)}/${stats.questions.latency_p95_ms.toFixed(0)}ms`, icon: Timer, tone: "text-teal-600" },
                          { label: "follow-ups", value: String(stats.questions.follow_ups_resolved), icon: History, tone: "text-violet-600" },
                          { label: "rewrites", value: String(stats.questions.query_rewrites), icon: RefreshCw, tone: "text-orange-600" },
                        ].map(({ label, value, icon: Icon, tone }) => (
                          <div key={label} className="rounded-lg border bg-muted/30 px-2.5 py-2 transition-colors hover:border-emerald-300/60 dark:hover:border-emerald-800/60">
                            <p className="flex items-center gap-1 text-[10px] uppercase tracking-wide text-muted-foreground">
                              <Icon className={cn("h-3 w-3", tone)} />
                              {label}
                            </p>
                            <p className="mt-0.5 font-mono text-sm font-semibold">
                              <AnimatedStat value={value} />
                            </p>
                          </div>
                        ))}
                      </div>

                      {stats.cache && (
                        <div className="space-y-1.5 rounded-lg border border-amber-200/50 bg-amber-50/20 p-2.5 dark:border-amber-900/50 dark:bg-amber-950/20">
                          <p className="flex items-center justify-between text-[10px] uppercase tracking-wide text-muted-foreground">
                            <span className="flex items-center gap-1">
                              <Zap className="h-3 w-3 text-amber-600 dark:text-amber-500" />
                              answer cache
                            </span>
                            <span className="flex items-center gap-1">
                              {stats.cache.persisted && (
                                <TooltipProvider delayDuration={150}>
                                  <Tooltip>
                                    <TooltipTrigger asChild>
                                      <span
                                        className="cursor-help rounded bg-amber-500/10 px-1 py-px font-mono text-[9px] normal-case text-amber-700 dark:text-amber-400"
                                        aria-label="Cache persisted to disk"
                                      >
                                        disk{typeof stats.cache.restored_entries === "number" && stats.cache.restored_entries > 0 ? ` +${stats.cache.restored_entries}` : ""}
                                      </span>
                                    </TooltipTrigger>
                                    <TooltipContent side="top">
                                      <p className="text-[10px]">
                                        cache persists to data/answer_cache.json and survives restarts —{" "}
                                        {stats.cache.restored_entries ?? 0} entr{stats.cache.restored_entries === 1 ? "y" : "ies"} restored
                                        from the last restart
                                      </p>
                                    </TooltipContent>
                                  </Tooltip>
                                </TooltipProvider>
                              )}
                              <TooltipProvider delayDuration={150}>
                                <Tooltip>
                                  <TooltipTrigger asChild>
                                    <span
                                      className="cursor-help rounded bg-amber-500/10 px-1 py-px font-mono text-[9px] normal-case text-amber-700 dark:text-amber-400"
                                      aria-label="Cache similarity threshold"
                                    >
                                      sim ≥ {stats.cache.sim_threshold.toFixed(2)}
                                    </span>
                                  </TooltipTrigger>
                                  <TooltipContent side="top">
                                    <p className="text-[10px]">cosine similarity threshold for a semantic cache match — grounded answers only</p>
                                  </TooltipContent>
                                </Tooltip>
                              </TooltipProvider>
                            </span>
                          </p>
                          <div className="grid grid-cols-4 gap-1.5 text-center">
                            {[
                              { label: "entries", value: `${stats.cache.entries}/${stats.cache.capacity}` },
                              { label: "hits", value: String(stats.questions.cache_hits) },
                              {
                                label: "hit rate",
                                value:
                                  stats.questions.cache_hit_rate !== null
                                    ? `${(stats.questions.cache_hit_rate * 100).toFixed(0)}%`
                                    : `${(stats.cache.hit_rate_last * 100).toFixed(0)}%`,
                              },
                              {
                                label: "ttl",
                                value:
                                  stats.cache.ttl_seconds >= 60
                                    ? `${Math.round(stats.cache.ttl_seconds / 60)}m`
                                    : `${stats.cache.ttl_seconds}s`,
                              },
                            ].map(({ label, value }) => (
                              <div key={label} className="rounded-md bg-background/70 px-1 py-1.5 dark:bg-background/50">
                                <p className="font-mono text-[11px] font-semibold">{value}</p>
                                <p className="text-[8.5px] uppercase tracking-wide text-muted-foreground">{label}</p>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Cross-encoder rerank stage telemetry (Task 1d). */}
                      {stats.rerank && (
                        <motion.div
                          initial={{ opacity: 0, y: 4 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ duration: 0.2, delay: 0.05 }}
                          className="space-y-1.5 rounded-lg border border-teal-200/50 bg-teal-50/20 p-2.5 dark:border-teal-900/50 dark:bg-teal-950/20"
                          aria-label="Cross-encoder rerank stage stats"
                        >
                          <p className="flex items-center justify-between text-[10px] uppercase tracking-wide text-muted-foreground">
                            <span className="flex items-center gap-1">
                              <ArrowDownUp className="h-3 w-3 text-teal-600 dark:text-teal-500" />
                              cross-encoder stage
                            </span>
                            <TooltipProvider delayDuration={150}>
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <span
                                    className="cursor-help rounded bg-teal-500/10 px-1 py-px font-mono text-[9px] normal-case text-teal-700 dark:text-teal-400"
                                    aria-label="Cross-encoder candidate pool size"
                                  >
                                    n={stats.rerank.n}
                                  </span>
                                </TooltipTrigger>
                                <TooltipContent side="top">
                                  <p className="max-w-56 text-[10px]">
                                    candidate pool per rerank call — the joint (query, chunk) scorer re-orders
                                    these pairs before top-k is cut
                                  </p>
                                </TooltipContent>
                              </Tooltip>
                            </TooltipProvider>
                          </p>
                          <div className="grid grid-cols-4 gap-1.5 text-center">
                            {[
                              { label: "calls", value: stats.rerank.calls.toLocaleString() },
                              { label: "reorders", value: stats.rerank.reorders.toLocaleString() },
                              {
                                label: "avg",
                                value:
                                  stats.rerank.avg_latency_ms !== null
                                    ? `${stats.rerank.avg_latency_ms.toFixed(2)}ms`
                                    : "—",
                              },
                              { label: "mode", value: stats.rerank.mode },
                            ].map(({ label, value }) => (
                              <div key={label} className="rounded-md bg-background/70 px-1 py-1.5 dark:bg-background/50">
                                <p className="font-mono text-[11px] font-semibold">
                                  <AnimatedStat value={value} />
                                </p>
                                <p className="text-[8.5px] uppercase tracking-wide text-muted-foreground">{label}</p>
                              </div>
                            ))}
                          </div>
                          <p className="text-[9px] leading-snug text-muted-foreground/80">
                            {stats.rerank.llm_scores > 0
                              ? `${stats.rerank.llm_scores.toLocaleString()} pair${stats.rerank.llm_scores !== 1 ? "s" : ""} scored by the LLM cross-encoder.`
                              : "Simulation mode: deterministic lexical surrogate (phrase containment, answer-likeness, IDF coverage)."}
                          </p>
                        </motion.div>
                      )}

                      {/* Golden-set growth telemetry (Task 17-b) — from /stats. */}
                      {/* Task 19-b: triage chips (open/resolved) from /stats. */}
                      {stats.golden && (
                        <GoldenGrowthCard golden={stats.golden} evalRuns={stats.eval_runs} triage={stats.triage} />
                      )}

                      <PrometheusCard
                        text={metricsText}
                        loading={metricsLoading}
                        error={metricsError}
                        onRefresh={() => void refreshMetrics()}
                        onGrafana={() => void downloadGrafana()}
                      />

                      {Object.keys(stats.questions.by_route).length > 0 && (
                        <div className="space-y-1.5 rounded-lg border bg-muted/30 p-2.5">
                          <p className="text-[10px] uppercase tracking-wide text-muted-foreground">routes</p>
                          {Object.entries(stats.questions.by_route).map(([route, count]) => {
                            const max = Math.max(...Object.values(stats!.questions.by_route));
                            return (
                              <div key={route} className="flex items-center gap-2 text-[11px]">
                                <span className="w-20 shrink-0 truncate font-mono text-muted-foreground">{route}</span>
                                <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
                                  <motion.div
                                    initial={{ width: 0 }}
                                    animate={{ width: `${(count / Math.max(1, max)) * 100}%` }}
                                    transition={{ duration: 0.5, ease: "easeOut" }}
                                    className={cn(
                                      "h-full rounded-full",
                                      route === "document" ? "bg-emerald-500" : route === "chitchat" ? "bg-rose-400" : "bg-amber-500"
                                    )}
                                  />
                                </div>
                                <span className="shrink-0 font-mono text-muted-foreground">{count}</span>
                              </div>
                            );
                          })}
                        </div>
                      )}

                      {stats.feedback.total > 0 && (
                        <div className="flex items-center gap-3 rounded-lg border bg-muted/30 p-2.5">
                          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full border-2 border-emerald-500/70">
                            <span className="font-mono text-xs font-bold text-emerald-600">
                              {stats.feedback.approval !== null ? Math.round(stats.feedback.approval * 100) : "—"}
                            </span>
                          </div>
                          <div className="min-w-0 flex-1 text-[11px]">
                            <p className="font-medium">answer approval</p>
                            <p className="flex items-center gap-2 font-mono text-[10px] text-muted-foreground">
                              <span className="text-emerald-600">▲ {stats.feedback.up}</span>
                              <span className="text-amber-600">▼ {stats.feedback.down}</span>
                              <span>· {stats.feedback.total} rated</span>
                            </p>
                          </div>
                        </div>
                      )}

                      {(stats.last_eval || stats.last_benchmark) && (
                        <div className="grid grid-cols-2 gap-2">
                          {stats.last_eval && (
                            <div className="rounded-lg border border-emerald-200/60 bg-emerald-50/40 px-2.5 py-2 text-[11px] dark:border-emerald-900/60 dark:bg-emerald-950/30">
                              <p className="flex items-center gap-1 font-medium text-emerald-700 dark:text-emerald-400">
                                <ClipboardCheck className="h-3 w-3" /> eval
                              </p>
                              <p className="mt-0.5 font-mono text-sm font-semibold">{stats.last_eval.passed}/{stats.last_eval.total}</p>
                              <p className="font-mono text-[9px] text-muted-foreground">
                                faith {stats.last_eval.faithfulness_avg !== null ? `${(stats.last_eval.faithfulness_avg * 100).toFixed(0)}%` : "—"}
                              </p>
                            </div>
                          )}
                          {stats.last_benchmark && (
                            <div className="rounded-lg border border-teal-200/60 bg-teal-50/40 px-2.5 py-2 text-[11px] dark:border-teal-900/60 dark:bg-teal-950/30">
                              <p className="flex items-center gap-1 font-medium text-teal-700 dark:text-teal-400">
                                <Activity className="h-3 w-3" /> bench
                              </p>
                              <p className="mt-0.5 font-mono text-sm font-semibold">{stats.last_benchmark.tokens_per_sec?.toFixed(0) ?? "—"} tok/s</p>
                              <p className="font-mono text-[9px] text-muted-foreground">
                                {stats.last_benchmark.ok}/{stats.last_benchmark.requests} ok
                              </p>
                            </div>
                          )}
                        </div>
                      )}

                      {stats.questions.recent.length > 0 && (
                        <div>
                          <p className="mb-1 text-[10px] uppercase tracking-wide text-muted-foreground">recent questions</p>
                          <div className="thin-scrollbar max-h-44 space-y-1 overflow-y-auto pr-1">
                            {stats.questions.recent.map((r, i) => (
                              <div key={i} className="flex items-center gap-1.5 rounded-md border border-border/50 bg-muted/20 px-2 py-1 text-[11px]">
                                <span
                                  className={cn(
                                    "h-1.5 w-1.5 shrink-0 rounded-full",
                                    r.route === "document" ? "bg-emerald-500" : r.route === "chitchat" ? "bg-rose-400" : "bg-amber-500"
                                  )}
                                  aria-hidden
                                />
                                <span className="min-w-0 flex-1 truncate">{r.question}</span>
                                <span className="shrink-0 font-mono text-[9px] text-muted-foreground">{r.latency_ms.toFixed(0)}ms</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      <p className="text-center text-[9px] text-muted-foreground">
                        refreshes every 15s · serving {stats.serving.mode} · {stats.corpus.documents ?? "—"} docs · {stats.corpus.chunks ?? "—"} chunks
                      </p>
                    </motion.div>
                  ) : (
                    <div className="space-y-2">
                      <Skeleton className="h-16 w-full" />
                      <Skeleton className="h-10 w-full" />
                      <Skeleton className="h-20 w-full" />
                    </div>
                  )}
                </CardContent>
              </Card>
              </TabPanel>
            </TabsContent>
          </Tabs>
        </aside>
      </div>

      {/* ---------------------------------------------------------- footer */}
      <footer className="mt-auto border-t border-border/70 bg-gradient-to-b from-background to-muted/30 pb-[env(safe-area-inset-bottom)]">
        <div className="mx-auto flex w-full max-w-6xl flex-wrap items-center justify-between gap-x-3 gap-y-1 px-4 py-2.5 text-[11px] text-muted-foreground">
          <span className="flex flex-wrap items-center gap-1.5">
            <span>RAG-Powered Agentic Q&amp;A · FastAPI + hybrid retrieval + verification graph · v{health?.version ?? "1.0.0"}</span>
            {stats && (
              <span className="flex items-center gap-1 rounded-full border border-border/60 bg-background/70 px-2 py-0.5 font-mono text-[10px]">
                <HeartPulse className="h-2.5 w-2.5 text-rose-500" />
                up {stats.service.uptime_human} · {stats.questions.total} Q
                {stats.questions.cache_hits > 0 && (
                  <span className="flex items-center gap-0.5 text-amber-600 dark:text-amber-400">
                    <Zap className="h-2.5 w-2.5" />
                    {stats.questions.cache_hits}
                  </span>
                )}
              </span>
            )}
            {evalReport && (
              <span className="flex items-center gap-1 rounded-full border border-emerald-300/60 bg-emerald-50/60 px-2 py-0.5 font-mono text-[10px] text-emerald-700 dark:border-emerald-900/60 dark:bg-emerald-950/40 dark:text-emerald-400">
                <ClipboardCheck className="h-2.5 w-2.5" />
                eval {evalReport.passed}/{evalReport.total_cases}
                {evalReport.faithfulness_avg !== null && ` · faith ${(evalReport.faithfulness_avg * 100).toFixed(0)}%`}
              </span>
            )}
          </span>
          <span className="font-mono">{health?.serving.embedding_backend ?? "feature-hashing-v1"}</span>
        </div>
      </footer>

      {/* -------------------------------------------------- image Q&A modal */}
      <Dialog open={imgDialog} onOpenChange={setImgDialog}>
        <DialogContent className="sm:max-w-lg">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <ImageIcon className="h-4 w-4 text-emerald-600" />
              Visual Q&amp;A
            </DialogTitle>
            <DialogDescription>
              Upload a chart, diagram or screenshot and ask a question about it. With a served
              VLM this uses the vision model; offline it falls back to OCR.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-3">
            <label className="flex cursor-pointer flex-col items-center justify-center gap-1.5 rounded-lg border border-dashed border-border bg-muted/30 px-4 py-6 text-center text-xs text-muted-foreground transition-colors hover:border-emerald-400 hover:bg-emerald-50/50 dark:hover:border-emerald-800">
              {imgPreview ? (
                <img src={imgPreview} alt="Upload preview" className="max-h-40 rounded-md object-contain" />
              ) : (
                <>
                  <Upload className="h-5 w-5" />
                  <span>Click to choose an image (PNG/JPG, max 8 MB)</span>
                </>
              )}
              <input type="file" accept="image/*" className="hidden" onChange={onPickImage} aria-label="Choose image" />
            </label>
            <Input
              value={imgQuestion}
              onChange={(e) => setImgQuestion(e.target.value)}
              placeholder="e.g. What warranty duration does this chart show?"
              aria-label="Image question"
            />
            {imgResult && (
              <motion.div
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                className="space-y-2 rounded-md border bg-muted/40 p-3 text-sm"
              >
                <Badge variant="outline" className="font-mono text-[10px]">
                  {imgResult.mode}
                </Badge>
                <p className="leading-relaxed">{imgResult.answer}</p>
                {imgResult.image_info?.dominant_colors && (
                  <div className="flex items-center gap-1.5 pt-1">
                    {imgResult.image_info.dominant_colors.slice(0, 5).map((c) => (
                      <span key={c} className="h-3 w-3 rounded-sm border" style={{ backgroundColor: c }} title={c} />
                    ))}
                  </div>
                )}
              </motion.div>
            )}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setImgDialog(false)}>
              Close
            </Button>
            <Button
              onClick={() => void submitImage()}
              disabled={!imgFile || !imgQuestion.trim() || imgBusy}
              className="gap-1.5 bg-emerald-600 hover:bg-emerald-700"
            >
              {imgBusy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
              Ask
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* ------------------------------------------------- source viewer modal */}
      <Dialog open={sourceOpen} onOpenChange={setSourceOpen}>
        <DialogContent className="sm:max-w-2xl">
          <DialogHeader>
            <DialogTitle className="flex flex-wrap items-center gap-2">
              <FileText className="h-4 w-4 text-emerald-600" />
              <span className="font-mono text-sm">{sourceDetail?.doc ?? "Loading source…"}</span>
              {sourceDetail && (
                <>
                  <Badge variant="secondary" className="font-mono text-[10px]">p.{sourceDetail.page}</Badge>
                  <Badge variant="outline" className="font-mono text-[10px]">{sourceDetail.doc_kind}</Badge>
                  {sourceDetail.is_figure_caption && (
                    <Badge variant="outline" className="border-amber-300 text-[10px] text-amber-700 dark:border-amber-800 dark:text-amber-400">
                      figure caption
                    </Badge>
                  )}
                </>
              )}
            </DialogTitle>
            <DialogDescription>
              The full indexed chunk behind a citation — verify the answer against its source.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-3">
            {sourceLoading || !sourceDetail ? (
              <div className="space-y-2">
                <Skeleton className="h-4 w-2/3" />
                <Skeleton className="h-32 w-full" />
                <Skeleton className="h-4 w-1/3" />
              </div>
            ) : (
              <>
                {sourceDetail.section && (
                  <p className="flex items-center gap-1.5 font-mono text-[11px] text-muted-foreground">
                    <Layers className="h-3 w-3" />
                    section: {sourceDetail.section}
                    <span className="ml-2 opacity-70">· ~{sourceDetail.token_count} tokens</span>
                  </p>
                )}
                <div className="thin-scrollbar max-h-72 overflow-y-auto rounded-md border bg-muted/30 p-3">
                  <pre className="whitespace-pre-wrap break-words font-mono text-[11.5px] leading-relaxed text-foreground/90">
                    {sourceDetail.text}
                  </pre>
                </div>
                <p className="truncate font-mono text-[10px] text-muted-foreground/70">
                  {sourceDetail.chunk_id}
                </p>
                {sourceDetail.neighbors.length > 0 && (
                  <div className="space-y-1.5">
                    <p className="text-[10px] uppercase tracking-wide text-muted-foreground">
                      Adjacent chunks in the same document
                    </p>
                    {sourceDetail.neighbors.map((n) => (
                      <button
                        key={n.chunk_id + n.relation}
                        onClick={() => void openSource(n.chunk_id)}
                        className="hover-lift flex w-full items-start gap-2 rounded-md border border-border/60 bg-muted/30 px-2.5 py-1.5 text-left hover:border-emerald-300 hover:bg-emerald-50/40 hover:shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500/60 focus-visible:ring-offset-1 dark:hover:border-emerald-800 dark:hover:bg-emerald-950/40"
                        aria-label={`Open ${n.relation} chunk`}
                      >
                        <Badge variant="outline" className="shrink-0 text-[9px] uppercase">
                          {n.relation}
                        </Badge>
                        <div className="min-w-0 flex-1">
                          <p className="truncate font-mono text-[10px] text-muted-foreground">
                            {n.chunk_id}
                            {n.section ? ` · ${n.section}` : ""}
                          </p>
                          <p className="line-clamp-2 text-[10px] text-muted-foreground/80">{n.preview}</p>
                        </div>
                      </button>
                    ))}
                  </div>
                )}
              </>
            )}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setSourceOpen(false)}>
              Close
            </Button>
            {sourceDetail && (
              <CopyButton text={sourceDetail.text} />
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
