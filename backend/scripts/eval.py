"""CLI: run the golden evaluation set against a live service.

Usage:
    python scripts/eval.py                     # run against http://localhost:3003
    python scripts/eval.py --base http://host:3003 --json out.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from app.eval.evaluator import EvalRunner
from app.agents.state import AgentState
from app.agents.workflow import build_graph
from app.rag.retriever import HybridRetriever
from app.rag.vector_store import VectorStore
from app.config import settings
from app.llm.service import ModelService


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run the RAG golden-set evaluation")
    parser.add_argument("--base", default="http://localhost:3003", help="Service base URL")
    parser.add_argument("--json", dest="json_out", default=None, help="Write full report JSON here")
    args = parser.parse_args()

    async with httpx.AsyncClient(base_url=args.base, timeout=120.0) as client:
        resp = await client.get("/health")
        resp.raise_for_status()
        mode = resp.json()["serving"]["mode"]
        print(f"Service: {args.base} (mode={mode})")

        async def ask(question: str, top_k: int | None = None) -> AgentState:
            r = await client.post("/ask", json={"question": question, "top_k": top_k})
            r.raise_for_status()
            d = r.json()
            st = AgentState(question=question)
            st.answer = d["answer"]
            st.route = d["route"]
            st.citations = d["citations"]
            st.fallback = d["fallback"]
            # rebuild trace for agent_path
            from app.agents.state import TraceRecord
            st.trace = [TraceRecord(node=t["node"], status=t["status"]) for t in d["trace"]]
            return st

        runner = EvalRunner(ask)
        report = await runner.run()

    print(f"\n{'='*64}")
    print(f"Golden set: {report['passed']}/{report['total_cases']} passed "
          f"(accuracy {report['accuracy']:.0%})")
    print(f"Retrieval hit rate: {report['retrieval_hit_rate']}")
    print(f"Fact support rate: {report['fact_support_rate']:.0%}")
    print(f"Latency p50/p95: {report['latency_p50_ms']:.0f}ms / {report['latency_p95_ms']:.0f}ms")
    print(f"{'='*64}")
    for r in report["results"]:
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"[{mark}] {r['id']:<28} {r['latency_ms']:>7.1f}ms  {r['answer_preview'][:70]}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\nReport written to {args.json_out}")
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
