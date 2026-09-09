#!/usr/bin/env python3
"""CLI smoke test: run questions through the full agentic graph without the API.

Usage:
    python scripts/ask.py "What is the X200 warranty period?"
    python scripts/ask.py --all          # run the built-in suggestion set
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.state import AgentState  # noqa: E402
from app.config import settings  # noqa: E402
from app.llm.service import ModelService  # noqa: E402
from app.rag.ingestion import IngestionPipeline  # noqa: E402
from app.rag.retriever import HybridRetriever  # noqa: E402
from app.rag.vector_store import VectorStore  # noqa: E402
from app.agents.workflow import build_graph  # noqa: E402

SAMPLES = [
    "What is the maximum warranty period mentioned for the X200 model?",
    "Which firmware version fixed the MODBUS-TCP timeout bug?",
    "What does a blinking red LED at 2 Hz indicate?",
    "Who won the 2022 World Cup?",  # must trigger the honest fallback
    "hello there!",                  # chit-chat path
]


async def ask(graph, question: str) -> None:
    state = AgentState(question=question)
    print(f"\n{'─' * 72}\nQ: {question}")
    result = await graph.run(state)
    print(f"A: {result.answer}")
    if result.citations:
        for c in result.citations[:3]:
            print(f"   ↳ {c['doc']} p.{c['page']} (score {c['score']})")
    print(f"   path: {' → '.join(result.agent_path)} | fallback={result.fallback} "
          f"| {result.elapsed_ms():.0f} ms")


async def main_async(questions: list[str]) -> None:
    store = VectorStore(settings.index_file)
    if not store.load() and settings.auto_ingest_on_boot:
        pipeline = IngestionPipeline(store)
        results = pipeline.ingest_corpus_dir(settings.corpus_dir)
        print(f"ingested {len(results)} documents from {settings.corpus_dir}")
    models = ModelService(corpus_vocab_provider=store.corpus_vocabulary)
    await models.refresh()
    retriever = HybridRetriever(store)
    graph = build_graph(models, retriever, models.mode)
    print(f"serving mode: {models.mode}")
    for q in questions:
        await ask(graph, q)
    await models.aclose()


def main() -> int:
    parser = argparse.ArgumentParser(description="Agentic Q&A smoke test")
    parser.add_argument("question", nargs="*", help="question(s) to ask")
    parser.add_argument("--all", action="store_true", help="run the built-in sample set")
    args = parser.parse_args()
    questions = args.question or SAMPLES if args.all or not args.question else args.question
    asyncio.run(main_async(questions))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
