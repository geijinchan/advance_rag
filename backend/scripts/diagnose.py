"""Diagnostic probe: dump retrieval + answer units for a question."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.llm.simulation import HeuristicBrain
from app.rag.vector_store import VectorStore
from app.config import settings
from app.rag.retriever import HybridRetriever

QUESTIONS = [
    "What is the standard warranty on the X200?",
    "What does LED status 2 Hz mean?",
    "What is the RMA processing time?",
]


async def main() -> None:
    store = VectorStore(settings.index_file); store.load()
    brain = HeuristicBrain()
    retriever = HybridRetriever(store)
    retriever.rebuild_lexical()
    for q in QUESTIONS:
        print(f"\n{'='*70}\nQ: {q}")
        hits = await retriever.search(q, k=6)
        for i, h in enumerate(hits):
            print(
                f"  [{i}] score={h.score:.3f} cov={h.coverage:.2f} "
                f"fig={h.chunk.is_figure_caption} {h.chunk.doc} p{h.chunk.page} "
                f":: {h.chunk.chunk_id}"
            )
            units = HeuristicBrain._extract_answer_units(h.chunk.text)
            for u in units[:4]:
                print(f"        unit: {u[:110]}")
        answer = await brain.generate_answer(
            q,
            [
                {"doc": h.chunk.doc, "page": h.chunk.page, "text": h.chunk.text, "score": h.score}
                for h in hits[:4]
            ],
        )
        print(f"  ANSWER: {answer[:260]}")


if __name__ == "__main__":
    asyncio.run(main())
