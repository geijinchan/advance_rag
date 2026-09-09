#!/usr/bin/env python3
"""CLI ingestion runner (Part B: "keep it re-runnable via a single command").

Usage:
    python scripts/ingest.py                     # ingest the bundled demo corpus
    python scripts/ingest.py path/to/docs ...    # ingest specific files/dirs
    python scripts/ingest.py --reset             # wipe the index first
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import BASE_DIR, settings  # noqa: E402
from app.rag.ingestion import IngestionPipeline  # noqa: E402
from app.rag.vector_store import VectorStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG index")
    parser.add_argument("paths", nargs="*", help="files or directories (default: bundled corpus)")
    parser.add_argument("--reset", action="store_true", help="clear the index before ingesting")
    args = parser.parse_args()

    store = VectorStore(settings.index_file)
    pipeline = IngestionPipeline(store)

    if args.reset:
        store.clear()
        print("index cleared")

    targets = [Path(p) for p in args.paths] or [BASE_DIR / "corpus"]
    total_chunks = 0
    for target in targets:
        if target.is_dir():
            results = pipeline.ingest_corpus_dir(target)
        else:
            results = [pipeline.ingest_path(target)]
        for r in results:
            print(f"  {r.doc:<40} {r.kind:<5} pages={r.pages:<3} chunks={r.chunks:<3} "
                  f"figures={r.figures:<2} words={r.words:<6} {r.duration_ms:.0f}ms")
            total_chunks += r.chunks

    stats = store.stats()
    print(f"\nDone: {stats['total_documents']} documents, {stats['total_chunks']} chunks "
          f"(dim={stats['embedding_dim']}, backend={stats['embedding_backend']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
