"""Numpy-backed vector store with cosine search, MMR, and JSON persistence.

Why not Qdrant/Chroma here: the assignment allows any vector store; an
in-process store keeps the demo self-contained (no extra containers) while
exposing the same logical interface. `docker-compose.yml` includes an
optional Qdrant profile and the README documents the swap path.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Callable

import numpy as np

from app.config import settings
from app.rag.chunker import Chunk
from app.rag.embeddings import cosine_matrix, get_embedder

logger = logging.getLogger(__name__)


class VectorStore:
    """Thread-safe in-memory vector store with disk persistence."""

    def __init__(self, index_file: Path) -> None:
        self.index_file = index_file
        self._lock = threading.RLock()
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._doc_stats: dict[str, dict] = {}
        self._embedder_name = ""

    # ------------------------------------------------------------------ state
    @property
    def size(self) -> int:
        with self._lock:
            return len(self._chunks)

    @property
    def dim(self) -> int:
        return self._matrix.shape[1] if self._matrix.ndim == 2 else 0

    def is_empty(self) -> bool:
        with self._lock:
            return not self._chunks

    def snapshot(self) -> list[Chunk]:
        """Consistent copy of all chunks (for BM25 rebuilds etc.)."""
        with self._lock:
            return list(self._chunks)

    def get_by_chunk_id(self, chunk_id: str) -> Chunk | None:
        with self._lock:
            for c in self._chunks:
                if c.chunk_id == chunk_id:
                    return c
        return None

    def corpus_vocabulary(self) -> set[str]:
        """All distinct content tokens in the corpus (used by the router)."""
        from app.rag.embeddings import tokenize

        with self._lock:
            text = " ".join(f"{c.text} {c.section} {c.doc}" for c in self._chunks)
        return set(tokenize(text))

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()
            self._matrix = np.zeros((0, 0), dtype=np.float32)
            self._doc_stats.clear()
        self._persist_locked()

    def has_doc(self, doc: str) -> bool:
        with self._lock:
            return doc in self._doc_stats

    def remove_doc(self, doc: str) -> int:
        with self._lock:
            keep = [i for i, c in enumerate(self._chunks) if c.doc != doc]
            removed = len(self._chunks) - len(keep)
            if removed:
                self._matrix = self._matrix[keep] if self._matrix.size else self._matrix
                self._chunks = [self._chunks[i] for i in keep]
                self._doc_stats.pop(doc, None)
                self._rebuild_doc_stats()
                self._persist_locked()
            return removed

    def _rebuild_doc_stats(self) -> None:
        """Recompute per-doc stats after structural changes."""
        agg: dict[str, dict] = {}
        for c in self._chunks:
            st = agg.setdefault(
                c.doc, {"chunks": 0, "pages": set(), "words": 0, "figures": 0, "kind": c.doc_kind}
            )
            st["chunks"] += 1
            st["pages"].add(c.page)
            st["words"] += len(c.text.split())
            if c.is_figure_caption:
                st["figures"] += 1
        self._doc_stats = {d: {**v, "pages": len(v["pages"])} for d, v in agg.items()}

    # -------------------------------------------------------------- write path
    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Embed and append chunks (documents already removed by caller)."""
        if not chunks:
            return 0
        embedder = get_embedder()
        vectors = embedder.encode([c.text for c in chunks])
        with self._lock:
            self._embedder_name = embedder.name
            if self._matrix.size == 0 or self._matrix.shape[0] == 0:
                self._matrix = vectors.astype(np.float32)
            else:
                if vectors.shape[1] != self._matrix.shape[1]:
                    raise ValueError(
                        f"Embedding dim mismatch: store={self._matrix.shape[1]} new={vectors.shape[1]}. "
                        "Re-ingest the corpus after changing embedding backend."
                    )
                self._matrix = np.vstack([self._matrix, vectors.astype(np.float32)])
            self._chunks.extend(chunks)
            self._rebuild_doc_stats()
            self._persist_locked()
        return len(chunks)

    # ------------------------------------------------------------- read path
    def dense_search(
        self, query: str, k: int, mmr_lambda: float | None = None
    ) -> list[tuple[Chunk, float]]:
        """Top-k dense retrieval with Maximal Marginal Relevance re-ranking.

        MMR trades a little relevance for diversity — prevents the top-k from
        being five near-duplicate chunks from the same page.
        """
        with self._lock:
            if not self._chunks:
                return []
            embedder = get_embedder()
            q = embedder.encode([query], is_query=True)[0]
            sims = cosine_matrix(q, self._matrix)
            lam = settings.mmr_lambda if mmr_lambda is None else mmr_lambda
            k = min(k, len(self._chunks))
            # Candidate pool 4x k, then MMR-select from it.
            pool = np.argsort(-sims)[: min(len(sims), k * 4)].tolist()
            selected: list[int] = []
            while pool and len(selected) < k:
                if not selected:
                    best = pool.pop(0)
                    selected.append(best)
                    continue
                best_i, best_score = None, -1e9
                for cand in pool:
                    relevance = float(sims[cand])
                    redundancy = max(float(np.dot(self._matrix[cand], self._matrix[s])) for s in selected)
                    mmr = lam * relevance - (1 - lam) * redundancy
                    if mmr > best_score:
                        best_i, best_score = cand, mmr
                pool.remove(best_i)
                selected.append(best_i)
            # Normalise raw cosine scores into [0, 1] via min-max over selection.
            raw = [float(sims[i]) for i in selected]
            lo, hi = (min(raw), max(raw)) if raw else (0.0, 1.0)
            span = (hi - lo) or 1.0
            return [
                (self._chunks[i], (s - lo) / span if hi > 0 else 0.0)
                for i, s in zip(selected, raw)
            ]

    def stats(self) -> dict:
        with self._lock:
            return {
                "total_documents": len(self._doc_stats),
                "total_chunks": len(self._chunks),
                "embedding_dim": int(self._matrix.shape[1]) if self._matrix.ndim == 2 else 0,
                "embedding_backend": self._embedder_name,
                "documents": [
                    {
                        "doc": doc,
                        "kind": st.get("kind", "text"),
                        "pages": st.get("pages", 0),
                        "chunks": st.get("chunks", 0),
                        "figures": st.get("figures", 0),
                        "words": st.get("words", 0),
                    }
                    for doc, st in sorted(self._doc_stats.items())
                ],
            }

    # ------------------------------------------------------------ persistence
    def _persist_locked(self) -> None:
        try:
            self.index_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": 1,
                "embedder": self._embedder_name,
                "chunks": [c.to_record() for c in self._chunks],
                "doc_stats": {d: {k: v for k, v in st.items()} for d, st in self._doc_stats.items()},
            }
            tmp = self.index_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.index_file)
        except Exception as exc:  # persistence must never kill the API
            logger.warning("Index persist failed: %s", exc)

    def load(self) -> bool:
        """Load persisted index. Returns False when rebuild is required
        (missing file or embedding-backend change invalidates vectors)."""
        if not self.index_file.exists():
            return False
        try:
            payload = json.loads(self.index_file.read_text(encoding="utf-8"))
            chunks = [Chunk.from_record(r) for r in payload.get("chunks", [])]
            if not chunks:
                return False
            embedder = get_embedder()
            with self._lock:
                self._chunks = chunks
                self._matrix = embedder.encode([c.text for c in chunks]).astype(np.float32)
                self._embedder_name = embedder.name
                self._rebuild_doc_stats()
            logger.info("Loaded index: %d chunks, %d docs", len(chunks), len(self._doc_stats))
            return True
        except Exception as exc:
            logger.warning("Index load failed (%s) — will re-ingest", exc)
            return False
