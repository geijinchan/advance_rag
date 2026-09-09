"""A minimal, dependency-free BM25 (Okapi) implementation."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from app.rag.embeddings import tokenize


@dataclass
class _DocStats:
    term_freqs: Counter = field(default_factory=Counter)
    length: int = 0


class BM25Index:
    """Okapi BM25 with k1=1.5, b=0.75 (Robertson/Sparck-Jones defaults).

    Kept hand-rolled on purpose: the assignment values understanding of the
    retrieval stack over gluing libraries together, and this keeps the
    service deployable with zero extra dependencies.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: list[_DocStats] = []
        self._df: Counter = Counter()  # document frequency per term
        self._avg_len = 0.0

    def __len__(self) -> int:
        return len(self._docs)

    def add(self, text: str) -> int:
        """Index one document; returns its internal id."""
        tokens = tokenize(text)
        stats = _DocStats(term_freqs=Counter(tokens), length=len(tokens))
        self._docs.append(stats)
        for term in stats.term_freqs:
            self._df[term] += 1
        n = len(self._docs)
        self._avg_len = sum(d.length for d in self._docs) / n
        return n - 1

    def _idf(self, term: str) -> float:
        n = len(self._docs) or 1
        df = self._df.get(term, 0)
        # BM25+ variant: adds 0.5 to df to avoid negative IDF for universal terms.
        return math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    def idf_map(self) -> dict[str, float]:
        """IDF per indexed term — corpus statistics for other components
        (the cross-encoder surrogate weights rare terms like BM25 does)."""
        return {t: self._idf(t) for t in self._df}

    def search(self, query: str) -> list[float]:
        """Return one BM25 score per indexed document (unnormalised)."""
        query_terms = tokenize(query)
        if not query_terms or not self._docs:
            return [0.0] * len(self._docs)
        scores = [0.0] * len(self._docs)
        avg = self._avg_len or 1.0
        for term in query_terms:
            idf = self._idf(term)
            for i, doc in enumerate(self._docs):
                tf = doc.term_freqs.get(term, 0)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * doc.length / avg)
                scores[i] += idf * (tf * (self.k1 + 1)) / denom
        return scores

    def search_top(self, query: str, k: int) -> list[tuple[int, float]]:
        """Top-k (doc_id, score) descending; zero scores excluded."""
        pairs = list(enumerate(self.search(query)))
        pairs = [(i, s) for i, s in pairs if s > 1e-9]
        pairs.sort(key=lambda p: p[1], reverse=True)
        return pairs[:k]
