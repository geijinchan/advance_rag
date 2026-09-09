"""Embedding backends.

Two interchangeable implementations behind one protocol:

* ``HashingEmbedder``  — deterministic feature hashing (word unigrams +
  bigrams + char trigrams) projected into a fixed dense vector. Zero
  dependencies, zero model downloads, fully reproducible. Used as the
  offline/sandbox fallback.
* ``SentenceTransformerEmbedder`` — real neural embeddings
  (BAAI/bge-base-en-v1.5 by default) loaded lazily when the optional
  ``sentence-transformers`` package is installed.

The factory :func:`get_embedder` picks the best available backend.
"""

from __future__ import annotations

import logging
import math
import re
from functools import lru_cache
from typing import Protocol

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    """a an and are as at be by for from has have how i in is it its of on or
    that the to was what when where which who will with you your do does did
    can could should would about into over under between each more most other
    some such only own same than too very s t just don now""".split()
)

_BIAS = 0x9E3779B1  # Knuth's multiplicative hash constant.


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    """Lowercase alphanumeric tokenizer shared by BM25 / embedders / grader."""
    tokens = _TOKEN_RE.findall(text.lower())
    if drop_stopwords:
        return [t for t in tokens if t not in _STOPWORDS]
    return tokens


@lru_cache(maxsize=1)
def _stopword_set() -> frozenset[str]:
    return _STOPWORDS


def _fnv1a(data: str) -> int:
    """32-bit FNV-1a hash — fast, stable across runs/platforms."""
    h = 0x811C9DC5
    for byte in data.encode("utf-8", errors="ignore"):
        h ^= byte
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


class EmbeddingBackend(Protocol):
    """Any embedder maps a list of texts to a (N, dim) float32 matrix."""

    name: str
    dim: int

    def encode(self, texts: list[str], *, is_query: bool = False) -> np.ndarray: ...


class HashingEmbedder:
    """Feature-hashing embedder (the "hashing trick").

    Signals hashed per text:
      * word unigrams (weight 1.0)
      * word bigrams  (weight 0.6) — captures short phrases like "warranty period"
      * char trigrams (weight 0.35) — robustness to typos / morphology

    Signed hashing (the sign bit of a second hash) reduces collision bias.
    Output is L2-normalised so cosine similarity == dot product.
    """

    def __init__(self, dim: int = 384) -> None:
        self.name = "feature-hashing-v1"
        self.dim = dim

    def _hash_to_slot(self, feature: str) -> tuple[int, int]:
        h = _fnv1a(feature)
        slot = h % self.dim
        sign = 1.0 if (h >> 31) & 1 == 0 else -1.0
        return slot, sign

    def _embed_one(self, text: str, is_query: bool = False) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = tokenize(text, drop_stopwords=False)
        content = [t for t in tokens if t not in _stopword_set()]
        if not content:  # degenerate input still yields a valid vector
            content = tokens or ["<empty>"]

        def bump(feature: str, weight: float) -> None:
            slot, sign = self._hash_to_slot(feature)
            vec[slot] += sign * weight

        # IDF-ish damping: rare terms matter more. Queries get a mild boost
        # on unigrams so retrieval emphasises the exact question terms.
        uni_w = 1.25 if is_query else 1.0
        for tok in content:
            bump(tok, uni_w)
        for a, b in zip(content, content[1:]):
            bump(f"{a}_{b}", 0.6)
        joined = " ".join(content)
        for i in range(max(0, len(joined) - 2)):
            bump(f"#{joined[i : i + 3]}", 0.35)

        norm = float(np.linalg.norm(vec))
        if norm > 1e-9:
            vec /= norm
        return vec

    def encode(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._embed_one(t, is_query) for t in texts])


class SentenceTransformerEmbedder:
    """Neural embedder used when `sentence-transformers` is installed.

    Prepends the BGE-style instruction to queries ("Represent this sentence
    for searching relevant passages: ...") — required by bge-* models.
    """

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(model_name)
        self.name = f"sentence-transformers:{model_name}"
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        payload = texts
        if is_query and any("bge" in self.name.lower() for _ in [self.name]):
            payload = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]
        vecs = self._model.encode(payload, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


_embedder: EmbeddingBackend | None = None


def get_embedder() -> EmbeddingBackend:
    """Singleton factory — prefers a neural backend, falls back to hashing."""
    global _embedder
    if _embedder is not None:
        return _embedder
    try:
        import sentence_transformers  # noqa: F401

        _embedder = SentenceTransformerEmbedder(settings.sentence_transformer_model)
        logger.info("Embedding backend: %s (dim=%d)", _embedder.name, _embedder.dim)
    except Exception as exc:  # not installed or model download blocked
        logger.info(
            "sentence-transformers unavailable (%s) — using HashingEmbedder", type(exc).__name__
        )
        _embedder = HashingEmbedder(settings.embedding_dim)
    return _embedder


def cosine_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity of `query` (d,) against `matrix` (N, d)."""
    if matrix.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    q = query / (np.linalg.norm(query) + 1e-9)
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
    return m @ q


def softmax(x: list[float], temperature: float = 1.0) -> list[float]:
    m = max(x) if x else 0.0
    exps = [math.exp((v - m) / max(temperature, 1e-9)) for v in x]
    total = sum(exps) or 1.0
    return [e / total for e in exps]
