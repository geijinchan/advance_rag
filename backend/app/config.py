"""Application configuration via environment variables (12-factor style)."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Central configuration. Every value can be overridden via env vars.

    See `.env.example` for the full documented list.
    """

    model_config = SettingsConfigDict(
        env_file=(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Service -----------------------------------------------------------
    service_name: str = Field("rag-agent", description="Service identifier")
    api_host: str = Field("0.0.0.0", alias="API_HOST")
    api_port: int = Field(3003, alias="API_PORT")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    # --- vLLM serving (Part A) ----------------------------------------------
    vllm_base_url: str = Field(
        "http://localhost:8001/v1",
        alias="VLLM_BASE_URL",
        description="OpenAI-compatible base URL of the vLLM server.",
    )
    vllm_api_key: str = Field("EMPTY", alias="VLLM_API_KEY", description="vLLM ignores this but the field is kept for OpenAI-client compat.")
    vllm_model_name: str = Field("Qwen/Qwen2.5-7B-Instruct", alias="VLLM_MODEL_NAME")
    vllm_vision_model_name: str = Field(
        "Qwen/Qwen2.5-VL-7B-Instruct", alias="VLLM_VISION_MODEL_NAME"
    )
    vllm_probe_timeout: float = Field(2.0, alias="VLLM_PROBE_TIMEOUT")
    vllm_request_timeout: float = Field(90.0, alias="VLLM_REQUEST_TIMEOUT")
    vllm_max_tokens: int = Field(768, alias="VLLM_MAX_TOKENS")
    vllm_temperature: float = Field(0.1, alias="VLLM_TEMPERATURE")

    # --- Embeddings & retrieval (Part B) ------------------------------------
    embedding_dim: int = Field(384, alias="EMBEDDING_DIM")
    sentence_transformer_model: str = Field(
        "BAAI/bge-base-en-v1.5", alias="EMBEDDING_MODEL"
    )
    chunk_target_tokens: int = Field(
        380, alias="CHUNK_TARGET_TOKENS", description="Target chunk size in ~tokens (whitespace words x 1.3)."
    )
    chunk_min_tokens: int = Field(60, alias="CHUNK_MIN_TOKENS")
    chunk_overlap_sentences: int = Field(1, alias="CHUNK_OVERLAP_SENTENCES")
    retrieval_top_k: int = Field(6, alias="RETRIEVAL_TOP_K")
    mmr_lambda: float = Field(0.7, alias="MMR_LAMBDA")
    rrf_k: int = Field(60, alias="RRF_K", description="k constant for Reciprocal Rank Fusion.")
    rerank_enabled: bool = Field(
        True, alias="RERANK_ENABLED",
        description="Cross-encoder rerank stage after RRF fusion.",
    )
    rerank_n: int = Field(
        10, alias="RERANK_N",
        description="How many fusion candidates the cross-encoder re-scores.",
    )

    # --- Agentic workflow (Part C) -------------------------------------------
    max_retrieval_retries: int = Field(2, alias="MAX_RETRIEVAL_RETRIES")
    grade_threshold: float = Field(
        0.35, alias="GRADE_THRESHOLD", description="Minimum relevance score for a chunk to count as relevant."
    )
    verification_support_threshold: float = Field(
        0.5, alias="VERIFICATION_THRESHOLD"
    )

    # --- Storage --------------------------------------------------------------
    corpus_dir: Path = Field(BASE_DIR / "corpus", alias="CORPUS_DIR")
    data_dir: Path = Field(BASE_DIR / "data", alias="DATA_DIR")
    index_file: Path = Field(BASE_DIR / "data" / "index.json", alias="INDEX_FILE")
    auto_ingest_on_boot: bool = Field(True, alias="AUTO_INGEST_ON_BOOT")

    # --- Benchmark (Part A) ---------------------------------------------------
    benchmark_default_concurrency: int = Field(8, alias="BENCH_CONCURRENCY")
    benchmark_default_requests: int = Field(24, alias="BENCH_REQUESTS")


settings = Settings()


def resolve(rel: str) -> Path:
    """Resolve a path relative to the service root (for CLI entry points)."""
    p = Path(rel)
    return p if p.is_absolute() else BASE_DIR / p
