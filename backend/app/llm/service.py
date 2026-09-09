"""ModelService: backend selection + brain wiring, one singleton for the app."""

from __future__ import annotations

import logging
from typing import Callable

from app.config import settings
from app.llm.base import LLMBackend
from app.llm.brain import LLMBrain
from app.llm.simulation import HeuristicBrain, SimulationBackend
from app.llm.vllm_backend import VLLMBackend

logger = logging.getLogger(__name__)


class ModelService:
    """Owns the active backend and the matching "brain" (decision engine).

    Selection logic:
      1. starts in simulation mode (safe default, no blocking network IO
         at import time);
      2. ``await refresh()`` (called from the FastAPI lifespan and /health)
         probes ``VLLM_BASE_URL`` and hot-swaps to the vLLM backend when a
         server is found — and back if it disappears.

    Both brains share method signatures, so agent nodes never branch on mode.
    """

    def __init__(self, corpus_vocab_provider: Callable[[], set[str]] | None = None) -> None:
        self.heuristic = HeuristicBrain(corpus_vocab_provider)
        self._sim = SimulationBackend(self.heuristic)
        self._vllm = VLLMBackend()
        self.backend: LLMBackend = self._sim
        self.brain: LLMBrain | HeuristicBrain = self.heuristic
        self.vllm_available = False
        self._apply_selection(False)

    def _apply_selection(self, vllm_reachable: bool) -> None:
        self.vllm_available = vllm_reachable
        if vllm_reachable:
            self.backend = self._vllm
            self.brain = LLMBrain(self._vllm, self.heuristic)
            logger.info("Model backend: vLLM at %s (%s)", settings.vllm_base_url, settings.vllm_model_name)
        else:
            self.backend = self._sim
            self.brain = self.heuristic
            logger.info(
                "Model backend: SIMULATION (no vLLM at %s) — deterministic extractive engine",
                settings.vllm_base_url,
            )

    async def refresh(self) -> bool:
        """Re-probe vLLM and hot-swap backends when availability changes."""
        reachable = await self._vllm.probe()
        if reachable != self.vllm_available:
            self._apply_selection(reachable)
        return reachable

    @property
    def mode(self) -> str:
        return "vllm" if self.vllm_available else "simulation"

    def serving_info(self) -> dict:
        return {
            "mode": self.mode,
            "model": settings.vllm_model_name if self.vllm_available else "simulation:extractive-v1",
            "vision_model": settings.vllm_vision_model_name,
            "vllm_base_url": settings.vllm_base_url,
            "vllm_reachable": self.vllm_available,
            "embedding_backend": self._embedding_name(),
        }

    def _embedding_name(self) -> str:
        try:
            from app.rag.embeddings import get_embedder

            return get_embedder().name
        except Exception:
            return "unknown"

    async def aclose(self) -> None:
        await self._vllm.aclose()
