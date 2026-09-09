"""ModelService: backend selection + brain wiring, one singleton for the app."""

from __future__ import annotations

import logging
from typing import Callable

from app.config import settings
from app.llm.base import LLMBackend
from app.llm.brain import LLMBrain
from app.llm.simulation import HeuristicBrain, SimulationBackend
from app.llm.vllm_backend import VLLMBackend
from app.llm.groq_backend import GroqBackend

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
        
        self._groq = None
        if settings.groq_api_key:
            self._groq = GroqBackend()
            
        self.backend: LLMBackend = self._sim
        self.brain: LLMBrain | HeuristicBrain = self.heuristic
        self.active_mode = "simulation"
        self._apply_selection("simulation")

    def _apply_selection(self, mode: str) -> None:
        self.active_mode = mode
        if mode == "vllm":
            self.backend = self._vllm
            self.brain = LLMBrain(self._vllm, self.heuristic)
            logger.info("Model backend: vLLM at %s (%s)", settings.vllm_base_url, settings.vllm_model_name)
        elif mode == "groq" and self._groq:
            self.backend = self._groq
            self.brain = LLMBrain(self._groq, self.heuristic)
            logger.info("Model backend: Groq at %s (%s)", self._groq.base_url, self._groq.model_name)
        else:
            self.backend = self._sim
            self.brain = self.heuristic
            logger.info("Model backend: SIMULATION — deterministic extractive engine")

    async def refresh(self) -> bool:
        """Re-probe vLLM and Groq, and hot-swap backends."""
        vllm_reachable = await self._vllm.probe()
        
        target_mode = "simulation"
        if vllm_reachable:
            target_mode = "vllm"
        elif self._groq and await self._groq.probe():
            target_mode = "groq"
            
        if target_mode != self.active_mode:
            self._apply_selection(target_mode)
            
        return target_mode != "simulation"

    @property
    def mode(self) -> str:
        return self.active_mode

    def serving_info(self) -> dict:
        model_name = "simulation:extractive-v1"
        vision_model_name = ""
        base_url = ""
        
        if self.active_mode in ("vllm", "groq"):
            model_name = self.backend.model_name
            vision_model_name = self.backend.vision_model_name
            base_url = self.backend.base_url

        return {
            "mode": self.active_mode,
            "model": model_name,
            "vision_model": vision_model_name,
            "vllm_base_url": base_url,
            "vllm_reachable": self.active_mode != "simulation",
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
        if self._groq:
            await self._groq.aclose()
