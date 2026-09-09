"""Groq backend — speaks the OpenAI-compatible API via Groq's cloud.

This is a thin wrapper around the vLLM backend, as Groq provides an
identical OpenAI-compatible API.
"""

from __future__ import annotations

from app.config import settings
from app.llm.vllm_backend import VLLMBackend

class GroqBackend(VLLMBackend):
    def __init__(self) -> None:
        super().__init__(
            base_url="https://api.groq.com/openai/v1",
            api_key=settings.groq_api_key,
            model_name=settings.groq_model_name,
            vision_model_name=settings.groq_vision_model_name,
        )
        self.name = f"groq:{self.model_name}"
        self.mode = "groq"
