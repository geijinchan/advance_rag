"""Shared types for the model layer."""

from __future__ import annotations

from typing import Any, AsyncIterator, Protocol


class ChatMessage(dict):
    """Typed-ish convenience for OpenAI-style chat messages."""

    @classmethod
    def user(cls, content: str) -> "ChatMessage":
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str) -> "ChatMessage":
        return cls(role="assistant", content=content)

    @classmethod
    def system(cls, content: str) -> "ChatMessage":
        return cls(role="system", content=content)


class StreamEvent(dict):
    """One streaming delta: {"token": str} or {"ttft_ms": float}."""

    @property
    def token(self) -> str:
        return self.get("token", "")


class LLMBackend(Protocol):
    """Anything that can chat, stream, and see images."""

    name: str
    mode: str  # "vllm" | "simulation"

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str: ...

    def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamEvent]: ...

    async def chat_vision(
        self, prompt: str, image_b64: str, *, mime: str = "image/png"
    ) -> str: ...

    async def probe(self) -> bool: ...

    def metrics(self) -> dict[str, Any]:
        return {}
