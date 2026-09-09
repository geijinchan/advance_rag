"""vLLM backend — speaks the OpenAI-compatible API exposed by `vllm serve`.

Supports chat completions, token streaming (for TTFT measurement and SSE),
and multimodal image payloads for Qwen2.5-VL-style vision models.
Retries transient failures with capped exponential backoff.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from typing import Any, AsyncIterator

import httpx

from app.config import settings
from app.llm.base import ChatMessage, StreamEvent

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BACKOFF_BASE = 0.4


class VLLMBackend:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model_name: str | None = None,
        vision_model_name: str | None = None,
    ) -> None:
        self.base_url = base_url or settings.vllm_base_url
        self.api_key = api_key or settings.vllm_api_key
        self.model_name = model_name or settings.vllm_model_name
        self.vision_model_name = vision_model_name or settings.vllm_vision_model_name

        self.name = f"vllm:{self.model_name}"
        self.mode = "vllm"
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=httpx.Timeout(connect=5.0, read=settings.vllm_request_timeout, write=10.0, pool=5.0),
        )
        self._vision_client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=httpx.Timeout(connect=5.0, read=settings.vllm_request_timeout, write=10.0, pool=5.0),
        )
        self._requests = 0
        self._errors = 0
        self._tokens = 0

    # ------------------------------------------------------------------ utils
    def _payload(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None,
        max_tokens: int | None,
        temperature: float | None,
        stream: bool,
    ) -> dict[str, Any]:
        return {
            "model": model or self.model_name,
            "messages": [dict(m) for m in messages],
            "max_tokens": max_tokens or settings.vllm_max_tokens,
            "temperature": settings.vllm_temperature if temperature is None else temperature,
            "stream": stream,
        }

    async def _with_retries(self, coro_factory):
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                return await coro_factory()
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
                last_exc = exc
                self._errors += 1
                wait = _BACKOFF_BASE * (2**attempt)
                logger.warning("vLLM retry %d/%d after %s (%.1fs)", attempt + 1, _MAX_RETRIES, exc, wait)
                await asyncio.sleep(wait)
        raise RuntimeError(f"vLLM unreachable after {_MAX_RETRIES} attempts: {last_exc}")

    # ------------------------------------------------------------------- chat
    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        self._requests += 1

        async def call() -> str:
            resp = await self._client.post(
                "/chat/completions",
                json=self._payload(messages, model=None, max_tokens=max_tokens,
                                   temperature=temperature, stream=False),
            )
            resp.raise_for_status()
            data = resp.json()
            self._tokens += data.get("usage", {}).get("completion_tokens", 0)
            return data["choices"][0]["message"]["content"] or ""

        return await self._with_retries(call)

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamEvent]:
        payload = self._payload(messages, model=None, max_tokens=max_tokens,
                                temperature=temperature, stream=True)
        t0 = time.perf_counter()
        first = True
        async with self._client.stream("POST", "/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                raw = line[6:].strip()
                if raw == "[DONE]":
                    break
                try:
                    delta = json.loads(raw)["choices"][0].get("delta", {})
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                token = delta.get("content") or ""
                if token:
                    if first:
                        yield StreamEvent(ttft_ms=round((time.perf_counter() - t0) * 1000, 1))
                        first = False
                    self._tokens += 1
                    yield StreamEvent(token=token)

    # ----------------------------------------------------------------- vision
    async def chat_vision(
        self, prompt: str, image_b64: str, *, mime: str = "image/png"
    ) -> str:
        """Send an image + prompt to the served VLM (Qwen2.5-VL style payload)."""
        self._requests += 1
        data_uri = f"data:{mime};base64,{image_b64}"
        messages = [
            ChatMessage.user(
                [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": prompt},
                ]
            )
        ]

        async def call() -> str:
            resp = await self._client.post(
                "/chat/completions",
                json=self._payload(messages, model=self.vision_model_name,
                                   max_tokens=None, temperature=None, stream=False),
            )
            resp.raise_for_status()
            data = resp.json()
            self._tokens += data.get("usage", {}).get("completion_tokens", 0)
            return data["choices"][0]["message"]["content"] or ""

        return await self._with_retries(call)

    # ------------------------------------------------------------------ probe
    async def probe(self) -> bool:
        try:
            resp = await self._client.get("/models", timeout=settings.vllm_probe_timeout)
            return resp.status_code == 200
        except Exception:
            return False

    def metrics(self) -> dict[str, Any]:
        return {"requests": self._requests, "errors": self._errors, "completion_tokens": self._tokens}

    async def aclose(self) -> None:
        await self._client.aclose()
