"""Multimodal Q&A (Part E).

Chosen option: **Option 2 — visual Q&A endpoint** (``POST /ask-image``),
with Option 1 (image-aware ingestion) also implemented via OCR-captioned
figures during indexing. See README for the trade-off discussion.

Behaviour:
  * vLLM live  → the image is sent to the served VLM (Qwen2.5-VL style).
  * offline    → deterministic fallback: OCR (tesseract) + PIL image
    analysis + corpus grounding. The response is honest about which path
    answered.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from collections import Counter

from PIL import Image

from app.config import settings
from app.rag.retriever import HybridRetriever

logger = logging.getLogger(__name__)

_MAX_IMAGE_BYTES = 8 * 1024 * 1024
_OCR_CHAR_TARGET = 1200


class ImageQAService:
    def __init__(self, models, retriever: HybridRetriever) -> None:
        self.models = models
        self.retriever = retriever

    async def answer(self, image_bytes: bytes, question: str, filename: str = "image") -> dict:
        t0 = time.perf_counter()
        if len(image_bytes) > _MAX_IMAGE_BYTES:
            raise ValueError("image too large (max 8 MB)")

        pil = Image.open(io.BytesIO(image_bytes))
        pil.load()
        image_info = self._basic_info(pil, filename)

        if self.models.active_mode != "simulation":
            answer = await self.models.brain.answer_image(
                question, base64.b64encode(image_bytes).decode(), mime=self._mime(pil)
            )
            mode = "vllm-vision"
            ocr_text = ""
        else:
            answer, ocr_text = await self._offline_answer(pil, question)
            mode = "ocr-fallback"

        return {
            "question": question,
            "answer": answer,
            "mode": mode,
            "ocr_text": ocr_text,
            "image_info": image_info,
            "citations": [],
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _mime(pil: Image.Image) -> str:
        return f"image/{(pil.format or 'PNG').lower()}"

    @staticmethod
    def _basic_info(pil: Image.Image, filename: str) -> dict:
        w, h = pil.size
        small = pil.convert("RGB")
        small.thumbnail((64, 64))
        pixels = list(small.getdata())
        common = Counter(pixels).most_common(5)
        return {
            "filename": filename,
            "format": pil.format or "unknown",
            "width": w,
            "height": h,
            "aspect_ratio": round(w / h, 2) if h else 0,
            "megapixels": round(w * h / 1e6, 2),
            "dominant_colors": ["#%02x%02x%02x" % c for c, _ in common],
        }

    async def _offline_answer(self, pil: Image.Image, question: str) -> tuple[str, str]:
        """OCR + corpus grounding. Deterministic, honest about its limits."""
        ocr_text = ""
        try:
            import pytesseract

            ocr_text = pytesseract.image_to_string(pil).strip()
        except Exception as exc:
            logger.debug("OCR unavailable: %s", exc)

        parts: list[str] = []
        info = self._basic_info(pil, "image")
        if ocr_text:
            compact = " ".join(ocr_text.split())
            parts.append(
                "I can read the following text in the image: "
                f"“{compact[:_OCR_CHAR_TARGET]}”"
                + ("…" if len(compact) > _OCR_CHAR_TARGET else "")
            )
            # Try grounding the question in the corpus as well.
            hits = await self.retriever.search(question, k=3)
            grounded = [h for h in hits if h.coverage >= 0.25]
            if grounded:
                best = grounded[0].chunk
                parts.append(
                    f"Related passage from the corpus: {best.preview(200)} "
                    f"[Source: {best.doc}, page {best.page}]"
                )
            else:
                parts.append(
                    "I couldn't connect the image content to the indexed documents with confidence."
                )
        else:
            parts.append(
                f"The image is {info['width']}×{info['height']} px "
                f"({info['format']}, dominant colours {', '.join(info['dominant_colors'][:3])}), "
                "but I couldn't extract readable text from it."
            )
        parts.append(
            "[Offline mode: served by the OCR fallback — start a vLLM vision model "
            "for full image understanding.]"
        )
        return " ".join(parts), " ".join(ocr_text.split())[:600]
