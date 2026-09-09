"""Ingestion pipeline: parse → figure captioning (Part E opt.1) →
chunk → embed → index. One re-runnable command for the whole corpus."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from app.parsing.pdf_parser import ParsedDocument, SUPPORTED_EXTENSIONS, parse_bytes, parse_file
from app.rag.chunker import Chunk, SentenceChunker
from app.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    doc: str
    kind: str
    pages: int
    chunks: int
    figures: int
    words: int
    duration_ms: float


class IngestionPipeline:
    def __init__(self, store: VectorStore, vlm_describer=None) -> None:
        """`vlm_describer`: optional callable(ParsedFigure) -> str used to
        caption figures during indexing (image-aware ingestion, Part E)."""
        self.store = store
        self.vlm_describer = vlm_describer
        self.chunker = SentenceChunker()

    # -------------------------------------------------------------- single doc
    def ingest_bytes(self, name: str, data: bytes, *, replace: bool = True) -> IngestResult:
        t0 = time.perf_counter()
        parsed = parse_bytes(data, name)
        return self._ingest_parsed(parsed, replace=replace, t0=t0)

    def ingest_path(self, path: Path, *, replace: bool = True) -> IngestResult:
        t0 = time.perf_counter()
        parsed = parse_file(path)
        return self._ingest_parsed(parsed, replace=replace, t0=t0)

    def _ingest_parsed(self, parsed: ParsedDocument, *, replace: bool, t0: float) -> IngestResult:
        if replace:
            self.store.remove_doc(parsed.name)
        chunks = self.chunker.chunk(parsed.name, parsed.kind, parsed.pages)

        # ---- image-aware ingestion: caption figures, index as pseudo-chunks --
        figure_count = 0
        for fig in parsed.figures:
            caption = self._caption_figure(parsed.name, fig)
            if not caption:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{parsed.name}::fig::p{fig.page}",
                    doc=parsed.name,
                    doc_kind=parsed.kind,
                    page=fig.page,
                    section="[Figure]",
                    text=f"[Figure on page {fig.page} of {parsed.name}] {caption}",
                    token_count=max(1, round(len(caption.split()) * 1.3)),
                    is_figure_caption=True,
                )
            )
            figure_count += 1

        self.store.add_chunks(chunks)
        return IngestResult(
            doc=parsed.name,
            kind=parsed.kind,
            pages=len(parsed.pages),
            chunks=len(chunks),
            figures=figure_count,
            words=parsed.words,
            duration_ms=round((time.perf_counter() - t0) * 1000, 1),
        )

    def _caption_figure(self, doc: str, fig) -> str:
        """Caption a figure via VLM when served; else OCR fallback."""
        if self.vlm_describer is not None:
            try:
                return self.vlm_describer(doc, fig)
            except Exception as exc:
                logger.warning("VLM captioning failed for %s p%d: %s", doc, fig.page, exc)
        return self._ocr_caption(doc, fig)

    def _ocr_caption(self, doc: str, fig) -> str:
        """Offline fallback: OCR text + geometric metadata → indexable text.

        Without a served VLM this is deliberately conservative: it indexes
        only what OCR can read (chart titles, axis labels) plus basic image
        facts, and the README documents this degradation honestly.
        """
        parts: list[str] = [f"Embedded figure ({fig.width}x{fig.height}px)."]
        try:
            import io as _io

            import pytesseract
            from PIL import Image

            pil = Image.open(_io.BytesIO(fig.image_bytes))
            text = pytesseract.image_to_string(pil).strip()
            if text:
                words = " ".join(text.split())[:600]
                parts.append(f"Visible text: {words}")
        except Exception as exc:  # tesseract missing or image unreadable
            logger.debug("OCR fallback unavailable for %s: %s", doc, exc)
        return " ".join(parts)

    # ---------------------------------------------------------------- corpus
    def ingest_corpus_dir(self, corpus_dir: Path) -> list[IngestResult]:
        """Ingest every supported file in a directory (skip /assets)."""
        results: list[IngestResult] = []
        files = sorted(
            p for p in corpus_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS and "assets" not in p.parts
        )
        for path in files:
            try:
                results.append(self.ingest_path(path))
            except Exception as exc:
                logger.error("Failed to ingest %s: %s", path.name, exc)
        return results
