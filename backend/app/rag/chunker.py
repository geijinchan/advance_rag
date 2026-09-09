"""Chunk model + sentence-aware chunker with overlap (Part B: ingestion)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any

from app.config import settings

_SENT_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")
_HEADING_RE = re.compile(r"^#{1,4}\s+.*$", re.MULTILINE)


def approx_tokens(text: str) -> int:
    """Cheap token estimate (~1.3 tokens/word for English technical prose)."""
    return max(1, round(len(text.split()) * 1.3))


@dataclass
class Chunk:
    """One indexed passage. `page` is 1-based; flat files always page 1."""

    chunk_id: str
    doc: str
    doc_kind: str
    page: int
    section: str = ""
    text: str = ""
    token_count: int = 0
    is_figure_caption: bool = False

    def to_record(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_record(cls, rec: dict[str, Any]) -> "Chunk":
        return cls(**rec)

    def preview(self, width: int = 240) -> str:
        snippet = " ".join(self.text.split())
        return snippet[: width - 1] + "…" if len(snippet) > width else snippet


class SentenceChunker:
    """Sentence-aware, structure-respecting chunker.

    Strategy (documented trade-off):
      * Split on sentence boundaries and markdown headings, never mid-sentence.
      * Accumulate sentences until ~CHUNK_TARGET_TOKENS, then flush.
      * Overlap: the last CHUNK_OVERLAP_SENTENCES sentences seed the next chunk,
        so facts spanning a boundary remain retrievable from both sides.
      * Headings start a new chunk and become the `section` attribute —
        in the typeset demo PDFs each H2 maps to its own page, which makes
        citations crisper.

    Chosen over fixed 512-token windows because citations then point at
    semantically complete passages rather than arbitrary character ranges.
    """

    def __init__(
        self,
        target_tokens: int | None = None,
        min_tokens: int | None = None,
        overlap_sentences: int | None = None,
    ) -> None:
        self.target = target_tokens or settings.chunk_target_tokens
        self.min_tokens = min_tokens or settings.chunk_min_tokens
        self.overlap = overlap_sentences if overlap_sentences is not None else settings.chunk_overlap_sentences

    def _split_units(self, text: str) -> list[str]:
        units: list[str] = []
        for block in _SENT_RE.split(text.strip()):
            block = block.strip()
            if not block:
                continue
            if _HEADING_RE.match(block):
                units.append(block)
                continue
            # A block containing markdown table rows must be split BY LINE so
            # every row stays atomic: table cells legitimately contain ";" and
            # ":" ("Wait 8 minutes; never cut power") which would otherwise be
            # mistaken for sentence boundaries, breaking the row structure and
            # losing header pairing downstream in the answer extractor.
            if "\n" in block and any(l.strip().startswith("|") for l in block.split("\n")):
                for line in block.split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("|"):
                        units.append(line)  # table row: atomic, any length
                    elif len(line) < 90:
                        units.append(line)
                    else:
                        units.extend(
                            s.strip() for s in re.split(r"(?<=[.!?;:])\s+", line) if s.strip()
                        )
                continue
            if len(block) < 90:
                units.append(block)
            else:
                units.extend(s.strip() for s in re.split(r"(?<=[.!?;:])\s+", block) if s.strip())
        return [u for u in units if u]

    def chunk(
        self,
        doc_name: str,
        doc_kind: str,
        pages: list[str],
    ) -> list[Chunk]:
        """Chunk per-page so `Chunk.page` is always truthful for citations."""
        chunks: list[Chunk] = []
        for page_no, page_text in enumerate(pages, start=1):
            units = self._split_units(page_text)
            if not units:
                continue
            section = self._current_section(units)
            buf: list[str] = []
            buf_tokens = 0
            idx = 0
            while idx < len(units):
                unit = units[idx]
                if _HEADING_RE.match(unit):
                    section = unit.lstrip("#").strip()
                    if buf and approx_tokens("\n".join(buf)) >= self.min_tokens:
                        chunks.append(self._make(doc_name, doc_kind, page_no, section, buf))
                        buf = buf[-self.overlap :] if self.overlap > 0 else []
                        buf_tokens = approx_tokens("\n".join(buf))
                buf.append(unit)
                buf_tokens += approx_tokens(unit)
                if buf_tokens >= self.target:
                    # Never cut a table mid-run: defer the flush while the next
                    # unit continues the same table (safety valve at 2x target
                    # so a pathological table cannot grow a chunk unbounded).
                    mid_table = unit.startswith("|") and idx + 1 < len(units) and units[idx + 1].startswith("|")
                    if not (mid_table and buf_tokens < self.target * 2):
                        chunks.append(self._make(doc_name, doc_kind, page_no, section, buf))
                        buf = buf[-self.overlap :] if self.overlap > 0 else []
                        buf_tokens = approx_tokens("\n".join(buf))
                idx += 1
            if buf and approx_tokens("\n".join(buf)) >= self.min_tokens:
                chunks.append(self._make(doc_name, doc_kind, page_no, section, buf))
            elif buf and chunks:
                # Tail fragment too small → merge into the previous chunk.
                chunks[-1].text += "\n" + "\n".join(buf)
                chunks[-1].token_count = approx_tokens(chunks[-1].text)
            elif buf:
                chunks.append(self._make(doc_name, doc_kind, page_no, section, buf))
        return chunks

    def _current_section(self, units: list[str]) -> str:
        for u in units:
            if _HEADING_RE.match(u):
                return u.lstrip("#").strip()
        return ""

    def _make(self, doc: str, kind: str, page: int, section: str, buf: list[str]) -> Chunk:
        # Newlines preserved: markdown tables stay row-aligned so the
        # extractive generator can turn rows into citable pseudo-sentences.
        text = "\n".join(buf)
        return Chunk(
            chunk_id=f"{doc}::p{page}::{abs(hash(text)) & 0xFFFF:04X}",
            doc=doc,
            doc_kind=kind,
            page=page,
            section=section,
            text=text,
            token_count=approx_tokens(text),
        )


def chunk_document(doc_name: str, doc_kind: str, pages: list[str]) -> list[Chunk]:
    return SentenceChunker().chunk(doc_name, doc_kind, pages)
