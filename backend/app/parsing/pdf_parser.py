"""Document parsing: PDF (pdfplumber, page-aware, figure extraction) and
plain text / markdown loaders (Part B ingestion + Part E image support)."""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pdfplumber
from PIL import Image

logger = logging.getLogger(__name__)

DocKind = Literal["pdf", "text"]

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt", ".markdown"}


@dataclass
class ParsedFigure:
    """An image extracted from a PDF page (candidates for VLM captioning)."""

    page: int
    image_bytes: bytes
    width: int
    height: int
    caption: str = ""


@dataclass
class ParsedDocument:
    name: str
    kind: DocKind
    pages: list[str] = field(default_factory=list)
    figures: list[ParsedFigure] = field(default_factory=list)
    words: int = 0

    def __post_init__(self) -> None:
        self.words = sum(len(p.split()) for p in self.pages)


def _clean_page_text(text: str) -> str:
    """Normalise whitespace while keeping headings/list structure readable."""
    lines = [ln.rstrip() for ln in (text or "").splitlines()]
    out: list[str] = []
    blank = 0
    for ln in lines:
        if not ln.strip():
            blank += 1
            if blank <= 1:
                out.append("")
            continue
        blank = 0
        out.append(ln)
    cleaned = "\n".join(out).strip()
    return cleaned


def _figure_from_pil(img: Image.Image, page: int) -> ParsedFigure:
    buf = io.BytesIO()
    rgb = img.convert("RGB")
    rgb.thumbnail((1024, 1024))
    rgb.save(buf, format="PNG")
    return ParsedFigure(page=page, image_bytes=buf.getvalue(), width=rgb.width, height=rgb.height)


def parse_pdf(data: bytes, name: str) -> ParsedDocument:
    """Extract per-page text + embedded raster figures via pdfplumber.

    Figures larger than 120x120 px are kept — smaller images are almost
    always icons, bullets or spacers.
    """
    pages: list[str] = []
    figures: list[ParsedFigure] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            text = _clean_page_text(page.extract_text() or "")
            pages.append(text)
            for img_meta in page.images or []:
                try:
                    srcsize = img_meta.get("srcsize", (0, 0))
                    w, h = float(srcsize[0]), float(srcsize[1])
                except Exception:
                    w, h = float(img_meta.get("width", 0)), float(img_meta.get("height", 0))
                if w < 120 or h < 120:
                    continue
                try:
                    region = page.crop((float(img_meta["x0"]), float(img_meta["top"]),
                                        float(img_meta["x1"]), float(img_meta["bottom"])))
                    pil = region.to_image(resolution=144).original
                    figures.append(_figure_from_pil(pil, page_no))
                except Exception as exc:
                    logger.debug("Figure extraction skipped on %s p%d: %s", name, page_no, exc)
    return ParsedDocument(name=name, kind="pdf", pages=pages, figures=figures)


def parse_text(data: bytes, name: str) -> ParsedDocument:
    """Text/markdown: whole file is one logical page (page=1 citations)."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1", errors="replace")
    return ParsedDocument(name=name, kind="text", pages=[_clean_page_text(text)])


def parse_file(path: Path) -> ParsedDocument:
    return parse_bytes(path.read_bytes(), path.name)


def parse_bytes(data: bytes, name: str) -> ParsedDocument:
    suffix = Path(name).suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(data, name)
    if suffix in {".md", ".markdown", ".txt"}:
        return parse_text(data, name)
    raise ValueError(f"Unsupported file type: {suffix!r} (allowed: {sorted(SUPPORTED_EXTENSIONS)})")
