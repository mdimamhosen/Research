"""PDF page rendering via PyMuPDF (no Poppler required)."""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image


def render_pdf_pages(
    pdf_path: Path | str,
    dpi: int = 300,
    page_start: int | None = None,
    page_end: int | None = None,
) -> list[tuple[int, Image.Image]]:
    """Render PDF pages to RGB PIL images.

    Page numbers are 1-based inclusive for page_start/page_end.
    Returns list of (page_number, image) tuples.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    pages: list[tuple[int, Image.Image]] = []
    with fitz.open(pdf_path) as doc:
        total = doc.page_count
        start = 1 if page_start is None else max(1, page_start)
        end = total if page_end is None else min(total, page_end)
        if start > end:
            raise ValueError(f"Invalid page range: {start}-{end} (doc has {total} pages)")

        for i in range(start - 1, end):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            pages.append((i + 1, image))

    return pages


def count_pages(pdf_path: Path | str) -> int:
    pdf_path = Path(pdf_path)
    with fitz.open(pdf_path) as doc:
        return doc.page_count
