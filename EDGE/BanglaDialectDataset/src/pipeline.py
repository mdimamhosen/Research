"""PDF → UTF-8 .txt OCR pipeline (PaddleOCR only)."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

from src.normalize import join_pages
from src.pdf_pages import count_pages, render_pdf_pages

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

EngineName = str
ProgressCallback = Callable[[int, int, int], None]  # page_number, index, total

# Single open-source engine for this project (no paid APIs).
ENGINES = ("paddle",)
OPEN_SOURCE_ENGINES = ("paddle",)


def resolve_default_engine() -> EngineName:
    """Always PaddleOCR unless OCR_ENGINE is explicitly set to a known engine."""
    forced = (os.getenv("OCR_ENGINE") or "").strip().lower()
    if forced in ENGINES:
        return forced
    return "paddle"


def _ocr_fn(engine: EngineName) -> Callable[[Image.Image], str]:
    """Lazy-import so missing paddle deps fail only when OCR starts."""
    engine = engine.lower().strip()
    if engine == "paddle":
        from src.ocr_paddle import ocr_page_paddle

        return ocr_page_paddle
    raise ValueError(f"Unknown engine '{engine}'. Only supported: {', '.join(ENGINES)}")


def iter_ocr_pages(
    pdf_path: Path | str,
    *,
    engine: EngineName = "paddle",
    dpi: int = 300,
    page_start: int | None = None,
    page_end: int | None = None,
    on_progress: ProgressCallback | None = None,
) -> Iterator[tuple[int, str]]:
    """Yield (page_number, text) for each rendered page."""
    pages = render_pdf_pages(
        pdf_path, dpi=dpi, page_start=page_start, page_end=page_end
    )
    ocr = _ocr_fn(engine)
    total = len(pages)
    for idx, (page_number, image) in enumerate(pages, start=1):
        text = ocr(image)
        if on_progress:
            on_progress(page_number, idx, total)
        yield page_number, text


def ocr_pdf_to_text(
    pdf_path: Path | str,
    *,
    engine: EngineName = "paddle",
    dpi: int = 300,
    page_start: int | None = None,
    page_end: int | None = None,
    on_progress: ProgressCallback | None = None,
) -> str:
    page_texts = list(
        iter_ocr_pages(
            pdf_path,
            engine=engine,
            dpi=dpi,
            page_start=page_start,
            page_end=page_end,
            on_progress=on_progress,
        )
    )
    return join_pages(page_texts)


def ocr_pdf_to_file(
    pdf_path: Path | str,
    output_path: Path | str,
    *,
    engine: EngineName = "paddle",
    dpi: int = 300,
    page_start: int | None = None,
    page_end: int | None = None,
    on_progress: ProgressCallback | None = None,
    force: bool = False,
) -> Path:
    pdf_path = Path(pdf_path)
    output_path = Path(output_path)
    if output_path.exists() and not force:
        return output_path

    text = ocr_pdf_to_text(
        pdf_path,
        engine=engine,
        dpi=dpi,
        page_start=page_start,
        page_end=page_end,
        on_progress=on_progress,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return output_path


def discover_pdfs(
    input_path: Path | str,
    *,
    recursive: bool = False,
) -> list[Path]:
    """Find one PDF file, or all PDFs in a folder (optionally recursive)."""
    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF: {path}")
        return [path]
    if path.is_dir():
        pattern = "**/*.pdf" if recursive else "*.pdf"
        found = {p.resolve(): p for p in path.glob(pattern) if p.is_file()}
        # Also catch *.PDF on case-sensitive filesystems when not recursive
        if not recursive:
            found.update({p.resolve(): p for p in path.glob("*.PDF") if p.is_file()})
        return sorted(found.values(), key=lambda p: str(p).lower())
    raise FileNotFoundError(f"Input not found: {path}")


def unique_path(directory: Path | str, stem: str, suffix: str) -> Path:
    """Return directory/stem.suffix, or stem_2.suffix, stem_3.suffix, ... if taken."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    candidate = directory / f"{stem}{suffix}"
    if not candidate.exists():
        return candidate
    n = 2
    while True:
        candidate = directory / f"{stem}_{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1


__all__ = [
    "ENGINES",
    "OPEN_SOURCE_ENGINES",
    "count_pages",
    "discover_pdfs",
    "iter_ocr_pages",
    "ocr_pdf_to_file",
    "ocr_pdf_to_text",
    "resolve_default_engine",
    "unique_path",
]
