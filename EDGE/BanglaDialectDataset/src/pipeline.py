"""PDF → UTF-8 .txt OCR pipeline."""

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


ENGINES = ("tesseract", "paddle", "deepseek", "gemini", "openai", "claude")
OPEN_SOURCE_ENGINES = ("tesseract", "paddle", "deepseek")


def resolve_default_engine() -> EngineName:
    """Prefer open-source local engines for the automated CLI pipeline."""
    forced = (os.getenv("OCR_ENGINE") or "").strip().lower()
    if forced in ENGINES:
        return forced
    return "tesseract"


def _ocr_fn(engine: EngineName) -> Callable[[Image.Image], str]:
    """Lazy-import backends so missing optional deps don't break startup."""
    engine = engine.lower().strip()
    if engine == "tesseract":
        from src.ocr_tesseract import ocr_page_tesseract

        return ocr_page_tesseract
    if engine == "paddle":
        from src.ocr_paddle import ocr_page_paddle

        return ocr_page_paddle
    if engine == "deepseek":
        from src.ocr_deepseek import ocr_page_deepseek

        return ocr_page_deepseek
    if engine == "gemini":
        from src.ocr_gemini import ocr_page_gemini

        return ocr_page_gemini
    if engine == "openai":
        from src.ocr_openai import ocr_page_openai

        return ocr_page_openai
    if engine == "claude":
        from src.ocr_claude import ocr_page_claude

        return ocr_page_claude
    raise ValueError(f"Unknown engine '{engine}'. Choose from: {', '.join(ENGINES)}")


def iter_ocr_pages(
    pdf_path: Path | str,
    *,
    engine: EngineName = "gemini",
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
    engine: EngineName = "gemini",
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
    engine: EngineName = "gemini",
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


def discover_pdfs(input_path: Path | str) -> list[Path]:
    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF: {path}")
        return [path]
    if path.is_dir():
        # Case-insensitive de-dupe (Windows may match *.pdf and *.PDF twice)
        found = {p.resolve(): p for p in path.glob("*.pdf")}
        found.update({p.resolve(): p for p in path.glob("*.PDF")})
        return sorted(found.values(), key=lambda p: p.name.lower())
    raise FileNotFoundError(f"Input not found: {path}")


def unique_path(directory: Path | str, stem: str, suffix: str) -> Path:
    """Return directory/stem.suffix, or stem_2.suffix, stem_3.suffix, ... if taken.

    Ensures each new book/upload gets its own file for a growing dataset.
    """
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
