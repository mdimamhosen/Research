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


ENGINES = ("openai", "claude", "tesseract")


def resolve_default_engine() -> EngineName:
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "claude"
    return "tesseract"


def _ocr_fn(engine: EngineName) -> Callable[[Image.Image], str]:
    """Lazy-import backends so missing optional deps don't break app startup."""
    engine = engine.lower().strip()
    if engine == "openai":
        from src.ocr_openai import ocr_page_openai

        return ocr_page_openai
    if engine == "claude":
        from src.ocr_claude import ocr_page_claude

        return ocr_page_claude
    if engine == "tesseract":
        from src.ocr_tesseract import ocr_page_tesseract

        return ocr_page_tesseract
    raise ValueError(f"Unknown engine '{engine}'. Choose from: {', '.join(ENGINES)}")


def iter_ocr_pages(
    pdf_path: Path | str,
    *,
    engine: EngineName = "openai",
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
    engine: EngineName = "openai",
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
    engine: EngineName = "openai",
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


__all__ = [
    "ENGINES",
    "count_pages",
    "discover_pdfs",
    "iter_ocr_pages",
    "ocr_pdf_to_file",
    "ocr_pdf_to_text",
    "resolve_default_engine",
]
