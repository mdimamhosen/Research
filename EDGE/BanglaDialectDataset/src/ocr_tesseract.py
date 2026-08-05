"""Local Tesseract OCR (Bengali + English)."""

from __future__ import annotations

from PIL import Image

try:
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None  # type: ignore


def ocr_page_tesseract(
    image: Image.Image,
    *,
    lang: str = "ben+eng",
) -> str:
    if pytesseract is None:
        raise RuntimeError("pytesseract is not installed. pip install pytesseract")

    try:
        text = pytesseract.image_to_string(image, lang=lang)
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "Tesseract is not installed or not on PATH. "
            "Install from https://github.com/UB-Mannheim/tesseract/wiki "
            "and include the Bengali (ben) language pack."
        ) from exc
    except pytesseract.TesseractError as exc:
        raise RuntimeError(
            f"Tesseract failed (is language pack '{lang}' installed?): {exc}"
        ) from exc

    return (text or "").strip()
