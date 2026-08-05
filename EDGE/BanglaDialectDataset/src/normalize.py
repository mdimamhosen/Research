"""Light text normalization for OCR output."""

from __future__ import annotations

import re
import unicodedata


def normalize_text(text: str) -> str:
    """Unicode NFC + collapse runaway whitespace. Keep Bangla and common punctuation."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    # Normalize newlines first, then collapse horizontal runs
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def page_marker(page_number: int) -> str:
    return f"\n\n--- page {page_number} ---\n\n"


def join_pages(pages: list[tuple[int, str]]) -> str:
    """Join (page_number, text) into a single UTF-8 document with page markers."""
    chunks: list[str] = []
    for page_number, text in pages:
        cleaned = normalize_text(text)
        if not cleaned:
            continue
        chunks.append(f"--- page {page_number} ---")
        chunks.append(cleaned)
    return "\n\n".join(chunks).strip() + ("\n" if chunks else "")
