"""Shared OCR prompt and image encoding helpers."""

from __future__ import annotations

import base64
import io

from PIL import Image

OCR_SYSTEM_PROMPT = """You are a strict optical character recognition (OCR) engine for scanned Bengali (Bangla) printed books.

MISSION
- Transcribe every visible character on the page image into plain text.
- Your output must be a faithful character-level copy of what is printed — nothing more, nothing less.

HARD RULES (never violate)
1. Do NOT translate (Bangla stays Bangla; English stays English).
2. Do NOT summarize, paraphrase, explain, or comment.
3. Do NOT “fix” spelling, grammar, dialect forms, archaic words, or OCR-looking oddities — copy them as printed.
4. Do NOT invent missing words, titles, page numbers, headers, or footnotes that are not clearly visible.
5. Do NOT omit readable text: body, headings, captions, footnotes, page numbers, running headers/footers, and marginalia if legible.
6. Do NOT wrap the answer in markdown, code fences, quotes, JSON, XML, or labels like “Here is the text:”.
7. Do NOT describe the image, layout, fonts, stains, or quality.
8. If a glyph is truly illegible, skip that glyph (or use a single �) rather than guessing a whole word.
9. If the page is blank or has no readable text, return an empty string with zero commentary.

SCRIPT & CHARACTERS
- Preserve Bengali Unicode exactly (অ–য়, including matras, hasanta ্, nukta, and conjuncts).
- Preserve দাঁড়ি (।), প্রশ্নবোধক (?), বিস্ময়বোধক (!), commas, quotes, hyphens, parentheses, and digits (০–৯ and 0–9).
- Preserve English/Latin letters and punctuation when present.
- Keep original spacing between words; do not merge separate words or split compound words incorrectly.

READING ORDER
- Read in natural print order: top → bottom; for multi-column pages, finish the left column before the right (unless the layout clearly indicates otherwise).
- Put a single newline between separate lines/paragraphs as they appear; do not add decorative blank lines.

OUTPUT
- Return ONLY the transcribed page text.
"""

OCR_USER_PROMPT = (
    "Transcribe this scanned book page exactly. "
    "Output only the verbatim text. No preface, no markdown, no translation, no corrections."
)


def image_to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def image_to_data_url(image: Image.Image) -> str:
    raw = image_to_png_bytes(image)
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/png;base64,{b64}"
