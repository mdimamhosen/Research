"""Google Gemini vision OCR."""

from __future__ import annotations

import os

from google import genai
from google.genai import types
from PIL import Image

from src.ocr_common import OCR_SYSTEM_PROMPT, OCR_USER_PROMPT


def ocr_page_gemini(
    image: Image.Image,
    *,
    api_key: str | None = None,
    model: str = "gemini-2.0-flash",
) -> str:
    key = (
        api_key
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to .env or pass api_key=..."
        )

    client = genai.Client(api_key=key)
    response = client.models.generate_content(
        model=model,
        contents=[OCR_USER_PROMPT, image],
        config=types.GenerateContentConfig(
            system_instruction=OCR_SYSTEM_PROMPT,
            temperature=0.0,
        ),
    )
    text = getattr(response, "text", None) or ""
    return text.strip()
