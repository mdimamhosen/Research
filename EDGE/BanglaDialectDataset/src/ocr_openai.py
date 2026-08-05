"""OpenAI GPT-4o vision OCR."""

from __future__ import annotations

import os

from openai import OpenAI
from PIL import Image

from src.ocr_common import OCR_SYSTEM_PROMPT, OCR_USER_PROMPT, image_to_data_url


def ocr_page_openai(
    image: Image.Image,
    *,
    api_key: str | None = None,
    model: str = "gpt-4o",
) -> str:
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to .env or pass api_key=..."
        )

    client = OpenAI(api_key=key)
    data_url = image_to_data_url(image)

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": OCR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": OCR_USER_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                ],
            },
        ],
    )
    content = response.choices[0].message.content or ""
    return content.strip()
