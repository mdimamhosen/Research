"""Anthropic Claude vision OCR."""

from __future__ import annotations

import base64
import os

from anthropic import Anthropic
from PIL import Image

from src.ocr_common import OCR_SYSTEM_PROMPT, OCR_USER_PROMPT, image_to_png_bytes


def ocr_page_claude(
    image: Image.Image,
    *,
    api_key: str | None = None,
    model: str = "claude-sonnet-4-5",
) -> str:
    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to .env or pass api_key=..."
        )

    client = Anthropic(api_key=key)
    png_b64 = base64.b64encode(image_to_png_bytes(image)).decode("ascii")

    message = client.messages.create(
        model=model,
        max_tokens=4096,
        temperature=0,
        system=OCR_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": png_b64,
                        },
                    },
                    {"type": "text", "text": OCR_USER_PROMPT},
                ],
            }
        ],
    )

    parts: list[str] = []
    for block in message.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "\n".join(parts).strip()
