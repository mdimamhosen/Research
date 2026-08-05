"""PaddleOCR open-source Bengali OCR (optional dependency)."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

_ocr: Any = None


def _get_ocr() -> Any:
    global _ocr
    if _ocr is not None:
        return _ocr
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise RuntimeError(
            "PaddleOCR is not installed. Install with:\n"
            "  pip install paddlepaddle paddleocr\n"
            "CPU: pip install paddlepaddle -i https://www.paddlepaddle.org.cn/packages/stable/cpu/"
        ) from exc

    # Prefer Bengali model; fall back to English if unavailable in this build.
    for lang in ("bengali", "en"):
        try:
            _ocr = PaddleOCR(lang=lang, show_log=False)
            return _ocr
        except TypeError:
            # Newer PaddleOCR dropped show_log
            try:
                _ocr = PaddleOCR(lang=lang)
                return _ocr
            except Exception:
                continue
        except Exception:
            continue
    raise RuntimeError("Could not initialize PaddleOCR (tried lang=bengali, en).")


def _lines_from_result(result: Any) -> list[str]:
    lines: list[str] = []
    if not result:
        return lines
    # PaddleOCR 2.x: list of pages → list of [box, (text, conf)]
    page = result[0] if isinstance(result, list) and result else result
    if page is None:
        return lines
    if isinstance(page, dict):
        # PaddleOCR 3.x style dict output
        texts = page.get("rec_texts") or page.get("texts") or []
        return [str(t).strip() for t in texts if str(t).strip()]
    for item in page:
        if not item:
            continue
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            payload = item[1]
            if isinstance(payload, (list, tuple)) and payload:
                text = str(payload[0]).strip()
            else:
                text = str(payload).strip()
            if text:
                lines.append(text)
    return lines


def ocr_page_paddle(image: Image.Image) -> str:
    ocr = _get_ocr()
    arr = np.asarray(image.convert("RGB"))
    try:
        result = ocr.ocr(arr, cls=True)
    except TypeError:
        result = ocr.ocr(arr)
    except Exception:
        # Some builds want a file path only
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "page.png"
            image.save(path)
            try:
                result = ocr.ocr(str(path), cls=True)
            except TypeError:
                result = ocr.ocr(str(path))
    return "\n".join(_lines_from_result(result)).strip()
