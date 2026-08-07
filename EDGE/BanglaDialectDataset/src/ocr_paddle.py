"""PaddleOCR open-source backend (optional dependency).

Default path for Bengali books: **PaddleOCR-VL** (~0.9B VLM, 109 languages
including Bengali). Classic PP-OCR multilingual models do not ship Bengali
recognition in current PP-OCRv5 builds.

Env:
  PADDLE_OCR_BACKEND     vl | ppocr | auto   (default: vl)
  PADDLE_OCR_DEVICE      cpu | gpu:0 | ...   (optional)
  PADDLE_OCR_LANG        lang for classic PP-OCR only
  PADDLE_OCR_FAST        1 = skip layout + shrink large pages (much faster on CPU)
  PADDLE_OCR_USE_LAYOUT  1 = enable layout detection (slower, default off)
  PADDLE_OCR_MAX_SIDE    max image side in px when fast (default 1280)
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_engine: Any = None
_backend_in_use: str | None = None

_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MD_HEADING = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_MD_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_MD_ITALIC = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")


def _device() -> str | None:
    raw = (os.getenv("PADDLE_OCR_DEVICE") or "").strip()
    return raw or None


def _wanted_backend() -> str:
    raw = (os.getenv("PADDLE_OCR_BACKEND") or "vl").strip().lower()
    if raw in ("vl", "ppocr", "auto"):
        return raw
    return "vl"


def _fast_mode() -> bool:
    return (os.getenv("PADDLE_OCR_FAST") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _use_layout() -> bool:
    """Layout detection is expensive; off by default for plain book pages."""
    raw = (os.getenv("PADDLE_OCR_USE_LAYOUT") or "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    # Fast mode always disables layout.
    return False


def _max_side() -> int:
    raw = (os.getenv("PADDLE_OCR_MAX_SIDE") or "").strip()
    if raw.isdigit():
        return max(640, int(raw))
    return 1280 if _fast_mode() else 0  # 0 = no shrink


def _shrink_for_speed(image: Image.Image) -> Image.Image:
    limit = _max_side()
    if limit <= 0:
        return image
    w, h = image.size
    m = max(w, h)
    if m <= limit:
        return image
    scale = limit / m
    return image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BILINEAR)


def _install_hint() -> str:
    return (
        "PaddleOCR / PaddleOCR-VL Python package is missing (paddlepaddle alone is not enough).\n"
        "Install BOTH:\n"
        "  # Kaggle / GPU:\n"
        "  pip install -q paddlepaddle-gpu==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/\n"
        "  pip install -q 'paddleocr>=3.0.0' 'paddlex[ocr]>=3.7.0' pymupdf Pillow python-dotenv tqdm\n"
        "  # Local CPU:\n"
        "  pip install paddlepaddle==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/\n"
        "  pip install -r requirements.txt\n"
        "Then:\n"
        "  python cli.py -i /kaggle/input/datasets/... -o /kaggle/working/txt -r --engine paddle\n"
        "  (no need to copy PDFs one-by-one — point -i at the folder)\n"
        "Optional: PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True"
    )


def _init_vl() -> Any:
    try:
        from paddleocr import PaddleOCRVL
    except ImportError as exc:
        raise RuntimeError(_install_hint()) from exc

    kwargs: dict[str, Any] = {
        # Speed: skip orientation / unwarp / layout for typical scanned books.
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_layout_detection": _use_layout(),
    }
    device = _device()
    if device:
        kwargs["device"] = device

    try:
        return PaddleOCRVL(**kwargs)
    except TypeError:
        # Older builds may not accept some flags — peel them off.
        for key in (
            "use_layout_detection",
            "use_doc_orientation_classify",
            "use_doc_unwarping",
        ):
            kwargs.pop(key, None)
        try:
            return PaddleOCRVL(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                **({"device": device} if device else {}),
            )
        except TypeError:
            return PaddleOCRVL(**({"device": device} if device else {}))


def _init_ppocr() -> Any:
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise RuntimeError(_install_hint()) from exc

    lang_env = (os.getenv("PADDLE_OCR_LANG") or "").strip()
    langs = [lang_env] if lang_env else ["bn", "bengali", "en"]
    device = _device()
    last_err: Exception | None = None

    for lang in langs:
        # PaddleOCR 3.x style kwargs first, then 2.x.
        attempts: list[dict[str, Any]] = [
            {
                "lang": lang,
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": False,
            },
            {"lang": lang, "show_log": False, "use_angle_cls": True},
            {"lang": lang, "use_angle_cls": True},
            {"lang": lang},
        ]
        for base in attempts:
            kwargs = dict(base)
            if device:
                kwargs["device"] = device
            try:
                return PaddleOCR(**kwargs)
            except TypeError as exc:
                last_err = exc
                continue
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                break  # try next lang
    detail = f" Last error: {last_err}" if last_err else ""
    raise RuntimeError(
        "Could not initialize classic PaddleOCR "
        f"(tried langs={langs}). Classic PP-OCRv5 has no Bengali model; "
        "set PADDLE_OCR_BACKEND=vl (default) for Bengali."
        f"{detail}"
    )


def _get_engine() -> tuple[Any, str]:
    global _engine, _backend_in_use
    if _engine is not None and _backend_in_use is not None:
        return _engine, _backend_in_use

    wanted = _wanted_backend()
    errors: list[str] = []

    if wanted in ("vl", "auto"):
        try:
            _engine = _init_vl()
            _backend_in_use = "vl"
            return _engine, _backend_in_use
        except Exception as exc:  # noqa: BLE001
            errors.append(f"vl: {exc}")
            if wanted == "vl":
                raise

    if wanted in ("ppocr", "auto"):
        try:
            _engine = _init_ppocr()
            _backend_in_use = "ppocr"
            return _engine, _backend_in_use
        except Exception as exc:  # noqa: BLE001
            errors.append(f"ppocr: {exc}")
            if wanted == "ppocr":
                raise

    raise RuntimeError(
        "PaddleOCR backend failed to start. Tried: "
        + " | ".join(errors)
        + "\n"
        + _install_hint()
    )


def _as_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    for attr in ("json", "res"):
        val = getattr(obj, attr, None)
        if callable(val):
            try:
                val = val()
            except TypeError:
                pass
        if isinstance(val, dict):
            return val
    return {}


def _markdown_to_plain(text: str) -> str:
    text = _MD_IMAGE.sub("", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_HEADING.sub("", text)
    text = _MD_BOLD.sub(r"\1", text)
    text = _MD_ITALIC.sub(r"\1", text)
    text = text.replace("```", "")
    lines = [ln.rstrip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln.strip()).strip()


def _lines_from_vl_result(result: Any) -> list[str]:
    """Extract plain text from PaddleOCR-VL predict() output."""
    pages = result if isinstance(result, list) else [result]
    chunks: list[str] = []
    for page in pages:
        if page is None:
            continue
        md_attr = getattr(page, "markdown", None)
        if isinstance(md_attr, dict):
            texts = md_attr.get("markdown_texts") or md_attr.get("text") or ""
            if isinstance(texts, list):
                joined = "\n".join(str(t) for t in texts if str(t).strip())
            else:
                joined = str(texts)
            plain = _markdown_to_plain(joined)
            if plain:
                chunks.append(plain)
                continue

        data = _as_dict(page)
        md = data.get("markdown")
        if isinstance(md, dict):
            texts = md.get("markdown_texts") or md.get("text") or ""
            if isinstance(texts, list):
                joined = "\n".join(str(t) for t in texts if str(t).strip())
            else:
                joined = str(texts)
            plain = _markdown_to_plain(joined)
            if plain:
                chunks.append(plain)
                continue

        # Fallback: classic-style recognition lists nested in VL json.
        for key in ("rec_texts", "texts"):
            vals = data.get(key)
            if isinstance(vals, (list, tuple)) and vals:
                chunks.extend(str(t).strip() for t in vals if str(t).strip())
                break
    return chunks


def _lines_from_ppocr_result(result: Any) -> list[str]:
    """Extract lines from classic PaddleOCR 2.x / 3.x outputs."""
    lines: list[str] = []
    if not result:
        return lines

    pages = result if isinstance(result, list) else [result]
    for page in pages:
        if page is None:
            continue
        data = _as_dict(page)
        if data:
            texts = data.get("rec_texts") or data.get("texts") or []
            if texts:
                lines.extend(str(t).strip() for t in texts if str(t).strip())
                continue
            # Sometimes nested under res
            nested = data.get("res")
            if isinstance(nested, dict):
                texts = nested.get("rec_texts") or nested.get("texts") or []
                if texts:
                    lines.extend(str(t).strip() for t in texts if str(t).strip())
                    continue

        if isinstance(page, dict):
            texts = page.get("rec_texts") or page.get("texts") or []
            lines.extend(str(t).strip() for t in texts if str(t).strip())
            continue

        # PaddleOCR 2.x: list of [box, (text, conf)]
        try:
            iterable = iter(page)
        except TypeError:
            continue
        for item in iterable:
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


def _run_predict(engine: Any, image: Image.Image, arr: np.ndarray) -> Any:
    """Call predict() / ocr() with numpy or temp-file fallback."""
    if hasattr(engine, "predict"):
        try:
            return engine.predict(arr)
        except Exception:
            with tempfile.TemporaryDirectory(prefix="paddle_ocr_") as tmp:
                path = Path(tmp) / "page.png"
                image.save(path)
                return engine.predict(str(path))

    # Classic 2.x
    try:
        return engine.ocr(arr, cls=True)
    except TypeError:
        try:
            return engine.ocr(arr)
        except Exception:
            with tempfile.TemporaryDirectory(prefix="paddle_ocr_") as tmp:
                path = Path(tmp) / "page.png"
                image.save(path)
                try:
                    return engine.ocr(str(path), cls=True)
                except TypeError:
                    return engine.ocr(str(path))


def ocr_page_paddle(image: Image.Image) -> str:
    """Run PaddleOCR-VL (default) or classic PP-OCR on one page image."""
    engine, backend = _get_engine()
    rgb = _shrink_for_speed(image.convert("RGB"))
    arr = np.asarray(rgb)
    result = _run_predict(engine, rgb, arr)

    if backend == "vl":
        lines = _lines_from_vl_result(result)
    else:
        lines = _lines_from_ppocr_result(result)

    return "\n".join(lines).strip()


def paddle_status() -> dict[str, Any]:
    """Lightweight status for UI / debugging (does not load models)."""
    info: dict[str, Any] = {
        "wanted_backend": _wanted_backend(),
        "device": _device() or "default",
        "fast": _fast_mode(),
        "use_layout": _use_layout(),
        "import_ok": False,
        "has_vl": False,
        "has_ppocr": False,
        "active_backend": _backend_in_use,
    }
    try:
        import paddleocr  # noqa: F401

        info["import_ok"] = True
        info["has_vl"] = hasattr(paddleocr, "PaddleOCRVL")
        info["has_ppocr"] = hasattr(paddleocr, "PaddleOCR")
    except ImportError:
        pass
    return info
