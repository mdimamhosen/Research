"""DeepSeek-OCR open VLM backend (optional, GPU recommended)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

_model: Any = None
_tokenizer: Any = None

PROMPT_FREE_OCR = "<image>\nFree OCR."


def _load() -> tuple[Any, Any]:
    global _model, _tokenizer
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "DeepSeek-OCR needs torch + transformers.\n"
            "  pip install -r requirements-deepseek.txt\n"
            "A CUDA GPU with ~8GB+ VRAM is strongly recommended."
        ) from exc

    model_name = "deepseek-ai/DeepSeek-OCR"
    try:
        _tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        # Prefer GPU bf16; fall back to CPU (very slow).
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "use_safetensors": True,
        }
        # flash_attn only on CUDA builds that have it
        if device == "cuda":
            try:
                kwargs["_attn_implementation"] = "flash_attention_2"
                _model = AutoModel.from_pretrained(model_name, **kwargs)
            except Exception:
                kwargs.pop("_attn_implementation", None)
                _model = AutoModel.from_pretrained(model_name, **kwargs)
        else:
            _model = AutoModel.from_pretrained(model_name, **kwargs)

        _model = _model.eval()
        if device == "cuda":
            _model = _model.cuda().to(dtype)
        else:
            _model = _model.to(dtype)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load DeepSeek-OCR ({model_name}). "
            "Check GPU/drivers and: pip install -r requirements-deepseek.txt\n"
            f"Details: {exc}"
        ) from exc

    return _model, _tokenizer


def ocr_page_deepseek(image: Image.Image) -> str:
    """Run DeepSeek-OCR Free OCR prompt on one page image."""
    model, tokenizer = _load()

    with tempfile.TemporaryDirectory(prefix="deepseek_ocr_") as tmp:
        img_path = Path(tmp) / "page.jpg"
        out_dir = Path(tmp) / "out"
        out_dir.mkdir()
        image.convert("RGB").save(img_path, quality=95)

        # Official API: model.infer(...); return type varies by revision.
        res = model.infer(
            tokenizer,
            prompt=PROMPT_FREE_OCR,
            image_file=str(img_path),
            output_path=str(out_dir),
            base_size=1024,
            image_size=640,
            crop_mode=True,
            save_results=False,
        )

    if res is None:
        return ""
    if isinstance(res, str):
        return res.strip()
    if isinstance(res, dict):
        for key in ("text", "ocr", "result", "output"):
            if key in res and res[key]:
                return str(res[key]).strip()
    return str(res).strip()
