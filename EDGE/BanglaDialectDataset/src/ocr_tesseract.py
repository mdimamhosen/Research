"""Local Tesseract OCR (Bengali + English) via CLI — fast CPU path."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
LOCAL_TESSDATA = ROOT / "tessdata"


def _tesseract_cmd() -> str:
    env = (os.getenv("TESSERACT_CMD") or "").strip()
    if env and Path(env).is_file():
        return env
    found = shutil.which("tesseract")
    if found:
        return found
    for candidate in (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        str(Path.home() / "AppData/Local/Programs/Tesseract-OCR/tesseract.exe"),
    ):
        if Path(candidate).is_file():
            return candidate
    raise RuntimeError(
        "Tesseract not found. Install then re-run:\n"
        "  winget install UB-Mannheim.TesseractOCR\n"
        "Or set TESSERACT_CMD in .env to the full path of tesseract.exe"
    )


def _tessdata_prefix() -> Path | None:
    """Prefer project ./tessdata if it has ben; else system default."""
    ben = LOCAL_TESSDATA / "ben.traineddata"
    if ben.is_file():
        return LOCAL_TESSDATA
    env = (os.getenv("TESSDATA_PREFIX") or "").strip()
    if env and Path(env).is_dir():
        return Path(env)
    return None


def tesseract_status() -> tuple[bool, str]:
    try:
        cmd = _tesseract_cmd()
    except RuntimeError as exc:
        return False, str(exc)
    prefix = _tessdata_prefix()
    extra = f" · tessdata={prefix}" if prefix else ""
    return True, f"Tesseract OK · {cmd}{extra}"


def ocr_page_tesseract(image: Image.Image) -> str:
    """OCR one page with Tesseract ben+eng."""
    cmd = _tesseract_cmd()
    env = os.environ.copy()
    prefix = _tessdata_prefix()
    if prefix is not None:
        env["TESSDATA_PREFIX"] = str(prefix)

    with tempfile.TemporaryDirectory(prefix="bangla_ocr_") as tmp:
        img_path = Path(tmp) / "page.png"
        image.convert("RGB").save(img_path)
        out_base = Path(tmp) / "out"
        try:
            proc = subprocess.run(
                [
                    cmd,
                    str(img_path),
                    str(out_base),
                    "-l",
                    "ben+eng",
                    "--oem",
                    "1",
                    "--psm",
                    "6",
                ],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Tesseract binary missing. winget install UB-Mannheim.TesseractOCR"
            ) from exc

        out_file = Path(str(out_base) + ".txt")
        if proc.returncode != 0 or not out_file.is_file():
            err = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(
                "Tesseract failed. Need Bengali language pack (ben).\n"
                f"Details: {err or proc.returncode}\n"
                "Install: winget install UB-Mannheim.TesseractOCR\n"
                "Or put ben.traineddata + eng.traineddata in ./tessdata/"
            )
        return out_file.read_text(encoding="utf-8", errors="replace").strip()
