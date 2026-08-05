"""Local Tesseract OCR (Bengali + English) via CLI — no pytesseract/pandas needed."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
PROJECT_TESSDATA = ROOT / "tessdata"


def _tesseract_exe() -> str:
    env_cmd = os.getenv("TESSERACT_CMD")
    candidates = [
        env_cmd,
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        "tesseract",
    ]
    for path in candidates:
        if not path:
            continue
        if path == "tesseract":
            return path
        if Path(path).is_file():
            return path
    return "tesseract"


def ocr_page_tesseract(
    image: Image.Image,
    *,
    lang: str = "ben+eng",
) -> str:
    exe = _tesseract_exe()
    tessdata = None
    if PROJECT_TESSDATA.is_dir() and (PROJECT_TESSDATA / "ben.traineddata").is_file():
        tessdata = str(PROJECT_TESSDATA)

    with tempfile.TemporaryDirectory(prefix="bangla_ocr_") as tmp:
        img_path = Path(tmp) / "page.png"
        out_base = Path(tmp) / "out"
        image.save(img_path, format="PNG")

        cmd = [exe, str(img_path), str(out_base), "-l", lang]
        if tessdata:
            cmd.extend(["--tessdata-dir", tessdata])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Tesseract is not installed or not on PATH. "
                "Install: winget install UB-Mannheim.TesseractOCR "
                "Or set TESSERACT_CMD in .env to tesseract.exe"
            ) from exc

        if result.returncode != 0:
            err = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(
                f"Tesseract failed (lang={lang}). "
                f"Is ben.traineddata in tessdata/? Details: {err}"
            )

        out_txt = Path(str(out_base) + ".txt")
        if not out_txt.is_file():
            raise RuntimeError("Tesseract produced no output file.")
        return out_txt.read_text(encoding="utf-8", errors="replace").strip()
