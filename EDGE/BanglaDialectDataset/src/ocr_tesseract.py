"""Local Tesseract OCR (Bengali + English) via CLI — no pytesseract needed."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
PROJECT_TESSDATA = ROOT / "tessdata"

# Ensure .env is loaded even if this module is imported first.
load_dotenv(ROOT / ".env")


def _tesseract_exe() -> str:
    """Resolve tesseract binary on Windows (local) or Linux (Streamlit Cloud)."""
    candidates: list[str] = []

    env_cmd = (os.getenv("TESSERACT_CMD") or "").strip().strip('"')
    if env_cmd:
        candidates.append(env_cmd)

    which = shutil.which("tesseract")
    if which:
        candidates.append(which)

    if sys.platform.startswith("win"):
        candidates.extend(
            [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                str(Path.home() / "AppData/Local/Programs/Tesseract-OCR/tesseract.exe"),
            ]
        )
    else:
        candidates.extend(
            [
                "/usr/bin/tesseract",
                "/usr/local/bin/tesseract",
            ]
        )

    tried: list[str] = []
    for path in candidates:
        if not path:
            continue
        tried.append(path)
        p = Path(path)
        if p.is_file():
            return str(p.resolve())
        # bare command name already checked via shutil.which
        if path == "tesseract" and which:
            return which

    # Last resort: hope PATH works at subprocess time
    if which:
        return which
    return "tesseract"


def _tessdata_dir() -> str | None:
    if PROJECT_TESSDATA.is_dir() and (PROJECT_TESSDATA / "ben.traineddata").is_file():
        return str(PROJECT_TESSDATA.resolve())
    # Streamlit Cloud / Linux apt packages
    for d in (
        Path("/usr/share/tesseract-ocr/5/tessdata"),
        Path("/usr/share/tesseract-ocr/4.00/tessdata"),
        Path("/usr/share/tessdata"),
    ):
        if d.is_dir() and (d / "ben.traineddata").is_file():
            return str(d)
    return None


def tesseract_status() -> tuple[bool, str]:
    """Return (ok, message) for UI health checks."""
    exe = _tesseract_exe()
    if exe != "tesseract" and not Path(exe).is_file():
        return False, f"Configured path missing: {exe}"
    try:
        result = subprocess.run(
            [exe, "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except FileNotFoundError:
        return (
            False,
            "Tesseract binary not found. Local: winget install UB-Mannheim.TesseractOCR. "
            "Cloud: ensure packages.txt has tesseract-ocr + tesseract-ocr-ben, then reboot.",
        )
    if result.returncode != 0:
        return False, (result.stderr or result.stdout or "tesseract --version failed").strip()
    ver = (result.stdout or result.stderr or "").splitlines()[0] if (result.stdout or result.stderr) else exe
    data = _tessdata_dir()
    ben = "ben OK" if data else "ben language data missing"
    return True, f"{ver} · {ben} · {exe}"


def ocr_page_tesseract(
    image: Image.Image,
    *,
    lang: str = "ben+eng",
) -> str:
    exe = _tesseract_exe()
    tessdata = _tessdata_dir()

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
                "Tesseract is not installed or not on PATH.\n"
                "• Local Windows: winget install UB-Mannheim.TesseractOCR\n"
                "• Then set in .env: TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe\n"
                "• Streamlit Cloud: packages.txt must list tesseract-ocr and tesseract-ocr-ben, then Reboot app\n"
                f"• Tried executable: {exe}"
            ) from exc

        if result.returncode != 0:
            err = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(
                f"Tesseract failed (lang={lang}). "
                f"tessdata={tessdata or 'default'}. Details: {err}"
            )

        out_txt = Path(str(out_base) + ".txt")
        if not out_txt.is_file():
            raise RuntimeError("Tesseract produced no output file.")
        return out_txt.read_text(encoding="utf-8", errors="replace").strip()
