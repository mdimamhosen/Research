"""Streamlit UI for Bengali book PDF OCR."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Streamlit Cloud runs from the repo root; ensure this package dir is importable.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.pipeline import (
    ENGINES,
    count_pages,
    discover_pdfs,
    ocr_pdf_to_text,
    resolve_default_engine,
    unique_path,
)

PDF_DIR = ROOT / "data" / "pdfs"
TXT_DIR = ROOT / "data" / "txt"


def _save_upload(uploaded) -> Path:
    """Save upload under a unique name so books never overwrite each other."""
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(uploaded.name).stem
    dest = unique_path(PDF_DIR, stem, ".pdf")
    dest.write_bytes(uploaded.getbuffer())
    return dest


def _txt_path_for(pdf_path: Path, *, overwrite: bool) -> Path:
    """One PDF → one .txt; if name taken, create book_2.txt unless overwrite."""
    TXT_DIR.mkdir(parents=True, exist_ok=True)
    if overwrite:
        return TXT_DIR / f"{pdf_path.stem}.txt"
    return unique_path(TXT_DIR, pdf_path.stem, ".txt")


def main() -> None:
    st.set_page_config(page_title="Bangla Book OCR", page_icon="📄", layout="wide")
    st.title("Bangla Book OCR")
    st.caption(
        "Scan Bengali book PDFs → UTF-8 `.txt` via Gemini, OpenAI, Claude, or Tesseract."
    )

    with st.sidebar:
        st.header("Settings")
        default_engine = resolve_default_engine()
        engine = st.selectbox(
            "OCR engine",
            options=list(ENGINES),
            index=list(ENGINES).index(default_engine),
        )
        dpi = st.slider("Render DPI", min_value=150, max_value=400, value=300, step=50)
        overwrite = st.checkbox(
            "Overwrite same-name .txt",
            value=False,
            help="Off (default): if book.txt exists, write book_2.txt, book_3.txt, …",
        )
        st.markdown("---")
        st.markdown(
            "Free engine: **tesseract** (no API key). "
            "Each upload gets its **own** `.txt` (numbered if the name already exists)."
        )

    tab_upload, tab_folder = st.tabs(["Upload PDF", "Process folder"])

    with tab_upload:
        uploaded = st.file_uploader("PDF file", type=["pdf"], accept_multiple_files=True)
        limit_pages = st.checkbox("Limit page range", value=False)
        page_start: int | None = None
        page_end: int | None = None
        if limit_pages:
            col1, col2 = st.columns(2)
            with col1:
                page_start = int(
                    st.number_input("Page start", min_value=1, value=1, step=1)
                )
            with col2:
                page_end = int(
                    st.number_input("Page end", min_value=1, value=1, step=1)
                )

        if st.button("Run OCR on upload", type="primary", disabled=not uploaded):
            for file in uploaded or []:
                pdf_path = _save_upload(file)
                out_path = _txt_path_for(pdf_path, overwrite=overwrite)
                _run_one(
                    pdf_path,
                    out_path,
                    engine=engine,
                    dpi=dpi,
                    page_start=page_start,
                    page_end=page_end,
                )

    with tab_folder:
        st.write(f"Looks for PDFs in `{PDF_DIR}`")
        st.caption(
            "Processes each PDF. If `book.txt` already exists, creates `book_2.txt` "
            "(unless overwrite is on)."
        )
        if st.button("Run OCR on folder", type="primary"):
            PDF_DIR.mkdir(parents=True, exist_ok=True)
            try:
                pdfs = discover_pdfs(PDF_DIR)
            except FileNotFoundError:
                st.error(f"Folder not found: {PDF_DIR}")
                return
            if not pdfs:
                st.warning("No PDFs in data/pdfs/. Drop files there or use Upload.")
                return
            for pdf_path in pdfs:
                out_path = _txt_path_for(pdf_path, overwrite=overwrite)
                _run_one(
                    pdf_path,
                    out_path,
                    engine=engine,
                    dpi=dpi,
                    page_start=None,
                    page_end=None,
                )


def _run_one(
    pdf_path: Path,
    out_path: Path,
    *,
    engine: str,
    dpi: int,
    page_start: int | None,
    page_end: int | None,
) -> None:
    try:
        n_pages = count_pages(pdf_path)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Cannot open {pdf_path.name}: {exc}")
        return

    st.write(
        f"**{pdf_path.name}** — {n_pages} page(s), engine=`{engine}` → `{out_path.name}`"
    )
    progress = st.progress(0.0, text="Starting…")
    status = st.empty()

    def on_progress(page_number: int, idx: int, total: int) -> None:
        progress.progress(idx / total, text=f"Page {page_number} ({idx}/{total})")
        status.caption(f"OCR page {page_number}…")

    text = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".txt", delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)

        text = ocr_pdf_to_text(
            pdf_path,
            engine=engine,
            dpi=dpi,
            page_start=page_start,
            page_end=page_end,
            on_progress=on_progress,
        )
        tmp_path.write_text(text, encoding="utf-8")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.replace(out_path)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed `{pdf_path.name}`: {exc}")
        return
    finally:
        progress.progress(1.0, text="Done")

    preview = text[:2000] + ("…" if len(text) > 2000 else "")
    st.success(f"Wrote `{out_path}` ({len(text):,} chars)")
    st.text_area("Preview", preview, height=240, key=f"preview-{out_path.name}")
    st.download_button(
        f"Download {out_path.name}",
        data=text,
        file_name=out_path.name,
        mime="text/plain",
        key=f"dl-{out_path.name}",
    )


if __name__ == "__main__":
    main()
