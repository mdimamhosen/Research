"""Streamlit UI for Bengali book PDF OCR — dark editorial workspace."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.ocr_tesseract import tesseract_status
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

ENGINE_HELP = {
    "tesseract": "Open source · free · local",
    "paddle": "Open source · free · local (pip install -r requirements-paddle.txt)",
    "gemini": "Cloud VLM · needs GEMINI_API_KEY",
    "openai": "Cloud VLM · needs OPENAI_API_KEY",
    "claude": "Cloud VLM · needs ANTHROPIC_API_KEY",
}


def _inject_styles() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Syne:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
  --bg: #0A0C0B;
  --bg-elev: #121514;
  --panel: #171B19;
  --line: #2A322E;
  --text: #E8EEE9;
  --muted: #8B9A92;
  --accent: #3DDC97;
  --accent-dim: #1F8F62;
  --danger: #E07A5F;
  --ok: #3DDC97;
}

html, body, [class*="css"]  {
  font-family: "IBM Plex Sans", sans-serif;
  color: var(--text);
}

.stApp {
  background:
    radial-gradient(900px 480px at 12% -8%, rgba(61, 220, 151, 0.12) 0%, transparent 55%),
    radial-gradient(700px 420px at 95% 8%, rgba(61, 220, 151, 0.06) 0%, transparent 50%),
    linear-gradient(165deg, #0A0C0B 0%, #0E1210 48%, #0A0C0B 100%);
  color: var(--text);
}

#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { right: 1rem; }

.block-container {
  padding-top: 1.25rem !important;
  padding-bottom: 3.5rem !important;
  max-width: 1040px !important;
}

/* ——— Hero ——— */
.bdd-hero {
  animation: bdd-in 0.65s cubic-bezier(.22,1,.36,1) both;
  margin-bottom: 1.75rem;
  padding-bottom: 1.5rem;
  border-bottom: 1px solid var(--line);
}
.bdd-kicker {
  font-family: "Syne", sans-serif;
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--accent);
  margin: 0 0 0.65rem 0;
}
.bdd-brand {
  font-family: "Instrument Serif", serif;
  font-weight: 400;
  font-size: clamp(2.75rem, 6vw, 4.1rem);
  line-height: 0.98;
  letter-spacing: -0.02em;
  color: var(--text);
  margin: 0;
}
.bdd-brand em {
  font-style: italic;
  color: var(--accent);
}
.bdd-tag {
  margin: 1rem 0 0 0;
  max-width: 34rem;
  font-size: 1.05rem;
  line-height: 1.6;
  color: var(--muted);
}
.bdd-steps {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.75rem;
  margin-top: 1.5rem;
  animation: bdd-in 0.75s cubic-bezier(.22,1,.36,1) 0.08s both;
}
.bdd-step {
  padding: 0.85rem 1rem;
  background: rgba(23, 27, 25, 0.85);
  border: 1px solid var(--line);
  border-radius: 8px;
  transition: border-color 0.2s ease, transform 0.2s ease;
}
.bdd-step:hover {
  border-color: rgba(61, 220, 151, 0.45);
  transform: translateY(-2px);
}
.bdd-step b {
  display: block;
  font-family: "Syne", sans-serif;
  font-size: 0.78rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--accent);
  margin-bottom: 0.35rem;
}
.bdd-step span {
  font-size: 0.88rem;
  color: var(--muted);
  line-height: 1.4;
}

/* ——— Sections ——— */
.bdd-section {
  animation: bdd-in 0.7s cubic-bezier(.22,1,.36,1) 0.1s both;
  margin: 0.25rem 0 1.1rem 0;
}
.bdd-section h2 {
  font-family: "Instrument Serif", serif;
  font-size: 1.65rem;
  font-weight: 400;
  margin: 0 0 0.4rem 0;
  color: var(--text);
}
.bdd-section p {
  margin: 0;
  color: var(--muted);
  font-size: 0.95rem;
  line-height: 1.5;
}

/* ——— Sidebar ——— */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0E1210 0%, #0A0C0B 100%) !important;
  border-right: 1px solid var(--line) !important;
}
section[data-testid="stSidebar"] .block-container {
  padding-top: 1.75rem !important;
}
.bdd-side-title {
  font-family: "Instrument Serif", serif;
  font-size: 1.45rem;
  color: var(--text);
  margin: 0 0 0.35rem 0;
}
.bdd-side-note {
  font-size: 0.82rem;
  color: var(--muted);
  line-height: 1.5;
  margin: 0 0 1.15rem 0;
}
.bdd-status {
  margin: 0.75rem 0 0.5rem 0;
  padding: 0.7rem 0.85rem;
  border-radius: 8px;
  border: 1px solid var(--line);
  font-size: 0.78rem;
  line-height: 1.45;
  color: var(--muted);
  background: var(--panel);
}
.bdd-status.ok {
  border-color: rgba(61, 220, 151, 0.35);
  color: #B8F0D4;
}
.bdd-status.bad {
  border-color: rgba(224, 122, 95, 0.45);
  color: #F0C4B8;
}
.bdd-status strong {
  display: block;
  font-family: "Syne", sans-serif;
  font-size: 0.7rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  margin-bottom: 0.25rem;
  color: inherit;
}

/* ——— Controls ——— */
.stButton > button {
  font-family: "Syne", sans-serif !important;
  font-weight: 600 !important;
  letter-spacing: 0.02em !important;
  border-radius: 8px !important;
  min-height: 2.85rem !important;
  transition: transform 0.15s ease, background 0.15s ease, box-shadow 0.15s ease !important;
}
.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {
  background: var(--accent) !important;
  color: #06140F !important;
  border: none !important;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="baseButton-primary"]:hover {
  background: #6AEBB3 !important;
  transform: translateY(-1px);
  box-shadow: 0 8px 24px rgba(61, 220, 151, 0.18) !important;
}
.stButton > button:disabled {
  opacity: 0.45 !important;
}

.stTabs [data-baseweb="tab-list"] {
  gap: 0.4rem;
  border-bottom: 1px solid var(--line);
  background: transparent;
}
.stTabs [data-baseweb="tab"] {
  font-family: "Syne", sans-serif;
  font-weight: 600;
  font-size: 0.85rem;
  color: var(--muted) !important;
  padding: 0.65rem 1.05rem;
  border-radius: 8px 8px 0 0;
  background: transparent !important;
}
.stTabs [aria-selected="true"] {
  color: var(--accent) !important;
  background: rgba(61, 220, 151, 0.08) !important;
}

div[data-testid="stFileUploader"] section {
  background: var(--panel) !important;
  border: 1px dashed var(--line) !important;
  border-radius: 10px !important;
  padding: 1rem !important;
  transition: border-color 0.2s ease;
}
div[data-testid="stFileUploader"] section:hover {
  border-color: rgba(61, 220, 151, 0.5) !important;
}

textarea, .stTextInput input, .stNumberInput input {
  background: var(--bg-elev) !important;
  color: var(--text) !important;
  border-radius: 8px !important;
  border-color: var(--line) !important;
}

div[data-testid="stExpander"] {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 10px;
}

/* Result */
.bdd-result {
  animation: bdd-in 0.45s ease both;
  margin: 1.1rem 0 0.75rem 0;
  padding: 1rem 1.2rem;
  background: var(--panel);
  border: 1px solid var(--line);
  border-left: 3px solid var(--accent);
  border-radius: 0 10px 10px 0;
}
.bdd-result h3 {
  font-family: "Instrument Serif", serif;
  font-size: 1.25rem;
  margin: 0 0 0.3rem 0;
  color: var(--text);
}
.bdd-result p {
  margin: 0;
  font-size: 0.86rem;
  color: var(--muted);
}
.bdd-result code {
  color: var(--accent);
  background: rgba(61, 220, 151, 0.1);
  padding: 0.12rem 0.4rem;
  border-radius: 4px;
  font-size: 0.8rem;
}

.bdd-done {
  animation: bdd-in 0.4s ease both;
  margin: 0.5rem 0 1rem 0;
  padding: 0.75rem 1rem;
  border-radius: 8px;
  background: rgba(61, 220, 151, 0.1);
  border: 1px solid rgba(61, 220, 151, 0.28);
  color: #B8F0D4;
  font-size: 0.9rem;
}

@keyframes bdd-in {
  from { opacity: 0; transform: translateY(12px); }
  to { opacity: 1; transform: translateY(0); }
}

@media (max-width: 800px) {
  .bdd-steps { grid-template-columns: 1fr; }
  .bdd-brand { font-size: 2.4rem; }
}
</style>
        """,
        unsafe_allow_html=True,
    )


def _save_upload(uploaded) -> Path:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(uploaded.name).stem
    dest = unique_path(PDF_DIR, stem, ".pdf")
    dest.write_bytes(uploaded.getbuffer())
    return dest


def _txt_path_for(pdf_path: Path, *, overwrite: bool) -> Path:
    TXT_DIR.mkdir(parents=True, exist_ok=True)
    if overwrite:
        return TXT_DIR / f"{pdf_path.stem}.txt"
    return unique_path(TXT_DIR, pdf_path.stem, ".txt")


def main() -> None:
    st.set_page_config(
        page_title="BanglaDialect · Book OCR",
        page_icon="ব",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_styles()

    st.markdown(
        """
<div class="bdd-hero">
  <p class="bdd-kicker">Dataset tooling</p>
  <p class="bdd-brand">Bangla<em>Dialect</em></p>
  <p class="bdd-tag">Scan Bengali book PDFs into faithful UTF-8 text — no translation, no rewriting. Built for a growing dialect corpus.</p>
  <div class="bdd-steps">
    <div class="bdd-step"><b>01 · Source</b><span>Upload PDFs or batch a folder of scanned books.</span></div>
    <div class="bdd-step"><b>02 · Extract</b><span>OCR each page with Tesseract or a cloud vision model.</span></div>
    <div class="bdd-step"><b>03 · Keep</b><span>Every book gets its own .txt — numbered, never silently skipped.</span></div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown('<p class="bdd-side-title">Controls</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="bdd-side-note">Tune the engine before you extract. Tesseract is free and local; cloud engines need API keys.</p>',
            unsafe_allow_html=True,
        )

        default_engine = resolve_default_engine()
        engine = st.selectbox(
            "OCR engine",
            options=list(ENGINES),
            index=list(ENGINES).index(default_engine),
            format_func=lambda e: f"{e}  ·  {ENGINE_HELP.get(e, '')}",
        )
        dpi = st.select_slider(
            "Page sharpness",
            options=[150, 200, 250, 300, 350, 400],
            value=300,
            help="DPI when rendering PDF pages. 300 is the sweet spot for most books.",
        )
        overwrite = st.toggle(
            "Overwrite same filename",
            value=False,
            help="Off (recommended): if book.txt exists, write book_2.txt instead.",
        )

        ok, status_msg = tesseract_status()
        cls = "ok" if ok else "bad"
        label = "Engine status" if ok else "Action needed"
        st.markdown(
            f'<div class="bdd-status {cls}"><strong>{label}</strong>{status_msg}</div>',
            unsafe_allow_html=True,
        )
        st.caption("Library → data/pdfs/")
        st.caption("Output → data/txt/")

    tab_upload, tab_folder = st.tabs(["Upload books", "Library folder"])

    with tab_upload:
        st.markdown(
            """
<div class="bdd-section">
  <h2>Upload books</h2>
  <p>Drop one or more scanned PDFs. Each file is stored uniquely and produces its own UTF-8 text.</p>
</div>
            """,
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            help="You can select multiple books at once.",
        )

        n_files = len(uploaded) if uploaded else 0
        if n_files:
            st.caption(f"{n_files} file(s) ready · engine `{engine}` · {dpi} DPI")

        with st.expander("Optional · limit page range for a quick test", expanded=False):
            limit_pages = st.checkbox("Only process some pages", value=False)
            page_start: int | None = None
            page_end: int | None = None
            if limit_pages:
                c1, c2 = st.columns(2)
                with c1:
                    page_start = int(
                        st.number_input("From page", min_value=1, value=1, step=1)
                    )
                with c2:
                    page_end = int(
                        st.number_input("To page", min_value=1, value=3, step=1)
                    )
            else:
                page_start = page_end = None

        run_upload = st.button(
            "Extract text from uploads",
            type="primary",
            disabled=not uploaded,
            use_container_width=True,
        )
        if run_upload and uploaded:
            st.markdown(
                '<div class="bdd-section"><h2>Results</h2><p>Live progress, then preview and download.</p></div>',
                unsafe_allow_html=True,
            )
            for file in uploaded:
                pdf_path = _save_upload(file)
                out_path = _txt_path_for(pdf_path, overwrite=overwrite)
                _run_one(
                    pdf_path,
                    out_path,
                    engine=engine,
                    dpi=int(dpi),
                    page_start=page_start,
                    page_end=page_end,
                )

    with tab_folder:
        st.markdown(
            """
<div class="bdd-section">
  <h2>Library folder</h2>
  <p>Batch every PDF already in <code>data/pdfs/</code>. Ideal once your corpus grows.</p>
</div>
            """,
            unsafe_allow_html=True,
        )
        PDF_DIR.mkdir(parents=True, exist_ok=True)
        try:
            existing = discover_pdfs(PDF_DIR)
        except FileNotFoundError:
            existing = []
        st.caption(f"{len(existing)} PDF(s) in library · `{PDF_DIR}`")

        if st.button(
            "Extract all books in folder",
            type="primary",
            use_container_width=True,
            disabled=not existing,
        ):
            for pdf_path in existing:
                out_path = _txt_path_for(pdf_path, overwrite=overwrite)
                _run_one(
                    pdf_path,
                    out_path,
                    engine=engine,
                    dpi=int(dpi),
                    page_start=None,
                    page_end=None,
                )
        elif not existing:
            st.warning("Folder is empty — upload books first, or copy PDFs into data/pdfs/.")


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

    st.markdown(
        f"""
<div class="bdd-result">
  <h3>{pdf_path.name}</h3>
  <p>{n_pages} pages · <code>{engine}</code> · {dpi} DPI → <code>{out_path.name}</code></p>
</div>
        """,
        unsafe_allow_html=True,
    )
    progress = st.progress(0.0, text="Preparing pages…")
    status = st.empty()

    def on_progress(page_number: int, idx: int, total: int) -> None:
        progress.progress(
            idx / total, text=f"Reading page {page_number} · {idx} / {total}"
        )
        status.caption(f"OCR in progress · page {page_number}")

    text = ""
    tmp_path: Path | None = None
    try:
        # Write temp file in the SAME directory as the output.
        # Path.replace()/rename fails on Streamlit Cloud (Errno 18) when
        # moving from /tmp to /mount/src (cross-device).
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.parent / f".{out_path.stem}.partial.txt"

        text = ocr_pdf_to_text(
            pdf_path,
            engine=engine,
            dpi=dpi,
            page_start=page_start,
            page_end=page_end,
            on_progress=on_progress,
        )
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(out_path)
        tmp_path = None
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed `{pdf_path.name}`: {exc}")
        return
    finally:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        progress.progress(1.0, text="Complete")
        status.empty()

    st.markdown(
        f'<div class="bdd-done">Saved <strong>{out_path.name}</strong> · {len(text):,} characters</div>',
        unsafe_allow_html=True,
    )
    st.text_area(
        "Preview",
        text[:2400] + ("…" if len(text) > 2400 else ""),
        height=280,
        key=f"preview-{out_path.name}",
    )
    st.download_button(
        f"Download {out_path.name}",
        data=text,
        file_name=out_path.name,
        mime="text/plain",
        key=f"dl-{out_path.name}",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
