# How BanglaDialectDataset OCR Was Built

A plain-language walkthrough of the tool under `EDGE/BanglaDialectDataset`: what it does, why each piece exists, and how data flows from a scanned PDF to a UTF-8 `.txt` file.

## 1. Goal

We have **scanned Bengali book PDFs** (images of pages, not selectable text). For the dataset paper we need:

- Extracted text in **UTF-8 `.txt`**
- Faithful transcription (no translation, no “cleanup” that changes dialect/spelling)
- A tool a software engineer can run (Streamlit UI + CLI), without training an ML model

OCR is done by a **vision-language model** (OpenAI GPT-4o by default; Claude optional) or local **Tesseract** as a free offline fallback.

## 2. High-level pipeline

```text
PDF file
  → PyMuPDF renders each page to a PNG image (default 300 DPI)
  → OCR engine reads the image (OpenAI / Claude / Tesseract)
  → Light Unicode NFC + whitespace normalize
  → Join pages with "--- page N ---" markers
  → Write book_name.txt (UTF-8)
```

Nothing here trains a model. We only call APIs (or Tesseract) page by page.

## 3. Folder map

```text
BanglaDialectDataset/
  app.py                 # Streamlit UI entrypoint
  cli.py                 # Batch command-line entrypoint
  requirements.txt       # Python packages for local + Streamlit Cloud
  .env / .env.example    # API keys (never commit .env)
  .gitignore             # Ignores .env, data/pdfs, data/txt, venv
  README.md              # Short usage
  GUIDE.md               # This document
  src/
    pdf_pages.py         # PDF → page images
    ocr_common.py        # Shared strict OCR prompt + PNG helpers
    ocr_openai.py        # GPT-4o vision call
    ocr_claude.py        # Claude vision call
    ocr_tesseract.py     # Local Tesseract ben+eng
    normalize.py         # NFC + whitespace; join pages
    pipeline.py          # Glue: discover PDFs, run OCR, write files
  data/
    pdfs/                # Drop scanned books here
    txt/                 # OCR outputs land here
```

## 4. Why each design choice

| Choice | Why |
|--------|-----|
| **Streamlit** | Same Python stack as OCR; fastest UI for a research lab; no React/Next needed |
| **PyMuPDF** | Renders PDF pages on Windows without installing Poppler |
| **GPT-4o / Claude** | Strong on messy scans and Bangla Unicode; better quality than classic OCR for books |
| **Tesseract fallback** | Offline / free / reproducible baseline for the paper methods section |
| **Lazy imports** | App still starts if `anthropic` isn’t installed; only the chosen engine is loaded |
| **Page markers** | Later cleaning can split by page; failures are easier to spot |
| **UTF-8 + NFC** | Matches conventions in sibling `BanglaDialectEmbedding` Bangla text pipeline |
| **`.env` for keys** | Keeps secrets out of git; Streamlit Cloud uses “Secrets” instead |

## 5. Module walkthrough

### `src/pdf_pages.py`

Opens the PDF with PyMuPDF (`fitz`), scales by `dpi/72`, and returns `(page_number, PIL.Image)` pairs. Optional `page_start` / `page_end` (1-based) for testing on a few pages.

### `src/ocr_common.py`

Holds the **strict system prompt** shared by OpenAI and Claude, plus helpers to encode a page image as PNG / data-URL. The prompt forbids translation, “fixing,” guessing, and markdown wrappers so the model behaves like an OCR engine, not a writing assistant.

### `src/ocr_openai.py` / `src/ocr_claude.py`

Send: system prompt + user prompt + page image.  
Temperature `0` for more deterministic transcription.  
Return the model’s text string only.

### `src/ocr_tesseract.py`

Calls `pytesseract.image_to_string(..., lang="ben+eng")`. Needs Tesseract installed on the machine with the Bengali language pack.

### `src/normalize.py`

- Unicode **NFC** (canonical composition — important for Bangla combining marks)
- Collapse runaway spaces/newlines
- `join_pages(...)` builds the final document with `--- page N ---` headers

### `src/pipeline.py`

Orchestrates everything:

1. Load `.env`
2. Pick default engine (`openai` if key exists, else `claude`, else `tesseract`)
3. Render pages → OCR each → join → write `.txt`
4. `discover_pdfs` finds one file or all `*.pdf` in a folder
5. Skip existing outputs unless `force=True` (resume-friendly batches)

### `app.py`

Streamlit UI:

- Sidebar: engine, DPI, overwrite
- Tab 1: upload PDFs
- Tab 2: process everything in `data/pdfs/`
- Progress bar per page; preview + download

Adds the project folder to `sys.path` so imports work when Streamlit Cloud runs from the **repo root**.

### `cli.py`

Same pipeline for scripts/CI:

```bash
.venv/Scripts/python cli.py -i data/pdfs -o data/txt --engine openai
```

## 6. Local run (correct Python)

Packages live in `.venv`. On Windows Git Bash, bare `streamlit` often still points at **Anaconda**. Always use:

```bash
cd EDGE/BanglaDialectDataset
.venv/Scripts/python -m streamlit run app.py
```

## 7. API keys

Local `.env`:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...   # optional
```

Never commit `.env`. If a key was pasted into chat, rotate it in the provider dashboard.

## 8. Streamlit Community Cloud

### Why you saw “This file does not exist”

Most of this project was still **untracked locally** — GitHub `main` did not yet contain `EDGE/BanglaDialectDataset/app.py`. Community Cloud only sees what is **pushed** to the repo.

Before deploying:

1. Commit and push `app.py`, `cli.py`, `src/`, `requirements.txt`, `.gitignore`, `.env.example`, docs  
2. Do **not** push `.env` or book PDFs

### Form fields to use

| Field | Select / enter |
|-------|----------------|
| Repository | `mdimamhosen/Research` |
| Branch | `main` |
| Main file path | `EDGE/BanglaDialectDataset/app.py` (**forward slashes**, not `\`) |
| App URL | Something like `bangla-book-ocr` → `bangla-book-ocr.streamlit.app` (pick a free subdomain; leave blank only if you accept the auto name) |
| Python version (Advanced) | 3.11 or 3.12 if offered |

`requirements.txt` next to `app.py` is enough; Cloud will install it.

### Secrets (Advanced / Settings → Secrets)

Do **not** put keys in the repo. In Streamlit Cloud secrets (TOML):

```toml
OPENAI_API_KEY = "sk-..."
# ANTHROPIC_API_KEY = "sk-ant-..."
```

`python-dotenv` reads `.env` locally; on Cloud, Streamlit injects secrets into the environment, which `os.getenv` already uses.

### Cloud limits to know

- Long books = many page API calls (cost + timeouts). Prefer CLI locally for full books; use Cloud for demos / short uploads.
- Working directory on Cloud is the **repo root**; `app.py` already fixes `sys.path` for `src.*` imports.
- Uploaded files on Cloud are ephemeral unless you wire external storage.

## 9. Paper methods blurb (short)

> Scanned Bengali book PDFs were rendered to page images with PyMuPDF at 300 DPI. Text was extracted with a vision-language model (OpenAI GPT-4o; optionally Anthropic Claude) prompted for verbatim OCR without translation or correction. A local Tesseract (`ben+eng`) path was retained as an offline baseline. Outputs were written as UTF-8 plain text with Unicode NFC normalization and explicit page markers.

## 10. What this tool does *not* do

- Dialect labeling or dialect↔standard parallel pairs (that’s later / sibling embedding work)
- Perfect table/column layout reconstruction
- Hosting your API keys or PDFs inside git

## 11. Suggested next steps for you

1. Push the project files to `mdimamhosen/Research` `main`
2. Redeploy Streamlit with path `EDGE/BanglaDialectDataset/app.py`
3. Add `OPENAI_API_KEY` under Cloud Secrets
4. Smoke-test with a 1–2 page PDF before full books
5. Rotate any key that was shared in chat
