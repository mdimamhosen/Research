# BanglaDialectDataset — Bengali Book OCR

Tool to OCR scanned Bengali (Bangla) book PDFs and save plain UTF-8 `.txt` files for the BanglaDialectDataset paper contribution.

No machine-learning training is required to run this tool.

- [GUIDE.md](GUIDE.md) — full walkthrough of how the tool was built (architecture, modules, Cloud deploy)

## Pipeline & data flow

```mermaid
flowchart LR
  User[User] --> UI[app.py Streamlit]
  User --> CLI[cli.py]
  UI --> Pipeline[pipeline.py]
  CLI --> Pipeline
  Pipeline --> Render[pdf_pages.py]
  Render --> Pages[Page images]
  Pages --> Engine{OCR engine}
  Engine -->|default| Gemini[ocr_gemini.py]
  Engine --> OpenAI[ocr_openai.py]
  Engine --> Claude[ocr_claude.py]
  Engine --> Tess[ocr_tesseract.py]
  Gemini --> Prompt[ocr_common.py]
  OpenAI --> Prompt
  Claude --> Prompt
  Gemini --> Raw[Page text]
  OpenAI --> Raw
  Claude --> Raw
  Tess --> Raw
  Raw --> Norm[normalize.py]
  Norm --> TXT[UTF-8 .txt]
```

```mermaid
flowchart TB
  PDF[Scanned PDF] --> Render[Render pages at DPI]
  Render --> Img["list of page_number, PIL.Image"]
  Img --> OCR["ocr_page_*(image) -> str"]
  OCR --> Pairs["list of page_number, text"]
  Pairs --> Join[join_pages NFC + markers]
  Join --> Out["data/txt/book.txt"]
```

## Features

- **Streamlit UI** for upload / folder batch OCR
- **CLI** for reproducible batch runs
- Engines:
  - `gemini` — Gemini 2.0 Flash vision (**default** when `GEMINI_API_KEY` is set)
  - `openai` — GPT-4o vision (optional)
  - `claude` — Claude vision (optional)
  - `tesseract` — local Bengali+English OCR (offline fallback)
- Output: one UTF-8 NFC-normalized `.txt` per PDF, with `--- page N ---` markers

## Setup

```bash
cd EDGE/BanglaDialectDataset
py -3.13 -m venv .venv
# Windows Git Bash / PowerShell:
source .venv/Scripts/activate   # or: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On Git Bash, `streamlit` may still resolve to Anaconda. Prefer the venv explicitly:

```bash
.venv/Scripts/python -m streamlit run app.py
```

Copy `.env.example` to `.env` and set keys:

```bash
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=your-gemini-api-key
```
### Optional: Tesseract (offline / free — recommended default)

Tesseract is already set up for this project if you installed it:

1. Install engine: `winget install UB-Mannheim.TesseractOCR`
2. Bengali data lives in `tessdata/ben.traineddata` (already downloaded here)
3. In Streamlit, choose engine **`tesseract`** (default when no cloud API keys are in `.env`)

No API key needed.

## Usage

### Streamlit

```bash
.venv/Scripts/python -m streamlit run app.py
```

Do **not** run bare `streamlit run app.py` if Anaconda is on your PATH — it will miss project deps.
- Upload PDFs, or drop files into `data/pdfs/` and use **Process folder**
- Results land in `data/txt/` — each upload gets its own file (`book.txt`, or `book_2.txt` if the name exists)
- PDFs are also stored uniquely under `data/pdfs/` so a big dataset never overwrites older books

### CLI

```bash
python cli.py --input data/pdfs --output data/txt --engine tesseract
python cli.py -i book.pdf -o data/txt --engine tesseract --force
python cli.py -i book.pdf -o data/txt --page-start 1 --page-end 5
```

By default the CLI writes a **new** numbered `.txt` if the name exists. Use `--force` to overwrite.

## Deploy to Streamlit Community Cloud

### What you need

1. **GitHub repo** with this code pushed (e.g. `mdimamhosen/Research`)
2. A free account at [share.streamlit.io](https://share.streamlit.io) linked to GitHub
3. **Do not** commit `.env` or book PDFs

### Deploy form

| Field | Value |
|-------|--------|
| Repository | `mdimamhosen/Research` |
| Branch | `main` |
| Main file path | `EDGE/BanglaDialectDataset/app.py` (use `/`, not `\`) |
| App URL | e.g. `bangla-book-ocr` |

### Files Cloud uses

- [`requirements.txt`](requirements.txt) — Python packages  
- [`packages.txt`](packages.txt) — installs system Tesseract + Bengali on Linux Cloud VMs  

### Secrets (only if using paid VLMs)

App settings → Secrets:

```toml
OCR_ENGINE = "tesseract"
# GEMINI_API_KEY = "..."
# OPENAI_API_KEY = "..."
```

For free Tesseract-only deploy you can leave Secrets empty and set nothing — default is tesseract when no cloud keys exist. Optional:

```toml
OCR_ENGINE = "tesseract"
```

### Important Cloud limits

- **Uploads are temporary** on free Cloud — download `.txt` results; don’t treat Cloud disk as your database
- Keep the real corpus locally / in Drive / Git LFS, not only on Streamlit Cloud
- Long 90-page books may hit timeouts; prefer **local CLI** for full books, Cloud for demos/short runs
- Push `tessdata/ben.traineddata` **or** rely on `packages.txt` (`tesseract-ocr-ben`). `packages.txt` is enough on Cloud Linux

### Before first deploy

```bash
git add EDGE/BanglaDialectDataset
git commit -m "Add Bangla book OCR tool"
git push origin main
```

Then open Streamlit Cloud → New app → fill the form above → Deploy.

## Output format

```text
--- page 1 ---

<extracted text>

--- page 2 ---

<extracted text>
```

- Encoding: UTF-8
- Normalization: Unicode NFC; collapsed runaway whitespace
- Bangla script, Latin, digits, and punctuation (including `।`) are preserved

## Methods blurb (paper)

> Scanned Bengali book PDFs were rendered to page images with PyMuPDF at 300 DPI. Text was extracted with a vision-language model (Google Gemini 2.0 Flash by default; optionally OpenAI GPT-4o or Anthropic Claude) prompted for verbatim OCR without translation or correction. A local Tesseract (`ben+eng`) path was retained as an offline baseline. Outputs were written as UTF-8 plain text with Unicode NFC normalization and explicit page markers.

## Layout

```text
BanglaDialectDataset/
  app.py              # Streamlit UI
  cli.py              # batch CLI
  requirements.txt
  .env.example
  src/
    pdf_pages.py
    ocr_openai.py
    ocr_claude.py
    ocr_tesseract.py
    normalize.py
    pipeline.py
  data/
    pdfs/             # drop scanned books here
    txt/              # OCR outputs
```

`data/pdfs/**` and `data/txt/**` are gitignored (except `.gitkeep`). Never commit `.env`.
