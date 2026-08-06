# BanglaDialectDataset — Bengali Book OCR (CLI-first)

Automated tool to OCR scanned Bengali (Bangla) book PDFs into UTF-8 `.txt` files for the dataset paper.

**Primary interface: CLI.** Open-source engines by default (`tesseract`, optional `paddle`). No paid API required. Python is the intentional language choice.

Optional Streamlit UI is for demos only (`requirements-ui.txt`).

## Quick start (CLI)

```bash
cd EDGE/BanglaDialectDataset
py -3.13 -m venv .venv
.venv/Scripts/activate          # Windows
pip install -r requirements.txt

# Install Tesseract OS package once:
#   winget install UB-Mannheim.TesseractOCR
# Bengali model files live in ./tessdata/ (ben + eng)

# Drop PDFs here, then run:
python cli.py -i data/pdfs -o data/txt --engine tesseract --skip-existing
```

### Useful CLI flags

```bash
# Resume a large corpus (skip books that already have .txt)
python cli.py -i data/pdfs -o data/txt --engine tesseract --skip-existing

# Overwrite same-named outputs
python cli.py -i data/pdfs -o data/txt --engine tesseract --force

# Smoke-test a few pages
python cli.py -i data/pdfs/book.pdf -o data/txt --page-start 1 --page-end 3

# Optional open-source alternative (after requirements-paddle.txt)
python cli.py -i data/pdfs -o data/txt --engine paddle --skip-existing

# DeepSeek-OCR open VLM (GPU recommended; smoke-test 1 page first)
pip install -r requirements-deepseek.txt
python cli.py -i data/pdfs/book.pdf -o data/txt --engine deepseek --page-start 1 --page-end 1
```

## Engines

| Engine | License / cost | Notes |
|--------|----------------|-------|
| `tesseract` | Open source · **default** | Local CLI; needs `ben` language data |
| `paddle` | Open source · optional | `pip install -r requirements-paddle.txt` |
| `deepseek` | Open source VLM · optional | `pip install -r requirements-deepseek.txt` · GPU ~8GB+ VRAM |
| `gemini` / `openai` / `claude` | Paid APIs · optional | Not part of the default paper pipeline |

Default resolution: `OCR_ENGINE` in `.env`, else **`tesseract`**.

## Layout

```text
BanglaDialectDataset/
  cli.py                 # ← main automated entrypoint
  app.py                 # optional Streamlit demo
  requirements.txt       # CLI core
  requirements-paddle.txt
  requirements-ui.txt
  src/
    pipeline.py
    pdf_pages.py
    ocr_tesseract.py
    ocr_paddle.py
    normalize.py
    ...
  data/pdfs/             # input books
  data/txt/              # UTF-8 outputs
  tessdata/              # Tesseract ben/eng models
```

## Output format

```text
--- page 1 ---

<extracted text>

--- page 2 ---

<extracted text>
```

- Encoding: UTF-8 · Unicode NFC
- One `.txt` per book; `--skip-existing` for safe re-runs on a large corpus

## Optional UI

```bash
pip install -r requirements-ui.txt
.venv/Scripts/python -m streamlit run app.py
```

## Methods blurb (paper)

> Scanned Bengali book PDFs were rendered with PyMuPDF (300 DPI) and transcribed with open-source OCR (Tesseract `ben+eng`; optionally PaddleOCR). Processing was automated via a command-line batch pipeline producing UTF-8 plain text with Unicode NFC normalization and page markers. No commercial vision API was required for the core corpus.

## Why Python

OCR libraries for Bangla (Tesseract bindings, PaddleOCR, PDF rendering) are Python-first. The contribution is an engineering pipeline, not a new ML model — Python keeps the stack small and reproducible.

## Docs

- [GUIDE.md](GUIDE.md) — module relations and data flow
