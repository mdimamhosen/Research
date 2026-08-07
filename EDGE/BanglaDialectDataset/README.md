# BanglaDialectDataset — Bengali Book OCR (CLI-first, Paddle only)

Automated tool: scanned Bengali (Bangla) book PDFs → UTF-8 `.txt` via **PaddleOCR-VL**.

- **No paid APIs** (no OpenAI / Claude / Gemini).
- **CLI auto-detects** one PDF or a whole folder → **one `.txt` per PDF**.
- **Team sharing:** run once via Docker on a shared host — see [TEAM_WORKFLOW.md](TEAM_WORKFLOW.md).
- Full architecture: [GUIDE.md](GUIDE.md).
- **Kaggle GPU:** [KAGGLE.md](KAGGLE.md) — no per-file PDF copy; use `--kaggle`.

## Quick start (local)

```bash
cd EDGE/BanglaDialectDataset
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install paddlepaddle==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
pip install -r requirements.txt

# Drop PDFs into data/pdfs/, then batch all of them:
python cli.py -i data/pdfs -o data/txt --skip-existing
```

### Useful commands

```bash
# One file
python cli.py -i data/pdfs/book.pdf -o data/txt

# Folder (auto-detect 1..N PDFs → N .txt files)
python cli.py -i data/pdfs -o data/txt --skip-existing

# Nested folders
python cli.py -i data/pdfs --recursive --skip-existing

# Smoke-test 1 page
python cli.py -i data/pdfs/book.pdf -o data/txt --page-start 1 --page-end 1

# Faster on CPU (no GPU): DPI 150 + skip layout + shrink pages
.\.venv\Scripts\python.exe cli.py -i data/pdfs -o data/txt --fast --force

# Fastest on CPU: Tesseract (install binary first — see below)
.\.venv\Scripts\python.exe cli.py -i data/pdfs -o data/txt -e tesseract --force
```

**Tesseract install (Windows):** download & run  
https://github.com/UB-Mannheim/tesseract/wiki  
During setup tick **Bengali**. Then:

```powershell
$env:TESSERACT_CMD="C:\Program Files\Tesseract-OCR\tesseract.exe"
.\.venv\Scripts\python.exe cli.py -i data/pdfs -o data/txt -e tesseract --force
```

Project already has `tessdata/ben.traineddata` + `eng.traineddata` for language data.

## Team / publish without everyone installing Paddle

**Most effective:** Docker Compose on one lab PC or VPS. Teammates only drop PDFs and collect `.txt`.

```bash
docker compose build
# put PDFs in ./data/pdfs/
docker compose run --rm ocr
# outputs in ./data/txt/
```

Details, roles, cron, and why not MongoDB/CI: **[TEAM_WORKFLOW.md](TEAM_WORKFLOW.md)**.

Anyone can still clone and run locally — same CLI.

## Engine

| Engine | Notes |
|--------|--------|
| `paddle` | **Only engine.** PaddleOCR-VL by default (`PADDLE_OCR_BACKEND=vl`) |

## Layout

```text
BanglaDialectDataset/
  cli.py                 # main automated entrypoint
  app.py                 # optional Streamlit demo
  Dockerfile / docker-compose.yml
  GUIDE.md               # full build + file relations
  TEAM_WORKFLOW.md       # shared host for the team
  src/
    pipeline.py          # discover PDFs + orchestrate
    pdf_pages.py
    ocr_paddle.py
    normalize.py
  data/pdfs/             # inputs
  data/txt/              # outputs
```

## Output format

```text
--- page 1 ---

<extracted text>

--- page 2 ---

<extracted text>
```

UTF-8 · Unicode NFC · `--skip-existing` for safe corpus resumes.

## Methods blurb (paper)

> Scanned Bengali book PDFs were rendered with PyMuPDF (300 DPI) and transcribed with open-source PaddleOCR-VL. Processing was automated via a command-line batch pipeline producing UTF-8 plain text with Unicode NFC normalization and page markers. No commercial vision API was used.
