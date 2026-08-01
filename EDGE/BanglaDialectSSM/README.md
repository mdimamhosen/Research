# BanglaDialectSSM

Beginner pipeline for the research idea:

> **Efficient Dialect-Aware Bangla Embedding Model Using State Space Models (SSMs)**

Turn dialect Bangla and standard Bangla into **similar vectors**, using a stateful encoder (GRU now → Mamba/SSM later).

## How to run

Open a terminal in the project folder:

```bash
cd c:/Error/Research/EDGE/BanglaDialectSSM
```

### First time only (setup)

Use **Python 3.13** (PyTorch may not support 3.14 yet).

```bash
py -3.13 -m venv .venv
.venv/Scripts/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

If `.venv` already exists, only activate:

```bash
.venv/Scripts/activate
```

### Every time (train + check)

```bash
# 1) Clean raw pairs -> corpus/processed/pairs.tsv
python -m src.preprocess

# 2) Build vocab (if needed), train, save checkpoints/toy_encoder.pt
python -m src.train

# 3) Check matched vs mismatched similarity
python -m src.evaluate
```

`python -m src.train` also auto-runs preprocess + vocab build if those files are missing.

### After you edit the dataset

If you change `corpus/raw/pairs.tsv`, delete old generated files so vocab rebuilds:

```bash
rm -f corpus/processed/pairs.tsv corpus/processed/vocab.json
python -m src.preprocess
python -m src.train
python -m src.evaluate
```

## Enrich the dataset (with AREA)

Edit [`corpus/raw/pairs.tsv`](corpus/raw/pairs.tsv). **3 columns**, tab-separated:

```text
dialect sentence<TAB>standard sentence<TAB>area
```

Examples:

```text
আই যাইয়্যার	আমি যাচ্ছি	chittagonian
তুই কী করর	তুমি কী করছ	sylheti
মোরে একটা কলম দাও	আমাকে একটা কলম দাও	barishal
```

Allowed areas (see [`src/regions.py`](src/regions.py)):

| Area id | Meaning |
|---------|---------|
| `sylheti` | সিলেটি / Sylhetia |
| `chittagonian` | চাটগাঁইয়া / Chittagong |
| `barishal` | বরিশালিয়া / Barishal |
| `noakhali` | নোয়াখাইল্যা |
| `mymensingh` | ময়মনসিংহ |
| `rangpur` / `rajshahi` / `dhaka` / ... | other regions |
| `unspecified` | unknown (temporary) |

Aliases work too: `barishalia` → `barishal`, `sylhetia` → `sylheti`, `chatgaiya` → `chittagonian`.

Rules:
- One pair per line
- Lines starting with `#` are comments
- If you omit the area column, it becomes `unspecified`
- For Kothon Excel: map Dialect + Standard Bangla columns, set area to `chittagonian` or `sylheti` per file

Later replace unlabeled rows with real ONUBAD / Kothon / lab data.

## Folder map

```
BanglaDialectSSM/
├── config.yaml
├── requirements.txt
├── README.md
├── LEARNING.md
├── corpus/
│   ├── raw/pairs.tsv
│   └── processed/          # created by scripts
├── checkpoints/            # saved model
└── src/
    ├── regions.py           # area names (sylheti, barishal, ...)
    ├── preprocess.py
    ├── tokenizer.py
    ├── dataset.py
    ├── model.py
    ├── train.py
    └── evaluate.py
```

## What each script does

| Command | Role |
|---------|------|
| `python -m src.preprocess` | Clean text |
| `python -m src.train` | Learn embeddings |
| `python -m src.evaluate` | Quick quality check |

## Honest limits

- Encoder is **GRU**, not real **Mamba/SSM** yet
- Tokenizer is simple **word-level**
- Sample pairs are for learning the pipeline, not paper results
