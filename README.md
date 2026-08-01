# Research Workspace

This workspace contains multiple research tracks.

## What Is Here

- `EDGE/BanglaDialectSSM`
  - A Bangla dialect research pipeline focused on dialect-to-standard Bangla pairs.
  - The goal is to learn useful text embeddings from paired data.
  - Current implementation uses a simple word-level tokenizer and a GRU-based encoder, with the longer-term idea of moving toward a state-space model such as Mamba.

- `MARVIS`
  - A separate research track related to medical work.
  - This folder is intentionally separated from the Bangla language work so the two projects stay independent.
  - If you add files here later, document the medical problem, dataset, and evaluation method inside that project folder.

## BanglaDialectSSM Overview

The BanglaDialectSSM project studies how to map dialect Bangla and standard Bangla into similar vectors.

The practical idea is:

1. Take a dialect sentence and its standard Bangla equivalent.
2. Clean and preprocess the pair.
3. Build a vocabulary from the corpus.
4. Train an encoder to produce similar embeddings for matched pairs.
5. Check whether matched sentences score higher than mismatched ones.

The project lives in:

`C:\Error\Research\EDGE\BanglaDialectSSM`

Key files:

- [`README.md`](EDGE/BanglaDialectSSM/README.md)
- [`config.yaml`](EDGE/BanglaDialectSSM/config.yaml)
- [`src/preprocess.py`](EDGE/BanglaDialectSSM/src/preprocess.py)
- [`src/tokenizer.py`](EDGE/BanglaDialectSSM/src/tokenizer.py)
- [`src/train.py`](EDGE/BanglaDialectSSM/src/train.py)
- [`src/evaluate.py`](EDGE/BanglaDialectSSM/src/evaluate.py)

## Data Format

The main dataset is:

`C:\Error\Research\EDGE\BanglaDialectSSM\corpus\raw\pairs.tsv`

It uses tab-separated rows with 3 columns:

```text
dialect sentence<TAB>standard sentence<TAB>area
```

Meaning:

- `dialect sentence` is the regional or dialect version
- `standard sentence` is the normalized standard Bangla version
- `area` is the regional label such as `sylheti`, `chottogrami`, `barishallia`, `khulnaia`, `rajshahi`, or `rangpuri`

## Why The Dataset Was Cleaned

The raw dataset originally had many synthetic or repetitive rows.

That caused two problems:

- The text quality was weak, so some pairs did not read naturally.
- The vocabulary stayed too small because too many sentences reused the same few words.

To improve this:

- obvious bad rows were removed
- the corpus was expanded with more real Bangla words
- vocabulary size was verified again using the actual tokenizer logic

## Current Bangla Corpus Status

The current corpus now builds a much larger vocabulary than before.

Important point:

- The tokenizer is word-level
- So vocabulary size depends entirely on the actual unique words present in the corpus
- After augmentation, the corpus crosses the 5,000-word target

## How The Bangla Pipeline Works

Run the pipeline from inside `EDGE/BanglaDialectSSM`:

```bash
python -m src.preprocess
python -m src.train
python -m src.evaluate
```

What each step does:

- `preprocess`
  - cleans the raw TSV
  - normalizes text
  - writes processed pairs

- `train`
  - builds or reloads the vocabulary
  - trains the encoder
  - saves checkpoints

- `evaluate`
  - compares matched and mismatched sentence similarity
  - gives a quick signal on whether the model is learning useful structure

## Supporting Scripts

There are also dataset-generation and augmentation scripts in the Bangla project directory, including:

- `generate_15k_dataset.py`
- `enrich_bangla_dataset.py`
- `enrich_bangla_dataset_v2.py`
- `enrich_bangla_dataset_v3.py`
- `enrich_final.py`
- `augment_vocab_from_wordlist.py`

These were used to expand the corpus and raise vocabulary coverage.

## MARVIS Note

`MARVIS` is a separate research area for medical work.

Because it is separate from the Bangla dialect project, the cleanest structure is:

- keep dataset and model code inside `MARVIS`
- keep Bangla language work inside `EDGE/BanglaDialectSSM`
- add a `README.md` inside `MARVIS` when the project contents are ready

If you want MARVIS documented in the same style as BanglaDialectSSM, I can create that README next.

## Repository Layout

```text
Research/
├── EDGE/
│   └── BanglaDialectSSM/
│       ├── corpus/
│       ├── checkpoints/
│       ├── src/
│       └── README.md
├── MARVIS/
└── README.md
```

## Short Summary

This workspace currently holds:

- one language research project about Bangla dialects
- one separate medical research track called MARVIS

The root README is meant to explain both so the workspace is easier to navigate and maintain.
