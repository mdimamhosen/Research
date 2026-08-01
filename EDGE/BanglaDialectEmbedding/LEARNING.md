# BanglaDialectSSM — Learning notes

Read this slowly.

## Research goal (one sentence)

Make a model that turns Bangla text into a list of numbers (**embedding**) so that a **dialect** sentence and its **standard Bangla** meaning get **similar** numbers — using an efficient **state** model (SSM/Mamba later; GRU toy now).

## Pipeline as a factory line

```text
raw pairs.tsv
    → preprocess (clean Unicode / Bangla)
    → build vocab + tokenize (words → IDs)
    → Dataset (one training example at a time)
    → Model (IDs → sentence vector)
    → Loss (pull matching pairs together)
    → Save checkpoint
    → Evaluate (are matches closer than random?)
```

## Glossary

| Word | Meaning |
|------|---------|
| **Token** | A piece of text (here: a word) |
| **Tokenizer** | Tool that maps text ↔ token IDs |
| **Vocab** | The list of known tokens + their IDs |
| **Embedding** | A vector (list of floats) representing meaning |
| **State** | A memory vector updated as we read each token |
| **SSM** | State Space Model — efficient stateful sequence model |
| **Mamba** | Popular modern SSM architecture |
| **Transformer** | Attention-based model (BERT/BanglaBERT family) |
| **Contrastive loss** | Train by making true pairs similar, false pairs dissimilar |
| **Batch** | Several examples processed together |
| **Epoch** | One full pass over the training data |
| **Checkpoint** | Saved model weights on disk |

## Why GRU now, Mamba later?

Both keep a **state** and update it token by token.

- **GRU**: built into PyTorch, easy to read, good for learning
- **Mamba/SSM**: research target for the “efficient” claim; harder to implement

When the lab picks an architecture, you replace `nn.GRU` inside `ToyStateEncoder` with a Mamba block — the dataset, loss, and train loop can stay similar.

## Why contrastive loss?

For each batch:

- dialect sentence *i* should match standard sentence *i*
- it should not match other standards in the batch

That teaches the model “same meaning → close vectors.”

## What to change when you get real data

1. Replace `corpus/raw/pairs.tsv` with real dialect↔standard pairs  
2. Re-run `python -m src.preprocess` then `python -m src.train`  
3. Later: better tokenizer, real SSM, proper eval metrics (STS, retrieval, dialect-ID)

## Python tips for this repo

- Run modules as `python -m src.train` from the project root (not from inside `src/`)
- `config.yaml` is the control panel — change batch size / epochs there
- `.venv` is your private package install folder — activate it before running
