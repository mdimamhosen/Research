#!/usr/bin/env python3
"""
Augment the Bangla dialect corpus with real Bangla vocabulary coverage.

This script appends simple, grammatical sentence pairs that introduce many
distinct Bangla words from a public word list. The goal is to raise the word
vocabulary size without inventing nonsense tokens.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


RAW_URL = "https://raw.githubusercontent.com/tahmid02016/bangla-wordlist/master/words.txt"
CORPUS_PATH = Path(r"C:\Error\Research\EDGE\BanglaDialectSSM\corpus\raw\pairs.tsv")


def load_existing_words(path: Path) -> set[str]:
    words: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            words.update(parts[0].split())
            words.update(parts[1].split())
    return words


def fetch_words(limit: int = 10000) -> list[str]:
    with urllib.request.urlopen(RAW_URL, timeout=30) as resp:
        text = resp.read().decode("utf-8", errors="ignore")
    words = []
    for line in text.splitlines():
        w = line.strip()
        if not w:
            continue
        if any(ch.isspace() for ch in w):
            continue
        words.append(w)
        if len(words) >= limit:
            break
    return words


def main() -> None:
    existing = load_existing_words(CORPUS_PATH)
    candidates = fetch_words()

    templates = [
        ("আমি {w} পড়ি", "আমি {w} পড়ি", "unspecified"),
        ("আমি {w} দেখি", "আমি {w} দেখি", "unspecified"),
        ("তুমি {w} পছন্দ করো", "তুমি {w} পছন্দ করো", "unspecified"),
        ("আমরা {w} ব্যবহার করি", "আমরা {w} ব্যবহার করি", "unspecified"),
        ("সে {w} খুঁজে পায়", "সে {w} খুঁজে পায়", "unspecified"),
    ]

    appended = []
    seen_new = set()
    for w in candidates:
        if w in existing or w in seen_new:
            continue
        if len(w) < 2:
            continue
        for d_tpl, s_tpl, region in templates:
            dialect = d_tpl.format(w=w)
            standard = s_tpl.format(w=w)
            appended.append((dialect, standard, region))
        seen_new.add(w)
        if len(seen_new) >= 6000:
            break

    if not appended:
        print("No new rows generated.")
        return

    backup = CORPUS_PATH.with_suffix(".tsv.pre_vocab_backup")
    backup.write_text(CORPUS_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    with CORPUS_PATH.open("a", encoding="utf-8", newline="\n") as f:
        for dialect, standard, region in appended:
            f.write(f"{dialect}\t{standard}\t{region}\n")

    print(f"Appended {len(appended)} rows using {len(seen_new)} unique Bangla words")
    print(f"Backup: {backup}")


if __name__ == "__main__":
    main()
