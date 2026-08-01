"""
tokenizer.py
------------
Turns Bangla text into integer IDs (and back).

We build a SIMPLE word-level vocabulary from OUR training pairs.
This is intentionally easy to understand — not BanglaBERT.

Later you can swap this for a pretrained subword tokenizer (BanglaBERT)
without changing the rest of the pipeline much.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


class WordTokenizer:
    """
    Word tokenizer = split on spaces, map each word → integer id.

    Special IDs:
      PAD (0) = padding filler so batches have equal length
      UNK (1) = unknown word not in vocabulary
    """

    def __init__(
        self,
        word_to_id: dict[str, int],
        pad_token: str = "<PAD>",
        unk_token: str = "<UNK>",
        max_length: int = 32,
    ):
        self.word_to_id = word_to_id
        self.id_to_word = {i: w for w, i in word_to_id.items()}
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.pad_id = word_to_id[pad_token]
        self.unk_id = word_to_id[unk_token]
        self.max_length = max_length

    @property
    def vocab_size(self) -> int:
        return len(self.word_to_id)

    def tokenize(self, text: str) -> list[str]:
        """Text → list of words."""
        return text.split()

    def encode(self, text: str) -> tuple[list[int], list[int]]:
        """
        Text → (input_ids, attention_mask)

        attention_mask: 1 for real tokens, 0 for PAD
        """
        words = self.tokenize(text)[: self.max_length]
        ids = [self.word_to_id.get(w, self.unk_id) for w in words]

        # Pad to fixed length so every batch row has the same shape
        mask = [1] * len(ids)
        while len(ids) < self.max_length:
            ids.append(self.pad_id)
            mask.append(0)

        return ids, mask

    def decode(self, ids: list[int]) -> str:
        """IDs → text (skips PAD)."""
        words = []
        for i in ids:
            if i == self.pad_id:
                continue
            words.append(self.id_to_word.get(i, self.unk_token))
        return " ".join(words)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "word_to_id": self.word_to_id,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "max_length": self.max_length,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "WordTokenizer":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            word_to_id=payload["word_to_id"],
            pad_token=payload["pad_token"],
            unk_token=payload["unk_token"],
            max_length=payload["max_length"],
        )


def build_vocab_from_pairs(
    pairs_path: Path,
    pad_token: str = "<PAD>",
    unk_token: str = "<UNK>",
    max_length: int = 32,
) -> WordTokenizer:
    """
    Read cleaned pairs and collect every unique word from both columns.
    Frequency is counted only for curiosity / debugging.
    """
    counter: Counter[str] = Counter()
    with pairs_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            # Only text columns go into the word vocab (not the area label)
            left, right = parts[0], parts[1]
            counter.update(left.split())
            counter.update(right.split())

    # Fixed special tokens first so IDs are stable
    word_to_id = {pad_token: 0, unk_token: 1}
    for word, _freq in sorted(counter.items()):
        if word not in word_to_id:
            word_to_id[word] = len(word_to_id)

    print(f"Vocab size: {len(word_to_id)} (including PAD/UNK)")
    return WordTokenizer(word_to_id, pad_token, unk_token, max_length)
