"""
dataset.py
----------
Loads cleaned pairs (dialect, standard, area) and returns tensors.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.regions import area_to_id, normalize_area
from src.tokenizer import WordTokenizer


class DialectPairDataset(Dataset):
    """
    Each item:
      dialect_ids / dialect_mask  — dialect sentence
      standard_ids / standard_mask — standard sentence
      area_id — which dialect region (sylheti, barishal, ...)
    """

    def __init__(self, pairs_path: Path, tokenizer: WordTokenizer):
        self.tokenizer = tokenizer
        # (dialect_text, standard_text, area_name)
        self.pairs: list[tuple[str, str, str]] = []

        with pairs_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) == 2:
                    dialect, standard, area = parts[0], parts[1], "unspecified"
                elif len(parts) >= 3:
                    dialect, standard, area = parts[0], parts[1], normalize_area(parts[2])
                else:
                    continue
                self.pairs.append((dialect, standard, area))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        dialect, standard, area = self.pairs[idx]

        d_ids, d_mask = self.tokenizer.encode(dialect)
        s_ids, s_mask = self.tokenizer.encode(standard)

        return {
            "dialect_ids": torch.tensor(d_ids, dtype=torch.long),
            "dialect_mask": torch.tensor(d_mask, dtype=torch.long),
            "standard_ids": torch.tensor(s_ids, dtype=torch.long),
            "standard_mask": torch.tensor(s_mask, dtype=torch.long),
            "area_id": torch.tensor(area_to_id(area), dtype=torch.long),
        }
