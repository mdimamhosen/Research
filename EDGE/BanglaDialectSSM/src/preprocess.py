"""
preprocess.py
-------------
Step 1 of the pipeline: clean raw text files.

Input:  corpus/raw/pairs.tsv
  dialect <TAB> standard <TAB> area
  (area is optional; missing area becomes "unspecified")

Output: corpus/processed/pairs.tsv  (same 3-column format, cleaned)
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

import yaml

from src.regions import normalize_area

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_config(path: str = "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_bangla_text(text: str) -> str:
    """Unicode NFC + keep Bangla letters/spaces/basic punctuation."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[^\u0980-\u09FF\s\?\!।,]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess_pairs(input_path: Path, output_path: Path) -> int:
    """Read TSV pairs, clean text columns, normalize area, write TSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    area_counts: dict[str, int] = {}

    with input_path.open(encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) == 2:
                dialect_raw, standard_raw, area_raw = parts[0], parts[1], "unspecified"
            elif len(parts) >= 3:
                dialect_raw, standard_raw, area_raw = parts[0], parts[1], parts[2]
            else:
                print(f"Skipping bad line {line_no}: need 2 or 3 tab fields")
                continue

            dialect = clean_bangla_text(dialect_raw)
            standard = clean_bangla_text(standard_raw)
            area = normalize_area(area_raw)

            if not dialect or not standard:
                continue

            fout.write(f"{dialect}\t{standard}\t{area}\n")
            area_counts[area] = area_counts.get(area, 0) + 1
            kept += 1

    print("Area counts:")
    for area, count in sorted(area_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {area}: {count}")

    return kept


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean Bangla dialect↔standard pairs (with area)"
    )
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    raw = Path(config["paths"]["raw_pairs"])
    out = Path(config["paths"]["processed_pairs"])

    if not raw.exists():
        raise FileNotFoundError(f"Missing raw file: {raw}")

    n = preprocess_pairs(raw, out)
    print(f"Cleaned {n} pairs")
    print(f"Wrote -> {out}")


if __name__ == "__main__":
    main()
