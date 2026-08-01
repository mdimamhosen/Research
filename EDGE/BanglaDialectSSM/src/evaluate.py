"""
evaluate.py
-----------
Check matched vs mismatched similarity, plus a per-area breakdown.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml

from src.dataset import DialectPairDataset
from src.model import ToyStateEncoder
from src.regions import area_to_id, id_to_area, num_areas
from src.tokenizer import WordTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    ckpt_path = Path(config["paths"]["checkpoint_dir"]) / "toy_encoder.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Train first. Missing: {ckpt_path}")

    tokenizer = WordTokenizer.load(Path(config["paths"]["vocab_file"]))
    dataset = DialectPairDataset(Path(config["paths"]["processed_pairs"]), tokenizer)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = ToyStateEncoder(
        vocab_size=ckpt["vocab_size"],
        embedding_dim=ckpt["model_config"]["embedding_dim"],
        hidden_dim=ckpt["model_config"]["hidden_dim"],
        num_layers=ckpt["model_config"]["num_layers"],
        dropout=0.0,
        pad_id=ckpt["pad_id"],
        num_area_labels=ckpt.get("num_areas", num_areas()),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    standard_area = area_to_id("standard")
    dialect_vecs = []
    standard_vecs = []
    area_names = []

    for i in range(len(dataset)):
        item = dataset[i]
        d = model(
            item["dialect_ids"].unsqueeze(0),
            item["dialect_mask"].unsqueeze(0),
            item["area_id"].unsqueeze(0),
        )
        s = model(
            item["standard_ids"].unsqueeze(0),
            item["standard_mask"].unsqueeze(0),
            torch.tensor([standard_area]),
        )
        dialect_vecs.append(d.squeeze(0))
        standard_vecs.append(s.squeeze(0))
        area_names.append(id_to_area(int(item["area_id"].item())))

    dialect_mat = torch.stack(dialect_vecs)
    standard_mat = torch.stack(standard_vecs)

    matched = (dialect_mat * standard_mat).sum(dim=-1)
    rolled = torch.roll(standard_mat, shifts=1, dims=0)
    mismatched = (dialect_mat * rolled).sum(dim=-1)

    print("=== Quick evaluation ===")
    print(f"Avg similarity (matched pairs):   {matched.mean().item():.4f}")
    print(f"Avg similarity (mismatched):      {mismatched.mean().item():.4f}")
    print("Matched should be HIGHER than mismatched if training worked.")
    print()

    # Per-area matched similarity
    by_area: dict[str, list[float]] = defaultdict(list)
    for sim, area in zip(matched.tolist(), area_names):
        by_area[area].append(sim)

    print("=== Per-area matched similarity ===")
    for area in sorted(by_area.keys()):
        vals = by_area[area]
        avg = sum(vals) / len(vals)
        print(f"  {area:14s}  n={len(vals):5d}  avg_sim={avg:.4f}")
    print()

    print("Examples (dialect [area] -> best matching standard):")
    # Cap retrieval demo size for speed on large datasets
    demo_n = min(5, len(dataset))
    sub_d = dialect_mat[:demo_n]
    sims = sub_d @ standard_mat.T
    for i in range(demo_n):
        best_j = int(sims[i].argmax().item())
        d_text, _, area = dataset.pairs[i]
        _, s_text, _ = dataset.pairs[best_j]
        ok = "OK" if best_j == i else "MISS"
        print(f"  [{ok}] [{area}] {d_text}")
        print(f"       -> {s_text}  (sim={sims[i, best_j].item():.3f})")


if __name__ == "__main__":
    main()
