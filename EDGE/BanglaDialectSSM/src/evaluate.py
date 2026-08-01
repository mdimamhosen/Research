"""
evaluate.py
-----------
Check matched vs mismatched similarity, plus a per-area breakdown.

Uses batched encoding so large corpora (40k+ pairs) finish quickly
and show a progress bar (does not look "stuck").
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="If >0, only evaluate first N pairs (faster smoke test)",
    )
    args = parser.parse_args()
    config = load_config(args.config)

    ckpt_path = Path(config["paths"]["checkpoint_dir"]) / "toy_encoder.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Train first. Missing: {ckpt_path}")

    print("Loading tokenizer, dataset, checkpoint...")
    tokenizer = WordTokenizer.load(Path(config["paths"]["vocab_file"]))
    dataset = DialectPairDataset(Path(config["paths"]["processed_pairs"]), tokenizer)

    if args.max_pairs and args.max_pairs < len(dataset):
        # Lightweight subset for quick checks
        dataset.pairs = dataset.pairs[: args.max_pairs]
        print(f"Evaluating subset: {len(dataset)} pairs (--max-pairs)")

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

    batch_size = int(config["training"].get("batch_size", 32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    standard_area = area_to_id("standard")
    dialect_vecs = []
    standard_vecs = []
    area_names = []

    print(f"Encoding {len(dataset)} pairs in batches of {batch_size}...")
    for batch in tqdm(loader, desc="Evaluate"):
        d = model(batch["dialect_ids"], batch["dialect_mask"], batch["area_id"])
        std_ids = torch.full_like(batch["area_id"], standard_area)
        s = model(batch["standard_ids"], batch["standard_mask"], std_ids)

        dialect_vecs.append(d)
        standard_vecs.append(s)
        for aid in batch["area_id"].tolist():
            area_names.append(id_to_area(int(aid)))

    dialect_mat = torch.cat(dialect_vecs, dim=0)
    standard_mat = torch.cat(standard_vecs, dim=0)

    matched = (dialect_mat * standard_mat).sum(dim=-1)
    rolled = torch.roll(standard_mat, shifts=1, dims=0)
    mismatched = (dialect_mat * rolled).sum(dim=-1)

    print("=== Quick evaluation ===")
    print(f"Avg similarity (matched pairs):   {matched.mean().item():.4f}")
    print(f"Avg similarity (mismatched):      {mismatched.mean().item():.4f}")
    print("Matched should be HIGHER than mismatched if training worked.")
    print()

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
    # Only search among first K standards for speed (full NxN is huge)
    demo_n = min(5, len(dataset))
    search_k = min(2000, len(dataset))
    sims = dialect_mat[:demo_n] @ standard_mat[:search_k].T
    for i in range(demo_n):
        best_j = int(sims[i].argmax().item())
        d_text, _, area = dataset.pairs[i]
        _, s_text, _ = dataset.pairs[best_j]
        ok = "OK" if best_j == i else "MISS"
        print(f"  [{ok}] [{area}] {d_text}")
        print(f"       -> {s_text}  (sim={sims[i, best_j].item():.3f})")


if __name__ == "__main__":
    main()
