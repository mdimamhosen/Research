"""
train.py
--------
Full training loop with dialect AREA labels.

Run from project root:
  python -m src.preprocess
  python -m src.train
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import DialectPairDataset
from src.model import ToyStateEncoder, contrastive_loss
from src.preprocess import preprocess_pairs
from src.regions import area_to_id, num_areas
from src.tokenizer import WordTokenizer, build_vocab_from_pairs

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_processed(config: dict) -> None:
    """Create cleaned pairs + vocab if they do not exist yet."""
    raw = Path(config["paths"]["raw_pairs"])
    processed = Path(config["paths"]["processed_pairs"])
    vocab_path = Path(config["paths"]["vocab_file"])

    # Always refresh processed from raw when training, so new area tags apply
    print("Running preprocess...")
    n = preprocess_pairs(raw, processed)
    print(f"Cleaned {n} pairs -> {processed}")

    print("Building vocab from processed pairs...")
    tok_cfg = config["tokenizer"]
    tokenizer = build_vocab_from_pairs(
        processed,
        pad_token=tok_cfg["pad_token"],
        unk_token=tok_cfg["unk_token"],
        max_length=tok_cfg["max_length"],
    )
    tokenizer.save(vocab_path)
    print(f"Saved vocab -> {vocab_path}")


def train(config_path: str = "config.yaml") -> None:
    config = load_config(config_path)
    ensure_processed(config)

    torch.manual_seed(config["training"]["seed"])

    tokenizer = WordTokenizer.load(Path(config["paths"]["vocab_file"]))
    dataset = DialectPairDataset(Path(config["paths"]["processed_pairs"]), tokenizer)

    batch_size = min(config["training"]["batch_size"], len(dataset))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model_cfg = config["model"]
    model = ToyStateEncoder(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=model_cfg["embedding_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
        pad_id=tokenizer.pad_id,
        num_area_labels=num_areas(),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    standard_area = area_to_id("standard")

    print(
        f"Examples: {len(dataset)} | Vocab: {tokenizer.vocab_size} "
        f"| Areas: {num_areas()} | Device: CPU"
    )
    model.train()

    for epoch in range(config["training"]["epochs"]):
        total_loss = 0.0
        steps = 0

        for batch in tqdm(loader, desc=f"Epoch {epoch + 1}/{config['training']['epochs']}"):
            # Dialect side: use the row's area (sylheti, barishal, ...)
            dialect_emb = model(
                batch["dialect_ids"],
                batch["dialect_mask"],
                batch["area_id"],
            )
            # Standard side: always conditioned as "standard"
            std_ids = torch.full_like(batch["area_id"], standard_area)
            standard_emb = model(
                batch["standard_ids"],
                batch["standard_mask"],
                std_ids,
            )

            if dialect_emb.size(0) == 1:
                loss = 1.0 - (dialect_emb * standard_emb).sum(dim=-1).mean()
            else:
                loss = contrastive_loss(dialect_emb, standard_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            steps += 1

        avg = total_loss / max(steps, 1)
        print(f"Epoch {epoch + 1} - avg loss: {avg:.4f}")

    ckpt_dir = Path(config["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "toy_encoder.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab_size": tokenizer.vocab_size,
            "model_config": model_cfg,
            "pad_id": tokenizer.pad_id,
            "num_areas": num_areas(),
        },
        ckpt_path,
    )
    print(f"Saved checkpoint -> {ckpt_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
