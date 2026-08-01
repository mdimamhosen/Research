"""
model.py
--------
Stateful sentence encoder (GRU stand-in for future SSM/Mamba).

Dialect-aware: an AREA embedding (sylheti, barishal, ...) is added to the
dialect-side vector so the model knows which region the sentence comes from.
The standard side uses the "standard" area id.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.regions import area_to_id, num_areas


class ToyStateEncoder(nn.Module):
    """Siamese encoder with optional dialect-area conditioning."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        pad_id: int = 0,
        num_area_labels: int | None = None,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.hidden_dim = hidden_dim
        n_areas = num_area_labels if num_area_labels is not None else num_areas()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.area_embedding = nn.Embedding(n_areas, hidden_dim)

        self.rnn = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def encode_tokens(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = self.dropout(x)
        outputs, _hidden = self.rnn(x)

        mask = attention_mask.unsqueeze(-1).float()
        summed = (outputs * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1.0)
        return summed / lengths

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        area_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        input_ids:      (batch, seq_len)
        attention_mask: (batch, seq_len)
        area_id:        (batch,) dialect region ids; if None, use "unspecified"
        returns:        (batch, hidden_dim) L2-normalized sentence embedding
        """
        pooled = self.encode_tokens(input_ids, attention_mask)

        if area_id is None:
            area_id = torch.full(
                (input_ids.size(0),),
                area_to_id("unspecified"),
                dtype=torch.long,
                device=input_ids.device,
            )

        # Add region signal (dialect-aware embedding)
        pooled = pooled + self.area_embedding(area_id)

        emb = self.out(pooled)
        emb = F.normalize(emb, p=2, dim=-1)
        return emb


def contrastive_loss(
    dialect_emb: torch.Tensor,
    standard_emb: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """InfoNCE-style: match pair i, push away other pairs in the batch."""
    logits = (dialect_emb @ standard_emb.T) / temperature
    batch_size = dialect_emb.size(0)
    labels = torch.arange(batch_size, device=dialect_emb.device)
    loss_d2s = F.cross_entropy(logits, labels)
    loss_s2d = F.cross_entropy(logits.T, labels)
    return (loss_d2s + loss_s2d) / 2.0
