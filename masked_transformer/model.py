# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 17:06:55 2025

@author: ayata
"""

from typing import Optional

import torch
import torch.nn as nn


class MaskedLatentTransformer(nn.Module):
    """
    Simple Transformer encoder over latent sequences.

    Inputs
    ------
    x: Tensor of shape (B, T, E)
        B = batch size
        T = number of tokens per sequence
        E = embed_dim (your "distil_d_embed_vec": latent + label one-hot + mask feature)

    Output
    ------
    logits: Tensor of shape (B, T, vocab_size)
        Per-token logits over discrete latent indices.
    """

    def __init__(
        self,
        embed_dim: int,
        vocab_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        max_position_embeddings: int,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings

        # Optional input projection (in case you later want different model_dim)
        self.input_proj = nn.Linear(embed_dim, embed_dim)

        # Positional embeddings (0 .. max_position_embeddings-1)
        self.pos_embedding = nn.Embedding(max_position_embeddings, embed_dim)

        # Standard Transformer encoder stack (batch_first=True for (B, T, E))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(embed_dim)

        # Final linear head to vocab logits
        self.head = nn.Linear(embed_dim, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, E)
        returns logits: (B, T, vocab_size)
        """
        # x is already "embeddings" (latent+label+mask feature)
        B, T, E = x.shape
        if T > self.max_position_embeddings:
            raise ValueError(
                f"Sequence length {T} > max_position_embeddings {self.max_position_embeddings}"
            )

        # (B, T, E) -> (B, T, E)
        h = self.input_proj(x)

        # Add positional embeddings
        pos_ids = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        pos_emb = self.pos_embedding(pos_ids)                    # (1, T, E)
        h = h + pos_emb

        h = self.dropout(h)
        h = self.encoder(h)      # (B, T, E)
        h = self.norm(h)
        logits = self.head(h)    # (B, T, vocab_size)

        return logits
