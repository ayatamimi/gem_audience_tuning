# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 17:06:55 2025

@author: ayata
"""

from typing import Optional

import torch
import torch.nn as nn


class MaskedLatentTransformer(nn.Module):
    def __init__(
        self,
        latent_dim: int,               # D_latent (channels of quantized latents)
        vocab_size: int,               # codebook size for logits
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        max_position_embeddings: int,  # T (H*W)
        num_classes: int = 10,         # classifier output size / label space
        label_dim: int = 8,            # learned conditioning embedding size
    ):
        super().__init__()
    
        # --- dims ---
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.label_dim = label_dim
        self.model_dim = latent_dim + label_dim  # transformer d_model
    
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
    
        # --- conditioning modules ---
        # hard label -> embedding (B,) -> (B,label_dim)
        self.label_emb = nn.Embedding(self.num_classes, self.label_dim)
    
        # soft probs/logits (B,num_classes) -> (B,label_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(self.num_classes, self.label_dim),
            nn.GELU(),
            nn.Linear(self.label_dim, self.label_dim),
        )
    
        # --- main transformer ---
        # keep everything in model_dim
        self.input_proj = nn.Linear(self.model_dim, self.model_dim)
        self.pos_embedding = nn.Embedding(self.max_position_embeddings, self.model_dim)
    
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
        self.norm = nn.LayerNorm(self.model_dim)
        self.head = nn.Linear(self.model_dim, self.vocab_size)
        self.dropout = nn.Dropout(dropout)


    def forward(
        self,
        x: torch.Tensor,                       # (B, T, latent_dim)
        labels: Optional[torch.Tensor] = None,  # (B,)
        probs: Optional[torch.Tensor] = None,   # (B, num_classes) probabilities or logits
        label_vec: Optional[torch.Tensor] = None,  # (B, label_dim)
    ) -> torch.Tensor:
        """
        Returns:
            logits: (B, T, vocab_size)
    
        Exactly one of (label_vec, probs, labels) must be provided.
        """
        if x.ndim != 3:
            raise ValueError(f"x must be (B,T,D), got {tuple(x.shape)}")
        B, T, D = x.shape
        if D != self.latent_dim:
            raise ValueError(f"latent_dim mismatch: x last dim {D} != self.latent_dim {self.latent_dim}")
    
        provided = sum(v is not None for v in (label_vec, probs, labels))
        if provided != 1:
            raise ValueError("Provide exactly one of: labels, probs, label_vec")
    
        # Build conditioning vector (B, label_dim)
        if label_vec is not None:
            if label_vec.shape != (B, self.label_dim):
                raise ValueError(f"label_vec must be (B,label_dim)={(B,self.label_dim)}, got {tuple(label_vec.shape)}")
            cond = label_vec
        elif probs is not None:
            if probs.shape != (B, self.num_classes):
                raise ValueError(f"probs must be (B,num_classes)={(B,self.num_classes)}, got {tuple(probs.shape)}")
            cond = self.cond_proj(probs)
        else:
            if labels.shape != (B,):
                raise ValueError(f"labels must be (B,), got {tuple(labels.shape)}")
            cond = self.label_emb(labels)
    
        # Expand to tokens and concat
        cond_tok = cond.unsqueeze(1).expand(B, T, self.label_dim)  # view, no copy
        h = torch.cat([x, cond_tok], dim=-1)  # (B, T, model_dim)
    
        if h.shape[-1] != self.model_dim:
            raise RuntimeError(f"internal model_dim mismatch: got {h.shape[-1]} expected {self.model_dim}")
    
        if T > self.max_position_embeddings:
            raise ValueError(f"Sequence length {T} > max_position_embeddings {self.max_position_embeddings}")
    
        # Input projection + pos emb + transformer
        h = self.input_proj(h)
        pos_ids = torch.arange(T, device=h.device).unsqueeze(0)     # (1,T)
        h = h + self.pos_embedding(pos_ids)                         # (B,T,model_dim)
    
        h = self.dropout(h)
        h = self.encoder(h)
        h = self.norm(h)
        logits = self.head(h)                                       # (B,T,vocab_size)
        return logits
