# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 17:07:05 2025

@author: ayata
"""


from typing import Optional

import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


# ------------------------------------------------------------
# Masking utility
# ------------------------------------------------------------

def mask_latents(
    latents: torch.Tensor,
    indices: torch.Tensor,
    mask_prob: float,
    mask_token_id: int = 0,
):
    """
    Randomly mask a fraction of tokens in the sequence.

    Parameters
    ----------
    latents : torch.Tensor
        Input latent features, shape (B, T, E).
        In your training script this is zq (masked_exp_mask_feat_*_quantizes).
    indices : torch.Tensor
        Discrete latent indices, shape (B, T), int64.
    mask_prob : float
        Probability of masking each token.
    mask_token_id : int, default 0
        Token id used as the "mask" label in the index space.

    Returns
    -------
    latents_masked : torch.Tensor
        Same shape as latents (B, T, E) but with masked positions zeroed.
    target : torch.Tensor
        Original target indices, shape (B, T), int64.
    mask : torch.Tensor
        Boolean mask of shape (B, T), True where the token was masked.

    Notes
    -----
    Your training loop does:
        zq_masked, target, mask = mask_latents(zq, idx, cfg.mask_prob)
        logits = model(zq_masked)
        if cfg.loss_masked:
            loss = criterion(logits[mask], target[mask])
        else:
            ...
    So:
        - `target` is always the original indices to predict.
        - `mask` selects the positions to include in the loss when loss_masked=True.
    """
    device = latents.device
    B, T, E = latents.shape

    if mask_prob <= 0.0:
        # no masking
        mask = torch.zeros(B, T, dtype=torch.bool, device=device)
        return latents, indices, mask

    # Sample mask: True with probability mask_prob
    # shape: (B, T)
    mask = torch.rand(B, T, device=device) < mask_prob

    # Copy original targets
    target = indices.clone()

    # Optionally, could also change indices at masked positions to mask_token_id.
    # Not strictly required for your current loss computation, but often useful.
    # masked_indices = indices.clone()
    # masked_indices[mask] = mask_token_id

    # Mask latent features: here we just zero them
    latents_masked = latents.clone()
    latents_masked[mask] = 0.0

    return latents_masked, target, mask


# ------------------------------------------------------------
# Scheduler utilities
# ------------------------------------------------------------

def _get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    """
    Linear warmup, then linear decay to 0.
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            # warmup
            return float(current_step) / max(1, num_warmup_steps)
        # decay
        return max(
            0.0,
            float(num_training_steps - current_step)
            / max(1, num_training_steps - num_warmup_steps),
        )

    return LambdaLR(optimizer, lr_lambda)


def _get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
) -> LambdaLR:
    """
    Cosine schedule with warmup (similar to HF transformers).
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / max(1, num_warmup_steps)
        progress = float(current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )
        return max(
            0.0,
            0.5 * (1.0 + math.cos(math.pi * 2.0 * num_cycles * progress)),
        )

    return LambdaLR(optimizer, lr_lambda)


def build_scheduler(
    optimizer: Optimizer,
    scheduler_name: Optional[str],
    num_training_steps: int,
    warmup_steps: int = 0,
):
    if scheduler_name is None:
        scheduler_name = "none"

    name = scheduler_name.lower()

    if name in ["none", "constant"]:
        # No scheduling; constant LR
        return LambdaLR(optimizer, lambda _: 1.0)
    elif name in ["linear_warmup", "linear", "linearwarmup"]:
        # Linear warmup + linear decay
        return _get_linear_schedule_with_warmup(
            optimizer, warmup_steps, num_training_steps
        )
    elif name == "cosine":
        return _get_cosine_schedule_with_warmup(
            optimizer, warmup_steps, num_training_steps
        )
    else:
        raise ValueError(f"Unknown scheduler '{scheduler_name}'")

