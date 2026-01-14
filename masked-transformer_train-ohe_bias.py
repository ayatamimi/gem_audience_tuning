import os
import argparse
import math
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import neptune

from logging_neptune import CheckpointManager, NeptuneRun
from distributed import set_seed
from configs.config import get_transformer_config
from masked_transformer.model_ohe import MaskedLatentTransformer
from masked_transformer.utils import mask_latents, build_scheduler

def attach_bias_ohe(labels, masked_quantizes, num_classes=10):
    """
    Returns: (B, T, D + num_classes)
    Exactly equivalent to:
        one_hot = F.one_hot(labels.unsqueeze(1).expand(B,T), num_classes).to(dtype)
        torch.cat([masked_quantizes, one_hot], dim=-1)
    but memory-safe (no one_hot tensor + no repeated cat allocations).
    """
    idx_t = torch.as_tensor(labels, dtype=torch.long, device=masked_quantizes.device)  # (B,)

    B, T, D = masked_quantizes.shape
    C = num_classes
    device = masked_quantizes.device
    dtype  = masked_quantizes.dtype

    # (B,T) view
    idx_t_exp = idx_t.unsqueeze(1).expand(B, T)

    # allocate only final tensor
    out = torch.empty((B, T, D + C), device=device, dtype=dtype)
    out[..., :D] = masked_quantizes

    tail = out[..., D:]          # (B,T,C) view
    tail.zero_()
    tail.scatter_(2, idx_t_exp.unsqueeze(-1), 1)

    return out


# =============================================================================
# def attach_bias_mask_feat(labels, masked_quantizes, mask):
#     
#     idx_t = torch.as_tensor(labels, dtype=torch.long, device=masked_quantizes.device)  # (N,)
# 
#     N, T, _ = masked_quantizes.shape
#     idx_t_exp = idx_t.unsqueeze(1).expand(N, T)                         # (N, T)
# 
#     one_hot_labels = torch.nn.functional.one_hot(idx_t_exp, num_classes=10).to(dtype=masked_quantizes.dtype)  # (N,T,C)
# 
#     masked_exp_quantizes = torch.cat([masked_quantizes, one_hot_labels], dim=2)
# 
# 
#     #print('masked_exp_quantizes.shape: ',masked_exp_quantizes.shape)  # torch.Size([N, 64, 27])
#     # masked_exp_train_quantizes: torch.FloatTensor (N, T, 27)
#     # mask_train: numpy bool array (N, T)  True = masked
# 
#     device = masked_exp_quantizes.device
#     dtype  = masked_exp_quantizes.dtype
#     
#    # 1) NumPy -> Torch, cast to float (1.0 masked, 0.0 unmasked)
#     mask_feat = torch.as_tensor(mask, device=device).to(dtype)   # (N, T)
# 
#     # 2) Add feature axis
#     mask_feat = mask_feat.unsqueeze(-1)                                 # (N, T, 1)
# 
#     # 3) Concatenate
#     masked_exp_mask_feat_quantizes = torch.cat([masked_exp_quantizes, mask_feat], dim=-1)  # (N, T, 27)
#     #print('masked_exp_mask_feat_quantizes.shape: ',masked_exp_mask_feat_quantizes.shape)
# 
#     return masked_exp_quantizes #masked_exp_mask_feat_quantizes
# =============================================================================

def train(cfg, resume_from=None):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Neptune
    neptune_run = NeptuneRun()
    neptune_run.init(project=os.getenv("NEPTUNE_PROJECT"), api_token=os.getenv("NEPTUNE_API_TOKEN"))
    run = neptune_run.run
    for k, v in vars(cfg).items():
        run[f"params/{k}"].log(v)
    if resume_from:
        run["params/resume_from"].log(resume_from)

    ckpt_mng = CheckpointManager(run)

    # Load latent data
    train_latents = np.load(cfg.train_latents)
    train_indices = np.load(cfg.train_indices)
    train_labels = np.load(cfg.train_labels)
    
    val_latents = np.load(cfg.val_latents)
    val_indices = np.load(cfg.val_indices)
    val_labels = np.load(cfg.val_labels)
    
    N, D, H, W = train_latents.shape
    train_latents = np.transpose(train_latents, (0, 2, 3, 1)).reshape(N, H * W, D)
    n_tokens = H * W
    print(f"Detected {n_tokens} tokens per sample")
    N, D, H, W = val_latents.shape
    val_latents = np.transpose(val_latents, (0, 2, 3, 1)).reshape(N, H * W, D)


    
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(train_latents, dtype=torch.float32),  # (N, T, D)
            torch.tensor(train_indices, dtype=torch.long),     # (N, T)
            torch.tensor(train_labels, dtype=torch.long),      # (N,)
        ),
        batch_size=cfg.bs,
        shuffle=True,
    )
    
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(val_latents, dtype=torch.float32),
            torch.tensor(val_indices, dtype=torch.long),
            torch.tensor(val_labels, dtype=torch.long),
        ),
        batch_size=cfg.bs,
        shuffle=False,
    )

    # Model
    model = MaskedLatentTransformer(
        embed_dim=cfg.embed_dim,
        vocab_size=cfg.vocab_size,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
        max_position_embeddings=n_tokens,
    ).to(device)

    num_training_steps = cfg.epochs * len(train_loader)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = build_scheduler(optimizer, cfg.scheduler, num_training_steps, warmup_steps=cfg.warmup_steps)
    criterion = nn.CrossEntropyLoss()

    # Resume training if requested
    if resume_from:
        local_ckpt = Path(cfg.run_dir) / f"{resume_from}.pt"
        print(f"🔁 Resuming from Neptune checkpoint: {resume_from}")
        try:
            run[f"model/{resume_from}"].download(str(local_ckpt))
            model.load_state_dict(torch.load(local_ckpt, map_location=device))
            print("✅ Checkpoint restored.")
        except Exception as e:
            print(f"⚠️ Failed to resume from checkpoint: {e}")

    best_train_loss = float("inf")
    best_val_loss = float("inf")

    # Training loop
    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        for zq, idx, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs} [Train]"):
            zq, idx, labels = zq.to(device), idx.to(device), labels.to(device)

            zq_masked, target, mask, mask_prob = mask_latents(zq, idx,0.5)
            masked_exp_train_quantizes = attach_bias_ohe(labels, zq_masked, num_classes=10)
            optimizer.zero_grad()
            logits = model(masked_exp_train_quantizes)
            if cfg.loss_masked:
                loss = criterion(logits[mask], target[mask])
            else:
                logits_flat = logits.view(-1, logits.size(-1))
                target_flat = target.view(-1)
                loss   = criterion(logits_flat, target_flat)           
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            run["train/loss"].log(loss.item())
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            run["train/lr"].log(current_lr)

        
        train_loss = total_loss / len(train_loader)
        print(f"Train Loss: {train_loss:.4f}")
        best_train_loss = ckpt_mng.save_and_upload(model, train_loss, best_train_loss, phase="train", epoch=epoch)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for zq, idx, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg.epochs} [Val]"):
                zq, idx, labels = zq.to(device), idx.to(device), labels.to(device)
                zq_masked, target, mask , _= mask_latents(zq, idx, mask_prob)
                masked_exp_train_quantizes = attach_bias_ohe(labels, zq_masked, num_classes=10)
                logits = model(masked_exp_train_quantizes)
                if cfg.loss_masked:
                    loss = criterion(logits[mask], target[mask])
                else:
                    logits_flat = logits.view(-1, logits.size(-1))
                    target_flat = target.view(-1)
                    loss   = criterion(logits_flat, target_flat)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        run["val/loss"].log(val_loss)
        print(f"Val Loss: {val_loss:.4f}")
        best_val_loss = ckpt_mng.save_and_upload(model, val_loss, best_val_loss, phase="val", epoch=epoch)

    run.stop()
    print("🏁 Training finished.")


# ------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a masked latent transformer (resumable).")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Checkpoint key to resume from: train_best, train_last, val_best, val_last.")
    args = parser.parse_args()

    cfg = get_transformer_config()
    train(cfg, resume_from=args.resume_from)



# =============================================================================
# torchrun --nproc_per_node=1 --master_port=29501 masked-transformer_train-ohe_bias.py
# 
# =============================================================================
