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
from masked_transformer.model import MaskedLatentTransformer
from masked_transformer.utils import mask_latents, build_scheduler


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
    val_latents = np.load(cfg.val_latents)
    val_indices = np.load(cfg.val_indices)

    N, D, H, W = train_latents.shape
    train_latents = np.transpose(train_latents, (0, 2, 3, 1)).reshape(N, H * W, D)
    n_tokens = H * W
    print(f"Detected {n_tokens} tokens per sample")
    N, D, H, W = val_latents.shape
    val_latents = np.transpose(val_latents, (0, 2, 3, 1)).reshape(N, H * W, D)

    train_loader = DataLoader(TensorDataset(
        torch.tensor(train_latents, dtype=torch.float32),
        torch.tensor(train_indices, dtype=torch.long)), batch_size=cfg.bs, shuffle=True)
    val_loader = DataLoader(TensorDataset(
        torch.tensor(val_latents, dtype=torch.float32),
        torch.tensor(val_indices, dtype=torch.long)), batch_size=cfg.bs, shuffle=False)


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
        for zq, idx in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs} [Train]"):
            zq, idx = zq.to(device), idx.to(device)
            zq_masked, target, mask = mask_latents(zq, idx, cfg.mask_prob)
            optimizer.zero_grad()
            logits = model(zq_masked)
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
            for zq, idx in tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg.epochs} [Val]"):
                zq, idx = zq.to(device), idx.to(device)
                zq_masked, target, mask = mask_latents(zq, idx, cfg.mask_prob)
                logits = model(zq_masked)
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
