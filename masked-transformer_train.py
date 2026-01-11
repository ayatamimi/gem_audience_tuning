import os
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import neptune

from logging_neptune import CheckpointManager, NeptuneRun
from distributed import set_seed
from configs.config import get_transformer_config
from masked_transformer.model import MaskedLatentTransformer
from masked_transformer.utils import mask_latents, build_scheduler


class LatentsNPYDataset(torch.utils.data.Dataset):
    """
    Memory-mapped dataset:
      latents: (N, D, H, W) float
      indices: (N, T) long  where T=H*W
      labels:  (N,) long
      probs:   (N, num_classes) float (optional) - classifier soft outputs or logits
    """
    def __init__(self, latents_path, indices_path, labels_path, probs_path=None, num_classes=10):
        self.latents = np.load(latents_path, mmap_mode="r")
        self.indices = np.load(indices_path, mmap_mode="r")
        self.labels  = np.load(labels_path,  mmap_mode="r")

        self.probs = None
        if probs_path is not None and str(probs_path).strip() != "":
            self.probs = np.load(probs_path, mmap_mode="r")
            if self.probs.ndim != 2 or self.probs.shape[1] != num_classes:
                raise ValueError(
                    f"probs must be (N,{num_classes}); got {self.probs.shape} from {probs_path}"
                )

        if self.latents.ndim != 4:
            raise ValueError(f"latents must be (N,D,H,W), got {self.latents.shape}")
        N, D, H, W = self.latents.shape
        self.N, self.D, self.H, self.W = N, D, H, W
        self.T = H * W
        self.num_classes = num_classes

        # basic sanity checks
        if self.indices.shape[0] != N:
            raise ValueError(f"indices N mismatch: {self.indices.shape[0]} vs {N}")
        if self.labels.shape[0] != N:
            raise ValueError(f"labels N mismatch: {self.labels.shape[0]} vs {N}")
        if self.indices.shape[1] != self.T:
            raise ValueError(f"indices T mismatch: {self.indices.shape[1]} vs H*W={self.T}")

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        z = self.latents[i]  # (D,H,W)
        # (D,H,W) -> (H,W,D) -> (T,D)
        z = np.transpose(z, (1, 2, 0)).reshape(self.T, self.D)

        z_t = torch.from_numpy(z).float()
        idx_t = torch.from_numpy(self.indices[i]).long()
        y_t = torch.tensor(int(self.labels[i])).long()

        if self.probs is None:
            return z_t, idx_t, y_t, None
        p = torch.from_numpy(np.asarray(self.probs[i], dtype=np.float32))  # (num_classes,)
        return z_t, idx_t, y_t, p


def _collate_with_optional_probs(batch):
    """
    batch items: (z, idx, y, p_or_none)
    Returns:
      z: (B,T,D)
      idx: (B,T)
      y: (B,)
      probs: (B,C) or None
    """
    zs, idxs, ys, ps = zip(*batch)
    z = torch.stack(zs, dim=0)
    idx = torch.stack(idxs, dim=0)
    y = torch.stack(ys, dim=0)

    if ps[0] is None:
        probs = None
    else:
        probs = torch.stack(ps, dim=0)
    return z, idx, y, probs


def train(cfg, resume_from=None):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Defaults for backward compatibility
    num_classes = getattr(cfg, "num_classes", 10)
    label_dim = getattr(cfg, "label_dim", 8)
    p_use_probs = float(getattr(cfg, "p_use_probs", 0.0))
    train_probs_path = getattr(cfg, "train_probs", None)
    val_probs_path = getattr(cfg, "val_probs", None)

    # Neptune
    neptune_run = NeptuneRun()
    neptune_run.init(project=os.getenv("NEPTUNE_PROJECT"), api_token=os.getenv("NEPTUNE_API_TOKEN"))
    run = neptune_run.run
    for k, v in vars(cfg).items():
        run[f"params/{k}"].log(v)
    run["params/num_classes"].log(num_classes)
    run["params/label_dim"].log(label_dim)
    run["params/p_use_probs"].log(p_use_probs)

    if resume_from:
        run["params/resume_from"].log(resume_from)

    ckpt_mng = CheckpointManager(run)

    # Datasets (mmap)
    train_ds = LatentsNPYDataset(cfg.train_latents, cfg.train_indices, cfg.train_labels,
                                probs_path=train_probs_path, num_classes=num_classes)
    val_ds   = LatentsNPYDataset(cfg.val_latents, cfg.val_indices, cfg.val_labels,
                                probs_path=val_probs_path, num_classes=num_classes)

    n_tokens = train_ds.T
    latent_dim = train_ds.D #latent_dim=D (the last dimension of zq_masked
    print(f"Detected tokens per sample: {n_tokens} (H*W), latent_dim: {latent_dim}")
    run["params/latent_dim"].log(latent_dim)
    run["params/n_tokens"].log(n_tokens)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.bs,
        shuffle=True,
        num_workers=getattr(cfg, "num_workers", 0),
        pin_memory=True,
        persistent_workers=(getattr(cfg, "num_workers", 0) > 0),
        collate_fn=_collate_with_optional_probs,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.bs,
        shuffle=False,
        num_workers=getattr(cfg, "num_workers", 0),
        pin_memory=True,
        persistent_workers=(getattr(cfg, "num_workers", 0) > 0),
        collate_fn=_collate_with_optional_probs,
        drop_last=False,
    )

    # Model (NEW API)
    model = MaskedLatentTransformer(
        latent_dim=latent_dim,
        vocab_size=cfg.vocab_size,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
        max_position_embeddings=n_tokens,
        num_classes=num_classes,
        label_dim=label_dim,
    ).to(device)

    num_training_steps = cfg.epochs * len(train_loader)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = build_scheduler(optimizer, cfg.scheduler, num_training_steps, warmup_steps=cfg.warmup_steps)
    criterion = nn.CrossEntropyLoss()

    # Resume training if requested
    if resume_from:
        local_ckpt = Path(cfg.run_dir) / f"{resume_from}.pt"
        print(f" Resuming from Neptune checkpoint: {resume_from}")
        try:
            run[f"model/{resume_from}"].download(str(local_ckpt))
            model.load_state_dict(torch.load(local_ckpt, map_location=device))
            print(" Checkpoint restored.")
        except Exception as e:
            print(f" Failed to resume from checkpoint: {e}")

    best_train_loss = float("inf")
    best_val_loss = float("inf")

    def _choose_conditioning(labels, probs):
        """
        Returns kwargs to pass to model: either {'labels': labels} or {'probs': probs}
        """
        if probs is None:
            return {"labels": labels}
        # mixed mode
        if p_use_probs <= 0.0:
            return {"labels": labels}
        if p_use_probs >= 1.0:
            return {"probs": probs}
        use_probs = (torch.rand((), device=labels.device) < p_use_probs)
        return {"probs": probs} if bool(use_probs) else {"labels": labels}

    for epoch in range(cfg.epochs):
        # --- Train ---
        model.train()
        total_loss = 0.0

        for zq, idx, labels, probs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs} [Train]"):
            zq = zq.to(device, non_blocking=True)
            idx = idx.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if probs is not None:
                probs = probs.to(device, non_blocking=True)

            zq_masked, target, mask, _ = mask_latents(zq, idx, cfg.mask_prob)

            optimizer.zero_grad(set_to_none=True)

            cond_kwargs = _choose_conditioning(labels, probs)
            logits = model(zq_masked, **cond_kwargs)

            if cfg.loss_masked:
                loss = criterion(logits[mask], target[mask])
            else:
                loss = criterion(logits.view(-1, logits.size(-1)), target.view(-1))

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += float(loss.item())
            run["train/loss"].log(float(loss.item()))
            run["train/lr"].log(float(optimizer.param_groups[0]["lr"]))

        train_loss = total_loss / len(train_loader)
        print(f"Train Loss: {train_loss:.4f}")
        best_train_loss = ckpt_mng.save_and_upload(model, train_loss, best_train_loss, phase="train", epoch=epoch)

        # --- Val ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for zq, idx, labels, probs in tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg.epochs} [Val]"):
                zq = zq.to(device, non_blocking=True)
                idx = idx.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                if probs is not None:
                    probs = probs.to(device, non_blocking=True)

                zq_masked, target, mask, _ = mask_latents(zq, idx, cfg.mask_prob)

                cond_kwargs = _choose_conditioning(labels, probs)
                logits = model(zq_masked, **cond_kwargs)

                if cfg.loss_masked:
                    loss = criterion(logits[mask], target[mask])
                else:
                    loss = criterion(logits.view(-1, logits.size(-1)), target.view(-1))

                val_loss += float(loss.item())

        val_loss /= len(val_loader)
        run["val/loss"].log(val_loss)
        print(f"Val Loss: {val_loss:.4f}")
        best_val_loss = ckpt_mng.save_and_upload(model, val_loss, best_val_loss, phase="val", epoch=epoch)

    run.stop()
    print(" Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a masked latent transformer (labels/probs conditioning).")
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Checkpoint key to resume from: train_best, train_last, val_best, val_last.",
    )
    args = parser.parse_args()

    cfg = get_transformer_config()
    train(cfg, resume_from=args.resume_from)
