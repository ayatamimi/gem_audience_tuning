from pathlib import Path
from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms, datasets

from configs.config import Config
from distributed import is_distributed

def _build_transforms(input_size: int):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

def build_dataloaders(cfg: Config) -> tuple[DataLoader, DataLoader, Optional[DistributedSampler]]:
    tfm = _build_transforms(cfg.input_size)

    train_dir = Path(cfg.data_root) / cfg.train_subdir
    val_dir   = Path(cfg.data_root) / cfg.val_subdir

    train_ds = datasets.ImageFolder(str(train_dir), transform=tfm)
    val_ds   = datasets.ImageFolder(str(val_dir),   transform=tfm)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_distributed() else None
    val_sampler   = DistributedSampler(val_ds,   shuffle=False) if is_distributed() else None

    train_loader = DataLoader(train_ds, batch_size=cfg.bs, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=cfg.num_workers, pin_memory=True, persistent_workers=(cfg.num_workers > 0), drop_last=True)
    val_loader = DataLoader( val_ds, batch_size=cfg.bs, shuffle=False, sampler=val_sampler, num_workers=cfg.num_workers, pin_memory=True, persistent_workers=(cfg.num_workers > 0))
    
    return train_loader, val_loader, train_sampler
