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


def build_dataloaders_subclasses(cfg: Config) -> tuple[DataLoader, DataLoader, Optional[DistributedSampler]]:
    tfm = _build_transforms(cfg.input_size)

    train_dir = Path(cfg.data_root) / cfg.train_subdir  # e.g. ".../structured/0"
    val_dir   = Path(cfg.data_root) / cfg.val_subdir    # same or another folder

    train_ds = FlatImageFolder(
        root=str(train_dir),
        transform=tfm,
        label=0
    )

    val_ds = FlatImageFolder(
        root=str(val_dir),
        transform=tfm,
        label=0
    )

    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_distributed() else None
    val_sampler   = DistributedSampler(val_ds, shuffle=False) if is_distributed() else None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.bs,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
        drop_last=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.bs,
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0)
    )

    return train_loader, val_loader, train_sampler



from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

class FlatImageFolder(Dataset):
    """
    Loads images directly from a single folder (no class subfolders).
    Assigns a constant label (e.g. 0).
    """
    def __init__(self, root: str, transform=None, label: int = 0):
        self.root = Path(root)
        self.transform = transform
        self.label = label

        self.samples = sorted([
            p for p in self.root.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = default_loader(self.samples[idx])
        if self.transform:
            img = self.transform(img)
        return img, self.label
