from pathlib import Path
from typing import Tuple, Optional
import re

from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms, datasets

from configs.config import Config
from distributed import is_distributed

from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision.datasets.folder import default_loader

IMG_EXTS = {".jpg", ".jpeg", ".png"}

def _infer_label_from_dirname(name: str) -> int:
    # handles "0", "0.", "7.", "class_3" (last integer found)
    m = re.findall(r"\d+", name)
    if not m:
        raise ValueError(f"Could not infer numeric label from folder name: {name!r}")
    return int(m[-1])

class SubdirClassImageFolder(Dataset):
    """
    Like ImageFolder, but:
      - does NOT require strict ImageFolder structure below class dirs
      - infers label from class dir name (e.g. '0', '0.')
    Expects:
      root/
        0/  (or 0.)
          img1.jpg
        1/
          img2.jpg
        ...
    """
    def __init__(self, root: str, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {root}")

        # each immediate subdir is a class
        class_dirs = [p for p in self.root.iterdir() if p.is_dir()]
        for cdir in sorted(class_dirs):
            label = _infer_label_from_dirname(cdir.name)
            for p in sorted(cdir.rglob("*")):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    self.samples.append((p, label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under class subfolders in {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = default_loader(path)
        if self.transform:
            img = self.transform(img)
        return img, y


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


def build_dataloaders_subclasses(cfg: Config):
    tfm = _build_transforms(cfg.input_size)

    train_dir = Path(cfg.data_root) / cfg.train_subdir
    val_dir   = Path(cfg.data_root) / cfg.val_subdir

    def _has_images_directly(d: Path) -> bool:
        return any(p.is_file() and p.suffix.lower() in IMG_EXTS for p in d.iterdir())

    # If images are directly in folder -> FlatImageFolder with inferred label
    # If folder contains class subfolders -> SubdirClassImageFolder (multi-class)
    if _has_images_directly(train_dir):
        train_label = _infer_label_from_dirname(train_dir.name)
        train_ds = FlatImageFolder(str(train_dir), transform=tfm, label=train_label)
    else:
        train_ds = SubdirClassImageFolder(str(train_dir), transform=tfm)

    if _has_images_directly(val_dir):
        val_label = _infer_label_from_dirname(val_dir.name)
        val_ds = FlatImageFolder(str(val_dir), transform=tfm, label=val_label)
    else:
        val_ds = SubdirClassImageFolder(str(val_dir), transform=tfm)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_distributed() else None
    val_sampler   = DistributedSampler(val_ds, shuffle=False) if is_distributed() else None

    train_loader = DataLoader(
        train_ds, batch_size=cfg.bs, shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=cfg.num_workers, pin_memory=True, persistent_workers=(cfg.num_workers > 0),
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.bs, shuffle=False, sampler=val_sampler,
        num_workers=cfg.num_workers, pin_memory=True, persistent_workers=(cfg.num_workers > 0)
    )
    return train_loader, val_loader, train_sampler


from torch.utils.data import Dataset


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