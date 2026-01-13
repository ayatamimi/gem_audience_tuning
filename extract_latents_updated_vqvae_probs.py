import os
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import neptune
from configs.config import Config
from model_def import build_model
from data import build_dataloaders


# -----------------------------
# Helpers for classifier + paths
# -----------------------------
class NormalizeMinus1To1(nn.Module):
    """
    Matches torchvision.transforms.Normalize([0.5]*3, [0.5]*3).
    Use ONLY if your classifier was trained with that normalization.
    """
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


def build_resnet50_classifier(num_classes: int = 10):
    from torchvision.models import resnet50
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class DatasetWithPaths(torch.utils.data.Dataset):
    """
    Wraps an ImageFolder-like dataset and returns (img, label, path).
    Works with:
      - torchvision.datasets.ImageFolder (has .samples or .imgs)
      - torch.utils.data.Subset of such datasets
    """
    def __init__(self, base):
        self.base = base

        # handle Subset
        self.is_subset = hasattr(base, "dataset") and hasattr(base, "indices")
        self.ds = base.dataset if self.is_subset else base
        self.indices = base.indices if self.is_subset else None

        # ImageFolder uses .samples; older may use .imgs
        self.samples = getattr(self.ds, "samples", None) or getattr(self.ds, "imgs", None)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, y = self.base[idx]
        path = ""
        if self.samples is not None:
            if self.is_subset:
                real_idx = self.indices[idx]
                path = self.samples[real_idx][0]
            else:
                path = self.samples[idx][0]
        return img, y, path


def _build_cfg_from_neptune(params: dict) -> Config:
    # Keep the same explicit mapping you already had (safer than passing all params blindly)
    return Config(**{k: v for k, v in {
        "data_root": params.get("data_root", "./data"),
        "train_subdir": params.get("train_subdir", "train"),
        "val_subdir": params.get("val_subdir", "val"),
        "input_size": int(params.get("input_size", 256)),
        "bs": int(params.get("bs", 4)),
        "epochs": int(params.get("epochs", 1)),
        "lr": float(params.get("lr", 3e-4)),
        "num_workers": int(params.get("num_workers", 4)),
        "beta": float(params.get("beta", 0.25)),
        "seed": int(params.get("seed", 42)),
        "model_type": params.get("model_type", "EnhancedFlatVQVAE"),
        "num_levels": int(params.get("num_levels", 1)),
        "codebook_size": int(params.get("codebook_size", 512)),
        "codebook_dim": int(params.get("codebook_dim", 64)),
        "embed_dim": int(params.get("embed_dim", 64)),
        "latent_channel": int(params.get("latent_channel", 144)),
        "rotation_trick": bool(params.get("rotation_trick", False)),
        "kmeans_init": bool(params.get("kmeans_init", False)),
        "decay": float(params.get("decay", 0.99)),
        "learnable_codebook": bool(params.get("learnable_codebook", False)),
        "ema_update": bool(params.get("ema_update", True)),
        "threshold_dead": None,
        "world_size": 1,
        "local_rank": 0,
        "run_dir": "./runs",
        "torch_compile": False,
    }.items()})


def extract_latents_from_neptune(
    run_id: str,
    clf_ckpt: str = None,
    num_classes: int = 10,
    apply_training_normalize: bool = False,
    amp: bool = False,
    device_vqvae: str = None,
    device_clf: str = None,
    save_full_paths: bool = False,
):
    # devices
    if device_vqvae is None:
        device_vqvae = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device_clf is None:
        device_clf = device_vqvae

    device_vqvae = torch.device(device_vqvae)
    device_clf = torch.device(device_clf)

    # Neptune
    run = neptune.init_run(with_id=run_id, project=os.getenv("NEPTUNE_PROJECT"))
    print(f"Connected to Neptune run: {run_id}")

    # pull params
    params = {}
    structure = run.get_structure()
    for k, _v in structure.get("params", {}).items():
        try:
            leaf = k.split("/")[-1]
            params[leaf] = run[f"params/{leaf}"].fetch_last()
        except Exception:
            pass

    # Build config + VQ-VAE
    cfg = _build_cfg_from_neptune(params)
    vqvae = build_model(cfg).to(device_vqvae)
    vqvae.eval()

    out_dir = Path(f"/local/altamabp/audience_tuning-gem/vqvae/{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = out_dir / "vqvae_val_best.pt"
    print("Downloading best validation checkpoint")
    run["model/val_checkpoint_best"].download(str(ckpt_path))
    vqvae.load_state_dict(torch.load(ckpt_path, map_location=device_vqvae))

    # Build dataloaders (then wrap datasets so we can fetch file names)
    train_loader, val_loader, _ = build_dataloaders(cfg)

    # Rebuild loaders with DatasetWithPaths and shuffle=False to make ordering stable + name-aligned
    def rebuild_loader(loader):
        ds = DatasetWithPaths(loader.dataset)
        return torch.utils.data.DataLoader(
            ds,
            batch_size=loader.batch_size,
            shuffle=False,
            num_workers=getattr(loader, "num_workers", 0),
            pin_memory=getattr(loader, "pin_memory", False),
            drop_last=getattr(loader, "drop_last", False),
            persistent_workers=getattr(loader, "persistent_workers", False),
        )

    train_loader = rebuild_loader(train_loader)
    val_loader = rebuild_loader(val_loader)

    # Optional classifier
    clf = None
    norm = None
    if clf_ckpt is not None and str(clf_ckpt).strip() != "":
        print(f"Loading classifier checkpoint: {clf_ckpt}")
        clf = build_resnet50_classifier(num_classes=num_classes).to(device_clf)
        clf.load_state_dict(torch.load(clf_ckpt, map_location=device_clf))
        clf.eval()
        norm = NormalizeMinus1To1().to(device_clf) if apply_training_normalize else None

    # Extraction loop
    def extract(loader, desc: str):
        all_latents, all_indices, all_labels = [], [], []
        all_probs = []  # (N, num_classes)
        all_logits= []
        all_names = []

        use_amp = (amp and device_clf.type == "cuda")

        for _k, batch in tqdm(enumerate(loader), total=len(loader), desc=desc):
            images, batch_labels, batch_paths = batch
            images = images.to(device_vqvae, non_blocking=True)

            with torch.no_grad():
                quant_b, _, id_b = vqvae.encode(images)

            all_latents.append(quant_b.detach().cpu().numpy())
            all_indices.append(id_b.detach().cpu().numpy())
            all_labels.append(batch_labels.cpu().numpy())

            # file names
            if save_full_paths:
                names = [str(p) for p in batch_paths]
            else:
                names = [os.path.basename(str(p)) for p in batch_paths]
            all_names.extend(names)

            # classifier probs on the VQ-VAE RECONSTRUCTED images (decoded from latents)
            if clf is not None:
                # 1) decode latents -> reconstructed images (range matches VQ-VAE training, typically [-1, 1])
                use_amp_vqvae = (amp and device_vqvae.type == "cuda")
                with torch.no_grad():
                    if use_amp_vqvae:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            recon = vqvae.decode(quant_b)
                    else:
                        recon = vqvae.decode(quant_b)

                # 2) run classifier on reconstructions
                imgs_clf = recon.to(device_clf, non_blocking=True)
                if norm is not None:
                    imgs_clf = norm(imgs_clf)

                with torch.no_grad():
                    if use_amp:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            logits = clf(imgs_clf)
                    else:
                        logits = clf(imgs_clf)
                    probs = F.softmax(logits, dim=-1)

                all_probs.append(probs.detach().float().cpu().numpy())
                all_logits.append(logits.detach().float().cpu().numpy())
                
                del recon, imgs_clf, logits, probs

            del images, quant_b, id_b
            if device_vqvae.type == "cuda" or device_clf.type == "cuda":
                torch.cuda.empty_cache()

        latents = np.concatenate(all_latents, axis=0)
        indices = np.concatenate(all_indices, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        names_arr = np.array(all_names, dtype=object)

        probs_arr = None
        logits_arr= None
        if clf is not None:
            probs_arr = np.concatenate(all_probs, axis=0)
            logits_arr = np.concatenate(all_logits, axis=0)

        return latents, indices, labels, probs_arr, logits_arr, names_arr

    latents_dir = out_dir / "latents"
    latents_dir.mkdir(parents=True, exist_ok=True)

    # VAL
    print("Extracting validation latents, labels, probs, and image names...")
    val_latents, val_indices, val_labels, val_probs, val_logits, val_names = extract(val_loader, "VAL")
    np.save(latents_dir / "val_latents.npy", val_latents)
    np.save(latents_dir / "val_indices.npy", val_indices)
    np.save(latents_dir / "val_labels.npy", val_labels)
    np.save(latents_dir / "val_img_names.npy", val_names)
    if val_probs is not None:
        np.save(latents_dir / "val_probs.npy", val_probs)

    # TRAIN
    print("Extracting training latents, labels, probs, and image names...")
    train_latents, train_indices, train_labels, train_probs, train_logits, train_names = extract(train_loader, "TRAIN")
    np.save(latents_dir / "train_latents.npy", train_latents)
    np.save(latents_dir / "train_indices.npy", train_indices)
    np.save(latents_dir / "train_labels.npy", train_labels)
    np.save(latents_dir / "train_img_names.npy", train_names)
    if train_probs is not None:
        np.save(latents_dir / "train_probs.npy", train_probs)

    print(f"Saved all outputs under {latents_dir}")
    if clf is None:
        print("[NOTE] classifier not provided => train_probs/val_probs were NOT saved. "
              "Pass --clf_ckpt to enable probs extraction.")

    run.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract latent representations (+ optional classifier probs and image names) from a trained VQ-VAE Neptune run.")
    parser.add_argument("--run_id", type=str, default="AUD-184", help="Neptune run ID")

    # classifier options
    parser.add_argument("--clf_ckpt", type=str, default="", help="Path to classifier .pth checkpoint (ResNet-50 head). If empty, probs aren't extracted.")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--apply_training_normalize", action="store_true",
                        help="Apply Normalize([0.5]*3,[0.5]*3) before classifier (ONLY if you trained that way).")
    parser.add_argument("--amp", action="store_true", help="Use autocast for classifier forward on CUDA.")

    # device options
    parser.add_argument("--device_vqvae", type=str, default=None, help="e.g. cuda:0 or cpu")
    parser.add_argument("--device_clf", type=str, default=None, help="e.g. cuda:1 or cpu")

    # naming options
    parser.add_argument("--save_full_paths", action="store_true",
                        help="Save full paths in *_img_names.npy instead of basenames.")

    args = parser.parse_args()

    extract_latents_from_neptune(
        run_id=args.run_id,
        clf_ckpt=args.clf_ckpt,
        num_classes=args.num_classes,
        apply_training_normalize=args.apply_training_normalize,
        amp=args.amp,
        device_vqvae=args.device_vqvae,
        device_clf=args.device_clf,
        save_full_paths=args.save_full_paths,
    )
