import os
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import neptune.new as neptune
from torchvision.models import resnet50
from tqdm import tqdm


from flat_models import EnhancedFlatVQVAE


def download_neptune_checkpoint(run, key: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run[key].download(str(out_path))


@torch.no_grad()
def decode_latents_streaming(vqvae, latents_np, device, batch_size=16, use_amp=True):
    """
    latents_np: np.ndarray or np.memmap of shape (N, D, H, W) float
    returns: generator of (start, end, recon_images_cpu) where recon_images_cpu is (B,3,Himg,Wimg) on CPU
    """
    vqvae.eval()
    N = latents_np.shape[0]

    use_cuda = (device.type == "cuda")
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if (use_cuda and use_amp) else nullcontext()

    for s in range(0, N, batch_size):
        e = min(N, s + batch_size)
        q = torch.from_numpy(np.asarray(latents_np[s:e])).float()  # CPU
        q = q.to(device, non_blocking=True)

        with amp_ctx:
            recon = vqvae.decode(q)  # output in same range as training ([-1,1])

        recon_cpu = recon.float().cpu()
        del q, recon
        if use_cuda:
            torch.cuda.empty_cache()
        yield s, e, recon_cpu


class NormalizeMinus1To1(nn.Module):
    """
    Equivalent to torchvision.transforms.Normalize([0.5]*3, [0.5]*3)
    when the input is already in [-1,1], it maps to [-3,1] (so DON'T do that).
    In your classifier training script, you applied Normalize(0.5,0.5) directly to
    VQ-VAE outputs. :contentReference[oaicite:2]{index=2}
    So we replicate that exactly, even if it's unusual.
    """
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.5, 0.5, 0.5]).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor([0.5, 0.5, 0.5]).view(1,3,1,1))

    def forward(self, x):
        return (x - self.mean) / self.std


# small helper because torch.autocast context manager differs on CPU
from contextlib import nullcontext


def build_classifier(num_classes=10):
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


@torch.no_grad()
@torch.no_grad()
def run_classifier_and_write(
    vqvae,
    clf,
    latents_path: Path,
    out_path: Path,
    batch_size_decode: int,
    batch_size_clf: int,
    device_vqvae,
    device_clf,
    output_kind: str,  # "probs" or "logits"
    use_amp: bool,
    apply_training_normalize: bool,
):
    latents = np.load(latents_path, mmap_mode="r")  # (N,D,H,W)
    N = latents.shape[0]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mm = np.lib.format.open_memmap(
        str(out_path), mode="w+", dtype=np.float32, shape=(N, 10)
    )

    clf.eval()
    vqvae.eval()
    norm = NormalizeMinus1To1().to(device_clf) if apply_training_normalize else None

    print(f"[INFO] Processing {latents_path} -> {out_path}  N={N}")

    # progress bar tracks number of samples written
    pbar = tqdm(total=N, desc=f"Saving {out_path.name}", unit="samples")

    try:
        for s, e, recon_cpu in decode_latents_streaming(
            vqvae,
            latents,
            device_vqvae,
            batch_size=batch_size_decode,
            use_amp=use_amp,
        ):
            B = recon_cpu.shape[0]

            for ss in range(0, B, batch_size_clf):
                ee = min(B, ss + batch_size_clf)
                imgs = recon_cpu[ss:ee].to(device_clf, non_blocking=True)

                if norm is not None:
                    imgs = norm(imgs)

                if device_clf.type == "cuda" and use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = clf(imgs)
                else:
                    logits = clf(imgs)

                out = F.softmax(logits, dim=-1) if output_kind == "probs" else logits

                mm[s + ss : s + ee] = out.detach().float().cpu().numpy()

                # update progress bar by number of samples written
                pbar.update(ee - ss)

                del imgs, logits, out

            del recon_cpu
            mm.flush()

        mm.flush()
        print(f"[OK] Wrote {out_path} shape={(N,10)} dtype=float32")

    except Exception:
        # remove partial output if something fails
        pbar.close()
        if out_path.exists():
            out_path.unlink()
        raise
    finally:
        pbar.close()



# =============================================================================
#     latents = np.load(latents_path, mmap_mode="r")  # (N,D,H,W)
#     N = latents.shape[0]
#     mm = np.lib.format.open_memmap(str(out_path), mode="w+", dtype=np.float32, shape=(N, 10))
# 
# 
#     clf.eval()
#     vqvae.eval()
# 
#     norm = NormalizeMinus1To1().to(device_clf) if apply_training_normalize else None
# 
#     print(f"[INFO] Processing {latents_path} -> {out_path}  N={N}")
# 
#     # decode batches on vqvae device, then classify on clf device
#     for s, e, recon_cpu in decode_latents_streaming(vqvae, latents, device_vqvae,
#                                                     batch_size=batch_size_decode, use_amp=use_amp):
#         # recon_cpu: (B,3,H,W)
#         B = recon_cpu.shape[0]
# 
#         # classify possibly in smaller batches
#         for ss in range(0, B, batch_size_clf):
#             ee = min(B, ss + batch_size_clf)
#             imgs = recon_cpu[ss:ee].to(device_clf, non_blocking=True)
# 
#             # IMPORTANT: match classifier training normalization
#             if norm is not None:
#                 imgs = norm(imgs)
# 
#             if device_clf.type == "cuda" and use_amp:
#                 with torch.autocast(device_type="cuda", dtype=torch.float16):
#                     logits = clf(imgs)
#             else:
#                 logits = clf(imgs)
# 
#             if output_kind == "probs":
#                 out = F.softmax(logits, dim=-1)
#             else:
#                 out = logits
# 
#             mm[s+ss:s+ee] = out.detach().float().cpu().numpy()
# 
#             del imgs, logits, out
#             if device_clf.type == "cuda":
#                 torch.cuda.empty_cache()
# 
#         del recon_cpu
#         mm.flush()
# 
#     mm.flush()
#     print(f"[OK] Wrote {out_path} shape={(N,10)} dtype=float32")
# =============================================================================


def sanity_check(out_path: Path, labels_path: Path = None):
    arr = np.load(out_path, mmap_mode="r")
    print(f"[CHECK] {out_path.name}: shape={arr.shape}, dtype={arr.dtype}")

    if arr.shape[1] == 10:
        row_sums = arr.sum(axis=1)
        print(f"[CHECK] row sum: mean={row_sums.mean():.4f} std={row_sums.std():.4f} "
              f"min={row_sums.min():.4f} max={row_sums.max():.4f}")

        p = np.clip(arr, 1e-8, 1.0)
        ent = -(p * np.log(p)).sum(axis=1)
        print(f"[CHECK] entropy: mean={ent.mean():.4f} std={ent.std():.4f} min={ent.min():.4f} max={ent.max():.4f}")

    if labels_path is not None and labels_path.exists():
        y = np.load(labels_path, mmap_mode="r").astype(int)
        pred = arr.argmax(axis=1)
        acc = (pred == y).mean()
        print(f"[CHECK] top1 acc vs labels: {acc:.4f}  (if this is ~0.1, normalization is wrong)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, default="altamimi.aya/UTKFaces")
    ap.add_argument("--vqvae_run", required=True, help="Neptune run id for VQ-VAE")

    ap.add_argument("--clf_run", required=True, help="Neptune run id for classifier")
    ap.add_argument("--clf_ckpt_key", default="model/val_checkpoint_best")

    ap.add_argument("--latents_root", required=True, type=Path,
                    help="Folder containing train_latents.npy, val_latents.npy, and labels")
    ap.add_argument("--out_root", required=True, type=Path)

    ap.add_argument("--output_kind", choices=["probs", "logits"], default="probs")
    ap.add_argument("--decode_bs", type=int, default=16)
    ap.add_argument("--clf_bs", type=int, default=64)
    ap.add_argument("--amp", action="store_true")

    ap.add_argument("--device_vqvae", default="cuda:0")
    ap.add_argument("--device_clf", default="cuda:0")

    ap.add_argument("--apply_training_normalize", action="store_true",
                    help="Apply the SAME Normalize([0.5]*3,[0.5]*3) used in train_classifier.py")

    args = ap.parse_args()

    device_vqvae = torch.device(args.device_vqvae)
    device_clf = torch.device(args.device_clf)


    # Download & load VQ-VAE
    vqvae_ckpt = Path(f"/local/altamabp/audience_tuning-gem/vqvae/{args.vqvae_run}/vqvae_val_best.pt")

    vqvae = EnhancedFlatVQVAE().to(device_vqvae)
    vqvae.load_state_dict(torch.load(vqvae_ckpt, map_location=device_vqvae))
    vqvae.eval()

    # Download & load classifier
    clf_ckpt = Path("/local/altamabp/audience_tuning-gem/classifier/weights_epoch99_AUD-91.pth")

    clf = build_classifier(num_classes=10).to(device_clf)
    clf.load_state_dict(torch.load(clf_ckpt, map_location=device_clf))
    clf.eval()

    # Paths
    train_latents = args.latents_root / "train_latents.npy"
    val_latents   = args.latents_root / "val_latents.npy"
    train_labels  = args.latents_root / "train_labels.npy"
    val_labels    = args.latents_root / "val_labels.npy"

    args.out_root.mkdir(parents=True, exist_ok=True)
    train_out = args.out_root / f"train_{args.output_kind}.npy"
    val_out   = args.out_root / f"val_{args.output_kind}.npy"

    # Generate
    run_classifier_and_write(
        vqvae=vqvae, clf=clf,
        latents_path=train_latents, out_path=train_out,
        batch_size_decode=args.decode_bs, batch_size_clf=args.clf_bs,
        device_vqvae=device_vqvae, device_clf=device_clf,
        output_kind=args.output_kind, use_amp=args.amp,
        apply_training_normalize=args.apply_training_normalize,
    )
    run_classifier_and_write(
        vqvae=vqvae, clf=clf,
        latents_path=val_latents, out_path=val_out,
        batch_size_decode=args.decode_bs, batch_size_clf=args.clf_bs,
        device_vqvae=device_vqvae, device_clf=device_clf,
        output_kind=args.output_kind, use_amp=args.amp,
        apply_training_normalize=args.apply_training_normalize,
    )

    # Verify normalization / sanity
    if args.output_kind == "probs":
        sanity_check(train_out, train_labels if train_labels.exists() else None)
        sanity_check(val_out, val_labels if val_labels.exists() else None)
    else:
        print("[NOTE] output_kind=logits; row sums/entropy checks not applicable. "
              "If you want, you can convert logits->probs later.")



if __name__ == "__main__":
    main()
