import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


#plot codebooks hist for each run
# =============================================================================
# aud_folders = [
#     "AUD-112", "AUD-113", "AUD-114", "AUD-115", "AUD-116",
#     "AUD-118", "AUD-119", "AUD-120", "AUD-121", "AUD-122"
# ]
# 
# root_dir = Path("/local/altamabp/audience_tuning-gem/vqvae")   # <-- change if needed
# latents_subdir = "latents"
# out_dir = Path("aud_histograms")
# out_dir.mkdir(parents=True, exist_ok=True)
# 
# def plot_hist(indices: np.ndarray, title: str, out_path: Path, bins: int = 32):
#     plt.figure(figsize=(10, 6))
#     plt.hist(indices.ravel(), bins=bins)
#     plt.title(title)
#     plt.xlabel("Codebook index")
#     plt.ylabel("Count")
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=150)
#     plt.close()
# 
# for aud in aud_folders:
#     lat_dir = root_dir / aud / latents_subdir
# 
#     train_path = lat_dir / "train_indices.npy"
#     val_path   = lat_dir / "val_indices.npy"
# 
#     if train_path.exists():
#         train_idx = np.load(train_path)
#         plot_hist(
#             train_idx,
#             f"{aud} — train_indices.npy",
#             out_dir / f"{aud}_train_indices_hist.png",
#             bins=32
#         )
#         print('max train index:', np.max(train_idx))
#         print('min train index:', np.min(train_idx))
#     else:
#         print(f"[WARN] Missing {train_path}")
# 
#     if val_path.exists():
#         val_idx = np.load(val_path)
#         plot_hist(
#             val_idx,
#             f"{aud} — val_indices.npy",
#             out_dir / f"{aud}_val_indices_hist.png",
#             bins=32
#         )
#         print('max val index:', np.max(val_idx))
#         print('min val index:', np.min(val_idx))
#     else:
#         print(f"[WARN] Missing {val_path}")
# 
# print(f"Saved plots to: {out_dir.resolve()}")
# =============================================================================


# plot classes hist to show which codebooks were used for each class
import gc
import subprocess
import sys

import argparse
ROOT = Path("/local/altamabp/audience_tuning-gem/vqvae")
OUT_DIR = Path("/home/altamabp/audience-tuning-gem/gem_audience_tuning-main/aud_histograms/histogram_codebooks_used_per_class")
OUT_DIR.mkdir(exist_ok=True)

RUNS = [
    "AUD-91", "AUD-129", "AUD-130", "AUD-131", "AUD-133",
    "AUD-134", "AUD-135", "AUD-136", "AUD-137", "AUD-138", "AUD-139",
]

latents_subdir = "latents"

SPLITS = ["train", "val"]
NUM_CLASSES = 10
VOCAB_SIZE = 32  # <-- codebook size (n_embed)

def unique_codes_per_class_fast(indices, labels, vocab_size, num_classes=10):
    # indices: (N,L) or (N,)
    if indices.ndim == 1:
        indices = indices[:, None]
    indices = indices.astype(np.int64, copy=False)

    out = np.zeros(num_classes, dtype=np.int32)
    for c in range(num_classes):
        m = (labels == c)
        if not m.any():
            continue
        idx_c = indices[m]              # memmap view
        seen = np.zeros(vocab_size, dtype=bool)
        seen[idx_c] = True              # mark used codes
        out[c] = int(seen.sum())
        del m, idx_c, seen
    return out


def worker(root, out_dir, run, split, vocab_size):
    run_dir = Path(root) / run /latents_subdir
    idx_path = run_dir / f"{split}_indices.npy"
    lbl_path = run_dir / f"{split}_labels.npy"

    if not idx_path.exists() or not lbl_path.exists():
        print(f"[WARN] Missing {split} files in {run}", flush=True)
        return 0

    # memmap to avoid loading entire arrays
    indices = np.load(idx_path, mmap_mode="r")
    labels = np.load(lbl_path, mmap_mode="r")

    uniq = unique_codes_per_class_fast(indices, labels, vocab_size, NUM_CLASSES)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8, 5))
    plt.bar(range(NUM_CLASSES), uniq)
    plt.xticks(range(NUM_CLASSES))
    plt.xlabel("Class label")
    plt.ylabel("Unique codebook indices used")
    plt.title(f"{run} — {split} unique-code usage per class")
    plt.tight_layout()

    out_path = out_dir / f"{run}_{split}_unique_codes.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved {out_path}", flush=True)

    # explicit cleanup inside worker (mostly redundant since process exits)
    del indices, labels, uniq, fig
    plt.close("all")
    gc.collect()
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root containing AUD-* folders")
    ap.add_argument("--out", default="unique_code_usage_histograms_subproc", help="Output directory")
    ap.add_argument("--vocab-size", type=int, required=True, help="Codebook size (n_embed)")
    ap.add_argument("--mode", choices=["master", "worker"], default="master")
    ap.add_argument("--run", default="")
    ap.add_argument("--split", default="")
    args = ap.parse_args()

    if args.mode == "worker":
        return worker(args.root, args.out, args.run, args.split, args.vocab_size)

    # MASTER: spawn a fresh python process per (run, split)
    script_path = Path(__file__).resolve()
    for run in RUNS:
        for split in SPLITS:
            cmd = [
                sys.executable, str(script_path),
                "--mode", "worker",
                "--root", args.root,
                "--out", args.out,
                "--vocab-size", str(args.vocab_size),
                "--run", run,
                "--split", split,
            ]
            print("Running:", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=False)

    print("Done.")


if __name__ == "__main__":
    raise SystemExit(main())
    
    
# =============================================================================
#     python unique_code_usage_subproc.py \
#   --root /local/altamabp/audience_tuning-gem/vqvae \
#   --vocab-size 32 \
#   --out unique_code_usage_histograms
# 
# =============================================================================
