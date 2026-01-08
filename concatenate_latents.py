#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np

FILES = [
    "train_indices.npy",
    "train_labels.npy",
    "train_latents.npy",
    "val_indices.npy",
    "val_labels.npy",
    "val_latents.npy",
]


import gc

def close_mmap(arr):
    """Best-effort close for np.memmap loaded via np.load(..., mmap_mode=...)."""
    try:
        mm = getattr(arr, "_mmap", None)
        if mm is not None:
            mm.close()
    except Exception:
        pass

def npy_exists(p: Path) -> bool:
    return p.exists() and p.is_file()

def load_npy_mmap(path: Path) -> np.ndarray:
    if not npy_exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    # mmap_mode keeps RAM low
    return np.load(path, mmap_mode="r", allow_pickle=False)

def compute_total_and_refshape(aud_dirs, fn: str, strict_shapes: bool):
    total0 = 0
    ref_shape = None
    ref_dtype = None
    for lat_dir in aud_dirs:
        arr = load_npy_mmap(lat_dir / fn)
        if ref_shape is None:
            ref_shape = arr.shape
            ref_dtype = arr.dtype
        else:
            if strict_shapes:
                if arr.ndim != len(ref_shape) or arr.shape[1:] != ref_shape[1:]:
                    raise ValueError(
                        f"Shape mismatch for {fn} in {lat_dir}:\n"
                        f"  got {arr.shape}\n"
                        f"  expected (*, {ref_shape[1:]}) based on first file {ref_shape}"
                    )
            # dtype mismatch can silently upcast; better to fail early
            if arr.dtype != ref_dtype:
                raise ValueError(
                    f"Dtype mismatch for {fn} in {lat_dir}:\n"
                    f"  got {arr.dtype}\n"
                    f"  expected {ref_dtype}"
                )
        total0 += arr.shape[0]
        # drop reference to memmap promptly
        del arr
    return total0, ref_shape, ref_dtype

def stream_concat_to_memmap(aud_dirs, fn: str, out_path: Path, strict_shapes: bool):
    total0, ref_shape, ref_dtype = compute_total_and_refshape(aud_dirs, fn, strict_shapes)
    if total0 == 0:
        print(f"[WARN] total size is zero for {fn}, skipping.")
        return

    out_shape = (total0, *ref_shape[1:])
    print(f"[INFO] Writing {fn}: out_shape={out_shape}, dtype={ref_dtype}")

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    out_mm = None
    try:
        out_mm = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=ref_dtype, shape=out_shape)

        offset = 0
        for lat_dir in aud_dirs:
            in_path = lat_dir / fn
            arr = None
            try:
                arr = load_npy_mmap(in_path)  # np.memmap
                n = arr.shape[0]
                out_mm[offset:offset + n] = arr
                offset += n
            finally:
                if arr is not None:
                    close_mmap(arr)
                    del arr

        out_mm.flush()

    except Exception:
        # If anything fails, don't leave a partial/corrupt tmp behind
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        raise

    finally:
        if out_mm is not None:
            # Ensure output memmap is closed promptly
            close_mmap(out_mm)
            del out_mm
        gc.collect()

    tmp_path.replace(out_path)
    print(f"[OK] Saved {fn} -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Path containing AUD-xxx folders")
    ap.add_argument(
        "--aud",
        nargs="+",
        default=[
            "AUD-129", "AUD-130", "AUD-131", "AUD-133", "AUD-134",
            "AUD-135", "AUD-136", "AUD-137", "AUD-138", "AUD-139",
        ],
        help="AUD folder names to include",
    )
    ap.add_argument("--latents-subdir", type=str, default="latents", help="Subfolder containing npy files")
    ap.add_argument("--out", type=str, required=True, help="Output directory")
    ap.add_argument("--strict-shapes", action="store_true", help="Enforce matching non-first dims across files")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    aud_dirs = []
    for name in args.aud:
        p = root / name / args.latents_subdir
        if not p.exists():
            raise FileNotFoundError(f"Latents folder not found: {p}")
        aud_dirs.append(p)

    print("Using folders:")
    for p in aud_dirs:
        print(" -", p)

    for fn in FILES:
        # Skip missing globally if none exist
        any_exist = any((d / fn).exists() for d in aud_dirs)
        if not any_exist:
            print(f"[WARN] {fn} not found in any folder, skipping.")
            continue

        out_path = out_dir / fn
        stream_concat_to_memmap(aud_dirs, fn, out_path, args.strict_shapes)

    print("Done.")

if __name__ == "__main__":
    main()


# =============================================================================
# python concatenate_latents.py \
#   --root /local/altamabp/audience_tuning-gem/vqvae \
#   --out  /local/altamabp/audience_tuning-gem/vqvae/AUD-concatenated_frozen-enc-dec \
#   --strict-shapes
# =============================================================================

