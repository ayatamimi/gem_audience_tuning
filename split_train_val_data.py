"""
Split an age-progression dataset into:
  out_root/train_data
  out_root/val_data

Rules enforced:
1) Split is by "group id" (identity/progression track), NOT by individual images.
2) A group id is everything before the LAST number in the filename (your rule):
     99_..._agegroup_0.png -> group id: 99_..._agegroup_
3) Every group placed into train or val must contain ALL classes 0..9 exactly (at least 1 each).
4) Train and val are disjoint.
5) Output keeps the original class subfolders (0..9) if your input is arranged like that,
   but it also works if all images are in one folder (it will create 0..9 in outputs).

Usage:
  python split_age_groups.py --src /path/to/original --out /path/to/output --val-frac 0.05 --seed 42

Expected input layouts supported:
A) src/
     0/  *.png
     1/  *.png
     ...
     9/  *.png
B) src/
     *.png   (mixed classes in filenames)

The class is taken from the LAST number in the filename (0..9).
"""

from __future__ import annotations
import argparse
import random
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Set

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Captures: (prefix_before_last_number)(last_number_0_to_9)(extension)
# Example: "99_..._agegroup_0.png" -> prefix="99_..._agegroup_", cls="0"
LAST_DIGIT_RE = re.compile(r"^(.*?)([0-9])(\.[^.]+)$")

# Optional: recognize class directories named "0".."9" or "class0".."class9"
CLASS_DIR_RE = re.compile(r"^(?:class)?([0-9])$")


def parse_group_and_class(filename: str) -> Tuple[str, int]:
    """
    Group id = everything before the LAST number (0..9) in the *basename* (excluding extension).
    Class     = that last number.

    This matches your example: ..._agegroup_0.png ..._agegroup_9.png
    """
    m = LAST_DIGIT_RE.match(filename)
    if not m:
        raise ValueError(f"Filename does not match pattern '*<digit>.<ext>': {filename}")
    prefix, digit, ext = m.group(1), m.group(2), m.group(3)
    cls = int(digit)
    if cls < 0 or cls > 9:
        raise ValueError(f"Last digit is not in 0..9 for: {filename}")
    return prefix, cls


def iter_images(src_root: Path) -> List[Path]:
    return [p for p in src_root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]


def detect_class_from_parent_dir(p: Path) -> int | None:
    """
    If file is under src_root/<class>/..., detect class from that folder name.
    Returns None if not in a class folder.
    """
    parent = p.parent.name
    m = CLASS_DIR_RE.match(parent)
    if m:
        return int(m.group(1))
    return None


def build_index(src_root: Path) -> Dict[str, Dict[int, List[Path]]]:
    """
    index[group_id][class] = [paths...]
    Class is primarily the LAST digit in filename, but if there is a class folder,
    we sanity check it matches.
    """
    index: Dict[str, Dict[int, List[Path]]] = {}

    for p in iter_images(src_root):
        group_id, cls_from_name = parse_group_and_class(p.name)

        cls_from_dir = detect_class_from_parent_dir(p)
        if cls_from_dir is not None and cls_from_dir != cls_from_name:
            raise RuntimeError(
                f"Class mismatch for {p}: folder says {cls_from_dir}, filename says {cls_from_name}"
            )

        index.setdefault(group_id, {}).setdefault(cls_from_name, []).append(p)

    return index


def keep_only_complete_groups(index: Dict[str, Dict[int, List[Path]]]) -> Tuple[Dict[str, Dict[int, List[Path]]], List[str]]:
    """
    Keep only groups that have at least one image for every class 0..9.
    Returns (kept, dropped_group_ids)
    """
    kept: Dict[str, Dict[int, List[Path]]] = {}
    dropped: List[str] = []

    for gid, cls_map in index.items():
        if all(c in cls_map and len(cls_map[c]) > 0 for c in range(10)):
            kept[gid] = cls_map
        else:
            dropped.append(gid)

    return kept, dropped


def split_groups(group_ids: List[str], val_frac: float, seed: int, val_count: int | None) -> Tuple[Set[str], Set[str]]:
    rnd = random.Random(seed)
    rnd.shuffle(group_ids)

    if val_count is not None:
        n_val = int(val_count)
    else:
        n_val = max(1, int(round(len(group_ids) * val_frac)))

    n_val = min(n_val, len(group_ids) - 1)  # keep at least 1 in train
    val = set(group_ids[:n_val])
    train = set(group_ids[n_val:])
    return train, val


def ensure_empty_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_group_to_split(
    group_id: str,
    cls_map: Dict[int, List[Path]],
    split_root: Path,
    copy_files: bool,
) -> None:
    """
    Writes files into split_root/<class>/filename
    """
    for cls, paths in cls_map.items():
        out_dir = split_root / str(cls)
        out_dir.mkdir(parents=True, exist_ok=True)

        for src in paths:
            dst = out_dir / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if copy_files:
                shutil.copy2(src, dst)
            else:
                shutil.move(str(src), str(dst))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path, help="Original dataset root folder")
    ap.add_argument("--out", required=True, type=Path, help="Output root folder")
    ap.add_argument("--val-frac", type=float, default=0.1, help="Validation fraction by group-id")
    ap.add_argument("--val-count", type=int, default=None, help="Validation group count (overrides --val-frac)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy", action="store_true", help="Copy files (default). If not set, files are MOVED.")
    ap.add_argument("--move", action="store_true", help="Move files instead of copying.")
    args = ap.parse_args()

    src_root: Path = args.src
    out_root: Path = args.out
    val_frac: float = args.val_frac
    seed: int = args.seed
    val_count: int | None = args.val_count

    if not src_root.exists():
        raise SystemExit(f"Source folder not found: {src_root}")

    if args.copy and args.move:
        raise SystemExit("Choose only one of --copy or --move.")
    copy_files = True
    if args.move:
        copy_files = False

    train_root = out_root / "train_data"
    val_root = out_root / "val_data"
    ensure_empty_dir(train_root)
    ensure_empty_dir(val_root)

    index = build_index(src_root)
    kept, dropped = keep_only_complete_groups(index)

    if not kept:
        raise SystemExit("No complete groups (0..9) found. Check filename pattern / data.")

    group_ids = sorted(kept.keys())
    train_groups, val_groups = split_groups(group_ids, val_frac=val_frac, seed=seed, val_count=val_count)

    # Write splits
    for gid in train_groups:
        copy_group_to_split(gid, kept[gid], train_root, copy_files=copy_files)

    for gid in val_groups:
        copy_group_to_split(gid, kept[gid], val_root, copy_files=copy_files)

    # Summary
    n_train_imgs = sum(len(p) for gid in train_groups for p in kept[gid].values())
    n_val_imgs = sum(len(p) for gid in val_groups for p in kept[gid].values())

    print("Done.")
    print(f"Total groups found: {len(index)}")
    print(f"Complete groups kept (0..9): {len(kept)}")
    print(f"Dropped incomplete groups: {len(dropped)}")
    print(f"Train groups: {len(train_groups)} | Train images: {n_train_imgs}")
    print(f"Val groups:   {len(val_groups)} | Val images:   {n_val_imgs}")
    print(f"Output written to: {out_root}")
    if dropped:
        # Print a small sample so you can inspect naming issues
        sample = dropped[:10]
        print(f"Sample dropped group_ids (missing some classes): {sample}")


if __name__ == "__main__":
    main()
