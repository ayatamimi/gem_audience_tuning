from pathlib import Path
from typing import Optional, Dict, Union
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# # ---------------- CONFIG ----------------
# ROOT_TMPL = Path("/local/altamabp/audience_tuning-gem/vqvae/{run}/latents")
# OUT_DIR = Path("/home/altamabp/audience-tuning-gem/gem_audience_tuning-main/aud_histograms/heat_map_per_class/histogram_codebooks_used_per_class")
# 
# NUM_CLASSES = 10
# CODEBOOK_SIZE = 32
# 
# RUN_INFO: Dict[str, Union[str, int]] = {
#     "AUD-91": "all",  # trained on all classes
#     "AUD-129": 0,
#     "AUD-130": 1,
#     "AUD-131": 2,
#     "AUD-133": 3,
#     "AUD-134": 4,
#     "AUD-135": 5,
#     "AUD-136": 6,
#     "AUD-137": 7,
#     "AUD-138": 8,
#     "AUD-139": 9,
# }
# # ---------------------------------------
# 
# 
# def indices_and_labels_to_class_dict(indices: np.ndarray, labels: np.ndarray, num_classes: int):
#     """
#     indices: (N,) or (N, T) or (N, ...)
#     labels:  (N,)
#     returns: dict[class] -> set(unique codebook indices)
#     """
#     indices = np.asarray(indices)
#     labels = np.asarray(labels).astype(int)
# 
#     if labels.ndim != 1:
#         raise ValueError(f"labels must be 1D (N,), got {labels.shape}")
#     if indices.shape[0] != labels.shape[0]:
#         raise ValueError(f"indices/labels mismatch: indices={indices.shape} labels={labels.shape}")
# 
#     flat = indices.reshape(indices.shape[0], -1) if indices.ndim > 1 else indices.reshape(-1, 1)
# 
#     out = {c: set() for c in range(num_classes)}
#     for cls in range(num_classes):
#         m = labels == cls
#         if np.any(m):
#             out[cls].update(map(int, np.unique(flat[m])))
#     return out
# 
# 
# def build_usage_matrix(indices_per_class, num_classes, codebook_size):
#     usage = np.zeros((num_classes, codebook_size), dtype=int)
#     for cls in range(num_classes):
#         for idx in indices_per_class.get(cls, []):
#             idx = int(idx)
#             if 0 <= idx < codebook_size:
#                 usage[cls, idx] = 1
#     return usage
# 
# 
# def mask_to_trained_class(usage: np.ndarray, trained_class: Optional[int]):
#     """
#     If trained_class is not None, keep only that class row and zero out others.
#     If None, keep all rows (AUD-91 case).
#     """
#     if trained_class is None:
#         return usage
#     masked = np.zeros_like(usage)
#     masked[trained_class, :] = usage[trained_class, :]
#     return masked
# 
# 
# def plot_heatmap(usage, title, save_path: Optional[Path] = None):
#     plt.figure(figsize=(16, 4))
#     plt.imshow(usage, aspect="auto", interpolation="nearest")
#     plt.colorbar(label="Used (1 = yes)")
#     plt.xlabel("Codebook index")
#     plt.ylabel("Class label")
#     plt.title(title)
#     plt.yticks(range(usage.shape[0]))
#     plt.tight_layout()
#     if save_path is not None:
#         save_path.parent.mkdir(parents=True, exist_ok=True)
#         plt.savefig(save_path, dpi=200)
#         plt.close()
#     else:
#         plt.show()
# 
# 
# def process_run(run_id: str, trained_on: Union[str, int]):
#     latents_dir = Path(str(ROOT_TMPL).format(run=run_id))
#     if not latents_dir.exists():
#         print(f"[WARN] Missing folder: {latents_dir}")
#         return
# 
#     # Explicit filenames (per your folder contents)
#     train_indices_path = latents_dir / "train_indices.npy"
#     train_labels_path  = latents_dir / "train_labels.npy"
#     val_indices_path   = latents_dir / "val_indices.npy"
#     val_labels_path    = latents_dir / "val_labels.npy"
# 
#     missing = [p for p in [train_indices_path, train_labels_path, val_indices_path, val_labels_path] if not p.exists()]
#     if missing:
#         print(f"[WARN] {run_id}: missing files: {', '.join(map(str, missing))}")
#         return
# 
#     train_indices = np.load(train_indices_path, allow_pickle=True)
#     train_labels  = np.load(train_labels_path,  allow_pickle=True)
#     val_indices   = np.load(val_indices_path,   allow_pickle=True)
#     val_labels    = np.load(val_labels_path,    allow_pickle=True)
# 
#     train_dict = indices_and_labels_to_class_dict(train_indices, train_labels, NUM_CLASSES)
#     val_dict   = indices_and_labels_to_class_dict(val_indices,   val_labels,   NUM_CLASSES)
# 
#     train_usage = build_usage_matrix(train_dict, NUM_CLASSES, CODEBOOK_SIZE)
#     val_usage   = build_usage_matrix(val_dict,   NUM_CLASSES, CODEBOOK_SIZE)
# 
#     # label + masking logic
#     if trained_on == "all":
#         trained_class = None
#         run_tag = "ALL_CLASSES"
#     else:
#         trained_class = int(trained_on)
#         run_tag = f"TRAINED_ON_CLASS_{trained_class}"
# 
#     train_usage_vis = mask_to_trained_class(train_usage, trained_class)
#     val_usage_vis   = mask_to_trained_class(val_usage,   trained_class)
# 
#     OUT_DIR.mkdir(parents=True, exist_ok=True)
# 
#     plot_heatmap(
#         train_usage_vis,
#         title=f"{run_id} ({run_tag}) — TRAIN codebook usage per class",
#         save_path=OUT_DIR / f"{run_id}_train_codebook_usage.png",
#     )
#     plot_heatmap(
#         val_usage_vis,
#         title=f"{run_id} ({run_tag}) — VAL codebook usage per class",
#         save_path=OUT_DIR / f"{run_id}_val_codebook_usage.png",
#     )
# 
#     print(f"[OK] {run_id}: saved train/val heatmaps to {OUT_DIR}")
# 
# 
# for run_id, trained_on in RUN_INFO.items():
#     process_run(run_id, trained_on)
# 
# =============================================================================




# =============================================================================
# from pathlib import Path
# import numpy as np
# import matplotlib.pyplot as plt
# 
# # ---------------- CONFIG ----------------
# RUN_TO_CLASS = {
#     "AUD-129": 0,
#     "AUD-130": 1,
#     "AUD-131": 2,
#     "AUD-133": 3,
#     "AUD-134": 4,
#     "AUD-135": 5,
#     "AUD-136": 6,
#     "AUD-137": 7,
#     "AUD-138": 8,
#     "AUD-139": 9,
# }
# 
# ROOT_TMPL = Path("/local/altamabp/audience_tuning-gem/vqvae/{run}/latents")
# OUT_PATH = Path(
#     "/home/altamabp/audience-tuning-gem/gem_audience_tuning-main/"
#     "aud_histograms/codebook_usage_heatmaps/"
#     "classes_vs_codebook_used_scatter.png"
# )
# 
# NUM_CLASSES = 10
# CODEBOOK_SIZE = 32
# 
# # Choose marker: "s"=square, "o"=circle
# MARKER = "s"
# 
# # If you want train-only or val-only, set SPLIT_MODE="train" or "val"
# SPLIT_MODE = "train|val"
# # --------------------------------------
# 
# 
# def load_flat_codes(latents_dir: Path, split: str) -> np.ndarray:
#     idx = np.load(latents_dir / f"{split}_indices.npy", allow_pickle=True)
#     return np.asarray(idx).reshape(-1)
# 
# 
# def codes_used_mask(codes_flat: np.ndarray, codebook_size: int) -> np.ndarray:
#     codes_flat = np.asarray(codes_flat)
#     codes_flat = codes_flat[(codes_flat >= 0) & (codes_flat < codebook_size)]
#     used = np.zeros(codebook_size, dtype=bool)
#     if codes_flat.size > 0:
#         used[np.unique(codes_flat).astype(int)] = True
#     return used
# 
# 
# # Build usage matrix U[class, code] = True if the run for that class used that code
# U = np.zeros((NUM_CLASSES, CODEBOOK_SIZE), dtype=bool)
# 
# for run, cls in RUN_TO_CLASS.items():
#     latents_dir = Path(str(ROOT_TMPL).format(run=run))
#     if not latents_dir.exists():
#         raise FileNotFoundError(f"Missing folder: {latents_dir}")
# 
#     if SPLIT_MODE == "train":
#         codes = load_flat_codes(latents_dir, "train")
#     elif SPLIT_MODE == "val":
#         codes = load_flat_codes(latents_dir, "val")
#     elif SPLIT_MODE == "train|val":
#         codes = np.concatenate([load_flat_codes(latents_dir, "train"),
#                                 load_flat_codes(latents_dir, "val")])
#     else:
#         raise ValueError(f"Unknown SPLIT_MODE: {SPLIT_MODE}")
# 
#     U[int(cls), :] = codes_used_mask(codes, CODEBOOK_SIZE)
# 
# # Create grid points (x=class, y=code)
# xs, ys = np.meshgrid(np.arange(NUM_CLASSES), np.arange(CODEBOOK_SIZE))
# xs = xs.reshape(-1)
# ys = ys.reshape(-1)
# 
# used_flat = U.T.reshape(-1)  # transpose so order matches ys (code major), not required but consistent
# 
# # Split used vs not used for two scatters (so we can color them differently)
# x_used = xs[used_flat]
# y_used = ys[used_flat]
# x_not  = xs[~used_flat]
# y_not  = ys[~used_flat]
# 
# plt.figure(figsize=(10, 8))
# 
# # Not used = red
# plt.scatter(x_not, y_not, marker=MARKER, c="red", s=60, linewidths=0)
# 
# # Used = green (draw on top)
# plt.scatter(x_used, y_used, marker=MARKER, c="green", s=60, linewidths=0)
# 
# plt.title(f"Codebook usage by class (single-class runs) — {SPLIT_MODE}")
# plt.xlabel("Class")
# plt.ylabel("Codebook index")
# 
# plt.xticks(np.arange(NUM_CLASSES))
# plt.yticks(np.arange(CODEBOOK_SIZE))
# 
# # Make it look like a clean grid
# plt.grid(True, which="both", linestyle="-", linewidth=0.5, alpha=0.3)
# plt.gca().set_axisbelow(True)
# 
# plt.tight_layout()
# OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
# plt.savefig(OUT_PATH, dpi=220)
# plt.close()
# 
# print(f"[OK] Saved: {OUT_PATH}")
# 
# =============================================================================

# =============================================================================
# 
# from pathlib import Path
# import numpy as np
# import matplotlib.pyplot as plt
# 
# # ---------------- CONFIG ----------------
# # Single-class runs (each run trained on ONE class)
# RUN_TO_CLASS = {
#     "AUD-129": 0,
#     "AUD-130": 1,
#     "AUD-131": 2,
#     "AUD-133": 3,
#     "AUD-134": 4,
#     "AUD-135": 5,
#     "AUD-136": 6,
#     "AUD-137": 7,
#     "AUD-138": 8,
#     "AUD-139": 9,
# }
# 
# # All-classes run
# ALL_CLASSES_RUN = "AUD-91"
# 
# ROOT_TMPL = Path("/local/altamabp/audience_tuning-gem/vqvae/{run}/latents")
# OUT_PATH = Path(
#     "/home/altamabp/audience-tuning-gem/gem_audience_tuning-main/"
#     "aud_histograms/codebook_usage_heatmaps/"
#     "classes_vs_codebook_used_scatter__single_vs_all.png"
# )
# 
# NUM_CLASSES = 10
# CODEBOOK_SIZE = 32
# 
# # "train", "val", or "train|val"
# SPLIT_MODE = "train|val"
# 
# # Marker: "s"=square, "o"=circle
# MARKER = "s"
# # --------------------------------------
# 
# 
# def load_flat_codes(latents_dir: Path, split: str) -> np.ndarray:
#     idx = np.load(latents_dir / f"{split}_indices.npy", allow_pickle=True)
#     return np.asarray(idx).reshape(-1)
# 
# 
# def load_labels(latents_dir: Path, split: str) -> np.ndarray:
#     y = np.load(latents_dir / f"{split}_labels.npy", allow_pickle=True)
#     return np.asarray(y).reshape(-1).astype(int)
# 
# 
# def codes_used_mask(codes_flat: np.ndarray, codebook_size: int) -> np.ndarray:
#     codes_flat = np.asarray(codes_flat)
#     codes_flat = codes_flat[(codes_flat >= 0) & (codes_flat < codebook_size)]
#     used = np.zeros(codebook_size, dtype=bool)
#     if codes_flat.size > 0:
#         used[np.unique(codes_flat).astype(int)] = True
#     return used
# 
# 
# def usage_by_class_from_indices_and_labels(indices: np.ndarray, labels: np.ndarray,
#                                           num_classes: int, codebook_size: int) -> np.ndarray:
#     """
#     Returns U[class, code] boolean for a single run that contains mixed classes (e.g., AUD-91).
#     indices: (N,) or (N,T) or (N,...) code indices
#     labels : (N,) class ids
#     """
#     indices = np.asarray(indices)
#     labels = np.asarray(labels).astype(int)
# 
#     if labels.ndim != 1:
#         raise ValueError(f"labels must be 1D (N,), got {labels.shape}")
#     if indices.shape[0] != labels.shape[0]:
#         raise ValueError(f"indices/labels mismatch: indices={indices.shape}, labels={labels.shape}")
# 
#     flat = indices.reshape(indices.shape[0], -1) if indices.ndim > 1 else indices.reshape(-1, 1)
# 
#     U = np.zeros((num_classes, codebook_size), dtype=bool)
#     for cls in range(num_classes):
#         m = labels == cls
#         if not np.any(m):
#             continue
#         codes = np.unique(flat[m])
#         codes = codes[(codes >= 0) & (codes < codebook_size)]
#         U[cls, codes.astype(int)] = True
#     return U
# 
# 
# def get_split_union_for_singleclass_run(latents_dir: Path, split_mode: str) -> np.ndarray:
#     if split_mode == "train":
#         return load_flat_codes(latents_dir, "train")
#     if split_mode == "val":
#         return load_flat_codes(latents_dir, "val")
#     if split_mode == "train|val":
#         return np.concatenate([load_flat_codes(latents_dir, "train"),
#                                load_flat_codes(latents_dir, "val")])
#     raise ValueError(f"Unknown SPLIT_MODE: {split_mode}")
# 
# 
# def get_split_union_for_allclasses_run(latents_dir: Path, split_mode: str):
#     if split_mode == "train":
#         idx = np.load(latents_dir / "train_indices.npy", allow_pickle=True)
#         lab = load_labels(latents_dir, "train")
#         return idx, lab
#     if split_mode == "val":
#         idx = np.load(latents_dir / "val_indices.npy", allow_pickle=True)
#         lab = load_labels(latents_dir, "val")
#         return idx, lab
#     if split_mode == "train|val":
#         idx_tr = np.load(latents_dir / "train_indices.npy", allow_pickle=True)
#         y_tr = load_labels(latents_dir, "train")
#         idx_va = np.load(latents_dir / "val_indices.npy", allow_pickle=True)
#         y_va = load_labels(latents_dir, "val")
# 
#         # concatenate along N (sample axis)
#         idx_tr = np.asarray(idx_tr)
#         idx_va = np.asarray(idx_va)
#         if idx_tr.ndim != idx_va.ndim:
#             raise ValueError(f"train_indices.ndim != val_indices.ndim: {idx_tr.ndim} vs {idx_va.ndim}")
#         idx = np.concatenate([idx_tr, idx_va], axis=0)
#         y = np.concatenate([y_tr, y_va], axis=0)
#         return idx, y
#     raise ValueError(f"Unknown SPLIT_MODE: {split_mode}")
# 
# 
# def plot_used_notused_grid(ax, U: np.ndarray, title: str, marker: str):
#     """
#     U shape: (NUM_CLASSES, CODEBOOK_SIZE) boolean
#     x = class (0..9), y = code (0..31)
#     green marker if used, red marker if not used
#     """
#     xs, ys = np.meshgrid(np.arange(NUM_CLASSES), np.arange(CODEBOOK_SIZE))
#     xs = xs.reshape(-1)
#     ys = ys.reshape(-1)
# 
#     used_flat = U.T.reshape(-1)  # align with ys (code-major)
#     x_used, y_used = xs[used_flat], ys[used_flat]
#     x_not,  y_not  = xs[~used_flat], ys[~used_flat]
# 
#     ax.scatter(x_not, y_not, marker=marker, c="red",   s=60, linewidths=0)
#     ax.scatter(x_used, y_used, marker=marker, c="green", s=60, linewidths=0)
# 
#     ax.set_title(title)
#     ax.set_xlabel("Class")
#     ax.set_ylabel("Codebook index")
#     ax.set_xticks(np.arange(NUM_CLASSES))
#     ax.set_yticks(np.arange(CODEBOOK_SIZE))
#     ax.grid(True, which="both", linestyle="-", linewidth=0.5, alpha=0.3)
#     ax.set_axisbelow(True)
# 
# 
# # --------- Build U for single-class runs (class -> run) ----------
# U_single = np.zeros((NUM_CLASSES, CODEBOOK_SIZE), dtype=bool)
# 
# for run, cls in RUN_TO_CLASS.items():
#     latents_dir = Path(str(ROOT_TMPL).format(run=run))
#     if not latents_dir.exists():
#         raise FileNotFoundError(f"Missing folder: {latents_dir}")
# 
#     codes = get_split_union_for_singleclass_run(latents_dir, SPLIT_MODE)
#     U_single[int(cls), :] = codes_used_mask(codes, CODEBOOK_SIZE)
# 
# # --------- Build U for AUD-91 (all classes, computed per class using labels) ----------
# latents_all = Path(str(ROOT_TMPL).format(run=ALL_CLASSES_RUN))
# if not latents_all.exists():
#     raise FileNotFoundError(f"Missing folder: {latents_all}")
# 
# idx_all, y_all = get_split_union_for_allclasses_run(latents_all, SPLIT_MODE)
# U_all = usage_by_class_from_indices_and_labels(idx_all, y_all, NUM_CLASSES, CODEBOOK_SIZE)
# 
# # --------- Plot: 2 panels ----------
# fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
# 
# plot_used_notused_grid(
#     axes[0],
#     U_single,
#     title=f"Single-class runs (AUD-129..139) — used vs not used ({SPLIT_MODE})",
#     marker=MARKER,
# )
# plot_used_notused_grid(
#     axes[1],
#     U_all,
#     title=f"{ALL_CLASSES_RUN} (trained on all classes) — used vs not used ({SPLIT_MODE})",
#     marker=MARKER,
# )
# 
# # Make y-label only once if you prefer:
# axes[1].set_ylabel("")
# 
# fig.suptitle("Classes (x) vs Codebook indices (y): green=used, red=not used", fontsize=14, y=0.98)
# fig.tight_layout(rect=[0, 0, 1, 0.96])
# 
# OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
# fig.savefig(OUT_PATH, dpi=220)
# plt.close(fig)
# 
# print(f"[OK] Saved: {OUT_PATH}")
# 
# =============================================================================



# =============================================================================
# from pathlib import Path
# import numpy as np
# import matplotlib.pyplot as plt
# 
# # ---------------- CONFIG ----------------
# # Single-class runs (each run trained on ONE class)
# RUN_TO_CLASS = {
#     "AUD-129": 0,
#     "AUD-130": 1,
#     "AUD-131": 2,
#     "AUD-133": 3,
#     "AUD-134": 4,
#     "AUD-135": 5,
#     "AUD-136": 6,
#     "AUD-137": 7,
#     "AUD-138": 8,
#     "AUD-139": 9,
# }
# 
# # All-classes run
# ALL_CLASSES_RUN = "AUD-91"
# 
# ROOT_TMPL = Path("/local/altamabp/audience_tuning-gem/vqvae/{run}/latents")
# OUT_PATH = Path(
#     "/home/altamabp/audience-tuning-gem/gem_audience_tuning-main/"
#     "aud_histograms/codebook_usage_heatmaps/"
#     "classes_vs_codebook_used_scatter__single_vs_all__freq.png"
# )
# 
# NUM_CLASSES = 10
# CODEBOOK_SIZE = 32
# 
# # "train", "val", or "train|val"
# SPLIT_MODE = "train|val"
# 
# # Marker: "s"=square, "o"=circle
# MARKER = "s"
# 
# # If True, marker SIZE encodes frequency (count). If False, fixed size.
# SIZE_BY_FREQUENCY = True
# 
# # Controls how big markers can get (tune if too small/large)
# SIZE_MIN = 10
# SIZE_MAX = 220
# 
# # If True, use relative frequency per class (row-normalized) instead of raw counts.
# # (Recommended when class sample sizes differ a lot.)
# USE_RELATIVE_FREQ = False
# # --------------------------------------
# 
# 
# def load_flat_codes(latents_dir: Path, split: str) -> np.ndarray:
#     idx = np.load(latents_dir / f"{split}_indices.npy", allow_pickle=True)
#     return np.asarray(idx).reshape(-1)
# 
# 
# def load_labels(latents_dir: Path, split: str) -> np.ndarray:
#     y = np.load(latents_dir / f"{split}_labels.npy", allow_pickle=True)
#     return np.asarray(y).reshape(-1).astype(int)
# 
# 
# def counts_mask_from_codes(codes_flat: np.ndarray, codebook_size: int) -> np.ndarray:
#     """
#     For single-class runs: returns counts[code] of occurrences in codes_flat.
#     """
#     codes_flat = np.asarray(codes_flat)
#     codes_flat = codes_flat[(codes_flat >= 0) & (codes_flat < codebook_size)]
#     if codes_flat.size == 0:
#         return np.zeros(codebook_size, dtype=np.int64)
#     return np.bincount(codes_flat.astype(int), minlength=codebook_size).astype(np.int64)
# 
# 
# def counts_by_class_from_indices_and_labels(indices: np.ndarray, labels: np.ndarray,
#                                            num_classes: int, codebook_size: int) -> np.ndarray:
#     """
#     Returns C[class, code] counts for a mixed-class run (e.g., AUD-91).
#     indices: (N,) or (N,T) or (N,...) code indices
#     labels : (N,) class ids
#     """
#     indices = np.asarray(indices)
#     labels = np.asarray(labels).astype(int)
# 
#     if labels.ndim != 1:
#         raise ValueError(f"labels must be 1D (N,), got {labels.shape}")
#     if indices.shape[0] != labels.shape[0]:
#         raise ValueError(f"indices/labels mismatch: indices={indices.shape}, labels={labels.shape}")
# 
#     flat = indices.reshape(indices.shape[0], -1) if indices.ndim > 1 else indices.reshape(-1, 1)
# 
#     C = np.zeros((num_classes, codebook_size), dtype=np.int64)
#     for cls in range(num_classes):
#         m = labels == cls
#         if not np.any(m):
#             continue
#         codes = flat[m].reshape(-1)
#         codes = codes[(codes >= 0) & (codes < codebook_size)]
#         if codes.size > 0:
#             C[cls] += np.bincount(codes.astype(int), minlength=codebook_size)
#     return C
# 
# 
# def get_split_union_for_singleclass_run(latents_dir: Path, split_mode: str) -> np.ndarray:
#     if split_mode == "train":
#         return load_flat_codes(latents_dir, "train")
#     if split_mode == "val":
#         return load_flat_codes(latents_dir, "val")
#     if split_mode == "train|val":
#         return np.concatenate([load_flat_codes(latents_dir, "train"),
#                                load_flat_codes(latents_dir, "val")])
#     raise ValueError(f"Unknown SPLIT_MODE: {split_mode}")
# 
# 
# def get_split_union_for_allclasses_run(latents_dir: Path, split_mode: str):
#     if split_mode == "train":
#         idx = np.load(latents_dir / "train_indices.npy", allow_pickle=True)
#         lab = load_labels(latents_dir, "train")
#         return idx, lab
#     if split_mode == "val":
#         idx = np.load(latents_dir / "val_indices.npy", allow_pickle=True)
#         lab = load_labels(latents_dir, "val")
#         return idx, lab
#     if split_mode == "train|val":
#         idx_tr = np.load(latents_dir / "train_indices.npy", allow_pickle=True)
#         y_tr = load_labels(latents_dir, "train")
#         idx_va = np.load(latents_dir / "val_indices.npy", allow_pickle=True)
#         y_va = load_labels(latents_dir, "val")
# 
#         # concatenate along N (sample axis)
#         idx_tr = np.asarray(idx_tr)
#         idx_va = np.asarray(idx_va)
#         if idx_tr.ndim != idx_va.ndim:
#             raise ValueError(f"train_indices.ndim != val_indices.ndim: {idx_tr.ndim} vs {idx_va.ndim}")
#         idx = np.concatenate([idx_tr, idx_va], axis=0)
#         y = np.concatenate([y_tr, y_va], axis=0)
#         return idx, y
#     raise ValueError(f"Unknown SPLIT_MODE: {split_mode}")
# 
# 
# def normalize_rows(C: np.ndarray) -> np.ndarray:
#     """
#     Row-normalize to relative frequencies per class (each row sums to 1 if nonzero).
#     """
#     C = C.astype(np.float64)
#     row_sums = C.sum(axis=1, keepdims=True)
#     row_sums[row_sums == 0] = 1.0
#     return C / row_sums
# 
# 
# def sizes_from_counts(C: np.ndarray, size_min: float, size_max: float) -> np.ndarray:
#     """
#     Map counts/frequencies to scatter sizes. Uses log1p for stability on raw counts.
#     If C is already frequency (0..1), it will still work.
#     """
#     X = C.copy().astype(np.float64)
#     # If values look like counts (can be >1), compress with log1p
#     if np.nanmax(X) > 1.0:
#         X = np.log1p(X)
#     vmax = np.nanmax(X)
#     if vmax <= 0:
#         return np.full_like(X, size_min, dtype=np.float64)
#     X = X / vmax
#     return size_min + X * (size_max - size_min)
# 
# 
# def plot_grid_with_frequency(ax, C: np.ndarray, title: str, marker: str, size_by_freq: bool):
#     """
#     C shape: (NUM_CLASSES, CODEBOOK_SIZE) counts (or frequencies if normalized)
#     x = class (0..9), y = code (0..31)
# 
#     Used -> green marker, size scales by frequency if size_by_freq=True
#     Not used -> red marker (small fixed size)
#     """
#     # Used mask
#     used = C > 0
# 
#     # Create coordinate grid
#     xs, ys = np.meshgrid(np.arange(NUM_CLASSES), np.arange(CODEBOOK_SIZE))
#     xs = xs.reshape(-1)
#     ys = ys.reshape(-1)
# 
#     # Flatten C in code-major to align with ys
#     C_flat = C.T.reshape(-1)
#     used_flat = used.T.reshape(-1)
# 
#     x_used, y_used = xs[used_flat], ys[used_flat]
#     x_not,  y_not  = xs[~used_flat], ys[~used_flat]
# 
#     # Sizes
#     if size_by_freq:
#         S = sizes_from_counts(C, SIZE_MIN, SIZE_MAX)
#         S_flat = S.T.reshape(-1)
#         s_used = S_flat[used_flat]
#         s_not = np.full(x_not.shape, SIZE_MIN, dtype=np.float64)
#     else:
#         s_used = np.full(x_used.shape, 60.0, dtype=np.float64)
#         s_not = np.full(x_not.shape, 60.0, dtype=np.float64)
# 
#     # Plot not used first (red, small)
#     ax.scatter(x_not, y_not, marker=marker, c="red", s=s_not, linewidths=0, alpha=0.8)
# 
#     # Plot used on top (green, size encodes frequency if enabled)
#     ax.scatter(x_used, y_used, marker=marker, c="green", s=s_used, linewidths=0, alpha=0.9)
# 
#     ax.set_title(title)
#     ax.set_xlabel("Class")
#     ax.set_ylabel("Codebook index")
#     ax.set_xticks(np.arange(NUM_CLASSES))
#     ax.set_yticks(np.arange(CODEBOOK_SIZE))
#     ax.grid(True, which="both", linestyle="-", linewidth=0.5, alpha=0.3)
#     ax.set_axisbelow(True)
# 
# 
# # --------- Build COUNT matrix for single-class runs: C_single[class, code] ----------
# C_single = np.zeros((NUM_CLASSES, CODEBOOK_SIZE), dtype=np.int64)
# 
# for run, cls in RUN_TO_CLASS.items():
#     latents_dir = Path(str(ROOT_TMPL).format(run=run))
#     if not latents_dir.exists():
#         raise FileNotFoundError(f"Missing folder: {latents_dir}")
# 
#     codes = get_split_union_for_singleclass_run(latents_dir, SPLIT_MODE)
#     C_single[int(cls), :] = counts_mask_from_codes(codes, CODEBOOK_SIZE)
# 
# # --------- Build COUNT matrix for AUD-91 (all classes): C_all[class, code] ----------
# latents_all = Path(str(ROOT_TMPL).format(run=ALL_CLASSES_RUN))
# if not latents_all.exists():
#     raise FileNotFoundError(f"Missing folder: {latents_all}")
# 
# idx_all, y_all = get_split_union_for_allclasses_run(latents_all, SPLIT_MODE)
# C_all = counts_by_class_from_indices_and_labels(idx_all, y_all, NUM_CLASSES, CODEBOOK_SIZE)
# 
# # Optional: convert counts to per-class relative frequency
# if USE_RELATIVE_FREQ:
#     C_single_plot = normalize_rows(C_single)
#     C_all_plot = normalize_rows(C_all)
#     freq_label = "relative frequency (row-normalized)"
# else:
#     C_single_plot = C_single
#     C_all_plot = C_all
#     freq_label = "counts"
# 
# # --------- Plot: 2 panels ----------
# fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
# 
# plot_grid_with_frequency(
#     axes[0],
#     C_single_plot,
#     title=f"Single-class runs (AUD-129..139) — used vs not used ({SPLIT_MODE})\nsize ∝ {freq_label}",
#     marker=MARKER,
#     size_by_freq=SIZE_BY_FREQUENCY,
# )
# plot_grid_with_frequency(
#     axes[1],
#     C_all_plot,
#     title=f"{ALL_CLASSES_RUN} (trained on all classes) — used vs not used ({SPLIT_MODE})\nsize ∝ {freq_label}",
#     marker=MARKER,
#     size_by_freq=SIZE_BY_FREQUENCY,
# )
# 
# # Make y-label only once if you prefer:
# axes[1].set_ylabel("")
# 
# fig.suptitle(
#     "Classes (x) vs Codebook indices (y): green=used (size ∝ frequency), red=not used",
#     fontsize=14,
#     y=0.98
# )
# fig.tight_layout(rect=[0, 0, 1, 0.94])
# 
# OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
# fig.savefig(OUT_PATH, dpi=220)
# plt.close(fig)
# 
# print(f"[OK] Saved: {OUT_PATH}")
# 
# # Optional console summary
# print("\nMax counts per class (single-class runs):", C_single.max(axis=1).tolist())
# print("Max counts per class (AUD-91):", C_all.max(axis=1).tolist())
# 
# =============================================================================



from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
RUN_TO_CLASS = {
    "AUD-129": 0, "AUD-130": 1, "AUD-131": 2, "AUD-133": 3, "AUD-134": 4,
    "AUD-135": 5, "AUD-136": 6, "AUD-137": 7, "AUD-138": 8, "AUD-139": 9,
}
ALL_CLASSES_RUN = "AUD-91"

ROOT_TMPL = Path("/local/altamabp/audience_tuning-gem/vqvae/{run}/latents")
OUT_PATH = Path(
    "/home/altamabp/audience-tuning-gem/gem_audience_tuning-main/"
    "aud_histograms/codebook_usage_heatmaps/"
    "classes_vs_codebook_used_scatter__single_vs_all__freq_annotated.png"
)

NUM_CLASSES = 10
CODEBOOK_SIZE = 32
SPLIT_MODE = "train|val"   # "train", "val", or "train|val"
MARKER = "s"               # "s"=square, "o"=circle

# Marker sizing
SIZE_BY_FREQUENCY = False #True
SIZE_MIN = 12
SIZE_MAX = 260

# Annotation controls
ANNOTATE_COUNTS = False #True                 # write the count on used cells
ANNOTATE_ONLY_IF_COUNT_AT_LEAST = 1    # set >1 to reduce clutter
ANNOTATION_FONTSIZE = 6                # tune if crowded
ANNOTATION_COLOR = "black"
# --------------------------------------

def pick_one_sample_per_class(indices: np.ndarray,
                              labels: np.ndarray,
                              num_classes: int,
                              seed: int = 0):
    """
    Returns dict[class] -> flattened code indices for ONE RANDOM sample of that class.
    Deterministic if seed is fixed.
    """
    
    rng = np.random.default_rng(seed)
    
    indices = np.asarray(indices)
    labels = np.asarray(labels).astype(int)

    out = {}
    for cls in range(num_classes):
        m = np.where(labels == cls)[0]
        if len(m) == 0:
            continue
        i = rng.choice(m)   # randomized selection
        sample_codes = indices[i]
        out[cls] = np.asarray(sample_codes).reshape(-1)
    return out


def load_flat_codes(latents_dir: Path, split: str) -> np.ndarray:
    idx = np.load(latents_dir / f"{split}_indices.npy", allow_pickle=True)
    return np.asarray(idx).reshape(-1)


def load_labels(latents_dir: Path, split: str) -> np.ndarray:
    y = np.load(latents_dir / f"{split}_labels.npy", allow_pickle=True)
    return np.asarray(y).reshape(-1).astype(int)


def counts_from_codes(codes_flat: np.ndarray, codebook_size: int) -> np.ndarray:
    """For single-class runs: counts[code] of occurrences."""
    codes_flat = np.asarray(codes_flat)
    codes_flat = codes_flat[(codes_flat >= 0) & (codes_flat < codebook_size)]
    if codes_flat.size == 0:
        return np.zeros(codebook_size, dtype=np.int64)
    return np.bincount(codes_flat.astype(int), minlength=codebook_size).astype(np.int64)


def counts_by_class_from_indices_and_labels(indices: np.ndarray, labels: np.ndarray,
                                           num_classes: int, codebook_size: int) -> np.ndarray:
    """For mixed-class run (AUD-91): C[class, code] = number of occurrences."""
    indices = np.asarray(indices)
    labels = np.asarray(labels).astype(int)

    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D (N,), got {labels.shape}")
    if indices.shape[0] != labels.shape[0]:
        raise ValueError(f"indices/labels mismatch: indices={indices.shape}, labels={labels.shape}")

    flat = indices.reshape(indices.shape[0], -1) if indices.ndim > 1 else indices.reshape(-1, 1)

    C = np.zeros((num_classes, codebook_size), dtype=np.int64)
    for cls in range(num_classes):
        m = labels == cls
        if not np.any(m):
            continue
        codes = flat[m].reshape(-1)
        codes = codes[(codes >= 0) & (codes < codebook_size)]
        if codes.size > 0:
            C[cls] += np.bincount(codes.astype(int), minlength=codebook_size)
    return C


def get_split_union_for_singleclass_run(latents_dir: Path, split_mode: str) -> np.ndarray:
    if split_mode == "train":
        return load_flat_codes(latents_dir, "train")
    if split_mode == "val":
        return load_flat_codes(latents_dir, "val")
    if split_mode == "train|val":
        return np.concatenate([load_flat_codes(latents_dir, "train"),
                               load_flat_codes(latents_dir, "val")])
    raise ValueError(f"Unknown SPLIT_MODE: {split_mode}")


def get_split_union_for_allclasses_run(latents_dir: Path, split_mode: str):
    if split_mode == "train":
        idx = np.load(latents_dir / "train_indices.npy", allow_pickle=True)
        lab = load_labels(latents_dir, "train")
        return idx, lab
    if split_mode == "val":
        idx = np.load(latents_dir / "val_indices.npy", allow_pickle=True)
        lab = load_labels(latents_dir, "val")
        return idx, lab
    if split_mode == "train|val":
        idx_tr = np.load(latents_dir / "train_indices.npy", allow_pickle=True)
        y_tr = load_labels(latents_dir, "train")
        idx_va = np.load(latents_dir / "val_indices.npy", allow_pickle=True)
        y_va = load_labels(latents_dir, "val")

        idx_tr = np.asarray(idx_tr)
        idx_va = np.asarray(idx_va)
        if idx_tr.ndim != idx_va.ndim:
            raise ValueError(f"train_indices.ndim != val_indices.ndim: {idx_tr.ndim} vs {idx_va.ndim}")
        idx = np.concatenate([idx_tr, idx_va], axis=0)
        y = np.concatenate([y_tr, y_va], axis=0)
        return idx, y
    raise ValueError(f"Unknown SPLIT_MODE: {split_mode}")


def sizes_from_counts(C: np.ndarray, size_min: float, size_max: float) -> np.ndarray:
    """Map counts to marker sizes using log1p scaling."""
    X = np.log1p(C.astype(np.float64))
    vmax = np.max(X) if X.size else 0.0
    if vmax <= 0:
        return np.full_like(X, size_min, dtype=np.float64)
    X = X / vmax
    return size_min + X * (size_max - size_min)


def plot_grid_with_counts(ax, C_counts: np.ndarray, title: str, marker: str,
                          size_by_freq: bool, annotate: bool):
    """
    C_counts: (NUM_CLASSES, CODEBOOK_SIZE) integer counts
    x = class (0..9), y = code (0..31)
    green marker if count>0, red marker if 0
    marker size ∝ count (optional)
    annotations show the count at each used cell
    """
    used = C_counts > 0

    # Coordinates
    xs, ys = np.meshgrid(np.arange(NUM_CLASSES), np.arange(CODEBOOK_SIZE))
    xs = xs.reshape(-1)
    ys = ys.reshape(-1)

    # Flatten in code-major order to align with ys
    C_flat = C_counts.T.reshape(-1)
    used_flat = used.T.reshape(-1)

    x_used, y_used = xs[used_flat], ys[used_flat]
    x_not,  y_not  = xs[~used_flat], ys[~used_flat]

    # Sizes
    if size_by_freq:
        S = sizes_from_counts(C_counts, SIZE_MIN, SIZE_MAX)
        S_flat = S.T.reshape(-1)
        s_used = S_flat[used_flat]
        s_not = np.full(x_not.shape, SIZE_MIN, dtype=np.float64)
    else:
        s_used = np.full(x_used.shape, 60.0, dtype=np.float64)
        s_not = np.full(x_not.shape, 60.0, dtype=np.float64)

    # Plot not used then used
    ax.scatter(x_not, y_not, marker=marker, c="red",   s=s_not, linewidths=0, alpha=0.8)
    ax.scatter(x_used, y_used, marker=marker, c="green", s=s_used, linewidths=0, alpha=0.9)

    # Annotate counts on used cells
    if annotate:
        # iterate over used points; write the integer count
        counts_used = C_flat[used_flat].astype(int)
        for x, y, cnt in zip(x_used, y_used, counts_used):
            if cnt >= ANNOTATE_ONLY_IF_COUNT_AT_LEAST:
                ax.text(
                    x, y, str(cnt),
                    ha="center", va="center",
                    fontsize=ANNOTATION_FONTSIZE,
                    color=ANNOTATION_COLOR
                )

    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Codebook index")
    ax.set_xticks(np.arange(NUM_CLASSES))
    ax.set_yticks(np.arange(CODEBOOK_SIZE))
    ax.grid(True, which="both", linestyle="-", linewidth=0.5, alpha=0.3)
    ax.set_axisbelow(True)


# --------- Build COUNT matrix for single-class runs: C_single[class, code] ----------
C_single = np.zeros((NUM_CLASSES, CODEBOOK_SIZE), dtype=np.int64)
for run, cls in RUN_TO_CLASS.items():
    latents_dir = Path(str(ROOT_TMPL).format(run=run))
    if not latents_dir.exists():
        raise FileNotFoundError(f"Missing folder: {latents_dir}")

    codes = get_split_union_for_singleclass_run(latents_dir, SPLIT_MODE)
    C_single[int(cls), :] = counts_from_codes(codes, CODEBOOK_SIZE)

# --------- Build COUNT matrix for AUD-91 (all classes): C_all[class, code] ----------
latents_all = Path(str(ROOT_TMPL).format(run=ALL_CLASSES_RUN))
if not latents_all.exists():
    raise FileNotFoundError(f"Missing folder: {latents_all}")

# =============================================================================
# idx_all, y_all = get_split_union_for_allclasses_run(latents_all, SPLIT_MODE)
# C_all = counts_by_class_from_indices_and_labels(idx_all, y_all, NUM_CLASSES, CODEBOOK_SIZE)
# =============================================================================

idx_all, y_all = get_split_union_for_allclasses_run(latents_all, SPLIT_MODE)

one_per_class = pick_one_sample_per_class(idx_all, y_all, NUM_CLASSES)

C_all = np.zeros((NUM_CLASSES, CODEBOOK_SIZE), dtype=np.int64)
for cls, codes in one_per_class.items():
    codes = codes[(codes >= 0) & (codes < CODEBOOK_SIZE)]
    C_all[cls, np.unique(codes).astype(int)] = 1




# --------- Plot: 2 panels ----------
fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)

plot_grid_with_counts(
    axes[0],
    C_single,
    title=f"Single-class runs (AUD-129..139) — green=used, red=not used ({SPLIT_MODE})\nsize ∝ count; text=count",
    marker=MARKER,
    size_by_freq=SIZE_BY_FREQUENCY,
    annotate=ANNOTATE_COUNTS,
)
plot_grid_with_counts(
    axes[1],
    C_all,
    title=f"{ALL_CLASSES_RUN} (trained on all classes) — green=used, red=not used ({SPLIT_MODE})\nsize ∝ count; text=count",
    marker=MARKER,
    size_by_freq=SIZE_BY_FREQUENCY,
    annotate=ANNOTATE_COUNTS,
)

axes[1].set_ylabel("")  # optional

fig.suptitle(
    "Classes (x) vs Codebook indices (y): green=used (size & text = count), red=not used",
    fontsize=14,
    y=0.98
)
fig.tight_layout(rect=[0, 0, 1, 0.94])

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH, dpi=220)
plt.close(fig)

print(f"[OK] Saved: {OUT_PATH}")
