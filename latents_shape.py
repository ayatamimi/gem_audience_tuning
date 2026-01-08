# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 16:18:55 2025

@author: ayata
"""

import os
import glob
import numpy as np

BASE_DIR = "vqvae"  # change if needed

def main():
    if not os.path.isdir(BASE_DIR):
        print(f"Base directory '{BASE_DIR}' not found.")
        return

    # Iterate over subdirectories like vqvae/AUD-13, vqvae/AUD-68, ...
    for run_dir in sorted(os.listdir(BASE_DIR)):
        full_run_path = os.path.join(BASE_DIR, run_dir)

        if not (run_dir.startswith("AUD-") and os.path.isdir(full_run_path)):
            continue

        run_id = run_dir  # e.g. "AUD-13"
        latents_dir = os.path.join(full_run_path, "latents")

        if not os.path.isdir(latents_dir):
            print(f"[{run_id}] No 'latents' directory found at {latents_dir}")
            continue

        npy_files = sorted(glob.glob(os.path.join(latents_dir, "*.npy")))
        if not npy_files:
            print(f"[{run_id}] No .npy files found in {latents_dir}")
            continue

        print(f"\nRun: {run_id}")
        for npy_path in npy_files:
            try:
                # mmap_mode='r' avoids loading fully into RAM if arrays are huge
                arr = np.load(npy_path, mmap_mode="r")
                fname = os.path.basename(npy_path)
                print(f"  {fname}: shape = {arr.shape}")
            except Exception as e:
                print(f"  {os.path.basename(npy_path)}: ERROR loading file -> {e}")

if __name__ == "__main__":
    main()
