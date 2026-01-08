# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 02:38:28 2025

@author: ayata
"""
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -----------------------------
# 1. Configuration
# -----------------------------

# Pattern to find your CSV files
# e.g. "logs/run_*.csv" or just "*.csv"
csv_pattern = "logs/*.csv"

# Column names in your CSVs
# Change these to whatever you actually have, e.g. "epoch", "step", "iteration"
x_col = "step"
y_col = "loss"


# -----------------------------
# 2. Define model functions
# -----------------------------

def exp_func(x, a, b, c):
    """Exponential: y = a * exp(b x) + c"""
    return a * np.exp(b * x) + c


def inv_func(x, a, b):
    """Inverse: y = a / x + b   (we'll pass in x_shifted so no division by 0)"""
    return a / x + b


def inv_sq_func(x, a, b):
    """Inverse squared: y = a / x^2 + b   (again using x_shifted)"""
    return a / (x ** 2) + b


# -----------------------------
# 3. Load all runs
# -----------------------------

csv_files = sorted(glob.glob(csv_pattern))
if not csv_files:
    raise FileNotFoundError(f"No CSV files found for pattern: {csv_pattern}")

runs_data = []  # list of dicts: { "name": ..., "x": np.array, "y": np.array }

for fpath in csv_files:
    df = pd.read_csv(fpath)

    if x_col not in df.columns or y_col not in df.columns:
        raise KeyError(f"{fpath} does not contain required columns '{x_col}' and '{y_col}'")

    # Drop NaNs and sort by x just in case
    df = df[[x_col, y_col]].dropna().sort_values(by=x_col)

    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)

    run_name = os.path.splitext(os.path.basename(fpath))[0]
    runs_data.append({"name": run_name, "x": x, "y": y})


# -----------------------------
# 4. Fit each run for each model
# -----------------------------

exp_fits = []     # list of dicts: { "name", "popt" }
inv_fits = []
inv_sq_fits = []

for run in runs_data:
    name, x, y = run["name"], run["x"], run["y"]

    # For numeric stability: shift x to start near 0
    x_norm = x - x[0]

    # ---- Exponential fit ----
    # crude initial guess; tweak if needed
    p0_exp = [y[0] if len(y) > 0 else 1.0, -0.1, y[-1] if len(y) > 0 else 0.0]
    try:
        popt_exp, _ = curve_fit(exp_func, x_norm, y, p0=p0_exp, maxfev=10000)
        exp_fits.append({"name": name, "x": x_norm, "y": y, "popt": popt_exp})
    except Exception as e:
        print(f"[WARN] Exp fit failed for {name}: {e}")

    # ---- Inverse (1/x) and 1/x^2 fits ----
    # make sure x_shifted starts at 1 (avoid division by 0)
    x_shifted = x_norm + 1.0

    # 1/x
    p0_inv = [ (y[0] - y[-1]) * x_shifted[0], y[-1] ]  # rough guess
    try:
        popt_inv, _ = curve_fit(inv_func, x_shifted, y, p0=p0_inv, maxfev=10000)
        inv_fits.append({"name": name, "x": x_shifted, "y": y, "popt": popt_inv})
    except Exception as e:
        print(f"[WARN] 1/x fit failed for {name}: {e}")

    # 1/x^2
    p0_inv_sq = [ (y[0] - y[-1]) * (x_shifted[0] ** 2), y[-1] ]
    try:
        popt_inv_sq, _ = curve_fit(inv_sq_func, x_shifted, y, p0=p0_inv_sq, maxfev=10000)
        inv_sq_fits.append({"name": name, "x": x_shifted, "y": y, "popt": popt_inv_sq})
    except Exception as e:
        print(f"[WARN] 1/x^2 fit failed for {name}: {e}")


# -----------------------------
# 5. Plotting helpers
# -----------------------------

def plot_model_fits(fits, model_func, title, x_label, y_label="loss"):
    plt.figure(figsize=(8, 6))

    for run in fits:
        name, x, y, popt = run["name"], run["x"], run["y"], run["popt"]

        # dense x for smooth curve
        x_dense = np.linspace(x.min(), x.max(), 500)
        y_pred = model_func(x_dense, *popt)

        # plot raw data
        plt.scatter(x, y, s=10, alpha=0.5, label=f"{name} (data)")

        # plot fit
        plt.plot(x_dense, y_pred, linewidth=2, label=f"{name} (fit)")

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()


# -----------------------------
# 6. Make the three plots
# -----------------------------

# Exponential (x normalized)
if exp_fits:
    plot_model_fits(
        exp_fits,
        exp_func,
        title="Exponential fits: loss ≈ a * exp(b * (x - x0)) + c",
        x_label="x - x0 (normalized step/epoch)",
    )

# 1/x (x shifted)
if inv_fits:
    plot_model_fits(
        inv_fits,
        inv_func,
        title="Inverse fits: loss ≈ a / (x_shifted) + b",
        x_label="x_shifted = (x - x0) + 1",
    )

# 1/x^2 (x shifted)
if inv_sq_fits:
    plot_model_fits(
        inv_sq_fits,
        inv_sq_func,
        title="Inverse squared fits: loss ≈ a / (x_shifted^2) + b",
        x_label="x_shifted = (x - x0) + 1",
    )

plt.show()
