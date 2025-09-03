#!/usr/bin/env python3
"""
Plot a 2D PCA scatter from a prepared X/y CSV saved by run_ml_pipe.

Behavior
- Loads a CSV containing feature columns and a target column.
- Optionally standardizes features.
- Runs PCA with 2 components (using dim_reduction.pca_reduction).
- Plots a scatter colored by target classes.

Defaults
- If --csv is not provided, tries to pick the most recent *_Xy.csv under
  <results_dir>/prepared.
"""

import argparse
import os
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt

from eeg_adhd_epilepsy_psychostimulant.utils.config import results_dir
from eeg_adhd_epilepsy_psychostimulant.viz.dim_reduction import pca_reduction


def find_latest_prepared_csv() -> Optional[str]:
    prep_dir = os.path.join(results_dir, "prepared")
    if not os.path.isdir(prep_dir):
        return None
    cands = [os.path.join(prep_dir, f) for f in os.listdir(prep_dir) if f.endswith("_Xy.csv")]
    if not cands:
        return None
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]


def main():
    parser = argparse.ArgumentParser(description="Plot 2D PCA scatter from prepared X/y CSV")
    parser.add_argument("--csv", default=None, help="Path to prepared Xy CSV (defaults to latest in <results_dir>/prepared)")
    parser.add_argument("--target", default=None, help="Target column name (required if not auto-detectable)")
    parser.add_argument("--standardize", action="store_true", help="Standardize features before PCA")
    parser.add_argument("--save", default=None, help="Path to save the PCA scatter (PNG)")
    parser.add_argument("--no-show", action="store_true", help="Do not show plot interactively")
    args = parser.parse_args()

    if not args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    csv_path = args.csv or find_latest_prepared_csv()
    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError("Could not locate prepared CSV. Provide --csv or ensure <results_dir>/prepared exists.")

    df = pd.read_csv(csv_path)
    # Infer target if not provided: prefer common names
    target = args.target
    if target is None:
        for cand in ("target", "TDAH", "Epilepsy", "TSA"):
            if cand in df.columns:
                target = cand
                break
    if target is None or target not in df.columns:
        raise KeyError("Target column not specified or not found. Use --target to specify it.")

    X = df.drop(columns=[target])
    y = df[target]

    # Optionally standardize
    if args.standardize:
        from sklearn.preprocessing import StandardScaler
        X_vals = StandardScaler().fit_transform(X.values)
    else:
        X_vals = X.values

    Z = pca_reduction(X_vals, n_components=2)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 6))
    classes = pd.Series(y).astype(str).unique()
    for cls in classes:
        mask = (y.astype(str) == cls).values
        ax.scatter(Z[mask, 0], Z[mask, 1], s=20, alpha=0.8, label=str(cls))
    
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title(f"PCA (2D) – {os.path.basename(csv_path)}")
    ax.legend(title=str(target))
    ax.grid(True, alpha=0.2)

    # Set x/y limits to include most points (use 1st/99th percentiles with small margin)
    x_q = pd.Series(Z[:, 0])
    y_q = pd.Series(Z[:, 1])
    x_lo, x_hi = x_q.quantile([0.01, 0.99]).values
    y_lo, y_hi = y_q.quantile([0.01, 0.99]).values
    # Fallback to min/max if percentiles collapse
    if x_hi == x_lo:
        x_lo, x_hi = x_q.min(), x_q.max()
    if y_hi == y_lo:
        y_lo, y_hi = y_q.min(), y_q.max()
    x_margin = max((x_hi - x_lo) * 0.05, 1e-6)
    y_margin = max((y_hi - y_lo) * 0.05, 1e-6)
    ax.set_xlim(x_lo - x_margin, x_hi + x_margin)
    ax.set_ylim(y_lo - y_margin, y_hi + y_margin)

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()

