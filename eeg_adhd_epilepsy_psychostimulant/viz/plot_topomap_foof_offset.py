#!/usr/bin/env python3
"""
Topomap for FOOOF offset per group, with stats and annotations.

Loads a prepared X/y CSV (from run_ml_pipe.py) and extracts the per-sensor
values for the specified feature key (default: foofOffset). Plots:
  - Group A mean topomap
  - Group B mean topomap
  - Difference (B - A) with significance mask (BH-FDR corrected)

Annotates the locations of Pz and P4 on the plots when present.

Example
-------
python -m eeg_adhd_epilepsy_psychostimulant.viz.plot_topomap_foof_offset \
  --csv data/results/ml/prepared/adhd_finetune_results_classification_hpsearch_all_Xy.csv \
  --target TDAH \
  --group-a 0 --group-b 1 \
  --save data/results/ml/figures/foof_offset_topomap.png
"""

from __future__ import annotations

import argparse
import os
import re
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mne


def _bh_fdr(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini–Hochberg FDR mask for significance.

    Returns boolean mask of pvals considered significant after FDR.
    """
    p = np.asarray(pvals, dtype=float)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    p_sorted = p[order]
    thresh = alpha * (np.arange(1, m + 1) / m)
    below = p_sorted <= thresh
    if not np.any(below):
        return np.zeros_like(p_sorted, dtype=bool)[np.argsort(order)]
    k = np.max(np.where(below)[0])
    mask_sorted = np.zeros_like(p_sorted, dtype=bool)
    mask_sorted[: k + 1] = True
    # unsort
    mask = np.zeros_like(mask_sorted, dtype=bool)
    mask[order] = mask_sorted
    return mask


def _ttest_or_permutation(x: np.ndarray, y: np.ndarray, n_perm: int = 2000, seed: int = 42) -> float:
    """Return two-sided p-value for difference in means between x and y.

    Tries SciPy Welch's t-test; if not available, falls back to a simple
    permutation test with n_perm permutations.
    """
    pval = None
    try:
        from scipy.stats import ttest_ind

        _, p = ttest_ind(x, y, equal_var=False, nan_policy="omit")
        pval = float(p)
    except Exception:
        # Permutation fallback
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]
        if x.size == 0 or y.size == 0:
            return np.nan
        rng = np.random.default_rng(seed)
        obs = abs(np.nanmean(y) - np.nanmean(x))
        pooled = np.concatenate([x, y])
        nx = x.size
        count = 0
        for _ in range(int(n_perm)):
            rng.shuffle(pooled)
            xa = pooled[:nx]
            yb = pooled[nx:]
            diff = abs(np.mean(yb) - np.mean(xa))
            if diff >= obs:
                count += 1
        pval = (count + 1) / (n_perm + 1)
    return pval


def _get_feature_columns(df: pd.DataFrame, feature_key: str, sensors: List[str]) -> Dict[str, str]:
    """Map sensor -> column name for the requested feature.

    Looks for columns like 'feature-<...feature_key...>.spaces-<SENSOR>'.
    Returns only sensors that have a matching column.
    """
    cols = list(df.columns)
    found: Dict[str, str] = {}
    # Regex to capture end '.spaces-<SENSOR>'
    for s in sensors:
        pat = re.compile(rf"feature-.*{re.escape(feature_key)}.*\.spaces-{re.escape(s)}$")
        match_cols = [c for c in cols if pat.search(str(c))]
        if match_cols:
            found[s] = match_cols[0]
    return found


def main():
    parser = argparse.ArgumentParser(description="Topomap for FOOOF offset with group stats and Pz/P4 labels")
    parser.add_argument("--csv", required=True, help="Path to prepared X/y CSV from run_ml_pipe.py")
    parser.add_argument("--target", default="TDAH", help="Target column name (default: TDAH)")
    parser.add_argument("--feature-key", default="foofOffset", help="Feature key to look for (default: foofOffset)")
    parser.add_argument("--group-a", type=float, default=None, help="Label value for group A (e.g., 0)")
    parser.add_argument("--group-b", type=float, default=None, help="Label value for group B (e.g., 1)")
    parser.add_argument("--alpha", type=float, default=0.05, help="Alpha for BH-FDR (default: 0.05)")
    parser.add_argument("--n-perm", type=int, default=2000, help="Permutations if SciPy unavailable (default: 2000)")
    parser.add_argument("--save", default=None, help="Output path to save figure (PNG)")
    parser.add_argument("--no-show", action="store_true", help="Do not display the figure interactively")

    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(args.csv)

    df = pd.read_csv(args.csv)
    if args.target not in df.columns:
        # try to infer last column as target
        warnings.warn(f"Target '{args.target}' not found; using last column as target")
        args.target = df.columns[-1]

    y = df[args.target].values
    # Choose groups
    uniq = pd.Series(y).dropna().unique()
    if args.group_a is None or args.group_b is None:
        uniq_sorted = np.sort(uniq)
        if uniq_sorted.size < 2:
            raise ValueError("Need at least two distinct classes in target to compare groups")
        gA, gB = uniq_sorted[0], uniq_sorted[-1]
    else:
        gA, gB = args.group_a, args.group_b

    # Sensor list (consistent with configs)
    sensors = ['C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'Fz', 'O1',
               'O2', 'P3', 'P4', 'Pz', 'T3', 'T4', 'T5', 'T6']

    # Find feature columns for feature-key; fallback across common spellings
    mapping = _get_feature_columns(df, args.feature_key, sensors)
    if not mapping:
        for alt_key in ("fooofOffset", "foofOffset"):
            if args.feature_key != alt_key:
                mapping = _get_feature_columns(df, alt_key, sensors)
                if mapping:
                    break
    if not mapping:
        # Give a helpful message
        candidates = [c for c in df.columns if isinstance(c, str) and ("foof" in c or "fooof" in c)]
        raise KeyError(
            f"Could not find columns for feature-key '{args.feature_key}'. Found candidates: {candidates[:8]}"
        )

    sensors_present = [s for s in sensors if s in mapping]
    cols = [mapping[s] for s in sensors_present]

    X_feat = df[cols].values
    # Group masks
    mask_A = (y == gA)
    mask_B = (y == gB)
    if not mask_A.any() or not mask_B.any():
        raise ValueError(f"Groups not found: group_a={gA}, group_b={gB}")

    mean_A = np.nanmean(X_feat[mask_A], axis=0)
    mean_B = np.nanmean(X_feat[mask_B], axis=0)
    diff = mean_B - mean_A

    # Per-sensor p-values
    pvals = []
    for j in range(X_feat.shape[1]):
        p = _ttest_or_permutation(X_feat[mask_B, j], X_feat[mask_A, j], n_perm=args.n_perm)
        pvals.append(p)
    pvals = np.array(pvals)
    sig_mask = _bh_fdr(pvals, alpha=args.alpha)

    # Build MNE info and positions
    info = mne.create_info(ch_names=sensors_present, sfreq=100.0, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info.set_montage(montage, on_missing="warn")

    # Build labels list so only Pz and P4 are shown (others blank)
    names_for_labels = [ch if ch in {"Pz", "P4"} else "" for ch in sensors_present]

    # Prepare figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = [f"Group {gA} mean", f"Group {gB} mean", f"Diff (B - A)"]
    datas = [mean_A, mean_B, diff]
    cmaps = ["viridis", "viridis", "RdBu_r"]
    vlim_shared = np.nanmax(np.abs(diff))
    for ax, data, title, cmap in zip(axes, datas, titles, cmaps):
        if cmap == "RdBu_r":
            vmin, vmax = -vlim_shared if np.isfinite(vlim_shared) and vlim_shared > 0 else None, vlim_shared if np.isfinite(vlim_shared) and vlim_shared > 0 else None
        else:
            vmin = vmax = None
        im, cn = mne.viz.plot_topomap(
            data,
            info,
            axes=ax,
            show=False,
            contours=6,
            cmap=cmap,
            vlim=(vmin, vmax),
            names=names_for_labels,
            # mask=(sig_mask if cmap == "RdBu_r" else None),
            # mask_params=dict(marker="o", markerfacecolor="none", markeredgecolor="k", markersize=8, linewidth=1.5) if cmap == "RdBu_r" else None,
        )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)

    fig.suptitle(f"FOOOF offset topomap — feature: {args.feature_key} — N_A={int(mask_A.sum())}, N_B={int(mask_B.sum())}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
