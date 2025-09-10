#!/usr/bin/env python3
"""
Compute 2D embeddings (PCA, UMAP, t-SNE) from features_patients.csv after
cleaning features as in run_ml_pipe (without balancing), and plot color-coded
scatter maps for:
  - TDAH category (excluding class 2)
  - psychostimulant category (only classes 0, 1, 2)

Usage examples
--------------
python eeg_adhd_epilepsy_psychostimulant/viz/plot_embeddings.py \
  --data data/results/merged/features_patients.csv \
  --config eeg_adhd_epilepsy_psychostimulant/ml/config_adhd.yml \
  --out-dir results/embeddings

Notes
-----
- Cleaning is performed using coco_pipe.io.clean_features with parameters
  loaded from the provided YAML config (defaults.clean_cfg). No class balancing
  is applied.
- Features are standardized before embeddings.
- Requires: pandas, numpy, scikit-learn, umap-learn, matplotlib, pyyaml,
  coco-pipe (for clean_features). If coco-pipe is unavailable, a minimal
  fallback cleaning is used.
"""

import argparse
import os
from typing import Dict, Any, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

try:
    import yaml
except Exception as e:  # pragma: no cover
    yaml = None

# Optional: use coco_pipe I/O for consistent loading/cleaning
try:
    from coco_pipe.io import load as coco_load
    from coco_pipe.io import clean_features as coco_clean_features
except Exception:  # pragma: no cover
    coco_load = None
    coco_clean_features = None

try:
    import umap  # type: ignore
except Exception as e:  # pragma: no cover
    umap = None


def read_config_clean_cfg(path: Optional[str]) -> Dict[str, Any]:
    """Load defaults.clean_cfg, sep, reverse from a YAML config, if provided.

    Returns a dict with keys: mode, sep, reverse, verbose, min_abs_value,
    min_abs_fraction. Missing values are filled with sensible defaults.
    """
    clean_cfg: Dict[str, Any] = {
        "mode": "sensor_wide",
        "sep": ".spaces-",
        "reverse": True,
        "verbose": True,
        "min_abs_value": 1e-6,
        "min_abs_fraction": 0.2,
    }
    if path and yaml is not None and os.path.exists(path):
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        defaults = (cfg or {}).get("defaults", {})
        if isinstance(defaults, dict):
            # Clean cfg
            cc = defaults.get("clean_cfg")
            if isinstance(cc, dict):
                clean_cfg.update({k: v for k, v in cc.items() if v is not None})
            # Also allow overriding sep/reverse from defaults
            if "sep" in defaults and defaults["sep"] is not None:
                clean_cfg["sep"] = defaults["sep"]
            if "reverse" in defaults and defaults["reverse"] is not None:
                clean_cfg["reverse"] = defaults["reverse"]
    return clean_cfg


def load_tabular(data_path: str) -> pd.DataFrame:
    """Load CSV using coco_pipe if available, else pandas.read_csv.
    """
    if coco_load is not None:
        return coco_load("tabular", data_path)
    return pd.read_csv(data_path)


def select_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Select feature columns starting with 'feature-'.
    Keeps order as in the DataFrame.
    """
    feat_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("feature-")]
    X = df[feat_cols].copy()
    return X


def clean_feature_matrix(X: pd.DataFrame, clean_cfg: Dict[str, Any]) -> pd.DataFrame:
    """Apply coco_pipe clean_features if available, else a minimal fallback.
    Returns the cleaned X.
    """
    if coco_clean_features is not None:
        # Filter kwargs to match function signature when possible
        kwargs = dict(
            mode=clean_cfg.get("mode", "sensor_wide"),
            sep=clean_cfg.get("sep", ".spaces-"),
            reverse=clean_cfg.get("reverse", True),
            verbose=clean_cfg.get("verbose", True),
            min_abs_value=clean_cfg.get("min_abs_value", 1e-6),
            min_abs_fraction=clean_cfg.get("min_abs_fraction", 0.2),
        )
        try:
            from inspect import signature
            sig = signature(coco_clean_features)
            kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        except Exception:
            pass
        Xc, report = coco_clean_features(X, **kwargs)
        # Optional: print brief report to console
        try:
            n_drop = len(report.get("dropped_columns", []))
            print(f"Cleaned features: dropped {n_drop} columns; {report.get('n_before')} -> {report.get('n_after')}")
        except Exception:
            pass
        return Xc

    # Fallback minimal cleaning: drop non-numeric, all-NaN, and near-constant columns
    Xn = X.apply(pd.to_numeric, errors="coerce")
    Xn = Xn.dropna(axis=1, how="all")
    # Drop columns with zero variance or almost all zeros
    nunique = Xn.nunique(dropna=True)
    keep = nunique[nunique > 1].index
    Xn = Xn[keep]
    zero_frac = (Xn == 0).sum() / max(1, len(Xn))
    Xn = Xn.loc[:, zero_frac < 0.95]
    print(f"[fallback] Cleaned features: kept {Xn.shape[1]} columns")
    return Xn


def standardize(X: pd.DataFrame) -> np.ndarray:
    scaler = StandardScaler(with_mean=True, with_std=True)
    return scaler.fit_transform(X.values)


def compute_embeddings(Xs: np.ndarray, random_state: int = 42) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    # PCA
    # PCA (deterministic given data; no need for random_state in most cases)
    pca = PCA(n_components=2)
    out["pca"] = pca.fit_transform(Xs)
    # UMAP (if available)
    if umap is not None:
        try:
            um = umap.UMAP(n_components=2, random_state=random_state)
            out["umap"] = um.fit_transform(Xs)
        except Exception as e:
            print(f"UMAP failed: {e}")
    else:
        print("UMAP not available; install umap-learn to enable.")
    # t-SNE
    try:
        tsne = TSNE(n_components=2, random_state=random_state, init="pca", learning_rate="auto")
        out["tsne"] = tsne.fit_transform(Xs)
    except Exception as e:
        print(f"t-SNE failed: {e}")
    return out


def plot_scatter(
    emb: np.ndarray,
    labels: pd.Series,
    title: str,
    out_path: Optional[str] = None,
    cmap: str = "tab10",
    alpha: float = 0.9,
    label_map: Optional[Dict[Any, str]] = None,
    class_order: Optional[Sequence[Any]] = None,
):
    """Scatter plot of a 2D embedding with consistent legend colors.

    - label_map: maps raw labels (e.g., 0/1/2) to display names.
    - class_order: sequence defining the order of classes/colors in legend.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    # Determine class order
    uniq_raw = list(pd.unique(labels))
    if class_order is None:
        # Preserve numeric sort if possible; fallback to unique order
        try:
            class_order = sorted(uniq_raw)
        except Exception:
            class_order = uniq_raw

    # Build a color for each class based on a discrete colormap
    cm = plt.cm.get_cmap(cmap, len(class_order))
    color_by_class: Dict[Any, Any] = {c: cm(i) for i, c in enumerate(class_order)}

    # Colors per point
    point_colors = [color_by_class.get(v, (0.5, 0.5, 0.5, 1.0)) for v in labels]

    ax.scatter(
        emb[:, 0], emb[:, 1],
        c=point_colors,
        s=35,
        alpha=alpha,
        edgecolors='none'
    )

    # Legend with display names
    handles = []
    for c in class_order:
        disp = label_map.get(c, str(c)) if label_map else str(c)
        handles.append(
            plt.Line2D(
                [0], [0], marker='o', color='w', label=disp,
                markerfacecolor=color_by_class[c], markersize=8
            )
        )
    ax.legend(handles=handles, title="Class", loc="best")
    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.grid(True, linestyle=":", alpha=0.3)
    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser(description="2D embeddings of features after coco-pipe cleaning")
    ap.add_argument("--data", default="data/results/merged/features_patients.csv", help="Path to features_patients.csv")
    ap.add_argument("--config", default=None, help="Path to YAML config to read defaults.clean_cfg (optional)")
    ap.add_argument("--out-dir", default="results/embeddings", help="Output directory for plots")
    ap.add_argument("--tdah-column", default="TDAH", help="Column name for TDAH label")
    ap.add_argument("--med-column", default="psychostimulant_category", help="Column name for psychostimulant category label")
    ap.add_argument("--exclude-tdah", nargs="*", type=int, default=[2], help="TDAH classes to exclude (default: 2)")
    ap.add_argument("--med-classes", nargs="*", type=int, default=[0, 1, 2], help="Psychostimulant classes to keep (default: 0 1 2)")
    ap.add_argument("--no-show", action="store_true", help="Do not show plots; save only")
    args = ap.parse_args()

    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    if not os.path.exists(args.data):
        raise FileNotFoundError(args.data)

    clean_cfg = read_config_clean_cfg(args.config)
    print(f"Using clean_cfg: {clean_cfg}")

    # Load data and pick features
    df = load_tabular(args.data)
    # Ensure labels exist
    if args.tdah_column not in df.columns:
        raise KeyError(f"Missing column '{args.tdah_column}' in data")
    if args.med_column not in df.columns:
        raise KeyError(f"Missing column '{args.med_column}' in data")

    X_raw = select_feature_matrix(df)
    # Clean features as in run_ml_pipe (no balancing)
    X = clean_feature_matrix(X_raw, clean_cfg)

    # Standardize for embeddings
    Xs = standardize(X)

    # Compute embeddings once on the full set
    emb_map = compute_embeddings(Xs, random_state=42)

    # Prepare label subsets
    y_tdah = pd.Series(df[args.tdah_column].values, index=df.index, name=args.tdah_column)
    mask_tdah = (~y_tdah.isin(args.exclude_tdah)) & (~y_tdah.isna())

    y_med = pd.Series(df[args.med_column].values, index=df.index, name=args.med_column)
    mask_med = y_med.isin(args.med_classes) & (~y_med.isna())

    # Align masks with embeddings (same order as X)
    # Note: X was subset of df rows only via cleaning by columns, so row order preserved
    idx = X.index
    y_tdah = y_tdah.loc[idx]
    y_med = y_med.loc[idx]
    mask_tdah = mask_tdah.loc[idx]
    mask_med = mask_med.loc[idx]

    # Plot for each embedding
    os.makedirs(args.out_dir, exist_ok=True)
    for name, emb in emb_map.items():
        # --- TDAH: compare only 0 vs 1 ---
        e_tdah = emb[mask_tdah.values]
        l_tdah = y_tdah[mask_tdah]
        # Restrict to {0,1} to ensure order and consistent colors
        tdah_keep = l_tdah.isin([0, 1])
        e_tdah = e_tdah[tdah_keep.values]
        l_tdah = l_tdah[tdah_keep]
        tdah_label_map = {0: "CTRL", 1: "ADHD"}
        title_t = f"{name.upper()} – TDAH: ADHD vs CTRL"
        out_t = os.path.join(args.out_dir, f"{name}_tdah_ADHD_vs_CTRL.png")
        plot_scatter(
            e_tdah, l_tdah, title_t, out_path=out_t,
            label_map=tdah_label_map, class_order=[0, 1]
        )

        # --- Psychostimulant: 0/1/2 ---
        e_med = emb[mask_med.values]
        l_med = y_med[mask_med]
        med_label_map = {0: "Non-Psychostim", 1: "MPH", 2: "AMPH"}
        title_m = f"{name.upper()} – Stimulants: Non-Psychostim vs MPH vs AMPH"
        out_m = os.path.join(args.out_dir, f"{name}_psych_0-1-2.png")
        plot_scatter(
            e_med, l_med, title_m, out_path=out_m,
            label_map=med_label_map, class_order=[0, 1, 2]
        )

        # --- Psychostimulant: only 1 vs 2 (exclude 0) ---
        med12 = l_med.isin([1, 2])
        e_med12 = e_med[med12.values]
        l_med12 = l_med[med12]
        title_m12 = f"{name.upper()} – Stimulants: MPH vs AMPH"
        out_m12 = os.path.join(args.out_dir, f"{name}_psych_1-2.png")
        plot_scatter(
            e_med12, l_med12, title_m12, out_path=out_m12,
            label_map=med_label_map, class_order=[1, 2]
        )

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
