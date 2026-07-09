"""Diagnose *why* PCA on the raw EEG separates into a few tight clusters.

Same site / same machine is already ruled out, so this focuses on the
remaining in-pipeline / per-recording candidates and ranks them by how well
each explains the cluster structure actually present in the embedding.

What it does
------------
1. Rebuilds the exact container the report uses (use_derivatives, target_col=None).
2. Fits PCA on the flattened channel x time voltages (same as the report) and
   clusters the top-PC scores with KMeans(k) to recover the visible blobs.
3. Computes, per epoch, a set of interpretable signal features (amplitude,
   band powers, artifact stats) and joins per-recording metadata (subject, run,
   block-ordinal y, sfreq/n_times, diagnosis, age, sex, source_dataset).
4. Ranks every candidate against cluster membership:
     - categorical  -> adjusted mutual information (0=none, 1=perfect)
     - continuous   -> one-way ANOVA eta^2   (0=none, 1=perfect)
5. Runs the decisive amplitude test: silhouette of the clusters on RAW vs
   per-epoch-STANDARDIZED data. If standardizing collapses the separation,
   the driver is amplitude/scale.

Usage
-----
    python diagnose_raw_clusters.py --bids_root /path/to/BIDS \
        --condition EO_baseline --k 6 --max_epochs_per_rec 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Per-epoch signal features (what dominates raw-EEG PCA variance)
# --------------------------------------------------------------------------- #
def epoch_features(X: np.ndarray, sfreq: float) -> pd.DataFrame:
    """X: (n_obs, n_ch, n_time) -> tidy per-epoch feature frame."""
    n_obs = X.shape[0]
    flat = X.reshape(n_obs, -1)

    # amplitude / scale
    global_std = flat.std(axis=1)
    chan_rms = np.sqrt((X**2).mean(axis=2)).mean(axis=1)  # mean per-channel RMS
    ptp = (X.max(axis=2) - X.min(axis=2)).mean(axis=1)
    dc = np.abs(X.mean(axis=2)).mean(axis=1)  # residual offset (should be ~0)
    kurt = pd.DataFrame(flat.T).kurt().to_numpy()  # spikiness / artifacts

    feats = {
        "log_global_std": np.log10(global_std + 1e-20),
        "log_chan_rms": np.log10(chan_rms + 1e-20),
        "log_ptp": np.log10(ptp + 1e-20),
        "dc_offset": dc,
        "kurtosis": kurt,
    }

    # spectral: relative band powers averaged over channels
    freqs = np.fft.rfftfreq(X.shape[-1], d=1.0 / sfreq)
    psd = (np.abs(np.fft.rfft(X, axis=-1)) ** 2).mean(axis=1)  # (n_obs, n_freq)
    total = psd.sum(axis=1) + 1e-20
    bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 45),
        "hi_gt45": (45, freqs.max() + 1),
    }
    for name, (lo, hi) in bands.items():
        m = (freqs >= lo) & (freqs < hi)
        feats[f"relpow_{name}"] = psd[:, m].sum(axis=1) / total
    feats["log_total_power"] = np.log10(total)
    return pd.DataFrame(feats)


# --------------------------------------------------------------------------- #
# Association measures
# --------------------------------------------------------------------------- #
def eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    """One-way ANOVA eta^2 of a continuous var across cluster groups."""
    values = np.asarray(values, float)
    ok = np.isfinite(values)
    values, groups = values[ok], np.asarray(groups)[ok]
    if values.size == 0 or np.allclose(values.var(), 0):
        return 0.0
    grand = values.mean()
    ss_total = ((values - grand) ** 2).sum()
    ss_between = sum(
        (values[groups == g].mean() - grand) ** 2 * (groups == g).sum() for g in np.unique(groups)
    )
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def rank_candidates(df: pd.DataFrame, cluster: np.ndarray) -> pd.DataFrame:
    """Rank every column of df by association with cluster membership."""
    from sklearn.metrics import adjusted_mutual_info_score

    rows = []
    for col in df.columns:
        s = df[col]
        if s.nunique(dropna=True) <= 1:
            continue
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 12:
            rows.append((col, "continuous", "eta^2", eta_squared(s.to_numpy(), cluster)))
        else:
            codes = s.astype("string").fillna("NA")
            rows.append(
                (col, "categorical", "AMI", adjusted_mutual_info_score(codes, cluster.astype(str)))
            )
    out = pd.DataFrame(rows, columns=["variable", "kind", "metric", "score"])
    return out.sort_values("score", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bids_root", required=True)
    p.add_argument("--condition", default="EO_baseline")
    p.add_argument("--k", type=int, default=6, help="clusters to recover (match the plot)")
    p.add_argument("--n_pcs", type=int, default=10)
    p.add_argument("--max_epochs_per_rec", type=int, default=5)
    p.add_argument("--max_records", type=int, default=None)
    args = p.parse_args()

    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    from eeg_adhd_epilepsy.analysis.dataset import build_container

    container = build_container(
        bids_root=Path(args.bids_root),
        use_derivatives=True,
        task="clinical",
        desc="base",
        condition=args.condition,
        target_col=None,
    )
    X = np.asarray(container.X)  # (n_obs, n_ch, n_time) after crop-to-min
    sfreq = float(container.meta.get("sfreq", 0.0)) or 256.0
    obs = container.observation_frame().reset_index(drop=True)
    obs["y_block_ordinal"] = np.asarray(container.y)
    print(
        f"[info] X={X.shape}  container.sfreq={sfreq}  "
        f"note: n_time is post-crop and constant here (crop already applied)."
    )

    # Subsample epochs per recording so PCA is tractable and balanced.
    key = next((c for c in ("subject", "obs_id", "sample_id") if c in obs.columns), None)
    if key is not None and args.max_epochs_per_rec:
        idx = (
            obs.groupby(key, group_keys=False)
            .apply(lambda g: g.sample(min(len(g), args.max_epochs_per_rec), random_state=0))
            .index.to_numpy()
        )
        if args.max_records:
            keep_keys = pd.Index(obs.loc[idx, key].unique()[: args.max_records])
            idx = idx[obs.loc[idx, key].isin(keep_keys).to_numpy()]
        X, obs = X[idx], obs.iloc[idx].reset_index(drop=True)
    print(f"[info] after subsample: X={X.shape}")

    flat = X.reshape(X.shape[0], -1)

    # ---- reproduce the report's raw PCA + recover clusters --------------- #
    raw_scores = PCA(n_components=args.n_pcs, random_state=0).fit_transform(flat)
    cluster = KMeans(n_clusters=args.k, n_init=10, random_state=0).fit_predict(raw_scores)

    # ---- interpretable features + metadata ------------------------------ #
    feats = epoch_features(X, sfreq)
    meta_cols = [c for c in obs.columns if obs[c].nunique(dropna=True) <= 400]
    candidates = pd.concat([obs[meta_cols].reset_index(drop=True), feats], axis=1)

    print("\n================ RANKED EXPLANATIONS FOR THE CLUSTERS ================")
    print("(AMI/eta^2 near 1.0 => that variable explains the split; near 0 => it doesn't)\n")
    print(rank_candidates(candidates, cluster).to_string(index=False))

    # ---- decisive amplitude test: raw vs per-epoch standardized --------- #
    sil_raw = silhouette_score(raw_scores, cluster)
    std_flat = StandardScaler().fit_transform(flat.T).T  # z-score each epoch
    std_scores = PCA(n_components=args.n_pcs, random_state=0).fit_transform(std_flat)
    std_clu = KMeans(n_clusters=args.k, n_init=10, random_state=0).fit_predict(std_scores)
    sil_std = silhouette_score(std_scores, std_clu)
    print("\n================ AMPLITUDE / SCALE TEST ================")
    print(f"silhouette (RAW voltages)              : {sil_raw:.3f}")
    print(f"silhouette (per-epoch z-scored)        : {sil_std:.3f}")
    if sil_std < 0.5 * sil_raw:
        print(">> Standardizing collapses the separation => AMPLITUDE/SCALE is the driver.")
    else:
        print(">> Clusters survive standardization => driver is SHAPE/SPECTRAL, not raw scale.")

    # ---- per-cluster fingerprint ---------------------------------------- #
    summary = candidates.assign(cluster=cluster).groupby("cluster")
    print("\n================ PER-CLUSTER FINGERPRINT (means) ================")
    show = [
        c
        for c in (
            "log_global_std",
            "log_total_power",
            "relpow_alpha",
            "relpow_hi_gt45",
            "dc_offset",
            "kurtosis",
            "y_block_ordinal",
        )
        if c in candidates.columns
    ]
    print(summary[show].mean(numeric_only=True).to_string())
    print("\ncluster sizes:", np.bincount(cluster).tolist())


if __name__ == "__main__":
    main()
