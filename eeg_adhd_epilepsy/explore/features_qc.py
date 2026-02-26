#!/usr/bin/env python
"""Comprehensive EEG feature quality control pipeline.

This script inspects extracted EEG features that follow the naming pattern
``feature-<feature_name>.spaces-<sensor>`` and generates a full QC report with
per-feature/per-subject statistics, visualizations, flagging logic, and
machine-readable summaries. It is designed for large studies (>=1000 subjects)
and follows the requirements provided in the project brief.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# Use a non-interactive backend to allow script execution on headless servers.
import matplotlib

matplotlib.use("Agg")

from pandas.plotting import parallel_coordinates
from scipy import signal, stats
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

try:
    import umap
except Exception:  # pragma: no cover - optional dependency
    umap = None  # type: ignore

DEFAULT_FEATURE_CATEGORIES: Dict[str, Dict[str, Any]] = {
    "frequency_raw": {
        "features": ["delta_raw", "theta_raw", "alpha_raw", "beta_raw", "gamma_raw"],
        "expected_range": [0, None],
        "description": "Raw frequency band powers",
    },
    "frequency_fooof": {
        "features": ["delta_fooof", "theta_fooof", "alpha_fooof", "beta_fooof", "gamma_fooof"],
        "expected_range": [0, None],
        "compare_to": "frequency_raw",
        "description": "FOOOF-corrected frequency band powers",
    },
    "entropy": {
        "features": ["sample_entropy", "perm_entropy", "spectral_entropy"],
        "expected_range": [0, 1],
        "description": "Entropy measures",
    },
    "complexity": {
        "features": ["fractal_dim", "hurst", "dfa", "lzc"],
        "expected_range": None,
        "description": "Complexity measures",
    },
}


class FeatureQCFlags:
    """Flag identifiers that are written to the QC outputs."""

    MISSING_HIGH = "high_missing_rate"
    CONSTANT = "constant_or_near_constant"
    NEGATIVE_POWER = "unexpected_negative_values"
    EXTREME_OUTLIERS = "extreme_outliers"
    SKEWED = "highly_skewed"
    INF_VALUES = "infinite_values"
    OUT_OF_BOUNDS = "out_of_expected_bounds"
    FOOOF_ISSUE = "fooof_correction_issue"
    CORRELATION_ISSUE = "unexpected_correlation"
    BIMODAL = "distribution_bimodal"


EXPECTED_BOUNDS = {
    "power": (0, None),
    "entropy_normalized": (0, 1),
    "entropy": (0, None),
    "hurst": (0, 1),
    "fractal_dim": (1, 2),
}

MAX_ABS_VALUE = 1e12


@dataclass
class FeatureMetadata:
    column: str
    base_feature: str
    sensor: Optional[str]
    category: Optional[str]
    feature_type: Optional[str]


def setup_logging(output_dir: Path, log_level: str) -> Path:
    """Configure logging to both stdout and a file."""

    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "features_qc.log"

    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(),
        ],
    )
    return log_path


def load_feature_categories(config_path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """Load feature categories from YAML if provided."""

    if not config_path:
        return DEFAULT_FEATURE_CATEGORIES

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyYAML is required to load a feature category config.") from exc

    config_path = str(config_path)
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    categories = config.get("feature_categories")
    if not categories:
        raise ValueError("feature_config is missing 'feature_categories' key.")
    return categories


def parse_feature_column_name(column: str) -> Tuple[str, Optional[str]]:
    """Split a feature column into base feature name and sensor name."""

    base = column
    sensor = None
    if column.startswith("feature-"):
        rest = column[len("feature-") :]
        if ".spaces-" in rest:
            base, sensor = rest.split(".spaces-", 1)
        else:
            base = rest
    base = base.replace(".", "_").replace("-", "_").lower()
    if sensor is not None:
        sensor = sensor.lower()
    return base, sensor


def determine_feature_type(base_feature: str) -> Optional[str]:
    """Infer a feature type to look up expected bounds."""

    bf = base_feature.lower()
    if any(band in bf for band in ("delta", "theta", "alpha", "beta", "gamma")):
        return "power"
    if "entropy" in bf:
        return "entropy_normalized" if "norm" in bf else "entropy"
    if "hurst" in bf:
        return "hurst"
    if "fractal" in bf or bf.startswith("fd") or "higuchi" in bf:
        return "fractal_dim"
    return None


def build_feature_metadata(
    columns: Iterable[str], feature_categories: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """Construct metadata for each feature column."""

    category_map: Dict[str, str] = {}
    for category, payload in feature_categories.items():
        for feat in payload.get("features", []):
            category_map[feat.lower()] = category

    records: List[FeatureMetadata] = []
    for col in columns:
        base, sensor = parse_feature_column_name(col)
        category = category_map.get(base)
        feature_type = determine_feature_type(base)
        records.append(
            FeatureMetadata(
                column=col,
                base_feature=base,
                sensor=sensor,
                category=category,
                feature_type=feature_type,
            )
        )
    return pd.DataFrame(records)


def sanitize_numeric_df(
    df: pd.DataFrame, *, min_non_na: int = 2, max_abs: float = MAX_ABS_VALUE
) -> pd.DataFrame:
    """Replace infinities, drop columns without enough data, fill gaps, and clip extremes."""

    sanitized = df.replace([np.inf, -np.inf], np.nan).copy()
    valid_cols = [col for col in sanitized.columns if sanitized[col].notna().sum() >= min_non_na]
    sanitized = sanitized[valid_cols]
    if sanitized.empty:
        return sanitized
    medians = sanitized.median()
    sanitized = sanitized.fillna(medians)
    sanitized = sanitized.clip(lower=-max_abs, upper=max_abs)
    return sanitized


def detect_outliers_iqr(series: pd.Series, k: float = 1.5) -> pd.Series:
    """Detect outliers using the IQR method."""

    series = series.replace([np.inf, -np.inf], np.nan)
    data = series.dropna()
    if data.empty:
        return pd.Series(False, index=series.index)
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return pd.Series(False, index=series.index)
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    mask = (series < lower) | (series > upper)
    return mask.fillna(False)


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Detect outliers using the Z-score method."""

    series = series.replace([np.inf, -np.inf], np.nan)
    data = series.dropna()
    if data.empty or data.std(ddof=0) == 0:
        return pd.Series(False, index=series.index)
    zscores = (series - data.mean()) / data.std(ddof=0)
    mask = zscores.abs() > threshold
    return mask.fillna(False)


def detect_multivariate_outliers(
    df: pd.DataFrame, contamination: float = 0.05, random_state: int = 42
) -> Tuple[pd.Series, pd.Series]:
    """Detect multivariate outliers using IsolationForest."""

    mask = pd.Series(False, index=df.index)
    scores = pd.Series(0.0, index=df.index)
    clean_df = sanitize_numeric_df(df)
    if clean_df.empty or clean_df.shape[0] < 20 or clean_df.shape[1] < 2:
        return mask, scores

    scaler = StandardScaler()
    scaled = scaler.fit_transform(clean_df)
    iso = IsolationForest(
        contamination=min(max(contamination, 0.001), 0.5),
        random_state=random_state,
        n_estimators=200,
    )
    preds = iso.fit_predict(scaled)
    mask.loc[clean_df.index] = preds == -1
    scores.loc[clean_df.index] = -iso.decision_function(scaled)
    return mask, scores


def detect_bimodality(data: np.ndarray) -> bool:
    """Detect bimodal distributions using KDE peaks."""

    data = np.asarray(data[np.isfinite(data)], dtype=float)
    if data.size < 20 or np.nanstd(data) == 0:
        return False
    try:
        kde = gaussian_kde(data)
    except (np.linalg.LinAlgError, ValueError):
        return False
    xs = np.linspace(np.nanmin(data), np.nanmax(data), 200)
    density = kde(xs)
    peaks, _ = signal.find_peaks(density)
    return len(peaks) >= 2


def compute_feature_statistics(
    df: pd.DataFrame,
    metadata: pd.DataFrame,
    missing_threshold: float,
    iqr_outliers: Optional[pd.DataFrame] = None,
    z_outliers: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute comprehensive statistics for each feature."""

    records = []
    for col in df.columns:
        series = df[col]
        sanitized = series.replace([np.inf, -np.inf], np.nan)
        data = sanitized.dropna()
        info = {
            "feature": col,
            "base_feature": metadata.loc[metadata["column"] == col, "base_feature"].iloc[0],
            "sensor": metadata.loc[metadata["column"] == col, "sensor"].iloc[0],
        }
        info["count"] = int(series.count())
        info["missing_count"] = int(series.isna().sum())
        info["missing_pct"] = float(series.isna().mean())

        if data.empty:
            stats_defaults = {
                "mean": np.nan,
                "median": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
                "skewness": np.nan,
                "kurtosis": np.nan,
            }
            info.update(stats_defaults)
            for pct in (1, 5, 25, 50, 75, 95, 99):
                info[f"pct_{pct}"] = np.nan
            info["zeros"] = 0
            info["negative_values"] = 0
            info["infinite_values"] = 0
            info["iqr_outliers"] = 0
            info["zscore_outliers"] = 0
            info["bimodal"] = False
            info["variance"] = np.nan
            records.append(info)
            continue

        info["mean"] = float(data.mean())
        info["median"] = float(data.median())
        info["std"] = float(data.std(ddof=1))
        info["variance"] = float(data.var(ddof=1))
        info["min"] = float(data.min())
        info["max"] = float(data.max())
        info["skewness"] = float(stats.skew(data)) if data.size > 2 else np.nan
        info["kurtosis"] = float(stats.kurtosis(data)) if data.size > 3 else np.nan
        for pct in (1, 5, 25, 50, 75, 95, 99):
            info[f"pct_{pct}"] = float(np.percentile(data, pct))
        info["zeros"] = int((series == 0).sum())
        info["negative_values"] = int((series < 0).sum())
        info["infinite_values"] = int(np.isinf(series).sum())
        base_series = series.replace([np.inf, -np.inf], np.nan)
        iqr_mask = detect_outliers_iqr(base_series) if iqr_outliers is None else iqr_outliers[col]
        z_mask = detect_outliers_zscore(base_series) if z_outliers is None else z_outliers[col]
        info["iqr_outliers"] = int(iqr_mask.sum())
        info["zscore_outliers"] = int(z_mask.sum())
        info["bimodal"] = bool(detect_bimodality(data.values))
        records.append(info)
    stats_df = pd.DataFrame.from_records(records)
    stats_df["is_constant"] = stats_df["variance"].fillna(0) < 1e-10
    stats_df["missing_flag"] = stats_df["missing_pct"] > missing_threshold
    return stats_df


def compute_sensor_group_stats(df: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sensor-specific columns into grouped statistics per base feature."""

    grouped = metadata.groupby("base_feature")["column"].apply(list)
    records = []
    for base, cols in grouped.items():
        data = df[cols].values.flatten()
        data = data[~np.isnan(data)]
        if data.size == 0:
            continue
        record = {
            "base_feature": base,
            "sensor_count": len(cols),
            "mean": float(np.mean(data)),
            "median": float(np.median(data)),
            "std": float(np.std(data, ddof=1)) if data.size > 1 else 0.0,
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "skewness": float(stats.skew(data)) if data.size > 2 else np.nan,
            "kurtosis": float(stats.kurtosis(data)) if data.size > 3 else np.nan,
        }
        records.append(record)
    return pd.DataFrame(records)


def get_expected_bounds(
    metadata_row: pd.Series, feature_categories: Dict[str, Dict[str, Any]]
) -> Optional[Tuple[Optional[float], Optional[float]]]:
    """Determine the expected bounds for a feature."""

    bounds = None
    feature_type = metadata_row["feature_type"]
    if feature_type and feature_type in EXPECTED_BOUNDS:
        bounds = EXPECTED_BOUNDS[feature_type]
    category = metadata_row["category"]
    if category and feature_categories.get(category, {}).get("expected_range") is not None:
        cat_bounds = feature_categories[category]["expected_range"]
        bounds = (cat_bounds[0], cat_bounds[1])
    return bounds


def flag_features(
    stats_df: pd.DataFrame,
    metadata: pd.DataFrame,
    feature_categories: Dict[str, Dict[str, Any]],
    thresholds: Dict[str, Any],
) -> pd.DataFrame:
    """Flag features based on QC criteria."""

    records = []
    skew_thr = thresholds.get("skewness", 3)
    kurt_thr = thresholds.get("kurtosis", 10)
    missing_thr = thresholds.get("missing", 0.05)
    variance_thr = thresholds.get("variance", 1e-10)

    for _, row in stats_df.iterrows():
        feature = row["feature"]
        meta_row = metadata.loc[metadata["column"] == feature].iloc[0]
        issues: List[str] = []
        if row["missing_pct"] > missing_thr:
            issues.append(FeatureQCFlags.MISSING_HIGH)
        if row["variance"] <= variance_thr or row["is_constant"]:
            issues.append(FeatureQCFlags.CONSTANT)
        if row["negative_values"] > 0 and determine_feature_type(meta_row["base_feature"]) == "power":
            issues.append(FeatureQCFlags.NEGATIVE_POWER)
        if row["iqr_outliers"] > thresholds.get("iqr_outliers", max(5, 0.05 * row["count"])):
            issues.append(FeatureQCFlags.EXTREME_OUTLIERS)
        if abs(row.get("skewness", 0) or 0) > skew_thr or abs(row.get("kurtosis", 0) or 0) > kurt_thr:
            issues.append(FeatureQCFlags.SKEWED)
        if row["infinite_values"] > 0:
            issues.append(FeatureQCFlags.INF_VALUES)
        bounds = get_expected_bounds(meta_row, feature_categories)
        if bounds:
            low, high = bounds
            min_val = row.get("min")
            max_val = row.get("max")
            if (low is not None and min_val is not None and min_val < low - thresholds.get("tolerance", 1e-8)) or (
                high is not None and max_val is not None and max_val > high + thresholds.get("tolerance", 1e-8)
            ):
                issues.append(FeatureQCFlags.OUT_OF_BOUNDS)
        if row.get("bimodal"):
            issues.append(FeatureQCFlags.BIMODAL)
        if issues:
            records.append(
                {
                    "feature": feature,
                    "base_feature": meta_row["base_feature"],
                    "sensor": meta_row["sensor"],
                    "issues": ";".join(issues),
                }
            )
    return pd.DataFrame(records)


def identify_subject_column(df: pd.DataFrame, subject_id_col: Optional[str]) -> Optional[str]:
    """Identify subject identifier column."""

    if subject_id_col and subject_id_col in df.columns:
        return subject_id_col
    candidates = ["subject", "subject_id", "participant", "participant_id", "id", "record_id"]
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def flag_subjects(
    df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    subject_ids: pd.Series,
    outlier_matrix: pd.DataFrame,
    thresholds: Dict[str, Any],
) -> pd.DataFrame:
    """Flag subjects based on QC criteria."""

    missing_fraction = df.isna().mean(axis=1)
    outlier_fraction = outlier_matrix.mean(axis=1)

    category_groups = metadata_df.groupby("category")["column"].apply(list).to_dict()
    critical_features = [col for col in df.columns if "alpha" in parse_feature_column_name(col)[0]]

    records = []
    for idx in df.index:
        reasons = []
        if missing_fraction.loc[idx] > thresholds.get("subject_missing", 0.1):
            reasons.append("high_missing_fraction")
        if outlier_fraction.loc[idx] > thresholds.get("subject_outliers", 0.2):
            reasons.append("many_outliers")

        if critical_features:
            if df.loc[idx, critical_features].isna().any():
                reasons.append("critical_feature_missing")

        missing_categories = []
        for category, cols in category_groups.items():
            if not cols or category is None:
                continue
            available_cols = [c for c in cols if c in df.columns]
            if not available_cols:
                continue
            if df.loc[idx, available_cols].isna().all():
                missing_categories.append(category)
        if missing_categories:
            reasons.append(f"missing_categories:{','.join(missing_categories)}")

        if reasons:
            records.append(
                {
                    "subject_index": idx,
                    "subject_id": subject_ids.loc[idx] if subject_ids is not None else idx,
                    "missing_fraction": float(missing_fraction.loc[idx]),
                    "outlier_fraction": float(outlier_fraction.loc[idx]),
                    "reasons": ";".join(reasons),
                }
            )
    return pd.DataFrame(records)


def identify_fooof_pairs(columns: Iterable[str]) -> List[Tuple[str, str, str]]:
    """Identify matching raw/FOOOF column pairs."""

    raw_map: Dict[str, str] = {}
    fooof_map: Dict[str, str] = {}
    for col in columns:
        base, _ = parse_feature_column_name(col)
        if base.endswith("_raw"):
            root = base[: -len("_raw")]
            raw_map[root] = col
        elif base.endswith("_fooof"):
            root = base[: -len("_fooof")]
            fooof_map[root] = col
    pairs = []
    for root, raw_col in raw_map.items():
        fooof_col = fooof_map.get(root)
        if fooof_col:
            pairs.append((raw_col, fooof_col, root))
    return pairs


def check_fooof_consistency(df: pd.DataFrame, pairs: List[Tuple[str, str, str]]) -> Dict[str, Any]:
    """Compare raw and FOOOF-corrected features."""

    if not pairs:
        return {
            "message": "No raw/FOOOF pairs detected.",
            "pair_count": 0,
            "correlations": {},
            "flagged_subjects": [],
            "extreme_ratio_count": 0,
        }

    correlations = {}
    flagged_subjects = set()
    extreme_ratio_count = 0
    negative_subjects = set()
    ratio_records = []

    for raw_col, fooof_col, root in pairs:
        subset = df[[raw_col, fooof_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if subset.empty:
            continue
        corr = subset[raw_col].corr(subset[fooof_col])
        correlations[root] = corr
        ratio = np.divide(
            subset[fooof_col],
            subset[raw_col].replace(0, np.nan),
        )
        ratio_records.append(ratio)
        extreme_mask = (ratio < 0.1) | (ratio > 10)
        extreme_ratio_count += int(extreme_mask.sum())
        flagged_subjects.update(subset.index[extreme_mask].tolist())
        negative_mask = subset[fooof_col] < 0
        negative_subjects.update(subset.index[negative_mask].tolist())

    ratio_concatenated = (
        pd.concat(ratio_records, axis=0).replace([np.inf, -np.inf], np.nan) if ratio_records else None
    )
    ratio_summary = ratio_concatenated.describe().to_dict() if ratio_concatenated is not None else {}

    return {
        "pair_count": len(correlations),
        "correlations": correlations,
        "flagged_subjects": sorted(flagged_subjects),
        "negative_subjects": sorted(negative_subjects),
        "extreme_ratio_count": extreme_ratio_count,
        "ratio_summary": ratio_summary,
    }


def generate_missing_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    """Generate heatmap of missing values."""

    sampling = min(500, len(df))
    sampled = df.sample(n=sampling, random_state=42) if len(df) > sampling else df
    plt.figure(figsize=(min(12, sampled.shape[1] * 0.25 + 4), 6))
    sns.heatmap(sampled.isna(), cbar=False)
    plt.title("Missing Data Heatmap")
    plt.xlabel("Features")
    plt.ylabel("Subjects")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def generate_outlier_heatmap(outlier_matrix: pd.DataFrame, output_path: Path) -> None:
    """Generate heatmap of outlier status per subject/feature."""

    if outlier_matrix.empty:
        return
    sampling = min(500, len(outlier_matrix))
    sampled = (
        outlier_matrix.sample(n=sampling, random_state=42) if len(outlier_matrix) > sampling else outlier_matrix
    )
    plt.figure(figsize=(min(12, sampled.shape[1] * 0.25 + 4), 6))
    sns.heatmap(sampled.astype(int), cmap="Reds", cbar=False)
    plt.title("Outlier Heatmap")
    plt.xlabel("Features")
    plt.ylabel("Subjects")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def clean_correlation_matrix(corr: pd.DataFrame) -> pd.DataFrame:
    """Ensure correlation matrix contains finite values and drop empty rows/columns."""

    if corr.empty:
        return corr
    corr = corr.replace([np.inf, -np.inf], np.nan)
    corr = corr.dropna(axis=0, how="all").dropna(axis=1, how="all")
    corr = corr.fillna(0.0)
    if corr.empty:
        return corr
    np.fill_diagonal(corr.values, 1.0)
    return corr


def generate_correlation_heatmap(corr: pd.DataFrame, output_path: Path) -> None:
    """Generate clustered correlation heatmap."""

    if corr.empty:
        return
    try:
        sns.clustermap(corr, cmap="coolwarm", center=0, figsize=(10, 10))
        plt.savefig(output_path, dpi=200)
        plt.close()
    except ValueError:
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, cmap="coolwarm", center=0)
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()


def generate_distribution_grid(df: pd.DataFrame, output_path: Path, max_features: int = 16) -> None:
    """Generate a grid of small histograms for an overview of distributions."""

    cols = df.columns[:max_features]
    if cols.empty:
        return
    rows = math.ceil(len(cols) / 4)
    fig, axes = plt.subplots(rows, 4, figsize=(16, rows * 3))
    axes = axes.flatten()
    for ax, col in zip(axes, cols):
        data = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if data.empty:
            ax.set_title(col)
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.axis("off")
            continue
        ax.hist(data, bins=30, color="steelblue", alpha=0.8)
        ax.set_title(col, fontsize=9)
    for ax in axes[len(cols) :]:
        ax.axis("off")
    fig.suptitle("Distribution Overview Grid", fontsize=16)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_feature_distribution(series: pd.Series, feature_name: str, output_path: Path) -> None:
    """Generate histogram/KDE, boxplot, QQ plot, and violin plot for a feature."""

    cleaned = series.replace([np.inf, -np.inf], np.nan).dropna()
    if cleaned.empty:
        return
    data = np.asarray(cleaned, dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return
    variance = np.var(data)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    # Histogram + KDE
    ax = axes[0, 0]
    ax.hist(data, bins=50, density=True, alpha=0.7, edgecolor="black")
    if variance > 0:
        try:
            kde = gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 100)
            ax.plot(x_range, kde(x_range), "r-", lw=2)
        except (np.linalg.LinAlgError, ValueError):
            pass
    mean_val = np.mean(data)
    median_val = np.median(data)
    ax.axvline(mean_val, color="blue", linestyle="--", label=f"Mean: {mean_val:.3f}")
    ax.axvline(median_val, color="green", linestyle="--", label=f"Median: {median_val:.3f}")
    ax.set_title("Histogram + KDE")
    ax.legend(fontsize=8)

    # Boxplot
    ax = axes[0, 1]
    ax.boxplot(data, vert=True)
    ax.set_title("Boxplot")
    ax.text(
        1.05,
        data.max() if data.max() != data.min() else 1,
        f"Outliers: {detect_outliers_iqr(series).sum()}",
        fontsize=9,
    )

    # QQ plot
    ax = axes[1, 0]
    if variance > 0:
        stats.probplot(data, dist="norm", plot=ax)
        ax.set_title("Q-Q Plot")
    else:
        ax.text(0.5, 0.5, "Constant values", ha="center", va="center")
        ax.set_axis_off()

    # Violin plot
    ax = axes[1, 1]
    if variance > 0:
        try:
            sns.violinplot(data=data, orient="h", ax=ax, color="lightgray", cut=0)
            ax.set_title("Violin Plot")
            ax.set_xlabel(feature_name)
        except (ValueError, np.linalg.LinAlgError):
            ax.text(0.5, 0.5, "Violin unavailable", ha="center", va="center")
            ax.set_axis_off()
    else:
        ax.text(0.5, 0.5, "Constant values", ha="center", va="center")
        ax.set_axis_off()

    plt.suptitle(feature_name)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def generate_parallel_coordinates(
    df: pd.DataFrame, outlier_mask: pd.Series, output_path: Path, max_features: int = 10
) -> None:
    """Create a parallel coordinates plot using a subset of features."""

    if df.empty:
        return
    selected_cols = df.columns[:max_features]
    subset = sanitize_numeric_df(df[selected_cols])
    if subset.empty:
        return
    scaled_array = StandardScaler().fit_transform(subset)
    scaled = pd.DataFrame(scaled_array, columns=subset.columns, index=subset.index)
    mask = outlier_mask.reindex(scaled.index).fillna(False)
    scaled["outlier"] = np.where(mask, "outlier", "normal")
    plt.figure(figsize=(12, 6))
    parallel_coordinates(scaled.reset_index(drop=True), "outlier", color=("#1f77b4", "#d62728"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def generate_pca_plot(
    df: pd.DataFrame, outlier_scores: pd.Series, output_path: Path, title: str = "PCA Outliers"
) -> None:
    """Generate PCA scatter plot colored by outlier score."""

    clean_df = sanitize_numeric_df(df)
    if clean_df.empty or clean_df.shape[1] < 2:
        return
    scaler = StandardScaler()
    transformed = scaler.fit_transform(clean_df)
    pca = PCA(n_components=2, random_state=42)
    comps = pca.fit_transform(transformed)
    plt.figure(figsize=(8, 6))
    scores = outlier_scores.reindex(clean_df.index).fillna(0)
    sc = plt.scatter(comps[:, 0], comps[:, 1], c=scores, cmap="viridis", s=20)
    plt.colorbar(sc, label="Outlier Score")
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def generate_umap_plot(
    df: pd.DataFrame, outlier_scores: pd.Series, output_path: Path, title: str = "UMAP Outliers"
) -> None:
    """Generate UMAP scatter plot if umap-learn is installed."""

    if umap is None or df.empty:
        return
    clean_df = sanitize_numeric_df(df)
    if clean_df.empty:
        return
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(clean_df)
    plt.figure(figsize=(8, 6))
    scores = outlier_scores.reindex(clean_df.index).fillna(0)
    sc = plt.scatter(embedding[:, 0], embedding[:, 1], c=scores, cmap="plasma", s=20)
    plt.colorbar(sc, label="Outlier Score")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def generate_html_report(results: Dict[str, Any], output_path: Path) -> None:
    """Generate comprehensive HTML report."""

    summary = results["summary"]
    flagged_features_html = results["flagged_features_df"].to_html(index=False, classes="table table-striped")
    flagged_subjects_html = results["flagged_subjects_df"].to_html(index=False, classes="table table-striped")
    stats_html = results["stats_df"].to_html(index=False, classes="table table-striped", float_format="%.5f")
    sensor_group_html = results["sensor_group_stats"].to_html(index=False, classes="table table-striped", float_format="%.5f")

    corr_pairs = results["high_corr_pairs"]
    corr_list_items = "".join(
        f"<li>{a} &mdash; {b}: {value:.3f}</li>" for a, b, value in corr_pairs[:20]
    )

    fooof = results["fooof"]
    fooof_items = "".join(
        f"<li>{k}: {v:.3f}</li>" for k, v in fooof.get("correlations", {}).items()
    )
    fooof_subjects = ", ".join(map(str, fooof.get("flagged_subjects", [])))

    html = f"""
    <html>
    <head>
        <meta charset="utf-8" />
        <title>EEG Features QC Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 2rem; }}
            h1, h2, h3 {{ color: #1f4e79; }}
            .section {{ margin-bottom: 2rem; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 0.4rem; font-size: 0.9rem; }}
            th {{ background-color: #f5f5f5; }}
        </style>
    </head>
    <body>
        <h1>EEG Features QC Report</h1>
        <div class="section">
            <h2>Executive Summary</h2>
            <p>Total subjects: {summary['total_subjects']}</p>
            <p>Total features: {summary['total_features']}</p>
            <p>Flagged subjects: {summary['flagged_subject_count']} ({summary['flagged_subject_pct']:.1f}%)</p>
            <p>Flagged features: {summary['flagged_feature_count']} ({summary['flagged_feature_pct']:.1f}%)</p>
            <p>Critical issues: {summary['critical_issues']}</p>
        </div>
        <div class="section">
            <h2>Dataset Overview</h2>
            {stats_html}
            <h3>Sensor Grouped Statistics</h3>
            {sensor_group_html}
        </div>
        <div class="section">
            <h2>Feature-Level Analysis</h2>
            <h3>Flagged Features</h3>
            {flagged_features_html}
            <h3>High Correlation Pairs</h3>
            <ul>{corr_list_items}</ul>
            <h3>FOOOF Correlations</h3>
            <ul>{fooof_items}</ul>
            <p>Flagged subjects (FOOOF): {fooof_subjects}</p>
        </div>
        <div class="section">
            <h2>Subject-Level Analysis</h2>
            {flagged_subjects_html}
        </div>
        <div class="section">
            <h2>Visualizations</h2>
            <ul>
                <li>Missing data heatmap: {results["figures"]["missing_data_heatmap"]}</li>
                <li>Outlier heatmap: {results["figures"]["outlier_heatmap"]}</li>
                <li>Correlation heatmap: {results["figures"]["correlation_heatmap"]}</li>
                <li>Distribution grid: {results["figures"]["distribution_grid"]}</li>
                <li>PCA outliers: {results["figures"]["pca_outliers"]}</li>
                <li>Parallel coordinates: {results["figures"]["parallel_coords"]}</li>
            </ul>
        </div>
        <div class="section">
            <h2>Recommendations</h2>
            <p>Review flagged features and consider removing or transforming them (log/sqrt for skewed).</p>
            <p>Inspect subjects listed above for potential exclusion.</p>
            <p>Cross-check FOOOF discrepancies and recalibrate the fitting parameters if necessary.</p>
        </div>
    </body>
    </html>
    """
    output_path.write_text(textwrap.dedent(html), encoding="utf-8")


def save_json_report(output_path: Path, results: Dict[str, Any]) -> None:
    """Save machine-readable JSON summary."""

    payload = {
        "summary": results["summary"],
        "flagged_features": results["flagged_features_df"].to_dict(orient="records"),
        "flagged_subjects": results["flagged_subjects_df"].to_dict(orient="records"),
        "fooof": results["fooof"],
        "high_corr_pairs": results["high_corr_pairs"],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def compute_correlation_pairs(corr: pd.DataFrame, threshold: float = 0.95) -> List[Tuple[str, str, float]]:
    """Identify highly correlated feature pairs."""

    pairs = []
    columns = corr.columns
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            value = corr.iloc[i, j]
            if abs(value) >= threshold:
                pairs.append((columns[i], columns[j], float(value)))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return pairs


def generate_reports(
    args: argparse.Namespace,
    df: pd.DataFrame,
    features_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Run the full QC pipeline and generate outputs."""

    # Outlier detection matrices
    iqr_out = pd.DataFrame(False, index=features_df.index, columns=features_df.columns)
    z_out = pd.DataFrame(False, index=features_df.index, columns=features_df.columns)
    for col in features_df.columns:
        iqr_out[col] = detect_outliers_iqr(features_df[col], k=args.iqr_k)
        z_out[col] = detect_outliers_zscore(features_df[col], threshold=args.zscore_threshold)

    stats_df = compute_feature_statistics(
        features_df,
        metadata_df,
        missing_threshold=args.missing_threshold,
        iqr_outliers=iqr_out,
        z_outliers=z_out,
    )
    sensor_group_stats = compute_sensor_group_stats(features_df, metadata_df)

    outlier_matrix = iqr_out | z_out if args.outlier_method == "both" else (iqr_out if args.outlier_method == "iqr" else z_out)

    subject_column = identify_subject_column(df, args.subject_id_col)
    subject_ids = df[subject_column] if subject_column else pd.Series(df.index, index=df.index)

    flagged_features_df = flag_features(
        stats_df,
        metadata_df,
        feature_categories=args.feature_categories,
        thresholds={
            "missing": args.missing_threshold,
            "variance": args.variance_threshold,
            "skewness": args.skew_threshold,
            "kurtosis": args.kurtosis_threshold,
            "iqr_outliers": args.feature_outlier_threshold,
            "tolerance": args.bound_tolerance,
        },
    )

    flagged_subjects_df = flag_subjects(
        features_df,
        metadata_df,
        subject_ids,
        outlier_matrix,
        thresholds={
            "subject_missing": args.subject_missing_threshold,
            "subject_outliers": args.subject_outlier_threshold,
        },
    )

    pairs = identify_fooof_pairs(features_df.columns)
    fooof = check_fooof_consistency(features_df, pairs)

    corr_matrix = features_df.replace([np.inf, -np.inf], np.nan).corr()
    corr = clean_correlation_matrix(corr_matrix)
    high_corr_pairs = compute_correlation_pairs(corr, threshold=args.corr_threshold) if not corr.empty else []

    # Multivariate outliers
    mv_outliers, mv_scores = detect_multivariate_outliers(features_df, contamination=args.contamination)
    lof = LocalOutlierFactor(
        n_neighbors=min(20, max(5, len(features_df) // 10)),
        contamination=min(max(args.contamination, 0.01), 0.5),
        novelty=False,
    )
    ml_ready = sanitize_numeric_df(features_df)
    if not ml_ready.empty:
        try:
            lof.fit_predict(ml_ready)
            lof_scores_series = pd.Series(-lof.negative_outlier_factor_, index=ml_ready.index)
        except ValueError:
            lof_scores_series = pd.Series(0.0, index=features_df.index)
    else:
        lof_scores_series = pd.Series(0.0, index=features_df.index)
    lof_scores = lof_scores_series.reindex(features_df.index).fillna(0.0)
    combined_scores = (mv_scores + lof_scores).reindex(features_df.index).fillna(0.0)

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True, parents=True)
    feature_dist_dir = figures_dir / "feature_distributions"
    feature_dist_dir.mkdir(exist_ok=True, parents=True)

    figures = {
        "missing_data_heatmap": figures_dir / "missing_data_heatmap.png",
        "outlier_heatmap": figures_dir / "outlier_heatmap.png",
        "correlation_heatmap": figures_dir / "correlation_heatmap.png",
        "distribution_grid": figures_dir / "distribution_grid.png",
        "pca_outliers": figures_dir / "pca_outliers.png",
        "parallel_coords": figures_dir / "parallel_coords.png",
        "umap_outliers": figures_dir / "umap_outliers.png",
    }

    generate_missing_heatmap(features_df, figures["missing_data_heatmap"])
    generate_outlier_heatmap(outlier_matrix, figures["outlier_heatmap"])
    generate_correlation_heatmap(corr, figures["correlation_heatmap"])
    generate_distribution_grid(features_df, figures["distribution_grid"])
    generate_pca_plot(features_df, combined_scores, figures["pca_outliers"])
    generate_parallel_coordinates(features_df, mv_outliers, figures["parallel_coords"])
    if args.generate_all_plots and umap is not None:
        generate_umap_plot(features_df, combined_scores, figures["umap_outliers"])

    if args.generate_all_plots:
        def _plot_feature(col: str) -> None:
            plot_feature_distribution(features_df[col], col, feature_dist_dir / f"{col}.png")

        joblib.Parallel(n_jobs=args.jobs)(
            joblib.delayed(_plot_feature)(col) for col in features_df.columns
        )

    # Save outputs
    stats_path = output_dir / "features_qc_summary.csv"
    stats_df.to_csv(stats_path, index=False)

    flagged_subjects_path = output_dir / "flagged_subjects.csv"
    flagged_subjects_df.to_csv(flagged_subjects_path, index=False)

    flagged_features_path = output_dir / "flagged_features.csv"
    flagged_features_df.to_csv(flagged_features_path, index=False)

    corr_path = output_dir / "correlation_matrix.csv"
    corr.to_csv(corr_path, index=True)

    # Summary for HTML/JSON
    summary = {
        "total_subjects": len(df),
        "total_features": features_df.shape[1],
        "flagged_subject_count": len(flagged_subjects_df),
        "flagged_subject_pct": (len(flagged_subjects_df) / len(df) * 100) if len(df) else 0,
        "flagged_feature_count": len(flagged_features_df),
        "flagged_feature_pct": (len(flagged_features_df) / features_df.shape[1] * 100)
        if features_df.shape[1]
        else 0,
        "critical_issues": len(fooof.get("flagged_subjects", [])),
    }

    report_results = {
        "summary": summary,
        "flagged_features_df": flagged_features_df,
        "flagged_subjects_df": flagged_subjects_df,
        "stats_df": stats_df,
        "sensor_group_stats": sensor_group_stats,
        "fooof": fooof,
        "high_corr_pairs": high_corr_pairs,
        "figures": {name: str(path) for name, path in figures.items()},
    }

    html_path = output_dir / "features_qc_report.html"
    generate_html_report(report_results, html_path)

    json_path = output_dir / "features_qc_report.json"
    save_json_report(json_path, report_results)

    logging.info("QC outputs:")
    logging.info(" - Summary CSV: %s", stats_path)
    logging.info(" - Flagged subjects: %s", flagged_subjects_path)
    logging.info(" - Flagged features: %s", flagged_features_path)
    logging.info(" - Correlation matrix: %s", corr_path)
    logging.info(" - HTML report: %s", html_path)
    logging.info(" - JSON report: %s", json_path)


def resolve_input_path(input_path: str) -> Path:
    """Resolve the user-provided path with fallbacks for common study layouts."""

    raw_path = Path(input_path).expanduser()
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2] if len(script_path.parents) >= 3 else script_path.parent
    search_dirs = [
        Path.cwd(),
        project_root,
        project_root / "data",
        project_root / "data" / "csv",
    ]

    def candidates_from(path_obj: Path, directories: List[Path]) -> List[Path]:
        cand: List[Path] = [path_obj]
        if path_obj.is_absolute():
            try:
                relative = path_obj.relative_to(path_obj.anchor)
            except ValueError:
                relative = None
            if relative:
                cand.extend([directory / relative for directory in directories])
        else:
            cand.extend([directory / path_obj for directory in directories])
        return cand

    candidates = candidates_from(raw_path, search_dirs)

    seen: Set[Path] = set()
    resolved_candidates: List[Path] = []
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate in seen:
            continue
        seen.add(candidate)
        resolved_candidates.append(candidate)

    for candidate in resolved_candidates:
        if candidate.exists():
            return candidate

    search_info = "\n".join(f"- {candidate}" for candidate in resolved_candidates)
    raise FileNotFoundError(
        f"Input file not found. Attempted locations:\n{search_info}\n"
        "Provide an absolute path or run the script from the project root."
    )


def load_input_file(path: Path) -> pd.DataFrame:
    """Load CSV or parquet file."""

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError("Unsupported input format. Use CSV or parquet.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EEG Feature QC report generator.")
    parser.add_argument("--input", required=True, help="Input CSV or parquet file.")
    parser.add_argument("--output_dir", required=True, help="Directory to store QC outputs.")
    parser.add_argument("--missing_threshold", type=float, default=0.05, help="Missing percentage threshold for features.")
    parser.add_argument("--outlier_method", choices=["iqr", "zscore", "both"], default="both", help="Outlier detection method.")
    parser.add_argument("--zscore_threshold", type=float, default=3.0, help="Z-score threshold.")
    parser.add_argument("--iqr_k", type=float, default=1.5, help="IQR multiplier k.")
    parser.add_argument("--generate_all_plots", action="store_true", help="Generate per-feature distribution plots.")
    parser.add_argument("--feature_config", type=str, help="Optional YAML config for feature categories.")
    parser.add_argument("--log_level", default="INFO", help="Logging level.")
    parser.add_argument("--subject_id_col", type=str, help="Column containing subject identifiers.")
    parser.add_argument("--variance_threshold", type=float, default=1e-10, help="Variance threshold for constant features.")
    parser.add_argument("--skew_threshold", type=float, default=3.0, help="Skewness threshold.")
    parser.add_argument("--kurtosis_threshold", type=float, default=10.0, help="Kurtosis threshold.")
    parser.add_argument("--feature_outlier_threshold", type=int, default=20, help="Absolute count threshold for per-feature outliers.")
    parser.add_argument("--subject_missing_threshold", type=float, default=0.1, help="Subject-level missing data threshold.")
    parser.add_argument("--subject_outlier_threshold", type=float, default=0.2, help="Subject-level outlier fraction threshold.")
    parser.add_argument("--bound_tolerance", type=float, default=1e-6, help="Tolerance for bound violations.")
    parser.add_argument("--contamination", type=float, default=0.05, help="IsolationForest contamination.")
    parser.add_argument("--corr_threshold", type=float, default=0.95, help="Correlation threshold for flagging redundant features.")
    parser.add_argument("--jobs", type=int, default=-1, help="Parallel jobs for distribution plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging(output_dir, args.log_level)
    logging.info("Logging to %s", log_path)
    input_path = resolve_input_path(args.input)
    logging.info("Loading data from %s", input_path)
    df = load_input_file(input_path)

    feature_categories = load_feature_categories(args.feature_config)
    setattr(args, "feature_categories", feature_categories)

    feature_cols = [col for col in df.columns if col.startswith("feature-")]
    if not feature_cols:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = numeric_cols
    features_df = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    metadata_df = build_feature_metadata(feature_cols, feature_categories)

    logging.info("Running QC on %d subjects and %d features.", len(df), len(feature_cols))
    generate_reports(args, df, features_df, metadata_df, output_dir)


if __name__ == "__main__":
    main()
