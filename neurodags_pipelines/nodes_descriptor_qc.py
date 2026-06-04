"""Descriptor QC node: converts NC features to QC DataFrames and generates HTML report.

AGGREGATOR NODE NOTE
--------------------
``generate_descriptor_qc_record`` is an aggregator node: it receives one NC file as input
but uses it only to locate the parent directory, then scans *all* NC files in that directory.
This is a workaround for neurodags lacking a gather/fan-in primitive — the proper architecture
would be a explicit multi-input dependency declared in the YAML. Until neurodags supports that,
this node couples itself to directory layout conventions instead of explicit DAG edges.

TODO: NaN CSVs missing for the 27 DescriptorQCRecord artifacts that ran before the NaN CSV
code was added (overwrite=False so they were skipped on re-run). To backfill: delete those
.json artifacts and re-run the step-1 pipeline.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neurodags.definitions import Artifact, NodeResult
from neurodags.nodes import register_node


# ---------------------------------------------------------------------------
# NC descriptor → column name mappings
# ---------------------------------------------------------------------------

_BAND_SUBTYPE = {
    "AbsBandPower": "abs",
    "RelBandPower": "rel",
    "LogBandPower": "log_abs",
    "CorrectedBandPower": "corr_abs",
    "CorrectedRelBandPower": "corr_rel",
    "CorrectedLogBandPower": "corr_log_abs",
}

_FOOOF_SCALAR_VAR_MAP = {
    "fooof_r_squared": "r_squared",
    "fooof_error": "fit_error",
    "fooof_aperiodic_offset": "offset",
    "fooof_aperiodic_exponent": "exponent",
    "fooof_aperiodic_knee": None,  # always NaN when aperiodic_mode='fixed' — skip
}

_FOOOF_PEAK_VAR_MAP = {
    "n_peaks": "peak_count",
    "dominant_peak_cf": "peak_freq_dom",
    "dominant_peak_pw": "peak_power_dom",
    "dominant_peak_bw": "peak_bandwidth_dom",
    "alpha_peak_cf": "alpha_peak_freq",
    "alpha_peak_pw": "alpha_peak_power",
    "alpha_peak_bw": "alpha_peak_bw",
}

_COMPLEXITY_NAME = {
    "AppEntropy": "app_entropy",
    "PermEntropy": "perm_entropy",
    "SVDEntropy": "svd_entropy",
    "SpectralEntropy": "spectral_entropy",
    "SampleEntropy": "sample_entropy",
    "LZivComplexity": "lziv_complexity",
    "NumZeroCross": "zero_crossings",
    "HiguchiFD": "higuchi_fd",
    "KatzFD": "katz_fd",
    "PetrosianFD": "petrosian_fd",
    "DetrendedFluctuation": "dfa",
    "Kurtosis": "kurtosis",
    "RMS": "rms",
    "FractalHurst": "hurst_exponent",
    "EntropyShannon": "shannon_entropy",
    "EntropyFuzzy": "fuzzy_entropy",
    "EntropyDispersion": "dispersion_entropy",
}

_VALUE_VAR_DESCRIPTORS = {"FractalHurst", "EntropyShannon", "EntropyFuzzy", "EntropyDispersion"}

_POOLED_FOOOF_MAP = {
    "FooofExponentPooled": ("fooof_aperiodic_exponent", "exponent"),
    "FooofOffsetPooled": ("fooof_aperiodic_offset", "offset"),
    "FooofRSquaredPooled": ("fooof_r_squared", "r_squared"),
}

_SKIP_SENSOR = {
    "SpectrumWelch", "FooofFit", "AbsBandPowerAgg", "CorrectedBandPowerAgg",
    "EntropyMultiscale",
}


def _sanitize_pair(pair_str: str) -> str:
    return pair_str.replace("/", "_over_")


def _nc_descriptor_name(nc_path: Path) -> str:
    """'sub-..._eeg.vhdr&CleanedPrepRaw.fif@AbsBandPower.nc' → 'AbsBandPower'"""
    stem = nc_path.stem
    return stem.split("@")[-1] if "@" in stem else stem


def _nc_to_sensor_df(nc_path: Path, descriptor: str) -> pd.DataFrame | None:
    """Flatten one sensor-level NC file to an epoch × feature DataFrame."""
    import xarray as xr

    try:
        ds = xr.open_dataset(nc_path)
    except Exception:
        return None

    if "epochs" not in ds.coords:
        ds.close()
        return None
    epoch_vals = ds.coords["epochs"].values

    data: dict[str, Any] = {}

    # Band power (freqbands dim)
    if descriptor in _BAND_SUBTYPE:
        subtype = _BAND_SUBTYPE[descriptor]
        vname = "spectrum" if "spectrum" in ds.data_vars else list(ds.data_vars)[0]
        arr = ds[vname]
        for space in arr.coords["spaces"].values:
            for band in arr.coords["freqbands"].values:
                data[f"band_{subtype}_{band}_ch-{space}"] = (
                    arr.sel(spaces=space, freqbands=band).values.astype(float)
                )

    # Band ratios (freqbandPairs dim)
    elif descriptor == "BandRatios":
        vname = "spectrum" if "spectrum" in ds.data_vars else list(ds.data_vars)[0]
        arr = ds[vname]
        for space in arr.coords["spaces"].values:
            for pair in arr.coords["freqbandPairs"].values:
                data[f"band_ratio_{_sanitize_pair(str(pair))}_ch-{space}"] = (
                    arr.sel(spaces=space, freqbandPairs=pair).values.astype(float)
                )

    # FOOOF scalars (multiple named vars)
    elif descriptor == "FooofScalarsDs":
        for vname in ds.data_vars:
            col_sfx = _FOOOF_SCALAR_VAR_MAP.get(vname, vname)
            if col_sfx is None:
                continue  # skip vars not applicable to the current fit mode
            for space in ds.coords["spaces"].values:
                data[f"param_{col_sfx}_ch-{space}"] = (
                    ds[vname].sel(spaces=space).values.astype(float)
                )

    # FOOOF peaks (multiple named vars)
    elif descriptor == "FooofPeaksDs":
        for vname in ds.data_vars:
            col_sfx = _FOOOF_PEAK_VAR_MAP.get(vname, vname)
            for space in ds.coords["spaces"].values:
                data[f"param_{col_sfx}_ch-{space}"] = (
                    ds[vname].sel(spaces=space).values.astype(float)
                )

    # Hjorth params (hjorthComponents extra dim)
    elif descriptor == "HjorthParams":
        vname = list(ds.data_vars)[0]
        arr = ds[vname]
        for comp in arr.coords["hjorthComponents"].values:
            for space in arr.coords["spaces"].values:
                data[f"complexity_hjorth_{comp}_ch-{space}"] = (
                    arr.sel(spaces=space, hjorthComponents=comp).values.astype(float)
                )

    # Descriptors using 'value' var (value + metadata structure)
    elif descriptor in _VALUE_VAR_DESCRIPTORS:
        name = _COMPLEXITY_NAME[descriptor]
        if "value" not in ds.data_vars:
            ds.close()
            return None
        arr = ds["value"]
        for space in arr.coords["spaces"].values:
            data[f"complexity_{name}_ch-{space}"] = (
                arr.sel(spaces=space).values.astype(float)
            )

    # Simple scalar: single var, (epochs, spaces)
    elif descriptor in _COMPLEXITY_NAME:
        name = _COMPLEXITY_NAME[descriptor]
        vname = list(ds.data_vars)[0]
        arr = ds[vname]
        if "spaces" not in arr.dims:
            ds.close()
            return None
        for space in arr.coords["spaces"].values:
            data[f"complexity_{name}_ch-{space}"] = (
                arr.sel(spaces=space).values.astype(float)
            )

    else:
        ds.close()
        return None

    ds.close()
    if not data:
        return None
    return pd.DataFrame(data, index=epoch_vals)


def _nc_to_pooled_df(nc_path: Path, descriptor: str) -> pd.DataFrame | None:
    """Flatten one pooled NC file to an epoch × feature DataFrame."""
    import xarray as xr

    try:
        ds = xr.open_dataset(nc_path)
    except Exception:
        return None

    if "epochs" not in ds.coords or "regions" not in ds.coords:
        ds.close()
        return None
    epoch_vals = ds.coords["epochs"].values
    region_vals = ds.coords["regions"].values

    data: dict[str, Any] = {}
    bare = descriptor.replace("Pooled", "")

    # Band power pooled
    if bare in _BAND_SUBTYPE:
        subtype = _BAND_SUBTYPE[bare]
        vname = "spectrum" if "spectrum" in ds.data_vars else list(ds.data_vars)[0]
        arr = ds[vname]
        for region in region_vals:
            for band in arr.coords["freqbands"].values:
                data[f"band_{subtype}_{band}_chgrp-{region}"] = (
                    arr.sel(regions=region, freqbands=band).values.astype(float)
                )

    # Band ratios pooled
    elif descriptor == "BandRatiosPooled":
        vname = "spectrum" if "spectrum" in ds.data_vars else list(ds.data_vars)[0]
        arr = ds[vname]
        for region in region_vals:
            for pair in arr.coords["freqbandPairs"].values:
                data[f"band_ratio_{_sanitize_pair(str(pair))}_chgrp-{region}"] = (
                    arr.sel(regions=region, freqbandPairs=pair).values.astype(float)
                )

    # FOOOF scalar pooled (single var)
    elif descriptor in _POOLED_FOOOF_MAP:
        var_nc_name, col_sfx = _POOLED_FOOOF_MAP[descriptor]
        if var_nc_name not in ds.data_vars:
            ds.close()
            return None
        arr = ds[var_nc_name]
        for region in region_vals:
            data[f"param_{col_sfx}_chgrp-{region}"] = (
                arr.sel(regions=region).values.astype(float)
            )

    else:
        ds.close()
        return None

    ds.close()
    if not data:
        return None
    return pd.DataFrame(data, index=epoch_vals)


def _build_sensor_dfs(nc_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    dfs: list[pd.DataFrame] = []
    for nc_path in sorted(nc_dir.glob("*.nc")):
        descriptor = _nc_descriptor_name(nc_path)
        if any(descriptor.endswith(s) for s in ("Pooled", "Agg", "Mean", "Median", "IQR", "MAD")):
            continue
        if descriptor in _SKIP_SENSOR:
            continue
        known = (
            descriptor in _BAND_SUBTYPE
            or descriptor in {"BandRatios", "FooofScalarsDs", "FooofPeaksDs", "HjorthParams"}
            or descriptor in _COMPLEXITY_NAME
        )
        if not known:
            continue
        df = _nc_to_sensor_df(nc_path, descriptor)
        if df is not None and not df.empty:
            dfs.append(df)

    if not dfs:
        return pd.DataFrame(), pd.DataFrame()

    sensor_epoch_df = pd.concat(dfs, axis=1)
    sensor_epoch_df = sensor_epoch_df.loc[:, ~sensor_epoch_df.columns.duplicated()]
    sensor_subject_df = sensor_epoch_df.mean(axis=0).to_frame().T.reset_index(drop=True)
    return sensor_epoch_df, sensor_subject_df


def _build_pooled_dfs(nc_dir: Path) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    dfs: list[pd.DataFrame] = []
    for nc_path in sorted(nc_dir.glob("*.nc")):
        descriptor = _nc_descriptor_name(nc_path)
        if not (descriptor.endswith("Pooled") or descriptor in _POOLED_FOOOF_MAP):
            continue
        df = _nc_to_pooled_df(nc_path, descriptor)
        if df is not None and not df.empty:
            dfs.append(df)

    if not dfs:
        return None, None

    pooled_epoch_df = pd.concat(dfs, axis=1)
    pooled_epoch_df = pooled_epoch_df.loc[:, ~pooled_epoch_df.columns.duplicated()]
    pooled_subject_df = pooled_epoch_df.mean(axis=0).to_frame().T.reset_index(drop=True)
    return pooled_epoch_df, pooled_subject_df


def _parse_sub_ses(nc_dir: Path) -> tuple[str, str]:
    subject, session = "unknown", "01"
    for part in nc_dir.parts:
        if part.startswith("sub-"):
            subject = part[4:]
        elif part.startswith("ses-"):
            session = part[4:]
    return subject, session


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(float(obj)) else float(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


@register_node
def generate_descriptor_qc_record(
    nc_file,
    condition_name: str,
    reports_root: str | None = None,
) -> NodeResult:
    """Generate descriptor QC record from NC feature files for one subject/condition.

    Parameters
    ----------
    nc_file
        Path to any saved NC file in the condition features directory.  Used to
        locate the directory containing all other NC files.
    condition_name
        Condition label (e.g. 'EO_baseline').
    reports_root
        Root directory for HTML reports.  Defaults to
        ``{deriv_root}/descriptor_qc_reports/`` derived from the NC path.
    """
    from eeg_adhd_epilepsy.qc.descriptor_qc import run_descriptor_subject_qc
    import xarray as xr

    # --- Resolve nc_file to a filesystem path ---
    if isinstance(nc_file, NodeResult):
        item = next(iter(nc_file.artifacts.values())).item
    else:
        item = nc_file

    if isinstance(item, xr.Dataset):
        src = item.encoding.get("source", "")
        nc_path = Path(src) if src else Path("unknown.nc")
    else:
        nc_path = Path(item)

    nc_dir = nc_path.parent
    subject, session = _parse_sub_ses(nc_dir)

    # --- Derive reports_root if not provided ---
    if reports_root is None:
        parts = list(nc_dir.parts)
        if "features_conditions" in parts:
            idx = parts.index("features_conditions")
            deriv_root = Path(*parts[:idx])
        else:
            deriv_root = nc_dir.parents[4]
        reports_root = str(deriv_root / "descriptor_qc_reports")

    reports_root_path = Path(reports_root)

    # --- Build DataFrames from NC files ---
    sensor_epoch_df, sensor_subject_df = _build_sensor_dfs(nc_dir)
    pooled_epoch_df, pooled_subject_df = _build_pooled_dfs(nc_dir)

    # --- Write NaN rate CSV ---
    qc_dir = nc_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    nan_parts: list[pd.DataFrame] = []
    if not sensor_epoch_df.empty:
        s = sensor_epoch_df.isna().mean()
        nan_parts.append(pd.DataFrame({
            "source": "sensor",
            "feature": s.index,
            "nan_rate": s.values,
            "n_nan": sensor_epoch_df.isna().sum().values,
            "n_epochs": len(sensor_epoch_df),
        }))
    if pooled_epoch_df is not None and not pooled_epoch_df.empty:
        p = pooled_epoch_df.isna().mean()
        nan_parts.append(pd.DataFrame({
            "source": "pooled",
            "feature": p.index,
            "nan_rate": p.values,
            "n_nan": pooled_epoch_df.isna().sum().values,
            "n_epochs": len(pooled_epoch_df),
        }))
    if nan_parts:
        nan_df = pd.concat(nan_parts, ignore_index=True)
        nan_df.to_csv(
            qc_dir / f"sub-{subject}_ses-{session}_{condition_name}_nan_rates.csv",
            index=False,
        )

    # --- Write feature column JSON files to qc/ ---
    se_cols = list(sensor_epoch_df.columns) if not sensor_epoch_df.empty else []
    ss_cols = list(sensor_subject_df.columns) if not sensor_subject_df.empty else []
    pe_cols = list(pooled_epoch_df.columns) if pooled_epoch_df is not None else []
    ps_cols = list(pooled_subject_df.columns) if pooled_subject_df is not None else []

    se_cols_path = qc_dir / "sensor_epoch_feature_columns.json"
    ss_cols_path = qc_dir / "sensor_subject_feature_columns.json"
    se_cols_path.write_text(json.dumps(se_cols), encoding="utf-8")
    ss_cols_path.write_text(json.dumps(ss_cols), encoding="utf-8")

    pe_cols_path: Path | None = None
    ps_cols_path: Path | None = None
    if pooled_epoch_df is not None:
        pe_cols_path = qc_dir / "pooled_epoch_feature_columns.json"
        ps_cols_path = qc_dir / "pooled_subject_feature_columns.json"
        pe_cols_path.write_text(json.dumps(pe_cols), encoding="utf-8")
        ps_cols_path.write_text(json.dumps(ps_cols), encoding="utf-8")

    config_snapshot = {
        "families": {
            "bands": {"enabled": True},
            "parametric": {"enabled": True},
            "complexity": {"enabled": True},
        }
    }

    # --- Write failures CSV (all-NaN features per family) ---
    from eeg_adhd_epilepsy.qc.descriptor_qc import compute_constant_feature_summary

    _tol = 1e-10
    failure_rows: list[dict] = []
    for _df, _feat_cols in [
        (sensor_epoch_df, list(sensor_epoch_df.columns) if not sensor_epoch_df.empty else []),
        (
            pooled_epoch_df,
            list(pooled_epoch_df.columns)
            if pooled_epoch_df is not None and not pooled_epoch_df.empty
            else [],
        ),
    ]:
        if not _feat_cols:
            continue
        _constant_df = compute_constant_feature_summary(_df, _feat_cols, _tol, config_snapshot)
        for _, _row in _constant_df[_constant_df["is_all_nan"]].iterrows():
            failure_rows.append(
                {
                    "condition": condition_name,
                    "subject": subject,
                    "channel_name": str(_row["column"]),
                    "family": str(_row["family"]) if _row["family"] is not None else "unknown",
                    "reason": "all_nan",
                }
            )

    failures_df = pd.DataFrame(
        failure_rows, columns=["condition", "subject", "channel_name", "family", "reason"]
    )
    failures_df.to_csv(
        qc_dir / f"sub-{subject}_ses-{session}_{condition_name}_failures.csv",
        index=False,
    )

    failure_df = failures_df.assign(exception_type="all_nan")[
        ["family", "channel_name", "exception_type", "condition"]
    ]

    summary_row = run_descriptor_subject_qc(
        shard_root=nc_dir,
        reports_root=reports_root_path,
        subject=subject,
        session=session,
        condition=condition_name,
        sensor_epoch_df=sensor_epoch_df,
        sensor_subject_df=sensor_subject_df,
        sensor_epoch_feature_columns_path=se_cols_path,
        sensor_subject_feature_columns_path=ss_cols_path,
        pooled_epoch_df=pooled_epoch_df,
        pooled_subject_df=pooled_subject_df,
        pooled_epoch_feature_columns_path=pe_cols_path,
        pooled_subject_feature_columns_path=ps_cols_path,
        failure_df=failure_df,
        config_snapshot=config_snapshot,
    )

    out_data = json.loads(json.dumps(summary_row, default=_json_default))
    out_path = qc_dir / f"sub-{subject}_ses-{session}_{condition_name}_descriptor_qc.json"
    out_path.write_text(json.dumps(out_data, indent=2), encoding="utf-8")

    return NodeResult(
        artifacts={
            "._descriptor_qc.json": Artifact(
                item=out_data,
                writer=lambda path, d=out_data: Path(path).write_text(
                    json.dumps(d, indent=2), encoding="utf-8"
                ),
            )
        }
    )
