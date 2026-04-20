from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from coco_pipe.io import DataContainer

from eeg_adhd_epilepsy.analysis import extract_descriptors, merge_descriptors


def _demo_container() -> DataContainer:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(4, 2, 64))
    time = np.linspace(0, 1, 64, endpoint=False)
    X[:, 0, :] += np.sin(2 * np.pi * 10 * time)
    X[:, 1, :] += np.sin(2 * np.pi * 6 * time)
    return DataContainer(
        X=X,
        y=np.array(["Control", "Control", "ADHD", "ADHD"], dtype=object),
        ids=np.array(["0001_ep0", "0001_ep1", "0002_ep0", "0002_ep1"], dtype=object),
        dims=("obs", "channel", "time"),
        coords={
            "obs": np.array(["0001_ep0", "0001_ep1", "0002_ep0", "0002_ep1"], dtype=object),
            "channel": np.array(["Fz", "Cz"], dtype=object),
            "time": time,
            "study_id": np.array(["0001", "0001", "0002", "0002"], dtype=object),
            "combined_diagnosis": np.array(
                ["Control", "Control", "ADHD", "ADHD"], dtype=object
            ),
            "age": np.array([10, 10, 13, 13], dtype=object),
            "sex": np.array(["F", "F", "M", "M"], dtype=object),
        },
        meta={"sfreq": 64.0},
    )


def _demo_container_for_subjects(subjects: list[str] | None = None) -> DataContainer:
    container = _demo_container()
    if not subjects:
        return container
    requested = {str(subject) for subject in subjects}
    obs_indices = [
        idx
        for idx, subject in enumerate(np.asarray(container.coords["study_id"], dtype=object))
        if str(subject) in requested
    ]
    return container.isel(obs=obs_indices)


def _demo_raw_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "study_id": ["0001", "0002"],
            "combined_diagnosis": ["Control", "ADHD"],
            "age": [10, 13],
            "sex": ["F", "M"],
        }
    )


def test_table_helpers_preserve_metadata_and_feature_names() -> None:
    container = _demo_container()
    metadata_df = container.obs_table(
        include_ids=True,
        include_y=True,
        y_col="combined_diagnosis",
    )
    metadata_df["obs_id"] = metadata_df["obs_id"].astype(str)
    metadata_df.insert(1, "condition", "EO_baseline")
    metadata_df["subject"] = metadata_df["study_id"].astype(str)
    ordered_columns = ["obs_id", "subject", "condition"]
    ordered_columns.extend(
        column for column in metadata_df.columns if column not in ordered_columns
    )
    metadata_df = metadata_df[ordered_columns]
    result = {
        "X": np.arange(28, dtype=float).reshape(4, 7),
        "descriptor_names": [
            "band_abs_alpha_ch-Fz",
            "band_abs_beta_ch-Fz",
            "band_corr_abs_alpha_ch-Fz",
            "band_corr_abs_beta_ch-Fz",
            "band_log_abs_alpha_ch-Fz",
            "band_log_abs_beta_ch-Fz",
            "param_offset_ch-Cz",
        ],
        "failures": [],
    }

    epoch_df = pd.concat(
        [
            metadata_df.reset_index(drop=True),
            pd.DataFrame(result["X"], columns=result["descriptor_names"]),
        ],
        axis=1,
    )
    assert "obs_id" in epoch_df.columns
    assert "subject" in epoch_df.columns
    assert "condition" in epoch_df.columns
    assert "band_abs_alpha_ch-Fz" in epoch_df.columns
    assert "band_abs_beta_ch-Fz" in epoch_df.columns
    assert "band_log_abs_alpha_ch-Fz" in epoch_df.columns
    assert "band_log_abs_beta_ch-Fz" in epoch_df.columns
    assert "param_offset_ch-Cz" in epoch_df.columns

    coords = {
        "obs": metadata_df["obs_id"].to_numpy(dtype=object),
        "feature": np.asarray(result["descriptor_names"], dtype=object),
    }
    for column in metadata_df.columns:
        if column == "obs_id":
            continue
        coords[column] = metadata_df[column].to_numpy(dtype=object)
    feature_container = DataContainer(
        X=result["X"],
        y=metadata_df["combined_diagnosis"].to_numpy(dtype=object),
        ids=metadata_df["obs_id"].to_numpy(dtype=object),
        dims=("obs", "feature"),
        coords=coords,
        meta={},
    )
    grouped_mean = feature_container.aggregate(
        by="subject",
        stats="mean",
        min_count=1,
        on_insufficient="raise",
    )
    agg_metadata_df = grouped_mean.obs_table(
        include_y=True,
        y_col="combined_diagnosis",
    )
    agg_metadata_df["condition"] = "EO_baseline"
    ordered_columns = ["subject", "condition", "epoch_count"]
    ordered_columns.extend(
        column
        for column in agg_metadata_df.columns
        if column not in ordered_columns
    )
    base_agg_feature_df = pd.DataFrame(
        grouped_mean.X,
        columns=list(np.asarray(grouped_mean.coords["feature"], dtype=object)),
    )
    grouped_features = feature_container.aggregate_groups(
        by="subject",
        groups=[
            {
                "name": "mean_export",
                "stats": "mean",
                "exclude_prefixes": ["band_abs_", "band_corr_abs_"],
            },
            {
                "name": "band_summaries",
                "prefixes": ["band_"],
                "exclude_prefixes": ["band_abs_", "band_corr_abs_"],
                "stats": ["median", "iqr"],
            },
            {
                "name": "param_summaries",
                "prefixes": ["param_"],
                "stats": ["median", "iqr"],
            },
        ],
    )
    agg_feature_df = pd.concat(
        [
            pd.DataFrame(
                grouped_features.X,
                columns=list(np.asarray(grouped_features.coords["feature"], dtype=object)),
            ),
            pd.DataFrame(
                {
                    "agg_band_ratio_alpha_beta_ch-Fz": np.divide(
                        base_agg_feature_df["band_abs_alpha_ch-Fz"].to_numpy(dtype=float),
                        base_agg_feature_df["band_abs_beta_ch-Fz"].to_numpy(dtype=float),
                        out=np.full(base_agg_feature_df.shape[0], np.nan, dtype=float),
                        where=base_agg_feature_df["band_abs_beta_ch-Fz"].to_numpy(dtype=float) > 0.0,
                    ),
                    "agg_band_corr_ratio_alpha_beta_ch-Fz": np.divide(
                        base_agg_feature_df["band_corr_abs_alpha_ch-Fz"].to_numpy(dtype=float),
                        base_agg_feature_df["band_corr_abs_beta_ch-Fz"].to_numpy(dtype=float),
                        out=np.full(base_agg_feature_df.shape[0], np.nan, dtype=float),
                        where=base_agg_feature_df["band_corr_abs_beta_ch-Fz"].to_numpy(dtype=float) > 0.0,
                    ),
                },
                index=base_agg_feature_df.index,
            ),
        ],
        axis=1,
    )
    agg_df = pd.concat(
        [
            agg_metadata_df[ordered_columns].reset_index(drop=True),
            agg_feature_df.reset_index(drop=True),
        ],
        axis=1,
    )
    assert agg_df["subject"].tolist() == ["0001", "0002"]
    assert "epoch_count" in agg_df.columns
    assert "mean_band_abs_alpha_ch-Fz" not in agg_df.columns
    assert "mean_band_log_abs_alpha_ch-Fz" in agg_df.columns
    assert "median_band_log_abs_alpha_ch-Fz" in agg_df.columns
    assert "iqr_band_log_abs_alpha_ch-Fz" in agg_df.columns
    assert "mean_param_offset_ch-Cz" in agg_df.columns
    assert "median_param_offset_ch-Cz" in agg_df.columns
    assert "iqr_param_offset_ch-Cz" in agg_df.columns
    assert "agg_band_ratio_alpha_beta_ch-Fz" in agg_df.columns
    assert "agg_band_corr_ratio_alpha_beta_ch-Fz" in agg_df.columns
    assert agg_df["epoch_count"].tolist() == [2, 2]


def test_epoch_metadata_frame_requires_container_ids() -> None:
    container = _demo_container()
    container.ids = None

    with pytest.raises(ValueError, match="DataContainer.ids"):
        container.obs_table(
            include_ids=True,
            include_y=True,
            y_col="combined_diagnosis",
        )


def test_extract_descriptors_cli_writes_epoch_and_subject_outputs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "descriptors.yaml"
    config_path.write_text(
        """
precision: float32
families:
  bands:
    enabled: true
    outputs: [absolute_power, corrected_absolute_power, log_absolute_power]
    ratio_pairs: [[alpha, beta]]
  parametric:
    enabled: false
  complexity:
    enabled: false
pooling:
  channel_groups:
    midline: [Fz, Cz]
runtime:
  execution_backend: sequential
  n_jobs: 1
  obs_chunk: 8
  on_error: collect
aggregation:
  descriptors:
    - name: mean_export
      stats: mean
      exclude_prefixes: [band_abs_, band_corr_abs_]
    - name: band_summaries
      prefixes: [band_]
      exclude_prefixes: [band_abs_, band_corr_abs_]
      stats: [median, iqr]
    - name: complexity_summaries
      prefixes: [complexity_]
      stats: [median, mad]
    - name: param_summaries
      prefixes: [param_]
      stats: [median, iqr]
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        extract_descriptors,
        "load",
        lambda metadata_path, sep=None: _demo_raw_metadata(),
    )
    monkeypatch.setattr(
        extract_descriptors,
        "validate_bids_coverage",
        lambda raw_meta_df, coverage_root, desc, suffix, subject_col: {
            "present_subjects": ["0001", "0002"]
        },
    )
    monkeypatch.setattr(
        extract_descriptors,
        "load_eeg_data",
        lambda **kwargs: _demo_container_for_subjects(kwargs.get("subjects")),
    )

    bids_root = tmp_path / "BIDS"
    argv = [
        "eeg-descriptors",
        "--bids_root",
        str(bids_root),
        "--metadata",
        str(tmp_path / "patients.csv"),
        "--config",
        str(config_path),
        "--conditions",
        "EO_baseline",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    extract_descriptors.main()

    derivative_root = bids_root / "derivatives" / "signal_features" / "descriptors"
    reports_root = tmp_path / "reports"
    assert (derivative_root / "dataset_description.json").exists()
    assert (derivative_root / "config_used.yaml").exists()

    subject_one_root = derivative_root / "sub-0001" / "ses-01" / "eeg" / "EO_baseline"
    subject_two_root = derivative_root / "sub-0002" / "ses-01" / "eeg" / "EO_baseline"
    for shard_root in (subject_one_root, subject_two_root):
        assert (shard_root / "_SUCCESS").exists()
        assert (shard_root / "sensor_descriptor_bundle.npz").exists()
        assert (shard_root / "sensor_epoch_features.csv").exists()
        assert (shard_root / "sensor_epoch_features.parquet").exists()
        assert (shard_root / "sensor_epoch_features_feature_columns.json").exists()
        assert (shard_root / "sensor_subject_features.csv").exists()
        assert (shard_root / "sensor_subject_features.parquet").exists()
        assert (shard_root / "sensor_subject_features_feature_columns.json").exists()
        assert (shard_root / "pooled_epoch_features.csv").exists()
        assert (shard_root / "pooled_epoch_features.parquet").exists()
        assert (shard_root / "pooled_epoch_features_feature_columns.json").exists()
        assert (shard_root / "pooled_subject_features.csv").exists()
        assert (shard_root / "pooled_subject_features.parquet").exists()
        assert (shard_root / "pooled_subject_features_feature_columns.json").exists()
        assert (shard_root / "failures.csv").exists()
        assert (shard_root / "qc" / "summary_row.csv").exists()
        assert (shard_root / "qc" / "summary_metrics.csv").exists()
        assert (shard_root / "qc" / "flags.csv").exists()
        assert (shard_root / "qc" / "failure_summary.csv").exists()
        assert (shard_root / "qc" / "feature_missingness.csv").exists()
        assert (shard_root / "qc" / "family_summary.csv").exists()
    assert (
        reports_root
        / "sub-0001"
        / "ses-01"
        / "descriptor_qc"
        / "sub-0001_ses-01_EO_baseline_descriptor_qc_report.html"
    ).exists()
    assert (
        reports_root
        / "sub-0002"
        / "ses-01"
        / "descriptor_qc"
        / "sub-0002_ses-01_EO_baseline_descriptor_qc_report.html"
    ).exists()

    sensor_epoch_df = pd.read_csv(subject_one_root / "sensor_epoch_features.csv")
    sensor_agg_df = pd.read_csv(subject_one_root / "sensor_subject_features.csv")
    pooled_epoch_df = pd.read_csv(subject_one_root / "pooled_epoch_features.csv")
    pooled_agg_df = pd.read_csv(subject_one_root / "pooled_subject_features.csv")
    assert len(sensor_epoch_df) == 2
    assert len(sensor_agg_df) == 1
    assert len(pooled_epoch_df) == 2
    assert len(pooled_agg_df) == 1
    assert "band_abs_delta_ch-Fz" in sensor_epoch_df.columns
    assert not any("chgrp-" in column for column in sensor_epoch_df.columns)
    assert "band_abs_delta_chgrp-midline" in pooled_epoch_df.columns
    assert not any(column.endswith("_ch-Fz") for column in pooled_epoch_df.columns)
    assert "epoch_count" in sensor_agg_df.columns
    assert sensor_agg_df["epoch_count"].tolist() == [2]
    assert pooled_agg_df["epoch_count"].tolist() == [2]
    assert any(column.startswith("mean_") for column in sensor_agg_df.columns)
    assert any(column.startswith("mean_band_log_abs_") for column in sensor_agg_df.columns)
    assert not any(column.startswith("mean_band_abs_") for column in sensor_agg_df.columns)
    assert any(column.startswith("median_band_") for column in sensor_agg_df.columns)
    assert any(column.startswith("iqr_band_") for column in sensor_agg_df.columns)
    assert any(column.startswith("agg_band_ratio_") for column in sensor_agg_df.columns)
    assert any(column.startswith("agg_band_corr_ratio_") for column in sensor_agg_df.columns)
    assert any(column.startswith("agg_band_ratio_") for column in pooled_agg_df.columns)
    assert any(
        column.startswith("agg_band_corr_ratio_") for column in pooled_agg_df.columns
    )
    assert not any(column.startswith("band_ratio_") for column in sensor_epoch_df.columns)
    pooled_log_col = "band_log_abs_alpha_chgrp-midline"
    assert pooled_log_col in pooled_epoch_df.columns
    assert pooled_agg_df.loc[0, f"median_{pooled_log_col}"] == pytest.approx(
        pooled_epoch_df[pooled_log_col].median()
    )
    assert not (derivative_root / "combined").exists()


def test_merge_descriptors_cli_writes_combined_outputs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "descriptors.yaml"
    config_path.write_text(
        """
precision: float32
families:
  bands:
    enabled: true
    outputs: [absolute_power, corrected_absolute_power, log_absolute_power]
    ratio_pairs: [[alpha, beta]]
  parametric:
    enabled: false
  complexity:
    enabled: false
pooling:
  channel_groups:
    midline: [Fz, Cz]
runtime:
  execution_backend: sequential
  n_jobs: 1
  obs_chunk: 8
  on_error: collect
aggregation:
  descriptors:
    - name: mean_export
      stats: mean
      exclude_prefixes: [band_abs_, band_corr_abs_]
    - name: band_summaries
      prefixes: [band_]
      exclude_prefixes: [band_abs_, band_corr_abs_]
      stats: [median, iqr]
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        extract_descriptors,
        "load",
        lambda metadata_path, sep=None: _demo_raw_metadata(),
    )
    monkeypatch.setattr(
        extract_descriptors,
        "validate_bids_coverage",
        lambda raw_meta_df, coverage_root, desc, suffix, subject_col: {
            "present_subjects": ["0001", "0002"]
        },
    )
    monkeypatch.setattr(
        extract_descriptors,
        "load_eeg_data",
        lambda **kwargs: _demo_container_for_subjects(kwargs.get("subjects")),
    )

    bids_root = tmp_path / "BIDS"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eeg-descriptors",
            "--bids_root",
            str(bids_root),
            "--metadata",
            str(tmp_path / "patients.csv"),
            "--config",
            str(config_path),
            "--conditions",
            "EO_baseline",
        ],
    )
    extract_descriptors.main()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eeg-merge-descriptors",
            "--bids_root",
            str(bids_root),
        ],
    )
    merge_descriptors.main()

    combined_root = (
        bids_root / "derivatives" / "signal_features" / "descriptors" / "combined"
    )
    reports_root = tmp_path / "reports"
    combined_sensor_epoch_df = pd.read_csv(combined_root / "sensor_epoch_features.csv")
    combined_sensor_agg_df = pd.read_csv(combined_root / "sensor_subject_features.csv")
    combined_pooled_epoch_df = pd.read_csv(combined_root / "pooled_epoch_features.csv")
    combined_pooled_agg_df = pd.read_csv(combined_root / "pooled_subject_features.csv")

    assert len(combined_sensor_epoch_df) == 4
    assert len(combined_sensor_agg_df) == 2
    assert len(combined_pooled_epoch_df) == 4
    assert len(combined_pooled_agg_df) == 2
    assert any(
        column.startswith("agg_band_ratio_") for column in combined_sensor_agg_df.columns
    )
    assert any(
        column.startswith("agg_band_corr_ratio_")
        for column in combined_sensor_agg_df.columns
    )
    assert any(
        column.startswith("agg_band_ratio_") for column in combined_pooled_agg_df.columns
    )
    assert any(
        column.startswith("agg_band_corr_ratio_")
        for column in combined_pooled_agg_df.columns
    )
    assert (combined_root / "failures.csv").exists()
    assert (combined_root / "sensor_epoch_features_feature_columns.json").exists()
    assert (combined_root / "sensor_subject_features_feature_columns.json").exists()
    assert (combined_root / "pooled_epoch_features_feature_columns.json").exists()
    assert (combined_root / "pooled_subject_features_feature_columns.json").exists()
    assert (combined_root / "qc" / "dataset_summary_metrics.csv").exists()
    assert (combined_root / "qc" / "dataset_flags.csv").exists()
    assert (combined_root / "qc" / "shard_qc_summary.csv").exists()
    assert (combined_root / "qc" / "failure_summary_by_family.csv").exists()
    assert (combined_root / "qc" / "failure_summary_by_channel.csv").exists()
    assert (combined_root / "qc" / "feature_missingness.csv").exists()
    assert (combined_root / "qc" / "feature_distribution_summary.csv").exists()
    assert (combined_root / "qc" / "low_variance_features.csv").exists()
    assert (
        reports_root / "summary" / "descriptor_qc" / "descriptor_qc_dataset_summary.html"
    ).exists()


def test_extract_descriptors_cli_skips_missing_subject_condition(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "descriptors.yaml"
    config_path.write_text(
        """
precision: float32
families:
  bands:
    enabled: true
    outputs: [absolute_power]
  parametric:
    enabled: false
  complexity:
    enabled: false
runtime:
  execution_backend: sequential
  n_jobs: 1
  obs_chunk: 8
  on_error: collect
aggregation:
  descriptors:
    - name: mean_export
      stats: mean
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        extract_descriptors,
        "load",
        lambda metadata_path, sep=None: _demo_raw_metadata(),
    )
    monkeypatch.setattr(
        extract_descriptors,
        "validate_bids_coverage",
        lambda raw_meta_df, coverage_root, desc, suffix, subject_col: {
            "present_subjects": ["0001"]
        },
    )

    def _load_demo_data(**kwargs):
        condition = kwargs.get("condition")
        if condition == "HV_EO":
            raise RuntimeError("No valid data found in /tmp/mock_preproc")
        return _demo_container_for_subjects(kwargs.get("subjects"))

    monkeypatch.setattr(extract_descriptors, "load_eeg_data", _load_demo_data)

    bids_root = tmp_path / "BIDS"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eeg-descriptors",
            "--bids_root",
            str(bids_root),
            "--metadata",
            str(tmp_path / "patients.csv"),
            "--config",
            str(config_path),
            "--conditions",
            "EO_baseline",
            "HV_EO",
        ],
    )
    extract_descriptors.main()

    derivative_root = bids_root / "derivatives" / "signal_features" / "descriptors"
    assert (derivative_root / "sub-0001" / "ses-01" / "eeg" / "EO_baseline" / "_SUCCESS").exists()
    assert not (derivative_root / "sub-0001" / "ses-01" / "eeg" / "HV_EO").exists()


def test_extract_descriptors_cli_resumes_completed_shards(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "descriptors.yaml"
    config_path.write_text(
        """
precision: float32
families:
  bands:
    enabled: true
    outputs: [absolute_power]
  parametric:
    enabled: false
  complexity:
    enabled: false
pooling:
  channel_groups:
    midline: [Fz, Cz]
runtime:
  execution_backend: sequential
  n_jobs: 1
  obs_chunk: 8
  on_error: collect
aggregation:
  descriptors:
    - name: mean_export
      stats: mean
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        extract_descriptors,
        "load",
        lambda metadata_path, sep=None: _demo_raw_metadata(),
    )
    monkeypatch.setattr(
        extract_descriptors,
        "validate_bids_coverage",
        lambda raw_meta_df, coverage_root, desc, suffix, subject_col: {
            "present_subjects": ["0001", "0002"]
        },
    )

    load_calls: list[tuple[str, ...]] = []

    def _load_demo_data(**kwargs):
        subjects = tuple(kwargs.get("subjects") or [])
        load_calls.append(subjects)
        return _demo_container_for_subjects(list(subjects))

    monkeypatch.setattr(extract_descriptors, "load_eeg_data", _load_demo_data)

    bids_root = tmp_path / "BIDS"
    argv = [
        "eeg-descriptors",
        "--bids_root",
        str(bids_root),
        "--metadata",
        str(tmp_path / "patients.csv"),
        "--config",
        str(config_path),
        "--conditions",
        "EO_baseline",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    extract_descriptors.main()

    assert load_calls == [("0001",), ("0002",)]

    incomplete_shard = (
        bids_root
        / "derivatives"
        / "signal_features"
        / "descriptors"
        / "sub-0001"
        / "ses-01"
        / "eeg"
        / "EO_baseline"
    )
    (incomplete_shard / "pooled_subject_features.csv").unlink()

    rerun_calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        extract_descriptors,
        "load_eeg_data",
        lambda **kwargs: rerun_calls.append(tuple(kwargs.get("subjects") or []))
        or _demo_container_for_subjects(kwargs.get("subjects")),
    )
    extract_descriptors.main()
    assert rerun_calls == [("0001",)]

    combined_root = (
        bids_root / "derivatives" / "signal_features" / "descriptors" / "combined"
    )
    assert not combined_root.exists()


def test_merge_descriptors_cli_writes_sensor_only_outputs_without_pooling(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "descriptors.yaml"
    config_path.write_text(
        """
precision: float32
families:
  bands:
    enabled: true
    outputs: [absolute_power]
  parametric:
    enabled: false
  complexity:
    enabled: false
runtime:
  execution_backend: sequential
  n_jobs: 1
  obs_chunk: 8
  on_error: collect
aggregation:
  descriptors:
    - name: mean_export
      stats: mean
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        extract_descriptors,
        "load",
        lambda metadata_path, sep=None: _demo_raw_metadata(),
    )
    monkeypatch.setattr(
        extract_descriptors,
        "validate_bids_coverage",
        lambda raw_meta_df, coverage_root, desc, suffix, subject_col: {
            "present_subjects": ["0001", "0002"]
        },
    )
    monkeypatch.setattr(
        extract_descriptors,
        "load_eeg_data",
        lambda **kwargs: _demo_container_for_subjects(kwargs.get("subjects")),
    )

    bids_root = tmp_path / "BIDS"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eeg-descriptors",
            "--bids_root",
            str(bids_root),
            "--metadata",
            str(tmp_path / "patients.csv"),
            "--config",
            str(config_path),
            "--conditions",
            "EO_baseline",
        ],
    )
    extract_descriptors.main()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eeg-merge-descriptors",
            "--bids_root",
            str(bids_root),
        ],
    )
    merge_descriptors.main()

    combined_root = (
        bids_root / "derivatives" / "signal_features" / "descriptors" / "combined"
    )
    reports_root = tmp_path / "reports"
    assert (combined_root / "sensor_epoch_features.csv").exists()
    assert (combined_root / "sensor_subject_features.csv").exists()
    assert not (combined_root / "pooled_epoch_features.csv").exists()
    assert not (combined_root / "pooled_subject_features.csv").exists()
    assert (
        reports_root / "summary" / "descriptor_qc" / "descriptor_qc_dataset_summary.html"
    ).exists()


def test_extract_descriptors_cli_raises_when_requested_subjects_are_unavailable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "descriptors.yaml"
    config_path.write_text(
        """
precision: float32
families:
  bands:
    enabled: true
    outputs: [absolute_power]
  parametric:
    enabled: false
  complexity:
    enabled: false
runtime:
  execution_backend: sequential
  n_jobs: 1
  obs_chunk: 8
  on_error: collect
aggregation:
  descriptors:
    - name: mean_export
      stats: mean
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        extract_descriptors,
        "load",
        lambda metadata_path, sep=None: _demo_raw_metadata(),
    )
    monkeypatch.setattr(
        extract_descriptors,
        "validate_bids_coverage",
        lambda raw_meta_df, coverage_root, desc, suffix, subject_col: {
            "present_subjects": ["0001"]
        },
    )

    def _unexpected_load(**kwargs):
        raise AssertionError("load_eeg_data should not be called when no subjects match")

    monkeypatch.setattr(extract_descriptors, "load_eeg_data", _unexpected_load)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eeg-descriptors",
            "--bids_root",
            str(tmp_path / "BIDS"),
            "--metadata",
            str(tmp_path / "patients.csv"),
            "--config",
            str(config_path),
            "--subjects",
            "9999",
            "--conditions",
            "EO_baseline",
        ],
    )

    with pytest.raises(ValueError, match="No matching saved-derivative subjects"):
        extract_descriptors.main()
