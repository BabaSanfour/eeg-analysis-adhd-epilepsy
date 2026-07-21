"""Regression tests for portable, representation-safe run identities."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from eeg_adhd_epilepsy.analysis.utils.decoding import scientific_config
from eeg_adhd_epilepsy.analysis.utils.dim_reduction import (
    build_input_signature,
    build_run_config_payload,
)


def _descriptor_dim_args(bids_root: Path, representation: str) -> SimpleNamespace:
    combined = bids_root / "derivatives" / "signal_features" / "descriptors" / "combined"
    stem = f"sensor_{representation}_features"
    return SimpleNamespace(
        dataset_name="cohort",
        run_label="cohort",
        input_mode="descriptors",
        analysis_mode="flat",
        conditions=["EO_baseline"],
        run_pooled=True,
        n_components_sweep=[2, 5],
        subject_col="study_id",
        subjects=None,
        filter_col=[],
        filter_val=[],
        group_filters=None,
        balance_target=None,
        balance_strategy="undersample",
        qc={},
        bids_root=str(bids_root),
        use_derivatives=True,
        task="clinical",
        representation=representation,
        descriptor_families=None,
        descriptor_table_path=str(combined / f"{stem}.parquet"),
        descriptor_feature_columns_path=str(combined / f"{stem}_feature_columns.json"),
        descriptor_max_abs_value=None,
        location_statistic="mean",
        embedding_derivative_root=None,
        embedding_aggregate_by=None,
        embedding_model_key=None,
        segment_duration=60.0,
        overlap=0.0,
        desc="base",
        window_source="auto",
    )


def test_decoding_scientific_paths_are_portable_across_mounts(tmp_path):
    first_root = tmp_path / "mount_a" / "BIDS"
    second_root = tmp_path / "mount_b" / "BIDS"
    relative_table = Path("derivatives/signal_features/descriptors/combined/features.parquet")
    relative_columns = relative_table.with_name("features_feature_columns.json")
    common = {
        "dataset_name": "cohort",
        "input_mode": "descriptors",
        "representation": "recording",
    }
    first = {
        **common,
        "bids_root": str(first_root),
        "metadata": str(tmp_path / "mount_a" / "metadata.csv"),
        "descriptor_table_path": str(first_root / relative_table),
        "descriptor_feature_columns_path": str(first_root / relative_columns),
    }
    second = {
        **common,
        "bids_root": str(second_root),
        "metadata": str(tmp_path / "mount_b" / "metadata.csv"),
        "descriptor_table_path": str(second_root / relative_table),
        "descriptor_feature_columns_path": str(second_root / relative_columns),
    }

    assert scientific_config(first) == scientific_config(second)
    assert scientific_config(first)["descriptor_table_path"] == (
        "bids:///derivatives/signal_features/descriptors/combined/features.parquet"
    )


def test_dim_reduction_descriptor_hash_payload_is_portable_and_representation_safe(tmp_path):
    first_root = tmp_path / "mount_a" / "BIDS"
    second_root = tmp_path / "mount_b" / "BIDS"
    eval_specs = [{"name": "clinical", "target_col": "diagnosis"}]

    first = build_run_config_payload(
        _descriptor_dim_args(first_root, "recording"), ["PCA"], eval_specs
    )
    second = build_run_config_payload(
        _descriptor_dim_args(second_root, "recording"), ["PCA"], eval_specs
    )
    epoch = build_run_config_payload(_descriptor_dim_args(first_root, "epoch"), ["PCA"], eval_specs)

    assert first == second
    assert first["representation"] == "recording"
    assert epoch["representation"] == "epoch"
    assert first != epoch


def test_dim_reduction_descriptor_fit_signature_is_portable(tmp_path):
    unit = {"unit_type": "flat", "unit_name": "all", "family": None}
    first_args = _descriptor_dim_args(tmp_path / "mount_a" / "BIDS", "recording")
    second_args = _descriptor_dim_args(tmp_path / "mount_b" / "BIDS", "recording")
    first_args.run_config_hash = second_args.run_config_hash = "samehash"

    first = build_input_signature(first_args, unit)
    second = build_input_signature(second_args, unit)

    assert first == second
    assert first["bids_root"] == "bids:///"
    assert first["representation"] == "recording"


def test_descriptor_launchers_share_epoch_recording_defaults():
    project_root = Path(__file__).resolve().parents[2]
    expected_defaults = {
        "cluster/12_batch_run_dim_reduction_descriptors.sh": (
            "REPRESENTATIONS=(${REPRESENTATIONS:-epoch recording})"
        ),
        "cluster/15_submit_classical_decode.sh": (
            "REPRESENTATIONS=(${REPRESENTATIONS:-epoch recording})"
        ),
        "cluster/20_submit_main_decoding.sh": (
            "CLASSICAL_REPRESENTATIONS=(${CLASSICAL_REPRESENTATIONS:-epoch recording})"
        ),
    }
    for relative_path, declaration in expected_defaults.items():
        contents = (project_root / relative_path).read_text(encoding="utf-8")
        assert declaration in contents

    dimred_main = (project_root / "cluster/19_submit_main_dim_reduction.sh").read_text(
        encoding="utf-8"
    )
    assert "for rep in epoch recording; do" in dimred_main
    decoding_main = (project_root / "cluster/20_submit_main_decoding.sh").read_text(
        encoding="utf-8"
    )
    assert "SAVED_REPRESENTATIONS=(${SAVED_REPRESENTATIONS:-epoch recording})" in decoding_main
