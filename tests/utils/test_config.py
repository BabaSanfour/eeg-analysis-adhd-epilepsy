"""Tests for the two-part (cohort + analysis) config loader."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from eeg_adhd_epilepsy.utils.config import (
    ConfigError,
    apply_overrides,
    load_cohort_analysis_config,
    resolve_cli_config,
)
from eeg_adhd_epilepsy.utils.yaml import load_yaml_config


def _write(path, body: str) -> str:
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return str(path)


def _cohort_yaml(extra: str = "") -> str:
    return f"""
    dataset_name: pooled_01_all_subjects_total
    conditions:
      - EO_baseline
      - EC_baseline
    group_filters:
      - adhd: [1]
    evals:
      - name: med_adhd_vs_ctrl
        target_col: combined_diagnosis
    {extra}
    """


def _analysis_yaml(extra: str = "") -> str:
    return f"""
    input_mode: descriptors
    models:
      logreg:
        estimator: LogisticRegression
    cv:
      n_splits: 5
    {extra}
    """


def test_merge_combines_cohort_and_analysis(tmp_path):
    cohort = _write(tmp_path / "cohort.yaml", _cohort_yaml())
    analysis = _write(tmp_path / "analysis.yaml", _analysis_yaml())

    merged = load_cohort_analysis_config(cohort, analysis)

    assert merged["dataset_name"] == "pooled_01_all_subjects_total"
    assert merged["input_mode"] == "descriptors"
    assert "logreg" in merged["models"]
    assert [e["name"] for e in merged["evals"]] == ["med_adhd_vs_ctrl"]


def test_analysis_conditions_override_cohort(tmp_path):
    cohort = _write(tmp_path / "cohort.yaml", _cohort_yaml())
    analysis = _write(
        tmp_path / "analysis.yaml",
        _analysis_yaml("conditions:\n      - EO_baseline"),
    )

    merged = load_cohort_analysis_config(cohort, analysis)

    # The analysis list replaces (does not concatenate) the cohort default.
    assert merged["conditions"] == ["EO_baseline"]


def test_missing_cohort_key_is_actionable(tmp_path):
    cohort = _write(
        tmp_path / "cohort.yaml",
        """
        evals:
          - name: x
        """,
    )
    analysis = _write(tmp_path / "analysis.yaml", _analysis_yaml())

    with pytest.raises(ConfigError, match="dataset_name"):
        load_cohort_analysis_config(cohort, analysis)


def test_analysis_without_method_block_is_rejected(tmp_path):
    cohort = _write(tmp_path / "cohort.yaml", _cohort_yaml())
    analysis = _write(tmp_path / "analysis.yaml", "input_mode: descriptors\n")

    with pytest.raises(ConfigError, match="method block"):
        load_cohort_analysis_config(cohort, analysis)


def test_analysis_must_not_define_evals(tmp_path):
    cohort = _write(tmp_path / "cohort.yaml", _cohort_yaml())
    analysis = _write(
        tmp_path / "analysis.yaml",
        _analysis_yaml("evals:\n      - name: leaked"),
    )

    with pytest.raises(ConfigError, match="evals"):
        load_cohort_analysis_config(cohort, analysis)


def test_selection_eval_name_must_exist_in_cohort_evals(tmp_path):
    cohort = _write(tmp_path / "cohort.yaml", _cohort_yaml())
    analysis = _write(
        tmp_path / "analysis.yaml",
        _analysis_yaml("selection_eval_name: not_a_real_eval"),
    )

    with pytest.raises(ConfigError, match="selection_eval_name"):
        load_cohort_analysis_config(cohort, analysis)


def test_selection_eval_name_matching_cohort_eval_passes(tmp_path):
    cohort = _write(tmp_path / "cohort.yaml", _cohort_yaml())
    analysis = _write(
        tmp_path / "analysis.yaml",
        _analysis_yaml("selection_eval_name: med_adhd_vs_ctrl"),
    )

    merged = load_cohort_analysis_config(cohort, analysis)
    assert merged["selection_eval_name"] == "med_adhd_vs_ctrl"


def test_decoding_analysis_configs_use_exported_session_column():
    for path in (
        Path("configs/analyses/decoding/classical.yaml"),
        Path("configs/analyses/decoding/foundation.yaml"),
    ):
        assert load_yaml_config(path)["session_col"] == "session"


def test_apply_overrides_ignores_none():
    config = {"bids_root": "/from/config", "n_jobs": 1}
    apply_overrides(config, bids_root=None, metadata="/cli/meta.csv", n_jobs=8)

    assert config["bids_root"] == "/from/config"  # None override ignored
    assert config["metadata"] == "/cli/meta.csv"
    assert config["n_jobs"] == 8


def test_resolve_cli_config_pair_with_path_override(tmp_path):
    cohort = _write(tmp_path / "cohort.yaml", _cohort_yaml())
    analysis = _write(tmp_path / "analysis.yaml", _analysis_yaml())

    config = resolve_cli_config(
        cohort_config=cohort,
        analysis_config=analysis,
        bids_root="/data/BIDS",
        metadata=None,
    )
    assert config["bids_root"] == "/data/BIDS"
    assert "logreg" in config["models"]


def test_resolve_cli_config_requires_bids_root(tmp_path):
    cohort = _write(tmp_path / "cohort.yaml", _cohort_yaml())
    analysis = _write(tmp_path / "analysis.yaml", _analysis_yaml())

    with pytest.raises(ConfigError, match="bids_root is required"):
        resolve_cli_config(cohort_config=cohort, analysis_config=analysis)


def test_resolve_cli_config_requires_both_config_roles(tmp_path):
    cohort = _write(tmp_path / "cohort.yaml", _cohort_yaml())

    with pytest.raises(ConfigError, match="--cohort_config and --analysis_config"):
        resolve_cli_config(
            cohort_config=cohort,
            analysis_config=None,
            bids_root="/data/BIDS",
        )
