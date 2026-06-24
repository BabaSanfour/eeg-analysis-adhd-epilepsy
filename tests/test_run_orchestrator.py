"""Tests for the eeg-run stage orchestrator (ordering, slicing, dry-run)."""

from __future__ import annotations

from eeg_adhd_epilepsy import run as orch


def test_stage_order_is_the_canonical_pipeline():
    assert orch.STAGE_NAMES == [
        "to-bids",
        "preprocess",
        "epochs",
        "descriptors",
        "merge",
        "dim-reduce",
        "classical-decode",
    ]


def test_select_slices_inclusive():
    names = [s.name for s in orch._select("preprocess", "merge")]
    assert names == ["preprocess", "epochs", "descriptors", "merge"]


def test_select_rejects_inverted_range():
    import pytest

    with pytest.raises(SystemExit):
        orch._select("merge", "preprocess")


def test_dry_run_prints_commands_and_runs_nothing(tmp_path, capsys, monkeypatch):
    called = {"ran": False}

    def _fail(*a, **k):  # subprocess.run must not be called in dry-run
        called["ran"] = True
        raise AssertionError("subprocess.run should not be called during --dry-run")

    monkeypatch.setattr(orch.subprocess, "run", _fail)

    rc = orch.main(
        [
            "--dry-run",
            "--from",
            "preprocess",
            "--to",
            "epochs",
            "--bids_root",
            str(tmp_path / "BIDS"),
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert called["ran"] is False
    assert "preproc.base" in out
    assert "preproc.epochs" in out


def test_consumer_stage_skips_when_configs_missing(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(orch.subprocess, "run", lambda *a, **k: None)
    # Select only the consumer stage without cohort/analysis configs.
    rc = orch.main(
        [
            "--dry-run",
            "--from",
            "classical-decode",
            "--bids_root",
            str(tmp_path),
            "--metadata",
            str(tmp_path / "m.csv"),
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "SKIP" in out and "--cohort_config" in out


def test_full_dry_run_uses_stage_specific_analysis_configs(tmp_path, capsys):
    rc = orch.main(
        [
            "--dry-run",
            "--from",
            "dim-reduce",
            "--to",
            "classical-decode",
            "--bids_root",
            str(tmp_path / "BIDS"),
            "--metadata",
            str(tmp_path / "meta.csv"),
            "--cohort_config",
            "cohort.yaml",
            "--dim_analysis_config",
            "dim.yaml",
            "--decode_analysis_config",
            "decode.yaml",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "dimensionality_reduction" in out and "--analysis_config dim.yaml" in out
    assert "classical_decoding" in out and "--analysis_config decode.yaml" in out


def test_list_exits_zero(capsys):
    rc = orch.main(["--list"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "to-bids" in out and "classical-decode" in out
