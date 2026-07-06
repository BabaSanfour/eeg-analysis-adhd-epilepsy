import pandas as pd
from coco_pipe.decoding import ExperimentResult
from coco_pipe.io.quality import QCResult, SubjectDropRecord
from coco_pipe.report import (
    best_rows,
    display_frame,
    primary_metric_column,
    signature_compatibility,
)

from eeg_adhd_epilepsy.reports.decoding import (
    generate_decoding_summary_report,
    generate_foundation_decoding_report,
    generate_head_to_head_report,
)


def _asset_urls():
    return {"plotly": "about:blank", "tailwind": "about:blank", "pako": "about:blank"}


def _write_result(tmp_path, name, *, accuracy=0.62, model="logreg"):
    output_dir = tmp_path / name
    result = ExperimentResult(
        {
            model: {
                "metrics": {
                    "accuracy": {
                        "mean": accuracy,
                        "std": 0.02,
                        "folds": [accuracy - 0.02, accuracy, accuracy + 0.02],
                    },
                    "balanced_accuracy": {
                        "mean": accuracy - 0.01,
                        "std": 0.02,
                        "folds": [accuracy - 0.03, accuracy - 0.01, accuracy + 0.01],
                    },
                },
                "predictions": [],
                "splits": [],
                "diagnostics": [],
                "metadata": [],
                "statistical_assessment": [],
                "importances": {
                    "mean": [0.3, 0.2],
                    "std": [0.01, 0.02],
                    "raw": [[0.3, 0.2], [0.31, 0.18], [0.29, 0.22]],
                    "feature_names": [
                        "Fp1_mean_log_abs_alpha",
                        "Fz_mean_entropy",
                    ],
                },
            }
        },
        meta={"task": "classification", "n_samples": 30, "n_features": 2},
    )
    result.save(output_dir / "result.joblib")
    return output_dir


def _foundation_records(tmp_path):
    records = []
    for target in ("adhd", "medication"):
        for model_key in ("labram", "cbramod"):
            for train_mode in ("linear_probe", "full", "lora"):
                output_dir = _write_result(
                    tmp_path,
                    f"{target}_{model_key}_{train_mode}",
                    accuracy=0.62,
                    model=f"{model_key}_{train_mode}",
                )
                records.append(
                    {
                        "condition": "EO",
                        "target": target,
                        "model_key": model_key,
                        "train_mode": train_mode,
                        "primary": train_mode == "linear_probe",
                        "model": f"{model_key}_{train_mode}",
                        "status": "success",
                        "accuracy_mean": 0.62,
                        "balanced_accuracy_mean": 0.6,
                        "f1_mean": 0.61,
                        "roc_auc_mean": 0.7,
                        "output_dir": str(output_dir),
                    }
                )
    records.append(
        {
            "condition": "EO",
            "target": "adhd",
            "model_key": "eegpt",
            "train_mode": "lora",
            "status": "skipped",
            "reason": "unsupported",
        }
    )
    return records


def test_decoding_report_metric_priority_prefers_balanced_accuracy():
    frame = pd.DataFrame(
        [
            {"accuracy_mean": 0.9, "balanced_accuracy_mean": 0.62},
            {"accuracy_mean": 0.8, "balanced_accuracy_mean": 0.71},
        ]
    )
    assert primary_metric_column(frame) == "balanced_accuracy_mean"


def test_decoding_report_best_rows_rank_by_primary_then_fdr():
    frame = pd.DataFrame(
        [
            {
                "scope": "EO",
                "target": "adhd",
                "model": "rf",
                "status": "success",
                "balanced_accuracy_mean": 0.7,
                "p_value_fdr": 0.04,
            },
            {
                "scope": "EO",
                "target": "adhd",
                "model": "logreg",
                "status": "success",
                "balanced_accuracy_mean": 0.7,
                "p_value_fdr": 0.02,
            },
        ]
    )
    best, metric = best_rows(
        frame,
        ("scope", "target"),
        tie_breakers=(("p_value_fdr", True), ("p_value", True)),
    )
    assert metric == "balanced_accuracy_mean"
    assert best.loc[0, "model"] == "logreg"


def test_decoding_report_cv_compatibility_detects_mismatches():
    frame = pd.DataFrame(
        [
            {"scope": "EO", "target": "adhd", "cv_signature": "cv-a"},
            {"scope": "EO", "target": "adhd", "cv_signature": "cv-b"},
            {"scope": "EC", "target": "adhd", "cv_signature": "cv-a"},
            {"scope": "EC", "target": "adhd", "cv_signature": "cv-a"},
        ]
    )
    compatibility = signature_compatibility(frame, ("scope", "target"))
    eo = compatibility[compatibility["scope"] == "EO"].iloc[0]
    ec = compatibility[compatibility["scope"] == "EC"].iloc[0]
    assert not bool(eo["paired_compatible"])
    assert eo["mismatched_fields"] == "cv_signature"
    assert bool(ec["paired_compatible"])


def test_foundation_visual_report_uses_coco_pipe_comparisons(tmp_path):
    figures_dir = tmp_path / "figures"
    output = generate_foundation_decoding_report(
        tmp_path / "visual.html",
        _foundation_records(tmp_path),
        title="Foundation Figures",
        config={"report_asset_urls": _asset_urls()},
        capability_records=[
            {
                "condition": "EO",
                "target": "adhd",
                "model_key": "labram",
                "train_mode": "lora",
                "status": "available",
                "reason": "",
            }
        ],
        figures_dir=figures_dir,
    )
    html = output.read_text(encoding="utf-8")
    assert "Scientific Overview" in html
    assert "Linear Probe Leaderboard" in html
    assert "Linear Probe" in html
    assert "Training-Mode Comparison" in html
    assert "Foundation Capability Matrix" in html
    assert "Per-Result Diagnostics" in html
    assert "Skipped and Failed Units" in html
    assert not list(figures_dir.glob("*.png"))


def test_foundation_visual_report_handles_empty_records(tmp_path):
    output = generate_foundation_decoding_report(
        tmp_path / "empty.html",
        [],
        title="Foundation Figures",
        config={"report_asset_urls": _asset_urls()},
    )
    assert "No foundation decoding units were produced." in output.read_text(encoding="utf-8")


def test_decoding_summary_is_grouped_by_scope_and_analysis_plan(tmp_path):
    records = []
    modes = [
        "flat",
        "sensor",
        "subfamily",
        "sensor_within_subfamily",
        "descriptor",
        "descriptor_sensor",
    ]
    for scope in ("EO_baseline", "EC_baseline", "pooled"):
        for mode in modes:
            records.append(
                {
                    "scope": scope,
                    "target": "adhd",
                    "status": "success",
                    "analysis_mode": mode,
                    "unit_name": "all",
                    "model": "logreg_l1",
                    "selection_mode": "baseline",
                    "primary": mode == "flat",
                    "balanced_accuracy_mean": 0.61,
                }
            )
    output = generate_decoding_summary_report(
        tmp_path / "summary.html",
        records,
        title="Summary",
        config={
            "conditions": ["EO_baseline", "EC_baseline"],
            "report_asset_urls": _asset_urls(),
        },
    )
    html = output.read_text(encoding="utf-8")
    assert "Scientific Overview" in html
    assert "Primary Leaderboard" in html
    titles = [
        "Full Analysis: All Sensors x All Features",
        "Sensor-wise Analyses",
        "Subfamily Analyses: All Sensors",
        "Sensor x Subfamily Analyses",
        "Single Descriptor (all stats): All Sensors",
        "Single Descriptor (all stats) x Single Sensor",
    ]
    first_scope_positions = [html.index(title) for title in titles]
    assert first_scope_positions == sorted(first_scope_positions)
    assert "EO_baseline" in html
    assert "EC_baseline" in html
    assert "POOLED" in html
    assert html.index("EO_baseline") < html.index("EC_baseline") < html.index("POOLED")


def test_classical_decoding_summary_uses_coco_pipe_comparisons(tmp_path):
    figures_dir = tmp_path / "figures"
    records = [
        {
            "scope": "EO_baseline",
            "target": "adhd",
            "status": "success",
            "analysis_mode": "flat",
            "unit_name": "all",
            "model": "logreg_l1",
            "selection_mode": "baseline",
            "accuracy_mean": 0.65,
            "accuracy_std": 0.03,
            "output_dir": str(_write_result(tmp_path, "flat", accuracy=0.65)),
        },
    ]
    for sensor, accuracy in zip(
        ("Fp1", "Fp2", "F3", "F4", "Cz"),
        (0.56, 0.57, 0.58, 0.59, 0.60),
    ):
        output_dir = _write_result(
            tmp_path,
            f"sensor_{sensor}",
            accuracy=accuracy,
        )
        records.append(
            {
                "scope": "EO_baseline",
                "target": "adhd",
                "status": "success",
                "analysis_mode": "sensor",
                "unit_name": sensor,
                "model": "logreg_l1",
                "selection_mode": "baseline",
                "accuracy_mean": accuracy,
                "output_dir": str(output_dir),
            }
        )
    records.append(
        {
            "scope": "EO_baseline",
            "target": "adhd",
            "status": "success",
            "analysis_mode": "sensor",
            "unit_name": "Fp1",
            "model": "logreg_l1",
            "selection_mode": "sfs",
            "accuracy_mean": 0.61,
            "output_dir": str(_write_result(tmp_path, "sensor_fp1_sfs", accuracy=0.61)),
        }
    )
    for subfamily, sensor, accuracy in (
        ("log_abs", "Fp1", 0.55),
        ("log_abs", "Fp2", 0.57),
        ("entropy", "Fp1", 0.61),
        ("entropy", "Fp2", 0.63),
    ):
        output_dir = _write_result(
            tmp_path,
            f"{subfamily}_{sensor}",
            accuracy=accuracy,
        )
        records.extend(
            [
                {
                    "scope": "EO_baseline",
                    "target": "adhd",
                    "status": "success",
                    "analysis_mode": "sensor_within_subfamily",
                    "unit_key": f"{subfamily}_{sensor}",
                    "unit_name": sensor,
                    "subfamily": subfamily,
                    "model": "logreg_l1",
                    "selection_mode": "baseline",
                    "accuracy_mean": accuracy,
                    "output_dir": str(output_dir),
                },
                {
                    "scope": "EO_baseline",
                    "target": "adhd",
                    "status": "success",
                    "analysis_mode": "subfamily",
                    "unit_key": f"{subfamily}_{sensor}",
                    "unit_name": subfamily,
                    "subfamily": subfamily,
                    "model": "logreg_l1",
                    "selection_mode": "baseline",
                    "accuracy_mean": accuracy,
                    "output_dir": str(output_dir),
                },
            ]
        )
    records.extend(
        [
            {
                "scope": "EO_baseline",
                "target": "adhd",
                "status": "success",
                "analysis_mode": "descriptor",
                "unit_key": "mean_log_abs_alpha",
                "unit_name": "mean_log_abs_alpha",
                "model": "logreg_l1",
                "selection_mode": "baseline",
                "accuracy_mean": 0.59,
            },
            {
                "scope": "EO_baseline",
                "target": "adhd",
                "status": "success",
                "analysis_mode": "descriptor_sensor",
                "unit_key": "mean_log_abs_alpha_Fp1",
                "unit_name": "mean_log_abs_alpha",
                "model": "logreg_l1",
                "selection_mode": "baseline",
                "accuracy_mean": 0.58,
            },
            {
                "scope": "EO_baseline",
                "target": "adhd",
                "status": "success",
                "analysis_mode": "descriptor_sensor",
                "unit_key": "mean_log_abs_alpha_Fp2",
                "unit_name": "mean_log_abs_alpha",
                "model": "logreg_l1",
                "selection_mode": "baseline",
                "accuracy_mean": 0.60,
            },
        ]
    )
    output = generate_decoding_summary_report(
        tmp_path / "summary.html",
        records,
        title="Summary",
        config={
            "conditions": ["EO_baseline"],
            "run_pooled": False,
            "report_asset_urls": _asset_urls(),
        },
        figures_dir=figures_dir,
    )

    assert not list(figures_dir.glob("*.png"))
    html = output.read_text(encoding="utf-8")
    assert "Scientific Overview" in html
    assert "Primary Leaderboard" in html
    assert "Model Heatmap" in html
    assert "Score Spread" in html
    assert "Feature-Selection Diagnostics" in html
    assert "Sensor-wise Accuracy: EO_baseline" in html
    assert "Subfamily Accuracy: EO_baseline" in html
    assert "Sensor x Subfamily Accuracy: EO_baseline" in html
    assert "Single-Descriptor Accuracy: EO_baseline" in html
    assert "Single Descriptor x Sensor Accuracy: EO_baseline" in html


def test_classical_decoding_summary_uses_full_result_artifacts(tmp_path):
    output_dir = _write_result(tmp_path, "full_result", accuracy=0.65)

    report_path = generate_decoding_summary_report(
        tmp_path / "summary.html",
        [
            {
                "scope": "EO_baseline",
                "target": "adhd",
                "status": "success",
                "analysis_mode": "flat",
                "unit_name": "all",
                "model": "logreg",
                "selection_mode": "baseline",
                "accuracy_mean": 0.65,
                "output_dir": str(output_dir),
            }
        ],
        title="Summary",
        config={
            "conditions": ["EO_baseline"],
            "run_pooled": False,
            "report_asset_urls": _asset_urls(),
        },
    )

    html = report_path.read_text(encoding="utf-8")
    assert "Performance" in html
    assert "Cross-Validation" in html
    assert "Features" in html
    assert "Download Fold Scores CSV" in html


def test_decoding_summary_keeps_configured_scope_with_no_rows(tmp_path):
    output = generate_decoding_summary_report(
        tmp_path / "summary.html",
        [
            {
                "scope": "EO_baseline",
                "status": "success",
                "analysis_mode": "flat",
                "selection_mode": "baseline",
            }
        ],
        title="Summary",
        config={
            "conditions": ["EO_baseline", "EC_baseline"],
            "run_pooled": False,
            "report_asset_urls": _asset_urls(),
        },
    )
    html = output.read_text(encoding="utf-8")
    assert "EC_baseline" in html
    assert "The full analysis is missing or incomplete." in html


def test_decoding_result_tables_lead_with_scores_and_hide_audit_noise():
    display = display_frame(
        pd.DataFrame(
            [
                {
                    "scope": "EO_baseline",
                    "analysis_mode": "flat",
                    "unit_key": "all",
                    "cohort_signature": "abc123",
                    "output_dir": "/tmp/result",
                    "target": "adhd",
                    "unit_name": "all",
                    "model": "rf",
                    "selection_mode": "baseline",
                    "status": "success",
                    "n_samples": 100,
                    "n_groups": 50,
                    "accuracy_mean": 0.7,
                    "accuracy_std": 0.02,
                    "balanced_accuracy_mean": 0.68,
                    "f1_mean": 0.69,
                    "p_value_fdr": 0.03,
                    "significant_fdr": True,
                }
            ]
        )
    )

    assert list(display.columns) == [
        "Target",
        "Analysis Unit",
        "Model",
        "Feature Selection",
        "Status",
        "N Observations",
        "N Subjects",
        "Accuracy",
        "Accuracy SD",
        "Balanced Accuracy",
        "F1",
        "FDR P Value",
        "FDR Significant",
    ]
    assert "scope" not in display
    assert "cohort_signature" not in display
    assert "output_dir" not in display


def test_decoding_summary_renders_family_scoped_qc(tmp_path):
    output = generate_decoding_summary_report(
        tmp_path / "summary.html",
        [],
        title="Summary",
        config={"report_asset_urls": _asset_urls()},
        qc_results=[
            (
                "EO_baseline",
                QCResult(
                    n_obs_in=6,
                    n_obs_out=6,
                    per_family_dropped={
                        "log_abs_alpha": [
                            SubjectDropRecord(
                                subject_id="0006",
                                outlier_fraction=1.0,
                                n_outlier_features=2.0,
                            )
                        ]
                    },
                    thresholds={"group_by": "measure"},
                ),
            )
        ],
    )

    html = output.read_text(encoding="utf-8")
    assert "Data Quality (QC): EO_baseline" in html
    assert "Conditional Drops by measure" in html
    assert "Per-Group Retention" in html
    assert "log_abs_alpha" in html
    # The misleading combined "Dropped Subjects" table must NOT appear.
    assert "Dropped Subjects" not in html


def test_decoding_summary_paginates_long_qc_tables(tmp_path):
    per_family_dropped = {
        f"measure_{index:02d}": [
            SubjectDropRecord(
                subject_id=f"{index:04d}",
                outlier_fraction=1.0,
                n_outlier_features=2.0,
            )
        ]
        for index in range(11)
    }
    output = generate_decoding_summary_report(
        tmp_path / "summary.html",
        [],
        title="Summary",
        config={"report_asset_urls": _asset_urls()},
        qc_results=[
            (
                "EO_baseline",
                QCResult(
                    n_obs_in=20,
                    n_obs_out=20,
                    per_family_dropped=per_family_dropped,
                    thresholds={"group_by": "measure"},
                ),
            )
        ],
    )

    html = output.read_text(encoding="utf-8")
    assert html.count('class="interactive-table"') == 2
    assert "&quot;selector_columns&quot;: [&quot;Group (measure)&quot;]" in html
    assert "&quot;page_size&quot;: 10" in html


def test_decoding_summary_only_shows_features_with_missingness(tmp_path):
    output = generate_decoding_summary_report(
        tmp_path / "summary.html",
        [],
        title="Summary",
        config={"report_asset_urls": _asset_urls()},
        qc_results=[
            (
                "EO_baseline",
                QCResult(
                    feature_missingness=pd.DataFrame(
                        {
                            "column": ["complete_feature", "missing_feature"],
                            "missing_count": [0, 2],
                            "missing_rate": [0.0, 0.1],
                        }
                    )
                ),
            )
        ],
    )

    html = output.read_text(encoding="utf-8")
    assert "Feature Missingness" in html
    assert "missing_feature" in html
    assert "complete_feature" not in html


def test_decoding_summary_omits_zero_feature_missingness(tmp_path):
    output = generate_decoding_summary_report(
        tmp_path / "summary.html",
        [],
        title="Summary",
        config={"report_asset_urls": _asset_urls()},
        qc_results=[
            (
                "EO_baseline",
                QCResult(
                    feature_missingness=pd.DataFrame(
                        {
                            "column": ["complete_feature"],
                            "missing_count": [0],
                            "missing_rate": [0.0],
                        }
                    )
                ),
            )
        ],
    )

    html = output.read_text(encoding="utf-8")
    assert "Feature Missingness" not in html
    assert "complete_feature" not in html


def test_head_to_head_report_includes_grouped_cv_signature(tmp_path):
    bids_root = tmp_path / "BIDS"
    result_root = bids_root / "derivatives" / "decoding" / "dataset" / "descriptors"
    result_root.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "status": "success",
                "input_mode": "descriptors",
                "analysis_mode": "flat",
                "selection_mode": "baseline",
                "scope": "EO_baseline",
                "target": "adhd",
                "model": "logreg",
                "balanced_accuracy_mean": 0.62,
                "cv_strategy": "stratified_group_kfold",
                "effective_n_splits": 5,
                "cv_random_state": 42,
                "cohort_signature": "abc123",
            },
            {
                "status": "success",
                "input_mode": "descriptors",
                "analysis_mode": "flat",
                "selection_mode": "baseline",
                "scope": "EC_baseline",
                "target": "adhd",
                "model": "logreg",
                "balanced_accuracy_mean": 0.6,
                "cv_strategy": "stratified_group_kfold",
                "effective_n_splits": 5,
                "cv_random_state": 42,
                "cohort_signature": "abc123",
            },
        ]
    ).to_csv(result_root / "sweep_results.csv", index=False)
    embedding_root = bids_root / "derivatives" / "decoding" / "dataset" / "foundation_embeddings"
    embedding_root.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "status": "success",
                "input_mode": "foundation_embeddings",
                "analysis_mode": "flat",
                "selection_mode": "baseline",
                "scope": "EC_baseline",
                "target": "adhd",
                "model": "ridge",
                "balanced_accuracy_mean": 0.66,
                "cv_strategy": "stratified_group_kfold",
                "effective_n_splits": 5,
                "cv_random_state": 42,
                "cohort_signature": "abc123",
            }
        ]
    ).to_csv(embedding_root / "sweep_results.csv", index=False)
    foundation_root = bids_root / "derivatives" / "decoding" / "dataset" / "foundation_linear"
    foundation_root.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "status": "success",
                "condition": "EO_baseline",
                "target": "adhd",
                "model_key": "labram",
                "train_mode": "linear_probe",
                "balanced_accuracy_mean": 0.67,
                "cv_strategy": "stratified_group_kfold",
                "effective_n_splits": 5,
                "cv_random_state": 99,
                "cohort_signature": "abc123",
            }
        ]
    ).to_csv(foundation_root / "foundation_results.csv", index=False)

    outputs = generate_head_to_head_report(
        bids_root=bids_root,
        reports_root=tmp_path / "reports",
        dataset_name="dataset",
        asset_urls=_asset_urls(),
    )
    assert outputs is not None
    comparison_path, report_path = outputs
    comparison = pd.read_csv(comparison_path)
    assert "cv_signature" in comparison
    assert "primary_metric" in comparison
    assert "comparison_family" in comparison
    assert set(comparison["comparison_family"]) == {
        "descriptor_flat_baseline",
        "foundation_embedding_flat_baseline",
        "foundation_linear_probe",
    }
    assert "stratified_group_kfold" in comparison.loc[0, "cv_signature"]
    assert comparison.loc[0, "cohort_signature"] == "abc123"
    html = report_path.read_text(encoding="utf-8")
    assert "Head-to-Head Comparison" in html
    assert "Comparison Compatibility" in html
    assert "Paired Comparisons Limited" in html
    assert "Paired Delta vs Descriptor Baseline" in html


def test_head_to_head_ignores_empty_failed_sweep(tmp_path):
    failed_root = tmp_path / "BIDS" / "derivatives" / "decoding" / "dataset" / "descriptors"
    failed_root.mkdir(parents=True)
    (failed_root / "sweep_results.csv").write_text("", encoding="utf-8")
    assert (
        generate_head_to_head_report(
            bids_root=tmp_path / "BIDS",
            reports_root=tmp_path / "reports",
            dataset_name="dataset",
            asset_urls=_asset_urls(),
        )
        is None
    )
