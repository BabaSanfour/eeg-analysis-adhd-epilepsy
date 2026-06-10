"""Pipeline comparison for DSS vs ICA branches with unified I/O and reports."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from eeg_adhd_epilepsy.io import bids
from eeg_adhd_epilepsy.reports.compare import (
    create_compare_dataset_report,
    create_compare_subject_report,
)
from eeg_adhd_epilepsy.utils.logs import setup_logging

from .utils import (
    benchmark_step,
    NumpyEncoder,
    load_stage_artifacts,
    select_subjects,
)


from .correct import ArtifactCorrectionConfig, run_correction_pipeline
from .denoise import ArtifactDenoisingConfig, run_denoising_pipeline
import eeg_adhd_epilepsy.viz.preproc_qc as viz_qc
import eeg_adhd_epilepsy.signal_quality.spectral as spectral

LOGGER = logging.getLogger(__name__)


def _collect_subject_log_traces(subject_id: str, reports_root: Path) -> Dict[str, List[str]]:
    """Collect matching log lines for audit trail (reuse mode)."""
    traces: Dict[str, List[str]] = {}
    log_files = [
        reports_root / "logs" / "correct_pipeline.log",
        reports_root / "logs" / "denoise_pipeline.log",
        reports_root / "logs" / "compare_pipeline.log",
    ]
    for log_path in log_files:
        if not log_path.exists():
            continue
        try:
            lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            hits = [line for line in lines if subject_id in line]
            if hits:
                traces[str(log_path)] = hits[-20:]
        except Exception:
            continue
    return traces


# -----------------------------------------------------------------------------
# Configurations
# -----------------------------------------------------------------------------

DSS_CONFIG = ArtifactCorrectionConfig(
    eog_method="blind-dss",
    ecg_method="dss",
    emg_method="dss",
)

ICA_CONFIG = ArtifactCorrectionConfig(
    eog_method="ica",
    ecg_method="ica",
    emg_method="ica",
    ica_exclude_prob=0.8,
    ica_n_components=20,
)

PIPELINE_CONFIGS = {
    "dss": ("correctDss", DSS_CONFIG),
    "ica": ("correctIca", ICA_CONFIG),
}


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


def _normalize_task_token(task: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "", str(task).strip())
    if not token:
        raise ValueError(f"Invalid condition/task value: {task!r}")
    return token


def _default_denoise_desc(correct_desc: str) -> str:
    """Map a correction desc token to its denoise desc token."""
    token = str(correct_desc)
    lower = token.lower()
    if lower.startswith("denoise"):
        return bids.validate_stage_desc(token)
    if lower.startswith("correct"):
        suffix = token[len("correct") :]
        return bids.validate_stage_desc(f"denoise{suffix}")
    return bids.validate_stage_desc(f"denoise{token[0].upper()}{token[1:]}")


def _compute_channel_correlation(raw_a: mne.io.BaseRaw, raw_b: mne.io.BaseRaw) -> Dict[str, float]:
    """Compute per-channel Pearson correlation between two raws."""
    picks = mne.pick_types(raw_a.info, eeg=True, exclude="bads")
    n_samples = min(raw_a.n_times, raw_b.n_times)
    data_a = raw_a.get_data(picks=picks)[:, :n_samples]
    data_b = raw_b.get_data(picks=picks)[:, :n_samples]

    corrs = []
    for ch_a, ch_b in zip(data_a, data_b):
        corr = np.corrcoef(ch_a, ch_b)[0, 1]
        corrs.append(float(corr))
    ch_names = [raw_a.ch_names[i] for i in picks]
    return dict(zip(ch_names, corrs))


def _compute_variance_removed(raw_before: mne.io.BaseRaw, raw_after: mne.io.BaseRaw) -> float:
    """Compute fraction of total variance removed (mean across channels)."""
    picks = mne.pick_types(raw_before.info, eeg=True, exclude="bads")
    n_samples = min(raw_before.n_times, raw_after.n_times)
    data_before = raw_before.get_data(picks=picks)[:, :n_samples]
    data_after = raw_after.get_data(picks=picks)[:, :n_samples]

    var_before = float(np.mean(np.var(data_before, axis=1)))
    var_after = float(np.mean(np.var(data_after, axis=1)))
    ratio = 1.0 - var_after / var_before if var_before > 0 else 0.0
    return float(np.clip(ratio, -1.0, 1.0))


def _extract_total_timing(provenance: Dict[str, Any], fallback: float = 0.0) -> float:
    """Get total timing from provenance benchmarks."""
    timing_map = provenance.get("benchmarks", {}).get("timing", {})
    if isinstance(timing_map, dict) and timing_map:
        return float(sum(float(v) for v in timing_map.values()))
    return float(fallback)


def _extract_component_counts(provenance: Dict[str, Any]) -> Tuple[int, int, int]:
    """Extract EOG/ECG/EMG component counts from correction provenance."""
    stats = provenance.get("correction_stats", {}) if isinstance(provenance, dict) else {}
    eog_comp = int(stats.get("eog", {}).get("n_components_removed", 0) or 0)
    ecg_comp = int(stats.get("ecg", {}).get("n_components_removed", 0) or 0)
    emg_comp = int(stats.get("emg", {}).get("n_components_removed", 0) or 0)
    return eog_comp, ecg_comp, emg_comp


# -----------------------------------------------------------------------------
# Main Compare Logic
# -----------------------------------------------------------------------------


def run_comparison(
    subjects: List[str],
    bids_root: Path,
    preproc_root: Path,
    reports_root: Path,
    compare_mode: str,
    dss_desc: str,
    ica_desc: str,
    strict_existing: bool,
    use_provenance_metrics: bool,
    condition_name: Optional[str] = None,
    train_condition: Optional[str] = None,
    denoise_config: Optional[ArtifactDenoisingConfig] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run compare workflow and return metrics + run metadata."""
    dss_desc = bids.validate_stage_desc(dss_desc)
    ica_desc = bids.validate_stage_desc(ica_desc)

    if compare_mode == "full":
        dss_compare_desc = _default_denoise_desc(dss_desc)
        ica_compare_desc = _default_denoise_desc(ica_desc)
    else:
        dss_compare_desc = dss_desc
        ica_compare_desc = ica_desc

    task_token = condition_name if condition_name else None

    all_metrics: List[Dict[str, Any]] = []
    timing_data: List[Dict[str, Any]] = []
    comp_data: List[Dict[str, Any]] = []
    var_data: List[Dict[str, Any]] = []
    global_plot_paths: Dict[str, str] = {}
    subject_reports: Dict[str, Path] = {}
    missing_subjects: List[str] = []
    reuse_log_traces: Dict[str, Dict[str, List[str]]] = {}

    for subject_id in subjects:
        LOGGER.info("%s", "=" * 60)
        LOGGER.info("COMPARING: %s (%s)", subject_id, compare_mode)
        LOGGER.info("%s", "=" * 60)

        raw_orig, _, base_issues = load_stage_artifacts(
            subject_id=subject_id,
            preproc_root=preproc_root,
            desc="base",
            task=task_token,
        )
        if raw_orig is None:
            LOGGER.error("Missing base input for %s: %s", subject_id, base_issues)
            missing_subjects.append(subject_id)
            continue

        method_raws: Dict[str, mne.io.BaseRaw] = {}
        method_provs: Dict[str, Dict[str, Any]] = {}
        metric_provs: Dict[str, Dict[str, Any]] = {}
        method_durations: Dict[str, float] = {}
        subject_issues: List[str] = []

        for method_name, corr_desc in (("dss", dss_desc), ("ica", ica_desc)):
            compare_desc = dss_compare_desc if method_name == "dss" else ica_compare_desc

            if compare_mode == "reuse":
                raw_obj, prov_obj, issues = load_stage_artifacts(
                    subject_id=subject_id,
                    preproc_root=preproc_root,
                    desc=compare_desc,
                    task=task_token,
                )
                if raw_obj is not None:
                    method_raws[method_name] = raw_obj
                    method_provs[method_name] = prov_obj
                    metric_provs[method_name] = prov_obj
                else:
                    subject_issues.extend(issues)

            elif compare_mode == "stage1":
                cfg = PIPELINE_CONFIGS[method_name][1]
                t_start = time.time()
                corr_result = run_correction_pipeline(
                    subject_id=subject_id,
                    bids_root=bids_root,
                    config=cfg,
                    preproc_root=preproc_root,
                    reports_root=reports_root,
                    condition_name=condition_name,
                    train_condition=train_condition,
                    output_desc=corr_desc,
                )
                method_durations[method_name] = time.time() - t_start
                if not corr_result.get("success"):
                    subject_issues.append(f"run_failed:{method_name}")
                    continue

                raw_obj, prov_obj, issues = load_stage_artifacts(
                    subject_id=subject_id,
                    preproc_root=preproc_root,
                    desc=compare_desc,
                    task=task_token,
                )
                if raw_obj is not None:
                    method_raws[method_name] = raw_obj
                    method_provs[method_name] = prov_obj
                    metric_provs[method_name] = prov_obj
                else:
                    subject_issues.extend(issues)

            elif compare_mode == "full":
                cfg = PIPELINE_CONFIGS[method_name][1]
                den_cfg = denoise_config if denoise_config is not None else ArtifactDenoisingConfig()

                t_start = time.time()
                corr_result = run_correction_pipeline(
                    subject_id=subject_id,
                    bids_root=bids_root,
                    config=cfg,
                    preproc_root=preproc_root,
                    reports_root=reports_root,
                    condition_name=condition_name,
                    train_condition=train_condition,
                    output_desc=corr_desc,
                )
                if not corr_result.get("success"):
                    subject_issues.append(f"run_failed:correct:{method_name}")
                    continue

                den_result = run_denoising_pipeline(
                    subject_id=subject_id,
                    bids_root=bids_root,
                    config=den_cfg,
                    preproc_root=preproc_root,
                    reports_root=reports_root,
                    condition_name=condition_name,
                    input_desc=corr_desc,
                    output_desc=compare_desc,
                )
                method_durations[method_name] = time.time() - t_start
                if not den_result.get("success"):
                    subject_issues.append(f"run_failed:denoise:{method_name}")
                    continue

                raw_obj, prov_obj, issues = load_stage_artifacts(
                    subject_id=subject_id,
                    preproc_root=preproc_root,
                    desc=compare_desc,
                    task=task_token,
                )
                if raw_obj is not None:
                    method_raws[method_name] = raw_obj
                    method_provs[method_name] = prov_obj
                    # For component counts in full mode, keep Stage 1 correction provenance.
                    _, corr_prov, _ = load_stage_artifacts(
                        subject_id=subject_id,
                        preproc_root=preproc_root,
                        desc=corr_desc,
                        task=task_token,
                    )
                    metric_provs[method_name] = corr_prov if corr_prov else prov_obj
                else:
                    subject_issues.extend(issues)

            else:
                subject_issues.append(f"unsupported_mode:{compare_mode}")

        if compare_mode == "reuse":
            reuse_log_traces[subject_id] = _collect_subject_log_traces(subject_id, reports_root)

        missing_method = "dss" not in method_raws or "ica" not in method_raws
        if missing_method:
            LOGGER.warning("Skipping %s due to missing method artifacts: %s", subject_id, subject_issues)
            missing_subjects.append(subject_id)
            if strict_existing and compare_mode == "reuse":
                raise RuntimeError(
                    f"Strict reuse check failed for {subject_id}: missing DSS/ICA artifacts ({subject_issues})"
                )
            continue

        raw_dss = method_raws["dss"]
        raw_ica = method_raws["ica"]

        # --- Enhanced Metrics: Spectral Preservation ---
        spectral_metrics = {}
        for name, raw in [("orig", raw_orig), ("dss", raw_dss), ("ica", raw_ica)]:
            try:
                summary = spectral.compute_stage_spectral_summary(
                    raw,
                    picks=None,
                    fmin=0.5,
                    fmax=50.0,
                )
                spectral_metrics[name] = {
                    "alpha_peak": float(summary["alpha_peak"]),
                    "slope": float(summary["aperiodic_slope"]),
                    "band_powers": {k: float(v) for k, v in summary.get("band_powers_mean", {}).items()},
                }
            except Exception as exc:
                LOGGER.warning("Spectral metrics failed for %s (%s): %s", subject_id, name, exc)
                spectral_metrics[name] = {"alpha_peak": float("nan"), "slope": float("nan"), "band_powers": {}}

        corr_map = _compute_channel_correlation(raw_dss, raw_ica)
        mean_corr = float(np.mean(list(corr_map.values()))) if corr_map else float("nan")

        subject_rows: List[Dict[str, Any]] = []
        for method_name, raw_method in (
            ("dss", raw_dss),
            ("ica", raw_ica),
        ):
            prov_for_timing = method_provs.get(method_name, {})
            prov_for_metrics = metric_provs.get(method_name, {})

            measured_duration = float(method_durations.get(method_name, 0.0))
            duration_sec = measured_duration
            if use_provenance_metrics or compare_mode == "reuse":
                duration_sec = _extract_total_timing(prov_for_timing, fallback=measured_duration)

            eog_comp, ecg_comp, emg_comp = _extract_component_counts(prov_for_metrics)
            variance_removed = _compute_variance_removed(raw_orig, raw_method) * 100.0

            row: Dict[str, Any] = {
                "subject": subject_id,
                "method": method_name,
                "mode": compare_mode,
                "duration_sec": duration_sec,
                "variance_removed_pct": variance_removed,
                "eog_components": eog_comp,
                "ecg_components": ecg_comp,
                "emg_components": emg_comp,
                "mean_dss_ica_corr": mean_corr,
                "alpha_peak_freq": spectral_metrics[method_name]["alpha_peak"],
                "alpha_peak_shift": abs(spectral_metrics[method_name]["alpha_peak"] - spectral_metrics["orig"]["alpha_peak"]),
                "aperiodic_slope": spectral_metrics[method_name]["slope"],
                "slope_distortion": abs(spectral_metrics[method_name]["slope"] - spectral_metrics["orig"]["slope"]),
                "source_desc": dss_compare_desc if method_name == "dss" else ica_compare_desc,
            }

            bp_method = spectral_metrics[method_name]["band_powers"]
            bp_orig_bands = spectral_metrics["orig"]["band_powers"]
            for band_name, val in bp_method.items():
                row[f"power_{band_name}"] = float(val)
            for band_name, orig_val in bp_orig_bands.items():
                if orig_val > 0:
                    row[f"power_{band_name}_reduction_pct"] = (
                        1.0 - bp_method.get(band_name, 0.0) / orig_val
                    ) * 100.0
                else:
                    row[f"power_{band_name}_reduction_pct"] = 0.0

            subject_rows.append(row)
            all_metrics.append(row)
            timing_data.append({"subject": subject_id, "method": method_name, "duration_sec": duration_sec})
            comp_data.append(
                {
                    "subject": subject_id,
                    "method": method_name,
                    "eog_components": eog_comp,
                    "ecg_components": ecg_comp,
                    "emg_components": emg_comp,
                }
            )
            var_data.append(
                {
                    "subject": subject_id,
                    "method": method_name,
                    "variance_removed_pct": variance_removed,
                }
            )

        subject_report_path = bids.get_subject_report_path(
            reports_root=reports_root,
            stage="compare",
            subject_id=subject_id,
            create_dir=True,
        )
        if compare_mode != "stage1":
            subject_report_path = subject_report_path.with_name(
                f"{subject_id}_compare_{compare_mode}_report.html"
            )

        subject_plots_dir = subject_report_path.parent / "figures"
        subject_plots_dir.mkdir(parents=True, exist_ok=True)

        subject_plot_paths: Dict[str, str] = {}
        try:
            subject_plot_paths["psd_comparison"] = viz_qc.plot_compare_psd(
                raw_orig, raw_dss, raw_ica, subject_id, subject_plots_dir
            )
        except Exception as exc:
            LOGGER.warning("PSD comparison plot failed for %s: %s", subject_id, exc)

        try:
            subject_plot_paths["band_power"] = viz_qc.plot_compare_band_power(
                spectral_metrics["orig"]["band_powers"],
                spectral_metrics["dss"]["band_powers"],
                spectral_metrics["ica"]["band_powers"],
                subject_id,
                subject_plots_dir,
            )
        except Exception as exc:
            LOGGER.warning("Band power plot failed for %s: %s", subject_id, exc)

        try:
            subject_plot_paths["channel_correlation"] = viz_qc.plot_compare_channel_correlation(
                corr_map, subject_id, subject_plots_dir
            )
        except Exception as exc:
            LOGGER.warning("Channel correlation plot failed for %s: %s", subject_id, exc)

        try:
            subject_plot_paths["butterfly"] = viz_qc.plot_compare_butterfly(
                raw_orig, raw_dss, raw_ica, subject_id, subject_plots_dir
            )
        except Exception as exc:
            LOGGER.warning("Butterfly plot failed for %s: %s", subject_id, exc)

        try:
            subject_plot_paths["variance_topomaps"] = viz_qc.plot_compare_variance_topomaps(
                raw_orig, raw_dss, raw_ica, subject_id, subject_plots_dir
            )
        except Exception as exc:
            LOGGER.warning("Variance topomaps failed for %s: %s", subject_id, exc)

        create_compare_subject_report(
            subject_id=subject_id,
            metrics_rows=subject_rows,
            correlation_map=corr_map,
            plot_paths=subject_plot_paths,
            subject_report_path=subject_report_path,
            mode=compare_mode,
            metadata={
                "dss_desc": dss_compare_desc,
                "ica_desc": ica_compare_desc,
                "condition": condition_name,
            },
        )
        subject_reports[subject_id] = subject_report_path

    metrics_df = pd.DataFrame(all_metrics)

    compare_paths = bids.get_compare_summary_paths(reports_root=reports_root, create_dir=True)
    if compare_mode == "stage1":
        summary_report_path = compare_paths["report_html"]
        metrics_csv_path = compare_paths["metrics_csv"]
        metadata_json_path = compare_paths["run_metadata_json"]
    else:
        summary_report_path = compare_paths["report_html"].with_name(
            f"compare_{compare_mode}_dataset_summary.html"
        )
        metrics_csv_path = compare_paths["metrics_csv"].with_name(
            f"compare_{compare_mode}_metrics.csv"
        )
        metadata_json_path = compare_paths["run_metadata_json"].with_name(
            f"compare_{compare_mode}_run_metadata.json"
        )

    metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_csv_path, index=False)
    LOGGER.info("Comparison CSV saved to %s", metrics_csv_path)

    global_plots_dir = summary_report_path.parent / "plots"
    global_plots_dir.mkdir(parents=True, exist_ok=True)

    if not metrics_df.empty:
        try:
            plot_path = viz_qc.plot_compare_timing(timing_data, global_plots_dir)
            if plot_path:
                global_plot_paths["timing_comparison"] = plot_path
        except Exception as exc:
            LOGGER.warning("Timing plot failed: %s", exc)

        try:
            plot_path = viz_qc.plot_compare_components_removed(comp_data, global_plots_dir)
            if plot_path:
                global_plot_paths["components_removed"] = plot_path
        except Exception as exc:
            LOGGER.warning("Components removed plot failed: %s", exc)

        try:
            plot_path = viz_qc.plot_compare_variance_removed(var_data, global_plots_dir)
            if plot_path:
                global_plot_paths["variance_removed"] = plot_path
        except Exception as exc:
            LOGGER.warning("Variance removed plot failed: %s", exc)

        try:
            plot_path = viz_qc.plot_compare_summary_dashboard(metrics_df, global_plots_dir)
            if plot_path:
                global_plot_paths["comparison_dashboard"] = plot_path
        except Exception as exc:
            LOGGER.warning("Dashboard plot failed: %s", exc)

    run_metadata = {
        "mode": compare_mode,
        "dss_desc": dss_desc,
        "ica_desc": ica_desc,
        "dss_compare_desc": dss_compare_desc,
        "ica_compare_desc": ica_compare_desc,
        "strict_existing": strict_existing,
        "use_provenance_metrics": use_provenance_metrics,
        "condition": condition_name,
        "train_condition": train_condition,
        "subjects_requested": subjects,
        "subjects_with_reports": sorted(subject_reports.keys()),
        "missing_subjects": sorted(set(missing_subjects)),
        "metrics_csv": str(metrics_csv_path),
        "summary_report": str(summary_report_path),
    }
    if compare_mode == "reuse":
        run_metadata["reuse_log_traces"] = reuse_log_traces

    with open(metadata_json_path, "w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2)
    LOGGER.info("Compare run metadata saved to %s", metadata_json_path)

    create_compare_dataset_report(
        metrics_df=metrics_df,
        summary_report_path=summary_report_path,
        mode=compare_mode,
        subject_reports=subject_reports,
        global_plot_paths=global_plot_paths,
        missing_subjects=missing_subjects,
        run_metadata=run_metadata,
        metrics_csv_path=metrics_csv_path,
    )

    return metrics_df, run_metadata


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare DSS vs ICA pipelines")
    parser.add_argument("--bids_root", type=str, required=True, help="Path to BIDS dataset root")
    parser.add_argument(
        "--preproc_root",
        type=str,
        default=None,
        help="Directory for stage FIF/provenance artifacts (default: <bids_root>/derivatives/preproc)",
    )
    parser.add_argument(
        "--reports_root",
        type=str,
        default=None,
        help="Directory for reports/logs (default: <cwd>/results/reports/preproc)",
    )

    parser.add_argument(
        "--compare-mode",
        type=str,
        default="stage1",
        choices=["stage1", "full", "reuse"],
        help="Compare mode: rerun stage1, rerun full (stage1+stage2), or reuse existing outputs",
    )
    parser.add_argument(
        "--reuse-existing-correct",
        action="store_true",
        help="Alias for --compare-mode reuse",
    )
    parser.add_argument("--dss-desc", type=str, default="correctDss", help="Desc token for DSS branch")
    parser.add_argument("--ica-desc", type=str, default="correctIca", help="Desc token for ICA branch")
    parser.add_argument(
        "--strict-existing",
        action="store_true",
        help="In reuse mode, require both DSS and ICA artifacts for every subject",
    )
    parser.add_argument(
        "--use-provenance-metrics",
        action="store_true",
        help="Use timing/components metrics from stored provenance when available",
    )

    parser.add_argument("--subjects", nargs="+", help="Specific subjects")
    parser.add_argument("--start-from", type=str, help="Start from this subject ID")
    parser.add_argument("--test", action="store_true", help="Test mode: first 5 subjects")
    parser.add_argument("--random", action="store_true", help="Random selection in test mode")
    parser.add_argument("--all", action="store_true", help="Process all subjects")

    parser.add_argument("--condition", type=str, help="Process specific condition/task")
    parser.add_argument("--train-condition", type=str, help="Training condition for Stage 1 reruns")
    parser.add_argument(
        "--transient-method",
        type=str,
        default="wiener",
        choices=["wiener", "asr", "dss", "none"],
        help="Stage 2 transient method when --compare-mode full",
    )

    args = parser.parse_args()

    compare_mode = "reuse" if args.reuse_existing_correct else args.compare_mode

    bids_root = Path(args.bids_root).expanduser()
    preproc_root = bids.get_preproc_root(bids_root)
    reports_root = bids.get_reports_root(bids_root)
    preproc_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)

    log_file = reports_root / "logs" / "compare_pipeline.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file, "INFO")

    if not bids_root.exists():
        LOGGER.error("BIDS root not found: %s", bids_root)
        sys.exit(1)

    task_token = _normalize_task_token(args.condition) if args.condition else None
    if task_token:
        base_pattern = f"*_task-{task_token}_desc-base_eeg.fif"
    else:
        base_pattern = "*_desc-base_eeg.fif"

    files = sorted(preproc_root.rglob(base_pattern))
    if not files:
        LOGGER.error("No base FIF files found in %s (pattern: %s)", preproc_root, base_pattern)
        sys.exit(1)

    subjects_found = sorted({bids.parse_subject_id(f) for f in files})
    LOGGER.info("Found %d base subjects.", len(subjects_found))

    subjects_to_process = select_subjects(
        subjects_found,
        selected_subjects=args.subjects,
        start_from=args.start_from,
        use_test=args.test,
        use_random_test=args.random,
        use_all=args.all,
    )
    if not subjects_to_process:
        if args.start_from:
            LOGGER.error("No subjects found starting from %s.", bids.normalize_subject_id(args.start_from))
            sys.exit(1)
        parser.print_help()
        sys.exit(0)
    subjects_to_process = set(subjects_to_process)

    subjects_sorted = sorted(subjects_to_process)
    LOGGER.info("Running compare (%s) for %d subjects: %s", compare_mode, len(subjects_sorted), subjects_sorted)

    denoise_config: Optional[ArtifactDenoisingConfig] = None
    if compare_mode == "full":
        denoise_config = ArtifactDenoisingConfig(
            transient_method=args.transient_method if args.transient_method != "none" else None
        )

    try:
        metrics_df, run_metadata = run_comparison(
            subjects=subjects_sorted,
            bids_root=bids_root,
            preproc_root=preproc_root,
            reports_root=reports_root,
            compare_mode=compare_mode,
            dss_desc=args.dss_desc,
            ica_desc=args.ica_desc,
            strict_existing=args.strict_existing,
            use_provenance_metrics=args.use_provenance_metrics,
            condition_name=args.condition,
            train_condition=args.train_condition,
            denoise_config=denoise_config,
        )
    except RuntimeError as exc:
        LOGGER.error("%s", exc)
        sys.exit(1)

    if metrics_df.empty:
        LOGGER.warning("No comparison rows were produced.")
        sys.exit(0)

    print("\n" + "=" * 70)
    print("  DSS vs ICA COMPARISON SUMMARY")
    print("=" * 70)
    print(f"  Mode: {run_metadata.get('mode')}")
    for method in ["dss", "ica"]:
        method_df = metrics_df[metrics_df["method"] == method]
        if method_df.empty:
            continue
        print(f"\n  {method.upper()}")
        print(f"    Avg Time:         {method_df['duration_sec'].mean():.1f}s")
        print(f"    Avg Var Removed:  {method_df['variance_removed_pct'].mean():.2f}%")
        print(f"    EOG Comp (avg):   {method_df['eog_components'].mean():.1f}")
        print(f"    ECG Comp (avg):   {method_df['ecg_components'].mean():.1f}")
        print(f"    EMG Comp (avg):   {method_df['emg_components'].mean():.1f}")

    corr_vals = metrics_df["mean_dss_ica_corr"].dropna()
    if not corr_vals.empty:
        print(f"\n  DSS-ICA Avg Correlation: {corr_vals.mean():.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
