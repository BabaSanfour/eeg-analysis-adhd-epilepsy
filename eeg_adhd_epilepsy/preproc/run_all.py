"""Single-entry orchestrator for Base -> Correct -> Denoise (optional Compare)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from collections import defaultdict

from eeg_adhd_epilepsy.io import bids
from eeg_adhd_epilepsy.qc import preproc_qc
from eeg_adhd_epilepsy.utils.logs import setup_logging

from .base import DEFAULT_HIGHPASS_HZ, DEFAULT_LOWPASS_HZ, run_base_record
from .compare import run_comparison
from .correct import ArtifactCorrectionConfig, run_correction_pipeline
from .denoise import ArtifactDenoisingConfig, run_denoising_pipeline

LOGGER = logging.getLogger("preproc_run_all")


def _normalize_subject_list(subjects: Sequence[str]) -> List[str]:
    return sorted({bids.normalize_subject_id(s) for s in subjects})


def _discover_input_files(bids_root: Path) -> List[Path]:
    """Discover all raw EEG input runs."""
    return bids.discover_bids_files(bids_root, suffix="eeg", extension=".vhdr")


def _select_subjects(
    subjects_found: Sequence[str],
    selected_subjects: Optional[Sequence[str]] = None,
    start_from: Optional[str] = None,
    use_test: bool = False,
    use_random_test: bool = False,
    use_all: bool = False,
) -> List[str]:
    """Apply standard subject selection logic."""
    found_sorted = sorted(set(subjects_found))
    if selected_subjects:
        return _normalize_subject_list(selected_subjects)

    if start_from:
        start_sid = bids.normalize_subject_id(start_from)
        chosen = [sid for sid in found_sorted if sid >= start_sid]
        return chosen

    if use_test:
        if use_random_test:
            import random

            random.seed(42)
            return sorted(random.sample(found_sorted, min(5, len(found_sorted))))
        return found_sorted[:5]

    if use_all:
        return found_sorted

    return []


def _build_base_config(
    *,
    bids_root: Path,
    preproc_root: Path,
    reports_root: Path,
    n_jobs: int,
    highpass: float,
    lowpass: float,
    resample: Optional[float],
    line_freq: float,
    adaptive: bool,
    segments_file: Optional[str],
) -> Dict:
    processing_cfg: Dict[str, object] = {
        "highpass_hz": highpass,
        "lowpass_hz": lowpass,
        "resample_hz": resample,
    }
    if segments_file:
        processing_cfg["segments_file"] = segments_file

    return {
        "n_jobs": int(n_jobs),
        "bids_root": str(bids_root),
        "preproc_root": str(preproc_root),
        "reports_root": str(reports_root),
        "processing": processing_cfg,
        "line_noise": {
            "line_freq": line_freq,
            "adaptive": bool(adaptive),
        },
    }


def _load_json(path: Optional[str]) -> Dict:
    if not path:
        return {}
    p = Path(path).expanduser()
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object JSON in {p}, got {type(data).__name__}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Base -> Correct -> Denoise sequentially with unified paths"
    )
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

    parser.add_argument("--all", action="store_true", help="Process all subjects")
    parser.add_argument("--test", action="store_true", help="Process first 5 subjects")
    parser.add_argument("--random", action="store_true", help="Random selection in test mode")
    parser.add_argument("--subjects", nargs="+", help="Specific subjects (e.g. sub-001 sub-002)")
    parser.add_argument("--start-from", type=str, help="Start from this subject ID")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip each stage when its expected output already exists",
    )

    parser.add_argument("--n_jobs", type=int, default=1, help="Internal parallel workers for stage methods")
    parser.add_argument("--highpass", type=float, default=DEFAULT_HIGHPASS_HZ, help="Base high-pass frequency (Hz)")
    parser.add_argument("--lowpass", type=float, default=DEFAULT_LOWPASS_HZ, help="Base low-pass frequency (Hz)")
    parser.add_argument("--resample", type=float, default=None, help="Base resampling frequency (Hz)")
    parser.add_argument("--line-freq", type=float, default=60.0, help="Line noise frequency (Hz)")
    parser.add_argument("--adaptive", action="store_true", help="Enable adaptive ZapLine mode")
    parser.add_argument(
        "--segments-file",
        type=str,
        default=None,
        help="Optional segments CSV path for base block annotation",
    )
    parser.add_argument(
        "--base-config-json",
        type=str,
        default=None,
        help="Optional JSON object merged into base config",
    )

    parser.add_argument(
        "--eog-method",
        type=str,
        default="dss",
        choices=["dss", "ica", "blind-dss", "none"],
        help="Stage 1 EOG method",
    )
    parser.add_argument(
        "--ecg-method",
        type=str,
        default="dss",
        choices=["dss", "ica", "quasiperiodic", "none"],
        help="Stage 1 ECG method",
    )
    parser.add_argument(
        "--emg-method",
        type=str,
        default="mwf",
        choices=["mwf", "wica", "ica", "dss", "none"],
        help="Stage 1 EMG method",
    )
    parser.add_argument("--correct-desc", type=str, default="correct", help="Stage 1 output desc token")
    parser.add_argument(
        "--correct-config-json",
        type=str,
        default=None,
        help="Optional JSON object merged into ArtifactCorrectionConfig",
    )

    parser.add_argument(
        "--transient-method",
        type=str,
        default="wiener",
        choices=["wiener", "asr", "dss", "none"],
        help="Stage 2 transient method",
    )
    parser.add_argument("--denoise-desc", type=str, default="denoise", help="Stage 2 output desc token")
    parser.add_argument(
        "--denoise-config-json",
        type=str,
        default=None,
        help="Optional JSON object merged into ArtifactDenoisingConfig",
    )

    parser.add_argument("--condition", type=str, default=None, help="Optional condition/task for correct/denoise")
    parser.add_argument(
        "--train-condition",
        type=str,
        default=None,
        help="Optional training condition for Stage 1 fitting",
    )

    parser.add_argument("--run-compare", action="store_true", help="Run compare after base/correct/denoise")
    parser.add_argument(
        "--compare-mode",
        type=str,
        default="stage1",
        choices=["stage1", "full", "reuse"],
        help="Compare mode",
    )
    parser.add_argument(
        "--reuse-existing-correct",
        action="store_true",
        help="Alias for --compare-mode reuse",
    )
    parser.add_argument("--dss-desc", type=str, default="correctDss", help="Compare DSS branch desc token")
    parser.add_argument("--ica-desc", type=str, default="correctIca", help="Compare ICA branch desc token")
    parser.add_argument(
        "--strict-existing",
        action="store_true",
        help="Compare reuse: fail if DSS or ICA artifacts are missing",
    )
    parser.add_argument(
        "--use-provenance-metrics",
        action="store_true",
        help="Compare: use stored provenance metrics where available",
    )
    parser.add_argument(
        "--compare-transient-method",
        type=str,
        default="wiener",
        choices=["wiener", "asr", "dss", "none"],
        help="Compare full mode transient method",
    )

    args = parser.parse_args()

    bids_root = Path(args.bids_root).expanduser()
    preproc_root = bids.get_preproc_root(bids_root)
    reports_root = bids.get_reports_root(bids_root)
    preproc_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)

    log_file = reports_root / "logs" / "run_all_pipeline.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file, "INFO")

    if not bids_root.exists():
        LOGGER.error("BIDS root not found: %s", bids_root)
        sys.exit(1)

    input_files = _discover_input_files(bids_root)
    if not input_files:
        LOGGER.error("No .vhdr EEG files found in BIDS root: %s", bids_root)
        sys.exit(1)

    subjects_found = sorted({bids.parse_subject_id(path) for path in input_files})
    subjects_sorted = _select_subjects(
        subjects_found=subjects_found,
        selected_subjects=args.subjects,
        start_from=args.start_from,
        use_test=args.test,
        use_random_test=args.random,
        use_all=args.all,
    )
    if not subjects_sorted:
        LOGGER.warning("No subjects selected. Use --all, --test, --subjects, or --start-from.")
        parser.print_help()
        sys.exit(0)

    missing_requested = sorted([sid for sid in subjects_sorted if sid not in subjects_found])
    if missing_requested:
        LOGGER.warning(
            "Requested subjects not found in discovered EEG files and will be skipped: %s",
            missing_requested,
        )
        subjects_sorted = [sid for sid in subjects_sorted if sid in subjects_found]
        if not subjects_sorted:
            LOGGER.error("No valid subjects remain after filtering missing requests.")
            sys.exit(1)

    selected_input_files = [path for path in input_files if bids.parse_subject_id(path) in set(subjects_sorted)]

    LOGGER.info(
        "Running full chain for %d subjects: %s",
        len(subjects_sorted),
        subjects_sorted,
    )

    correct_desc = bids.validate_stage_desc(args.correct_desc)
    denoise_desc = bids.validate_stage_desc(args.denoise_desc)
    task_token = args.condition if args.condition else None

    base_config = _build_base_config(
        bids_root=bids_root,
        preproc_root=preproc_root,
        reports_root=reports_root,
        n_jobs=args.n_jobs,
        highpass=args.highpass,
        lowpass=args.lowpass,
        resample=args.resample,
        line_freq=args.line_freq,
        adaptive=args.adaptive,
        segments_file=args.segments_file,
    )
    base_overrides = _load_json(args.base_config_json)
    if base_overrides:
        base_config.update(base_overrides)

    correct_cfg = ArtifactCorrectionConfig(
        eog_method=args.eog_method if args.eog_method != "none" else None,
        ecg_method=args.ecg_method if args.ecg_method != "none" else None,
        emg_method=args.emg_method if args.emg_method != "none" else None,
    )
    correct_overrides = _load_json(args.correct_config_json)
    if correct_overrides:
        correct_cfg = ArtifactCorrectionConfig(**{**correct_cfg.__dict__, **correct_overrides})

    denoise_cfg = ArtifactDenoisingConfig(
        transient_method=args.transient_method if args.transient_method != "none" else None
    )
    denoise_overrides = _load_json(args.denoise_config_json)
    if denoise_overrides:
        denoise_cfg = ArtifactDenoisingConfig(**{**denoise_cfg.__dict__, **denoise_overrides})

    base_success: List[str] = []
    base_failed: List[str] = []
    correct_success: List[str] = []
    correct_failed: List[str] = []
    denoise_success: List[str] = []
    denoise_failed: List[str] = []
    raw_lookup = preproc_qc.load_raw_pre_base_lookup(reports_root)
    base_lookup = preproc_qc.load_stage_run_lookup(
        reports_root,
        preproc_qc.get_preproc_qc_stage_name("base", "base"),
    )
    correct_lookup = preproc_qc.load_stage_run_lookup(
        reports_root,
        preproc_qc.get_preproc_qc_stage_name("correct", correct_desc),
    )
    base_profile = preproc_qc.get_preproc_qc_profile("base")
    correct_profile = preproc_qc.get_preproc_qc_profile("correct")
    denoise_profile = preproc_qc.get_preproc_qc_profile("denoise")
    base_qc_records: List[dict[str, object]] = []
    correct_qc_records: List[dict[str, object]] = []
    denoise_qc_records: List[dict[str, object]] = []
    base_qc_groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    correct_qc_groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    denoise_qc_groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)

    for input_file in selected_input_files:
        input_ids = bids.build_bids_report_ids(input_file)
        input_comps = bids.parse_bids_components(input_file)
        subject_id = bids.parse_subject_id(input_file)
        run_label = str(input_ids["run_prefix"])
        effective_task = task_token or input_comps.get("task")

        LOGGER.info("%s", "=" * 72)
        LOGGER.info("RUN %s", run_label)
        LOGGER.info("%s", "=" * 72)

        base_out = bids.get_stage_output_path(
            subject_id=subject_id,
            preproc_root=preproc_root,
            desc="base",
            session=input_comps.get("session"),
            task=input_comps.get("task"),
            run=input_comps.get("run"),
        )
        corr_out = bids.get_stage_output_path(
            subject_id=subject_id,
            preproc_root=preproc_root,
            desc=correct_desc,
            session=input_comps.get("session"),
            task=effective_task,
            run=input_comps.get("run"),
        )
        den_out = bids.get_stage_output_path(
            subject_id=subject_id,
            preproc_root=preproc_root,
            desc=denoise_desc,
            session=input_comps.get("session"),
            task=effective_task,
            run=input_comps.get("run"),
        )

        stage_base_ok = False
        stage_correct_ok = False
        stage_denoise_ok = False

        if args.skip_existing and base_out.exists():
            LOGGER.info("Skipping base (existing): %s", base_out)
            stage_base_ok = True
            try:
                base_result = {
                    "success": True,
                    "qc_record": preproc_qc.collect_existing_preproc_qc_record(
                        profile=base_profile,
                        reports_root=reports_root,
                        filepath=base_out,
                        output_desc="base",
                        raw_lookup=raw_lookup,
                    ),
                }
            except Exception as exc:
                LOGGER.error("Failed rebuilding base QC for %s: %s", subject_id, exc, exc_info=True)
                stage_base_ok = False
                base_result = {"success": False, "qc_record": None}
        else:
            try:
                base_result = run_base_record(
                    subject_id=subject_id,
                    source_path=input_file,
                    bids_root=bids_root,
                    config=base_config,
                    reports_root=reports_root,
                    raw_lookup=raw_lookup,
                )
                stage_base_ok = bool(base_result.get("success"))
            except Exception as exc:
                LOGGER.error("Base failed for %s: %s", subject_id, exc, exc_info=True)
                stage_base_ok = False
                base_result = {"success": False, "qc_record": None}

        if stage_base_ok:
            base_success.append(run_label)
            if base_result.get("qc_record") is not None:
                record = base_result["qc_record"]
                preproc_qc.update_run_lookup(base_lookup, record)
                base_qc_records.append(record)
                base_qc_groups[record["subject_session_key"]].append(record)
                preproc_qc.write_subject_preproc_qc_report(
                    reports_root,
                    base_qc_groups[record["subject_session_key"]],
                    profile=base_profile,
                    output_desc="base",
                )
        else:
            base_failed.append(run_label)
            correct_failed.append(run_label)
            denoise_failed.append(run_label)
            continue

        if args.skip_existing and corr_out.exists():
            LOGGER.info("Skipping correct (existing): %s", corr_out)
            try:
                correct_result = {
                    "success": True,
                    "qc_record": preproc_qc.collect_existing_preproc_qc_record(
                        profile=correct_profile,
                        reports_root=reports_root,
                        filepath=corr_out,
                        output_desc=correct_desc,
                        previous_output_desc="base",
                        raw_lookup=raw_lookup,
                        previous_lookup=base_lookup,
                    ),
                }
                stage_correct_ok = True
            except Exception as exc:
                LOGGER.error("Failed rebuilding correct QC for %s: %s", subject_id, exc, exc_info=True)
                correct_result = {"success": False, "qc_record": None}
                stage_correct_ok = False
        else:
            correct_result = run_correction_pipeline(
                subject_id=subject_id,
                bids_root=bids_root,
                config=correct_cfg,
                preproc_root=preproc_root,
                reports_root=reports_root,
                input_path=base_out,
                condition_name=effective_task,
                train_condition=args.train_condition,
                output_desc=correct_desc,
                raw_lookup=raw_lookup,
                previous_lookup=base_lookup,
            )
            stage_correct_ok = bool(correct_result.get("success"))

        if stage_correct_ok:
            correct_success.append(run_label)
            if correct_result.get("qc_record") is not None:
                record = correct_result["qc_record"]
                preproc_qc.update_run_lookup(correct_lookup, record)
                correct_qc_records.append(record)
                correct_qc_groups[record["subject_session_key"]].append(record)
                preproc_qc.write_subject_preproc_qc_report(
                    reports_root,
                    correct_qc_groups[record["subject_session_key"]],
                    profile=correct_profile,
                    output_desc=correct_desc,
                )
        else:
            correct_failed.append(run_label)
            denoise_failed.append(run_label)
            continue

        if args.skip_existing and den_out.exists():
            LOGGER.info("Skipping denoise (existing): %s", den_out)
            try:
                denoise_result = {
                    "success": True,
                    "qc_record": preproc_qc.collect_existing_preproc_qc_record(
                        profile=denoise_profile,
                        reports_root=reports_root,
                        filepath=den_out,
                        output_desc=denoise_desc,
                        previous_output_desc=correct_desc,
                        raw_lookup=raw_lookup,
                        previous_lookup=correct_lookup,
                    ),
                }
                stage_denoise_ok = True
            except Exception as exc:
                LOGGER.error("Failed rebuilding denoise QC for %s: %s", subject_id, exc, exc_info=True)
                denoise_result = {"success": False, "qc_record": None}
                stage_denoise_ok = False
        else:
            denoise_result = run_denoising_pipeline(
                subject_id=subject_id,
                bids_root=bids_root,
                config=denoise_cfg,
                preproc_root=preproc_root,
                reports_root=reports_root,
                input_path=corr_out,
                condition_name=effective_task,
                input_desc=correct_desc,
                output_desc=denoise_desc,
                raw_lookup=raw_lookup,
                previous_lookup=correct_lookup,
            )
            stage_denoise_ok = bool(denoise_result.get("success"))

        if stage_denoise_ok:
            denoise_success.append(run_label)
            if denoise_result.get("qc_record") is not None:
                record = denoise_result["qc_record"]
                denoise_qc_records.append(record)
                denoise_qc_groups[record["subject_session_key"]].append(record)
                preproc_qc.write_subject_preproc_qc_report(
                    reports_root,
                    denoise_qc_groups[record["subject_session_key"]],
                    profile=denoise_profile,
                    output_desc=denoise_desc,
                )
        else:
            denoise_failed.append(run_label)

    base_success = sorted(set(base_success))
    base_failed = sorted(set(base_failed))
    correct_success = sorted(set(correct_success))
    correct_failed = sorted(set(correct_failed))
    denoise_success = sorted(set(denoise_success))
    denoise_failed = sorted(set(denoise_failed))

    LOGGER.info("Base   : success=%d failed=%d", len(base_success), len(base_failed))
    LOGGER.info("Correct: success=%d failed=%d", len(correct_success), len(correct_failed))
    LOGGER.info("Denoise: success=%d failed=%d", len(denoise_success), len(denoise_failed))

    preproc_qc.write_preproc_qc_aggregate_reports(reports_root, base_qc_records, profile=base_profile, output_desc="base")
    preproc_qc.write_preproc_qc_aggregate_reports(reports_root, correct_qc_records, profile=correct_profile, output_desc=correct_desc)
    preproc_qc.write_preproc_qc_aggregate_reports(reports_root, denoise_qc_records, profile=denoise_profile, output_desc=denoise_desc)

    if args.run_compare:
        compare_mode = "reuse" if args.reuse_existing_correct else args.compare_mode
        compare_den_cfg: Optional[ArtifactDenoisingConfig] = None
        if compare_mode == "full":
            compare_den_cfg = ArtifactDenoisingConfig(
                transient_method=(
                    args.compare_transient_method
                    if args.compare_transient_method != "none"
                    else None
                )
            )
        try:
            run_comparison(
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
                denoise_config=compare_den_cfg,
            )
        except RuntimeError as exc:
            LOGGER.error("Compare failed: %s", exc)
            sys.exit(1)


if __name__ == "__main__":
    main()
