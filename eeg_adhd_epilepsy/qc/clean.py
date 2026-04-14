"""Fallback rebuild utility for post-preprocessing QC from existing derivatives."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from eeg_adhd_epilepsy.io import bids
from eeg_adhd_epilepsy.qc import preproc_qc
from eeg_adhd_epilepsy.utils.logs import setup_logging

LOGGER = logging.getLogger("preproc_qc_rebuild")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild stage QC reports from existing derivative FIF files.")
    parser.add_argument("--bids_root", required=True, type=Path, help="Path to the BIDS dataset root.")
    parser.add_argument(
        "--preproc_root",
        type=Path,
        default=None,
        help="Directory containing stage FIF/provenance outputs (default: <bids_root>/derivatives/preproc).",
    )
    parser.add_argument(
        "--reports_root",
        type=Path,
        default=None,
        help="Reports root (default: sibling 'reports' next to BIDS).",
    )
    parser.add_argument("--stage", required=True, choices=["base", "correct", "denoise"], help="Stage to rebuild.")
    parser.add_argument("--output_desc", default=None, help="Stage desc token to rebuild.")
    parser.add_argument("--condition", default=None, help="Optional task/condition token.")
    parser.add_argument("--subjects", nargs="+", help="Optional subject filter.")
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bids_root = Path(args.bids_root).expanduser()
    preproc_root = bids.get_preproc_root(
        bids_root=bids_root,
        preproc_root=Path(args.preproc_root).expanduser() if args.preproc_root else None,
    )
    reports_root = bids.get_reports_root(
        bids_root=bids_root,
        reports_root=Path(args.reports_root).expanduser() if args.reports_root else None,
    )
    reports_root.mkdir(parents=True, exist_ok=True)
    setup_logging(reports_root / "logs" / "preproc_qc_rebuild.log", args.log_level)

    profile = preproc_qc.get_preproc_qc_profile(args.stage)
    output_desc = bids.validate_stage_desc(args.output_desc or profile.default_output_desc)
    task_token = args.condition if args.condition else None

    pattern = f"*_desc-{output_desc}_eeg.fif" if not task_token else f"*_task-{task_token}_desc-{output_desc}_eeg.fif"
    files = sorted(preproc_root.rglob(pattern))
    if not files:
        LOGGER.error("No stage files found in %s for pattern %s", preproc_root, pattern)
        sys.exit(1)

    subjects_filter = {bids.normalize_subject_id(subject) for subject in args.subjects} if args.subjects else None
    raw_lookup = preproc_qc.load_raw_pre_base_lookup(reports_root)
    qc_run_records: list[dict[str, object]] = []

    for filepath in files:
        subject_id = bids.parse_subject_id(filepath)
        if subjects_filter and subject_id not in subjects_filter:
            continue
        try:
            record = preproc_qc.collect_existing_preproc_qc_record(
                profile=profile,
                bids_root=bids_root,
                preproc_root=preproc_root,
                reports_root=reports_root,
                filepath=filepath,
                output_desc=output_desc,
                raw_lookup=raw_lookup,
            )
            qc_run_records.append(record)
            preproc_qc.write_subject_preproc_qc_report(
                reports_root,
                [record],
                profile=profile,
                output_desc=output_desc,
            )
        except Exception as exc:
            LOGGER.error("Failed rebuilding QC for %s: %s", filepath, exc, exc_info=True)

    summary_dir = preproc_qc.write_preproc_qc_aggregate_reports(
        reports_root,
        qc_run_records,
        profile=profile,
        output_desc=output_desc,
    )
    if summary_dir is None:
        LOGGER.error("No QC records were rebuilt.")
        sys.exit(1)
    LOGGER.info("Rebuilt %d records. Summary written to %s", len(qc_run_records), summary_dir)


if __name__ == "__main__":
    main()
