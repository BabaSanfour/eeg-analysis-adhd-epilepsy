"""Verify completeness and QC status of extracted descriptor shards.

Audits the per-subject descriptor shards produced by ``extract_descriptors``
*before* the merge step, so failed or partial array tasks are caught early.

Three independent checks are performed:

1. **Per-shard completeness** — every ``sub-*/ses-*/eeg/<condition>/`` directory
   must hold its epoch table and ``shard.json``, and the subject must have its QC
   report. ``shard.json`` is written last by the producer, so a directory missing it
   is a crashed/partial run.
2. **QC status rollup** — the QC summary in each ``shard.json`` is read and the
   ``pass`` / ``warn`` / ``fail`` verdicts are tallied.
3. **Coverage** — when ``--metadata`` is supplied, subjects of its rows (all of
   them, or those in ``--rows``) that produced *no* shard are listed. These are
   candidate failures, though some are legitimate skips (no epochs / fewer than
   ``min_obs`` epochs survived MAD rejection for every condition) — cross-check
   the SLURM ``.err`` logs for those.

Run on the machine that holds the descriptor derivatives, e.g.::

    python -m eeg_adhd_epilepsy.analysis.verify_descriptors \\
        --derivative_root $SCRATCH/BIDS/derivatives/signal_features/descriptors \\
        --reports_root $SCRATCH/reports \\
        --metadata participants.tsv

Exits non-zero when any shard is incomplete or any expected subject is missing
(use ``--strict`` to also fail when any shard's QC status is ``fail``).
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from eeg_adhd_epilepsy.analysis.utils.descriptor_shards import (
    SHARD_FILES,
    SHARD_RECORD,
    find_shards,
    read_shard_record,
)
from eeg_adhd_epilepsy.io.bids import (
    DerivativeStage,
    bids_subject,
    get_derivative_root,
    read_participants,
)
from eeg_adhd_epilepsy.io.report_paths import (
    ReportStage,
    default_reports_root,
    descriptor_qc_report_name,
    subject_report_dir,
    summary_report_dir,
)

LOGGER = logging.getLogger(__name__)


def _parse_rows(spec: str) -> list[int]:
    """Parse a row spec like ``"1-1000"`` or ``"1-1000,1100,1200-1241"``."""
    rows: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            low, high = chunk.split("-", 1)
            rows.extend(range(int(low), int(high) + 1))
        else:
            rows.append(int(chunk))
    return rows


def collect_shard_records(
    derivative_root: Path,
    reports_root: Path | None,
) -> pd.DataFrame:
    """Audit every shard directory under *derivative_root*.

    Returns one row per ``sub-*/ses-*/eeg/<condition>/`` directory with its
    completion state, missing files, QC status, and failure count.
    """
    records: list[dict[str, object]] = []
    for shard_dir in find_shards(derivative_root):
        rel = shard_dir.relative_to(derivative_root)
        subject = rel.parts[0].removeprefix("sub-")
        session = rel.parts[1].removeprefix("ses-")
        condition = rel.parts[3]

        has_record = (shard_dir / SHARD_RECORD).exists()
        missing_files = [name for name in SHARD_FILES if not (shard_dir / name).exists()]

        report_missing: bool | None = None
        if reports_root is not None:
            report_path = subject_report_dir(
                reports_root=reports_root,
                subject=subject,
                session=session,
                stage=ReportStage.DESCRIPTOR_QC,
            ) / descriptor_qc_report_name(subject, session)
            report_missing = not report_path.exists()

        qc_status = ""
        n_failures: int | None = None
        if has_record:
            summary = read_shard_record(shard_dir)["qc_summary"]
            qc_status = str(summary["qc_status"])
            n_failures = int(summary["n_failures_total"])

        complete = not missing_files and not bool(report_missing)
        records.append(
            {
                "subject": subject,
                "session": session,
                "condition": condition,
                "complete": complete,
                "has_record": has_record,
                "n_missing_files": len(missing_files),
                "missing_files": ";".join(missing_files),
                "report_missing": bool(report_missing) if report_missing is not None else "",
                "qc_status": qc_status,
                "n_failures_total": n_failures,
                "shard": str(rel),
            }
        )
    columns = [
        "subject",
        "session",
        "condition",
        "complete",
        "has_record",
        "n_missing_files",
        "missing_files",
        "report_missing",
        "qc_status",
        "n_failures_total",
        "shard",
    ]
    return pd.DataFrame(records, columns=columns)


def summarize_reasons(derivative_root: Path, shard_df: pd.DataFrame) -> dict[str, Counter]:
    """Roll up *why* shards fail: QC flag codes and extractor-failure breakdown.

    Reads the QC flags and failures in each shard's ``shard.json`` and accumulates:

    - ``flags``       : ``(level, code)`` -> occurrences across shards
    - ``flag_scope``  : ``(code, scope)`` -> occurrences (scope = family, etc.)
    - ``fail_family`` : extractor-failure ``family`` -> occurrences
    - ``fail_exc``    : extractor-failure ``exception_type`` -> occurrences
    - ``fail_msg``    : extractor-failure ``message`` (truncated) -> occurrences
    """
    flags: Counter = Counter()
    flag_scope: Counter = Counter()
    fail_family: Counter = Counter()
    fail_exc: Counter = Counter()
    fail_msg: Counter = Counter()

    for rel in shard_df["shard"][shard_df["has_record"].astype(bool)]:
        record = read_shard_record(derivative_root / rel)
        for flag in record["qc_flags"]:
            flags[(flag["level"], flag["code"])] += 1
            if flag["scope"]:
                flag_scope[(flag["code"], flag["scope"])] += 1
        for failure in record["failures"]:
            fail_family[str(failure["family"])] += 1
            fail_exc[str(failure["exception_type"])] += 1
            fail_msg[str(failure["message"])[:80]] += 1

    return {
        "flags": flags,
        "flag_scope": flag_scope,
        "fail_family": fail_family,
        "fail_exc": fail_exc,
        "fail_msg": fail_msg,
    }


def _print_counter(title: str, counter: Counter, top_n: int = 15) -> None:
    print(f"{title} (top {top_n}):" if len(counter) > top_n else f"{title}:")
    if not counter:
        print("  (none)")
        return
    for key, count in counter.most_common(top_n):
        label = " / ".join(str(part) for part in key) if isinstance(key, tuple) else str(key)
        print(f"  {count:>9,}  {label}")


def expected_subjects(
    metadata_path: Path, subject_col: str, rows: list[int] | None = None
) -> list[str]:
    """Bare BIDS subject labels for the one-based *rows* of ``participants.tsv`` (all rows
    by default)."""
    meta_df = read_participants(metadata_path)
    if rows is None:
        rows = list(range(1, len(meta_df) + 1))
    labels: list[str] = []
    for row in rows:
        position = row - 1
        if 0 <= position < len(meta_df):
            labels.append(bids_subject(meta_df.iloc[position][subject_col]))
    return sorted(set(labels))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify completeness and QC status of extracted descriptor shards."
    )
    parser.add_argument(
        "--derivative_root",
        default=None,
        help="Descriptor derivative root. Defaults to "
        "<bids_root>/derivatives/signal_features/descriptors.",
    )
    parser.add_argument(
        "--bids_root",
        default=None,
        help="BIDS dataset root (used to derive --derivative_root/--reports_root if unset).",
    )
    parser.add_argument(
        "--reports_root",
        default=None,
        help="Reports root. When set, per-subject QC report HTML presence is also checked.",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="participants.tsv: every subject in it (or in --rows) must have a shard.",
    )
    parser.add_argument("--subject_col", default="subject", help="Subject id column in metadata.")
    parser.add_argument(
        "--rows",
        default=None,
        help="One-based metadata rows to expect, e.g. '1-10' or '1-10,1200-1218' (default: all).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Per-shard CSV destination (default: the descriptor QC summary reports folder).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Also exit non-zero when any shard's QC status is 'fail'.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if args.derivative_root:
        derivative_root = Path(args.derivative_root).expanduser()
    elif args.bids_root:
        derivative_root = get_derivative_root(
            Path(args.bids_root).expanduser(), DerivativeStage.DESCRIPTORS
        )
    else:
        parser.error("Provide --derivative_root or --bids_root.")
    if not derivative_root.exists():
        parser.error(f"Derivative root does not exist: {derivative_root}")

    if args.reports_root:
        reports_root: Path | None = Path(args.reports_root).expanduser()
    elif args.bids_root:
        reports_root = default_reports_root(Path(args.bids_root).expanduser())
    else:
        reports_root = None
        LOGGER.warning("No --reports_root/--bids_root: skipping QC report HTML presence check.")

    config_path = derivative_root / "config_used.yaml"
    if not config_path.exists():
        parser.error(f"config_used.yaml not found under {derivative_root}; extraction never ran?")
    shard_df = collect_shard_records(derivative_root, reports_root)

    # Next to the descriptor QC reports, not inside the derivative it checks.
    default_dir = (
        summary_report_dir(reports_root, ReportStage.DESCRIPTOR_QC, create=True)
        if reports_root is not None
        else derivative_root
    )
    out_path = Path(args.out).expanduser() if args.out else default_dir / "verification_report.csv"
    shard_df.to_csv(out_path, index=False)

    n_shards = len(shard_df)
    n_complete = int(shard_df["complete"].sum()) if n_shards else 0
    incomplete_df = shard_df[~shard_df["complete"]] if n_shards else shard_df
    status_counts = (
        shard_df["qc_status"].replace("", "unknown").value_counts().to_dict() if n_shards else {}
    )
    subjects_with_complete = (
        set(shard_df.loc[shard_df["complete"], "subject"]) if n_shards else set()
    )

    print("=" * 72)
    print(f"Descriptor extraction verification — {derivative_root}")
    print("-" * 72)
    print(f"Shards found        : {n_shards}")
    print(f"Complete            : {n_complete}")
    print(f"Incomplete/partial  : {n_shards - n_complete}")
    print(f"QC status breakdown : {status_counts}")
    failures_series = pd.to_numeric(shard_df["n_failures_total"], errors="coerce").fillna(0)
    total_failures = int(failures_series.sum())
    print(f"Total extractor failures across shards: {total_failures}")

    # Aggregate *why* across all shards — the actionable signal when failures
    # are systematic rather than a handful of bad subjects.
    reasons = summarize_reasons(derivative_root, shard_df)
    print("-" * 72)
    _print_counter("QC flags raised  [level / code]", reasons["flags"])
    if reasons["flag_scope"]:
        print()
        _print_counter("QC flags by  [code / scope]", reasons["flag_scope"])
    print()
    _print_counter("Extractor failures by family", reasons["fail_family"])
    print()
    _print_counter("Extractor failures by exception_type", reasons["fail_exc"])
    print()
    _print_counter("Extractor failures by message", reasons["fail_msg"])

    if not incomplete_df.empty:
        print("-" * 72)
        print("INCOMPLETE shards (missing files or report):")
        for _, row in incomplete_df.iterrows():
            reason = []
            if row["n_missing_files"]:
                reason.append(f"missing {row['n_missing_files']} file(s): {row['missing_files']}")
            if row["report_missing"] is True:
                reason.append("missing QC report")
            print(f"  {row['shard']}: {'; '.join(reason)}")

    failed = shard_df[shard_df["qc_status"] == "fail"]
    if not failed.empty:
        print("-" * 72)
        print(f"Shards with QC status 'fail' ({len(failed)}; full list in CSV):")
        for _, row in failed.head(20).iterrows():
            print(f"  {row['shard']}")
        if len(failed) > 20:
            print(f"  … and {len(failed) - 20} more")

    missing_subjects: list[str] = []
    if args.metadata:
        rows = _parse_rows(args.rows) if args.rows else None
        wanted = expected_subjects(Path(args.metadata).expanduser(), args.subject_col, rows)
        missing_subjects = [s for s in wanted if s not in subjects_with_complete]
        print("-" * 72)
        print(f"Coverage: {len(wanted)} expected subjects from rows {args.rows or 'all'}")
        print(f"  with ≥1 complete shard : {len(wanted) - len(missing_subjects)}")
        print(f"  with NO complete shard : {len(missing_subjects)}")
        if missing_subjects:
            print("  (candidate failures — also check SLURM .err logs; some may be")
            print("   legitimate skips: no epochs / too few survived MAD rejection)")
            print("   " + ", ".join(f"sub-{s}" for s in missing_subjects))
    elif args.rows:
        LOGGER.warning("--rows needs --metadata for the coverage check; skipping.")

    print("-" * 72)
    print(f"Per-shard CSV written to: {out_path}")
    print("=" * 72)

    problems = (n_shards - n_complete) > 0 or bool(missing_subjects)
    if args.strict:
        problems = problems or bool((shard_df["qc_status"] == "fail").any())
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
