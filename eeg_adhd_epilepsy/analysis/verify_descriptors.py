"""Verify completeness and QC status of extracted descriptor shards.

Audits the per-subject descriptor shards produced by ``extract_descriptors``
*before* the merge step, so failed or partial array tasks are caught early.

Three checks are performed:

1. **Per-shard completeness** — every ``sub-*/ses-*/eeg/<condition>/`` directory
   must hold its epoch table and ``shard.json``, and the subject must have its QC
   report. ``shard.json`` is written last by the producer, so a directory missing it
   is a crashed/partial run.
2. **QC status rollup** — the QC summary, flags and failures in each ``shard.json``
   are tallied; epochs dropped as outliers are counted apart from extraction failures.
3. **Coverage** — every subject of ``participants.tsv`` (or of its ``--rows``) without
   a shard is listed, unless none of its saved recordings has ``min_obs`` epochs of
   the extracted conditions (the extraction skips those).

Exits non-zero when any shard is incomplete or any expected subject is missing
(use ``--strict`` to also fail when any shard's QC status is ``fail``).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import mne

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
    descriptor_qc_report_name,
    subject_report_dir,
)
from eeg_adhd_epilepsy.utils.yaml import load_yaml_config


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


def _print_counter(title: str, counter: Counter, top_n: int = 15) -> None:
    print(f"{title} (top {top_n}):" if len(counter) > top_n else f"{title}:")
    if not counter:
        print("  (none)")
        return
    for key, count in counter.most_common(top_n):
        label = " / ".join(str(part) for part in key) if isinstance(key, tuple) else str(key)
        print(f"  {count:>9,}  {label}")


def has_extractable_epochs(
    bids_root: Path, subject: str, desc: str, conditions: list[str], min_obs: int
) -> bool:
    """Whether a saved recording of *subject* has *min_obs* epochs of one of *conditions*.

    Reads the epoch files' headers only. A subject without such a recording has no
    shard by design (the extraction skips it), so it is not a failure.
    """
    preproc_root = get_derivative_root(bids_root, DerivativeStage.PREPROC)
    for path in sorted(preproc_root.glob(f"sub-{subject}/*/eeg/*_desc-{desc}_epo.fif")):
        epochs = mne.read_epochs(path, preload=False, verbose="ERROR")
        if any(len(epochs[c]) >= min_obs for c in conditions if c in epochs.event_id):
            return True
    return False


def expected_subjects(metadata_path: Path, rows: list[int] | None = None) -> list[str]:
    """Bare BIDS subject labels for the one-based *rows* of ``participants.tsv`` (all rows
    by default)."""
    meta_df = read_participants(metadata_path)
    if rows is None:
        rows = list(range(1, len(meta_df) + 1))
    return sorted(
        {bids_subject(meta_df.iloc[row - 1]["subject"]) for row in rows if 0 < row <= len(meta_df)}
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify completeness and QC status of extracted descriptor shards."
    )
    parser.add_argument("--derivative_root", required=True, help="Root of the descriptor shards.")
    parser.add_argument("--reports_root", required=True, help="Root of the subject QC reports.")
    parser.add_argument(
        "--metadata",
        required=True,
        help="participants.tsv: every subject in it (or in --rows) must have a shard.",
    )
    parser.add_argument("--bids_root", required=True, help="BIDS root with the saved epochs.")
    parser.add_argument(
        "--conditions",
        nargs="+",
        required=True,
        help=(
            "Conditions the extraction ran on. A subject without a shard is a failure only if "
            "a saved recording has min_obs epochs of one of them."
        ),
    )
    parser.add_argument(
        "--rows",
        default=None,
        help="One-based metadata rows to expect, e.g. '1-10' or '1-10,1200-1218' (default: all).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Also exit non-zero when any shard's QC status is 'fail'.",
    )
    args = parser.parse_args()

    derivative_root = Path(args.derivative_root).expanduser()
    reports_root = Path(args.reports_root).expanduser()
    config_path = derivative_root / "config_used.yaml"
    if not config_path.exists():
        parser.error(f"config_used.yaml not found under {derivative_root}; extraction never ran?")
    config_used = load_yaml_config(config_path)

    status_counts: Counter = Counter()
    flags: Counter = Counter()
    fail_family: Counter = Counter()
    fail_exc: Counter = Counter()
    fail_msg: Counter = Counter()
    rejected_family: Counter = Counter()
    n_rejected = 0
    incomplete: list[str] = []
    failed_shards: list[str] = []
    subjects_with_complete: set[str] = set()
    shards = find_shards(derivative_root)
    for shard_dir in shards:
        rel = shard_dir.relative_to(derivative_root)
        subject = rel.parts[0].removeprefix("sub-")
        session = rel.parts[1].removeprefix("ses-")
        missing = [name for name in SHARD_FILES if not (shard_dir / name).exists()]
        report = subject_report_dir(
            reports_root=reports_root,
            subject=subject,
            session=session,
            stage=ReportStage.DESCRIPTOR_QC,
        ) / descriptor_qc_report_name(subject, session)
        if not report.exists():
            missing.append("QC report")
        if missing:
            incomplete.append(f"{rel}: missing {', '.join(missing)}")
        else:
            subjects_with_complete.add(subject)
        if not (shard_dir / SHARD_RECORD).exists():
            continue
        record = read_shard_record(shard_dir)
        status = str(record["qc_summary"]["qc_status"])
        status_counts[status] += 1
        if status == "fail":
            failed_shards.append(str(rel))
        for flag in record["qc_flags"]:
            flags[(flag["level"], flag["code"], flag["scope"] or "")] += 1
        # An epoch dropped by two families has a record for each.
        n_rejected += len(
            {
                failure["obs_id"]
                for failure in record["failures"]
                if str(failure["family"]).startswith("MAD_Rejection")
            }
        )
        for failure in record["failures"]:
            family = str(failure["family"])
            if family.startswith("MAD_Rejection"):
                rejected_family[family.removeprefix("MAD_Rejection:")] += 1
                continue
            fail_family[family] += 1
            fail_exc[str(failure["exception_type"])] += 1
            fail_msg[str(failure["message"])[:80]] += 1

    print("=" * 72)
    print(f"Descriptor extraction verification — {derivative_root}")
    print("-" * 72)
    print(f"Shards found        : {len(shards)}")
    print(f"Complete            : {len(shards) - len(incomplete)}")
    print(f"Incomplete/partial  : {len(incomplete)}")
    print(f"QC status breakdown : {dict(status_counts)}")
    print(f"Extraction failures : {sum(fail_family.values())}")
    print(f"Rejected epochs     : {n_rejected}")
    print("-" * 72)
    _print_counter("QC flags  [level / code / scope]", flags)
    print()
    _print_counter("Rejected epochs by the family that dropped them", rejected_family)
    print()
    _print_counter("Extraction failures by family", fail_family)
    if fail_family:
        print()
        _print_counter("Extraction failures by exception_type", fail_exc)
        print()
        _print_counter("Extraction failures by message", fail_msg)

    if incomplete:
        print("-" * 72)
        print("INCOMPLETE shards:")
        for line in incomplete:
            print(f"  {line}")

    if failed_shards:
        print("-" * 72)
        print(f"Shards with QC status 'fail' ({len(failed_shards)}):")
        for shard in failed_shards:
            print(f"  {shard}")

    rows = _parse_rows(args.rows) if args.rows else None
    wanted = expected_subjects(Path(args.metadata).expanduser(), rows)
    missing_subjects = [s for s in wanted if s not in subjects_with_complete]
    min_obs = int(config_used["qc"].get("min_obs", 3))
    without_epochs = [
        s
        for s in missing_subjects
        if not has_extractable_epochs(
            Path(args.bids_root).expanduser(), s, config_used["desc"], args.conditions, min_obs
        )
    ]
    missing_subjects = [s for s in missing_subjects if s not in without_epochs]
    print("-" * 72)
    print(f"Coverage: {len(wanted)} expected subjects from rows {args.rows or 'all'}")
    print(f"  with ≥1 complete shard : {len(wanted) - len(missing_subjects) - len(without_epochs)}")
    if without_epochs:
        print(
            f"  no recording with {', '.join(args.conditions)} epochs (skipped by the "
            f"extraction): {len(without_epochs)}"
        )
    print(f"  with NO complete shard : {len(missing_subjects)}")
    if missing_subjects:
        print("  (candidate failures — check the SLURM logs; a subject is also skipped")
        print("   when outlier rejection leaves no recording with min_obs epochs)")
        print("   " + ", ".join(f"sub-{s}" for s in missing_subjects))
    print("=" * 72)

    problems = bool(incomplete) or bool(missing_subjects)
    if args.strict:
        problems = problems or bool(failed_shards)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
