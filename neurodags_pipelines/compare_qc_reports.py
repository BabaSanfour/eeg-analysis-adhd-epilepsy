#!/usr/bin/env python3
"""Compare old (eeg_adhd_epilepsy/scripts) vs new (neurodags_pipelines) QC HTML reports.

Old:  /home/yorguin/datasets/eeg-adhd-epilepsy/reports/sub-{ID}/ses-01/{stage}_qc/sub-{ID}_ses-01_{stage}_qc_report.html
New:  /home/yorguin/datasets/eeg-adhd-epilepsy/derivatives/preprocessing/sub-{ID}/ses-01/eeg/
        sub-{ID}_ses-01_task-clinical_run-{RUN}_eeg.vhdr@{Stage}QCReport._{stage}_qc_report.html

Usage:
    python compare_qc_reports.py                     # all subjects, all stages
    python compare_qc_reports.py --sub 0001          # single subject
    python compare_qc_reports.py --stage base        # single stage (base/correct/denoise)
    python compare_qc_reports.py --sub 0001 --stage base --verbose
"""

import argparse
import glob
import os
import re
from html.parser import HTMLParser
from pathlib import Path

OLD_ROOT = Path("/home/yorguin/datasets/eeg-adhd-epilepsy/reports")
NEW_ROOT = Path("/home/yorguin/datasets/eeg-adhd-epilepsy/derivatives/preprocessing")
STAGES = ["base", "correct", "denoise"]
STAGE_CLASS = {"base": "BaseQCReport", "correct": "CorrectQCReport", "denoise": "DenoiseQCReport"}


# ── HTML parser ────────────────────────────────────────────────────────────────

class TableExtractor(HTMLParser):
    """Extract all tables from HTML, each tagged with the preceding heading."""

    def __init__(self):
        super().__init__()
        self.tables: list[dict] = []
        self._table = None
        self._row = None
        self._cell = None
        self._in_h = False
        self._h_buf = ""
        self._last_h = ""

    def handle_starttag(self, tag, attrs):
        if tag in ("h2", "h3", "h4", "h5"):
            self._in_h, self._h_buf = True, ""
        elif tag == "table":
            self._table = {"heading": self._last_h, "rows": []}
        elif tag == "tr" and self._table is not None:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = ""

    def handle_endtag(self, tag):
        if tag in ("h2", "h3", "h4", "h5"):
            self._in_h = False
            self._last_h = self._h_buf.strip()
        elif tag == "table" and self._table is not None:
            self.tables.append(self._table)
            self._table = None
        elif tag == "tr" and self._row is not None:
            if self._table is not None:
                self._table["rows"].append(self._row)
            self._row = None
        elif tag in ("td", "th") and self._cell is not None:
            if self._row is not None:
                self._row.append(self._cell.strip())
            self._cell = None

    def handle_data(self, data):
        if self._in_h:
            self._h_buf += data
        elif self._cell is not None:
            self._cell += data


def parse_html(path: Path) -> list[dict]:
    p = TableExtractor()
    p.feed(path.read_text(errors="replace"))
    return p.tables


def tables_by_heading(tables: list[dict]) -> dict[str, list]:
    """Return {heading: rows_list} — last table wins on duplicate headings."""
    out = {}
    for t in tables:
        out[t["heading"]] = t["rows"]
    return out


# ── Path helpers ───────────────────────────────────────────────────────────────

def old_report_path(sub: str, stage: str) -> Path | None:
    p = OLD_ROOT / f"sub-{sub}" / "ses-01" / f"{stage}_qc" / f"sub-{sub}_ses-01_{stage}_qc_report.html"
    return p if p.exists() else None


def new_report_paths(sub: str, stage: str) -> list[Path]:
    """Return all runs for this sub/stage (sorted)."""
    cls = STAGE_CLASS[stage]
    pattern = str(NEW_ROOT / f"sub-{sub}" / "ses-01" / "eeg" / f"sub-{sub}_ses-01_task-clinical_run-*_eeg.vhdr@{cls}._{stage}_qc_report.html")
    return sorted(Path(p) for p in glob.glob(pattern))


def discover_subjects() -> list[str]:
    subs = set()
    for p in OLD_ROOT.glob("sub-*/ses-01"):
        m = re.search(r"sub-(\d+)", str(p))
        if m:
            subs.add(m.group(1))
    for p in NEW_ROOT.glob("sub-*/ses-01"):
        m = re.search(r"sub-(\d+)", str(p))
        if m:
            subs.add(m.group(1))
    return sorted(subs)


# ── Metric extraction ──────────────────────────────────────────────────────────

OVERVIEW_FIELDS = ["Raw Duration", "Retained Duration", "QC Status"]
RETENTION_FIELDS = ["Condition segment retention"]
RESIDUAL_FIELDS = ["Mean amplitude", "Max amplitude", "Flat channels"]
GLOBAL_FIELDS = ["Line noise ratio", "Alpha peak frequency", "Aperiodic slope"]


def extract_kv_table(rows: list[list[str]]) -> dict[str, str]:
    """Parse a 2-column key/value table."""
    out = {}
    for row in rows[1:]:  # skip header
        if len(row) >= 2:
            out[row[0]] = row[1]
    return out


def extract_overview_row(rows: list[list[str]]) -> dict[str, str]:
    if len(rows) < 2:
        return {}
    headers = rows[0]
    values = rows[1]
    return {h: v for h, v in zip(headers, values)}


def extract_metrics(path: Path) -> dict:
    tables = parse_html(path)
    by_h = tables_by_heading(tables)

    metrics = {}

    # Overview row
    if "Overview" in by_h:
        ov = extract_overview_row(by_h["Overview"])
        for f in OVERVIEW_FIELDS:
            metrics[f] = ov.get(f, "—")

    # Retention table (key/value)
    if "Retention" in by_h:
        kv = extract_kv_table(by_h["Retention"])
        for f in RETENTION_FIELDS:
            metrics[f] = kv.get(f, "—")

    # Residual Metrics (key/value)
    if "Residual Metrics" in by_h:
        kv = extract_kv_table(by_h["Residual Metrics"])
        for f in RESIDUAL_FIELDS:
            metrics[f] = kv.get(f, "—")

    # Global Metrics (key/value) — present in some stages
    for heading in ("Global Metrics", "Global Signal Quality"):
        if heading in by_h:
            kv = extract_kv_table(by_h[heading])
            for f in GLOBAL_FIELDS:
                if f in kv:
                    metrics[f] = kv[f]

    return metrics


# ── Display ────────────────────────────────────────────────────────────────────

def fmt_val(v: str) -> str:
    return v if v else "—"


def diff_marker(old: str, new: str) -> str:
    if old == new:
        return "="
    if old == "—" or old == "":
        return "+"   # new has it, old doesn't
    if new == "—" or new == "":
        return "-"   # old had it, new doesn't
    return "~"       # both present but differ


def compare_subject_stage(sub: str, stage: str, verbose: bool = False) -> list[str]:
    lines = []
    old_path = old_report_path(sub, stage)
    new_paths = new_report_paths(sub, stage)

    label = f"sub-{sub} | {stage}"

    if not old_path and not new_paths:
        lines.append(f"  {label}: BOTH MISSING")
        return lines

    if not old_path:
        lines.append(f"  {label}: OLD MISSING  (new: {len(new_paths)} run(s))")
        return lines

    if not new_paths:
        lines.append(f"  {label}: NEW MISSING  (old exists)")
        return lines

    old_m = extract_metrics(old_path)

    for new_path in new_paths:
        run_m = re.search(r"run-(\d+)", new_path.name)
        run = run_m.group(1) if run_m else "??"
        new_m = extract_metrics(new_path)

        all_keys = list(dict.fromkeys(list(old_m.keys()) + list(new_m.keys())))

        diffs = [(k, old_m.get(k, "—"), new_m.get(k, "—")) for k in all_keys]
        changed = [d for d in diffs if d[1] != d[2]]

        header = f"  sub-{sub} | {stage} | run-{run}"
        if not changed:
            lines.append(f"{header}: ALL METRICS MATCH")
        else:
            lines.append(f"{header}: {len(changed)} metric(s) differ")
            col = max(len(k) for k, _, _ in diffs) + 2
            for k, o, n in diffs:
                mark = diff_marker(o, n)
                if verbose or mark != "=":
                    lines.append(f"    [{mark}] {k:<{col}} OLD={fmt_val(o)!r:30s}  NEW={fmt_val(n)!r}")

    return lines


def run_comparison(subjects: list[str], stages: list[str], verbose: bool = False):
    print(f"\n{'='*70}")
    print("  QC Report Comparison: OLD vs NEW")
    print(f"  OLD: {OLD_ROOT}")
    print(f"  NEW: {NEW_ROOT}")
    print(f"{'='*70}\n")

    for sub in subjects:
        print(f"Subject: sub-{sub}")
        for stage in stages:
            for line in compare_subject_stage(sub, stage, verbose=verbose):
                print(line)
        print()


def main():
    parser = argparse.ArgumentParser(description="Compare old vs new QC HTML reports")
    parser.add_argument("--sub", help="Subject ID (e.g. 0001). Default: all")
    parser.add_argument("--stage", choices=STAGES, help="Stage. Default: all")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show matching metrics too")
    args = parser.parse_args()

    subjects = [args.sub] if args.sub else discover_subjects()
    stages = [args.stage] if args.stage else STAGES

    run_comparison(subjects, stages, verbose=args.verbose)


if __name__ == "__main__":
    main()
