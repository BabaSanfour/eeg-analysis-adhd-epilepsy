# Gap Closure Plan: neurodags pipeline vs old pipeline

**Created:** 2026-06-04  
**Status tracking:** each gap has a `[ ]` / `[x]` checkbox. Update when done.

Each gap is self-contained — a new Claude Code session can pick up any single item by reading:
1. This file (the gap description + approach)
2. The files listed under "Key files" for that gap
3. `COMPARISON.md` §referenced for background

---

## P1 — ICA artifact correction method `[x]`

**Severity:** significant  
**Ref:** `COMPARISON.md §2.11`, gap AD in §4 summary

### Problem

Old pipeline (`correct.py → run_source_correction`):
- EOG/ECG: DSS (Denoising Source Separation), profile-based, auto-tunes aggressiveness, falls back to blind-DSS / quasiperiodic if DSS skipped
- EMG: MWF (Multi-channel Wiener Filter)

New pipeline (`nodes_ica.py → ica_artifact_correction`):
- EOG/ECG: basic `find_bads_eog` / `find_bads_ecg`
- EMG: **nothing**

### Approach

Same wrapping pattern as `nodes_denoise.py` → `run_residual_denoising`:
wrap the existing `run_source_correction` from `eeg_adhd_epilepsy/preproc/correct.py`.

**Step 1 — new node in `nodes_ica.py`:**

```python
@register_node
def source_correction(
    mne_object,
    eog_method: str = "dss",
    ecg_method: str = "dss",
    emg_method: str = "mwf",
    ica_n_components: int = 20,
    random_state: int = 42,
) -> NodeResult:
    from eeg_adhd_epilepsy.preproc.correct import ArtifactCorrectionConfig, run_source_correction
    # extract subject_id + output_dir from reference_base path
    # build ArtifactCorrectionConfig from args
    # call run_source_correction(raw, config, output_dir, subject_id, ...)
    # return NodeResult with .fif + figure artifacts
```

`output_dir` and `subject_id` must be extracted from the neurodags `reference_base` path
(same pattern used in `nodes_qc.py` — look at how `build_base_qc_record` derives paths).

**Step 2 — update `step-0_pipeline@preprocessing.yml`:**

```yaml
CorrectRaw:
  nodes:
    - id: 0
      derivative: CleanedPrepRaw.fif
    - id: 1
      node: source_correction       # was: ica_artifact_correction
      args:
        mne_object: id.0
        eog_method: dss
        ecg_method: dss
        emg_method: mwf
        ica_n_components: 20
        random_state: 42
```

**Step 3** — delete existing `CleanedPrepRaw`-derived `CorrectRaw` artifacts for all subjects
(they were computed with basic ICA and need recomputation). Then rerun step-0.

### Key files

- `neurodags_pipelines/nodes_ica.py` — replace `ica_artifact_correction` or add `source_correction`
- `eeg_adhd_epilepsy/preproc/correct.py` — `run_source_correction`, `ArtifactCorrectionConfig`
- `neurodags_pipelines/nodes_denoise.py` — reference pattern for wrapping old code
- `neurodags_pipelines/nodes_qc.py` — reference pattern for extracting paths from `reference_base`
- `neurodags_pipelines/step-0_pipeline@preprocessing.yml` — swap node name in `CorrectRaw`

### Definition of done

- `source_correction` node calls `run_source_correction` with DSS+MWF defaults
- YAML updated, old `CorrectRaw` artifacts deleted, pipeline reruns cleanly
- QC reports show new correct/denoise stages without errors
- `COMPARISON.md` AD gap updated to `DONE`

---

## P2 — Dataset-level preprocessing QC report `[x]`

**Severity:** significant  
**Ref:** `ECOSYSTEM_REPORT.md §2.4`, `COMPARISON.md §2.3`

### Problem

Old pipeline generates a dataset-level HTML summary across all subjects (aggregates per-subject QC records into one report). Not implemented in neurodags.

`run_preproc_dataset_qc` already exists in `eeg_adhd_epilepsy/qc/preproc_qc.py` but is not wired.

### Approach

Aggregator node (same pattern as `DescriptorQCRecord`):
- Receives one `*@CleanedPrepRaw_denoise_qc.json` as trigger
- Globs all `*@CleanedPrepRaw_denoise_qc.json` + `_base_qc.json` + `_correct_qc.json` across all subjects under the preprocessing derivatives root
- Calls `run_preproc_dataset_qc` from the package
- Writes dataset HTML report

**Step 1 — new node in `nodes_qc.py`:**

```python
@register_node
def generate_preproc_dataset_qc_report(qc_record_file, ...) -> NodeResult:
    # locate derivatives/preprocessing root from qc_record_file path
    # glob all *_denoise_qc.json (one per subject/run)
    # load each, call run_preproc_dataset_qc(records, reports_root, ...)
    # return NodeResult with HTML artifact
```

**Step 2 — add derivative to `step-0_pipeline@preprocessing.yml`:**

```yaml
PreprocDatasetQCReport:
  overwrite: False
  for_dataframe: False
  nodes:
    - id: 0
      derivative: DenoiseQCRecord._denoise_qc.json
    - id: 1
      node: generate_preproc_dataset_qc_report
      args:
        qc_record_file: id.0
```

Note: aggregator node limitation — only ONE subject's file triggers it; it scans all siblings.
This means it will re-run once per subject but produce the same output (overwrite: False protects).
Better: add a `skip_if_not_last: true` mechanism or accept the redundancy.

### Key files

- `neurodags_pipelines/nodes_qc.py` — add new node
- `eeg_adhd_epilepsy/qc/preproc_qc.py` — `run_preproc_dataset_qc`
- `neurodags_pipelines/nodes_descriptor_qc.py` — reference aggregator pattern
- `neurodags_pipelines/step-0_pipeline@preprocessing.yml` — add derivative

### Definition of done

- `generate_preproc_dataset_qc_report` node produces dataset HTML summary
- Report covers all subjects × runs × stages (base/correct/denoise)
- `ECOSYSTEM_REPORT.md §2.4` dataset summary row updated
- `COMPARISON.md §2.3` updated

---

## P3a — Dataset-level descriptor QC report (as node) `[x]`

**Severity:** significant  
**Ref:** `COMPARISON.md §2.3`, `ECOSYSTEM_REPORT.md §2.4`

### Problem

`merge_descriptors.py` already calls `run_descriptor_dataset_qc` but runs as a standalone
script outside neurodags. The aggregated report (across all subjects × conditions) is not
wired into the pipeline.

### Approach (two options — pick one)

**Option A (recommended now): document `merge_descriptors.py` as the explicit post-pipeline step.**

Add to `MIGRATION_GUIDE.md §2` run sequence:

```bash
# 3. Merge descriptor shards + generate dataset QC report
python -m eeg_adhd_epilepsy.analysis.merge_descriptors \
    --bids_root /home/yorguin/datasets/eeg-adhd-epilepsy
```

Zero new code. Works today.

**Option B (later): wire as aggregator node in `step-1_pipeline@extraction.yml`.**

Same pattern as `PreprocDatasetQCReport` above — receives one `DescriptorQCRecord` artifact,
globs siblings, calls `run_descriptor_dataset_qc`.

### Key files

- `eeg_adhd_epilepsy/analysis/merge_descriptors.py` — already functional
- `neurodags_pipelines/MIGRATION_GUIDE.md` — add step to §2 run sequence

### Definition of done (Option A)

- `MIGRATION_GUIDE.md` updated with `merge_descriptors` step
- `COMPARISON.md §2.3` updated to reflect partial coverage
- `ECOSYSTEM_REPORT.md §2.4` updated

---

## P4 — Structured failure rows / intra-file NaN `[x]`

**Severity:** significant  
**Ref:** `COMPARISON.md §2.3`, gap Z in §4 summary

### Problem

Old pipeline writes `failures.csv` per condition with columns:
`condition, subject, obs_id, obs_index, channel_index, channel_name, family, exception_type, message`

New pipeline has:
- Per-condition NaN rate CSV (`sub-{sub}_ses-{ses}_{cond}_nan_rates.csv`) — added 2026-06-04
- No failure log for channels that are all-NaN per family
- Complete-file failures covered by `neurodags status --list-errors` / `.error` markers

### Approach

Extend `generate_descriptor_qc_record` in `nodes_descriptor_qc.py` to also write a
`failures.csv` derived from the NaN data:

```python
# After building sensor_epoch_df and pooled_epoch_df:
# For each family, find columns where all values are NaN
# → one row per all-NaN column: {condition, subject, channel_name, family, reason="all_nan"}
# Also: for each NC file that was expected but missing (not in nc_dir)
# → one row per missing file: {condition, subject, family, reason="missing_nc"}
# Write to qc/sub-{sub}_ses-{ses}_{cond}_failures.csv
```

This covers:
- All-NaN features (FOOOF fit failures, entropy compute failures)
- Missing NC files (complete derivative failures that didn't write .error)

Does NOT cover: per-epoch partial failures within a feature that has some valid values.
That would require changes to extraction nodes themselves — out of scope for now.

### Key files

- `neurodags_pipelines/nodes_descriptor_qc.py` — extend `generate_descriptor_qc_record`
  (around line 381 where NaN CSV is written — add failures CSV alongside it)
- `eeg_adhd_epilepsy/qc/descriptor_qc.py` — `summarize_failures` already handles this format

### Definition of done

- `qc/sub-{sub}_ses-{ses}_{cond}_failures.csv` written per subject/condition
- Columns match old `failures.csv` schema (at minimum: condition, subject, channel_name, family, reason)
- `COMPARISON.md §2.3` gap Z updated to reflect improved coverage

---

## P5 — ZapLine `n_removed` in provenance `[ ]`

**Severity:** minor  
**Ref:** `COMPARISON.md §2.6`, gap Y in §4 summary

### Problem

Old pipeline stores `n_removed` (number of DSS components removed by ZapLine) in `prov.json`.
New `zapline_denoise` node does not track this.

### Approach

Check ZapLine API for component count exposure. If accessible:

```python
# nodes_preprocessing.py → zapline_denoise
# After fit_transform, extract n_removed from ZapLine object
# Add to NodeResult provenance: {"zapline_n_removed": n_removed}
```

If ZapLine doesn't expose this directly (likely — it's a DSS-based method), compute as:
`n_removed = n_components_removed` from the internal DSS object, or skip and mark as
"not exposed by ZapLine API".

### Key files

- `neurodags_pipelines/nodes_preprocessing.py` — `zapline_denoise` node
- Check `meegkit` / `zapline-python` ZapLine class API

### Definition of done

- `@CleanedPrepRaw_prov.json` includes `zapline_n_removed: N` or explicit `zapline_n_removed: null` with comment
- `COMPARISON.md §2.6` gap Y updated

---

## P6 — Run-aware aggregation utility `[ ]`

**Severity:** minor  
**Ref:** `COMPARISON.md §2.2`, gap V in §4 summary

### Problem

Old pipeline groups by `recording_id = sub_ses_run` so multi-run subjects get one row per run
in the final feature CSV. `neurodags dataframe` already produces one row per source file
(= one row per run), so the data is correct — but there is no utility to further aggregate
runs into one row per subject when desired.

### Approach

Small post-pipeline script:

```python
# eeg_adhd_epilepsy/analysis/aggregate_runs.py
# --input features_all_conditions.csv
# --output features_subject_level.csv
# --agg mean  (or median)
# groups by [subject, session, condition], averages numeric columns across runs
```

Alternatively: just document the pandas one-liner in `MIGRATION_GUIDE.md`:

```python
df.groupby(["subject", "session", "condition"]).mean(numeric_only=True).reset_index()
```

### Key files

- `neurodags_pipelines/MIGRATION_GUIDE.md` — document groupby pattern in §4
- (optional) `eeg_adhd_epilepsy/analysis/aggregate_runs.py` — new utility

### Definition of done

- `MIGRATION_GUIDE.md §4` documents run aggregation pattern
- `COMPARISON.md §2.2` gap V updated

---

## Not planned

| Gap | Reason |
|-----|--------|
| AR plot per-chunk (T) | Combined plot sufficient; effort > value |
| Float32 round-trip (P) | Below EEG noise floor; MNE behavior |

---

## Implementation order

```
P1  ICA method          ← data quality, most impactful
P2  Preproc dataset QC  ← reporting
P3a Descriptor QC doc   ← zero code, just docs (do alongside P2)
P4  Structured failures ← extends recent NaN CSV work
P5  ZapLine n_removed   ← small, do when touching nodes_preprocessing.py
P6  Run aggregation doc ← one-liner in MIGRATION_GUIDE.md
```

After P1 completes: re-run `compare_qc_reports.py` to verify new correct/denoise outputs
and update `QC_PIPELINE_COMPARISON_REPORT.md §3` with updated numbers.
