# QC Pipeline Comparison Report
# Old (`eeg_adhd_epilepsy/preproc/`) vs New (`neurodags_pipelines/`)

**Date:** 2026-05-28  
**Author:** yorguin  
**Subjects verified:** sub-0001, sub-0002, sub-0100, sub-0337  
**Comparison tool:** `neurodags_pipelines/compare_qc_reports.py`

---

## 1. Architecture Overview

### Old pipeline

Monolithic Python scripts orchestrated by a single CLI entry point.

```
eeg_adhd_epilepsy/preproc/
  run_all.py     (604 lines)   ← entry point: Base → Correct → Denoise
  base.py       (1034 lines)   ← cleaning: filter, ZapLine, RANSAC, CAR, AR
  correct.py     (900 lines)   ← ICA artifact correction
  denoise.py     (747 lines)   ← residual Wiener denoise

eeg_adhd_epilepsy/qc/
  preproc_qc.py  (792 lines)   ← metrics computation + multi-run aggregation

eeg_adhd_epilepsy/reports/
  preproc_qc.py  (493 lines)   ← HTML report rendering
```

**Total preprocessing + QC code:** ~4 570 lines Python.

Intermediate results are held in memory or written to disk with ad-hoc path construction. Re-running requires deleting outputs or explicit skip-if-exists guards scattered throughout each script. Subject parallelism uses `joblib.Parallel` embedded in `run_all.py`.

### New pipeline (neurodags)

DAG-driven: YAML declares what to compute; node functions declare how.

```
neurodags_pipelines/
  step-0_pipeline@preprocessing.yml   ← 10 derivatives, caching automatic
  step-0_dataset.yml                  ← subject/session list (16 lines)

  nodes_annotations.py    (144 lines)
  nodes_preprocessing.py  (287 lines)
  nodes_autoreject.py     (551 lines)
  nodes_ica.py            ( 78 lines)
  nodes_bad_channels.py   + nodes_denoise.py
  nodes_qc.py             (623 lines)
```

**Total preprocessing + QC code:** ~1 700 lines Python + YAML.

Each node is a pure function. The framework handles caching, dependency resolution, and incremental recomputation automatically. Rerunning only recomputes stale or missing derivatives.

---

## 2. Bugs Found and Fixed (Old → New)

### 2.1 Raw Duration always `'0s'`

**Severity:** display/trust — misleads QC reviewers.

**Root cause:** `raw_duration_sec` was looked up from a CSV that was either missing or written with a mismatched key. The Overview table always showed `Raw Duration = 0s`.

**Fix:** `compute_raw_qc_metrics` writes duration to `_raw_qc.json`; `build_base_qc_record` loads it directly by path. All subjects now show correct values.

| Subject | Old Raw Duration | New Raw Duration |
|---------|-----------------|-----------------|
| sub-0001 | `0s` | `25m 51.99s` |
| sub-0002 run-01 | `0s` | `19m 25.99s` |
| sub-0002 run-02 | `0s` | `44m 47.99s` |
| sub-0100 | `0s` | `20m 53.99s` |
| sub-0337 | `0s` | `17m 29.99s` |

---

### 2.2 Channel positions were all NaN (critical signal bug)

**Severity:** critical — silently broke RANSAC, AutoReject, and Wiener denoise.

**Root cause:** `inject_block_annotations` (and the old `base.py` equivalent) never loaded the BIDS `*_electrodes.tsv` sidecar. All 19 EEG channels had `NaN` montage positions throughout preprocessing.

**Downstream failures:**
- RANSAC inside `NoisyChannels` raised `ValueError` or `OSError` — caught by a bare `except (ValueError, OSError): pass` — silently yielding zero bad channels for all subjects.
- AutoReject requires valid channel positions for spatial interpolation; it crashed silently.
- Residual Wiener denoise crashed because it performs spatial filtering that requires a montage.

**Fix:** `inject_block_annotations` now loads `*_electrodes.tsv`, constructs a `DigMontage` via `make_dig_montage`, and applies it with `set_montage(on_missing="ignore")`. This must be the first step in the pipeline.

**Visible effect:** sub-0001 denoise stage: flat channel changed from `Pz` (old) to `Cz` (new). AR aggressiveness differences across all subjects also reflect AutoReject now operating on valid spatial data.

---

### 2.3 `_annotation_intervals` counted per-channel BAD_ marks as global bad (retention collapse)

**Severity:** critical — retained_duration collapsed to near-zero for some subjects.

**Root cause:** `_annotation_intervals` iterated all `BAD_*` annotations without checking `ch_names`. AutoReject writes per-channel span annotations (`BAD_{cond}` with a non-empty `ch_names` list). These mark bad time for a single channel, not for the full recording. Summing them as global bad time inflated the total bad duration — measured: **18,693 s of "bad" on a 1,166 s recording** — collapsing `retained_duration_sec` to ~3 s (0.28%).

**Fix (commit 889f77c, 2026-05-25):**
```python
# _annotation_intervals in eeg_adhd_epilepsy/qc/preproc_qc.py
for annot in raw.annotations:
    if str(annot["description"]).startswith("BAD_"):
        if annot.get("ch_names"):  # channel-specific — not globally bad
            continue
```

This is also the root cause of the colleague's `Retained Duration = 2m 50.44s` on sub-0002 (see §4).

---

### 2.4 ICA `n_components` crash

**Severity:** pipeline crash for datasets with fewer than 20 channels.

**Root cause:** `ica_artifact_correction` called `mne.preprocessing.ICA(n_components=20, ...)` unconditionally. When `len(picks) < 20` (this dataset has 19 EEG channels), MNE raises:
```
ValueError: ica.n_components (20) cannot be greater than len(picks) (19)
```

**Fix:**
```python
n_components = min(n_components, len(picks))
```

---

### 2.5 Annotations setter crash (`_strip_bad_annotations`)

**Severity:** pipeline crash on BrainVision-sourced data.

**Root cause:** `raw_clean.annotations = new_annotations` raises `AttributeError` on BrainVision-loaded Raw objects — `annotations` is a read-only property.

**Fix:** replaced with `raw_clean.set_annotations(new_annotations)`.

---

### 2.6 "Mean Dur (s)" column mislabeled

**Severity:** display/interpretation — value was total duration, not mean duration per epoch.

**Root cause (commit 81a0659):** `build_condition_comparison_table` wrote `out["Mean Dur (s)"]` but used `total_duration_post_sec` as the value.

**Fix (commit a48bc36):** renamed to `out["Total Dur (s)"]`.

---

### 2.7 Multi-run subjects treated as single recording

**Severity:** significant — metrics for sub-0002 were meaningless in old pipeline.

**Root cause:** sub-0002 has two runs (`run-01`: 19m26s, `run-02`: 44m48s). Old pipeline processed both as one concatenated file, producing a single mixed-run report.

**New behavior:** Each run processed independently by neurodags (one report per run). The `eeg_adhd_epilepsy` QC layer supports proper multi-run subject aggregation via `_aggregate_subject_metrics`, which:
- Sums `retained_duration_sec` and `usable_condition_coverage_sec` across runs.
- Takes duration-weighted averages for amplitude, line noise, etc.
- Takes the MAX across runs for `amplitude_max_uv`, `n_flat_channels`, `n_noisy_channels`.
- Takes `records[0]` channel diagnostics (topomaps are combined via weighted aggregation).

---

## 3. Metric Comparison Across Subjects (Base Stage)

### Universal differences (expected / feature improvements)

| Metric | Old | New | Verdict |
|--------|-----|-----|---------|
| Raw Duration | `'0s'` | Correct | Bug fixed |
| Condition segment retention | `''` (missing) | Real % | New feature |
| Per-condition pre-values | `'—'` | Real values | New feature |
| Effect Vs Raw delta table | Absent | Present | New feature |

### Subject-level residual differences

| Subject | Stage | Retained Δ | Amplitude Δ | Notes |
|---------|-------|-----------|------------|-------|
| sub-0001 | base/correct | +2m05s (new) | <1% | Fewer epochs rejected by AR (now valid positions) |
| sub-0001 | denoise | +24s (new) | +3.6% | Flat ch: old=Pz → new=Cz (positions fix side effect) |
| sub-0002 | all | Not comparable | — | Different run structure (old: merged; new: run-01 + run-02) |
| sub-0100 | base/correct | +2s (new) | <0.3% | Effectively matching |
| sub-0100 | denoise | +10s (new) | <0.3% | Effectively matching |
| sub-0337 | base/correct | +1m12s (new) | <2% | AR aggressiveness, expected |
| sub-0337 | denoise | −22s (new) | <1% | Within expected variance |

Small retained-duration differences (~1–2 min) are expected and represent a signal quality improvement: new pipeline runs AutoReject on spatially-aware data (valid montage), producing better-calibrated rejection thresholds.

---

## 4. Colleague's Pipeline Analysis (sub-0002, base stage)

The colleague (Hamza Abdelhedi, `hamza.abdelhedii@gmail.com`) is running `eeg_adhd_epilepsy/` at approximately commit `a5053a1` — after multi-run aggregation was added (commit `81a0659`) but before two critical fixes.

### Colleague's report: key values

| Metric | Colleague | New run-01 | New run-02 |
|--------|-----------|-----------|-----------|
| Raw Duration | `0s` | `19m 25.99s` | `44m 47.99s` |
| Retained Duration | `2m 50.44s` | `12m 10s` | `37m 41s` |
| Mean amplitude | `2748.28 uV` | `1504.92 uV` | `2723.19 uV` |
| Max amplitude | `7089.30 uV` | `3345.83 uV` | `7147.37 uV` |
| Flat channel | `C3` | `C3` | `C4` |
| Noisy channel | `Fp1` | `Fp1` | `P3` |
| Condition col. header | `Mean Dur (s)` | `Total Dur (s)` | `Total Dur (s)` |

### Diagnosis

| Issue in colleague's report | Root cause | Fixed in commit |
|-----------------------------|-----------|----------------|
| `Retained Duration = 2m 50.44s` | `_annotation_intervals` counts per-channel BAD_ ch_names as global bad → inflated bad total → retention collapse | `889f77c` (2026-05-25) |
| `"Mean Dur (s)"` column label | Mislabeled in original implementation | `a48bc36` |
| `Raw Duration = '0s'` | `raw_qc_pre_base` CSV missing or wrong lookup key | Separate issue |

**Mixed channel/amplitude signals** (flat=C3 matches run-01, amplitude matches run-02) are consistent with `_aggregate_subject_metrics` using `records[0].channel_diagnostics` (from run-01) while amplitude is duration-weighted (dominated by the longer run-02, 44m vs 19m).

**Recommendation for colleague:** pull current branch — commit `889f77c` is the critical fix. The retention collapse makes all QC metrics unreliable.

---

## 5. Feature Gap: Per-Run Summary Table

The colleague's subject-level report includes a **Per-Run Summary** table listing each run's QC status, retention, bad channels, line noise, and HF/LF ratio in one place. This table is absent from the neurodags per-run reports.

**Verdict: not worth adding to neurodags.**

Reasons:

1. **Neurodags already generates one report per run.** sub-0002 has `run-01` and `run-02` reports, each with full detail. The summary table would just duplicate information already visible by opening both reports side by side.

2. **Only 1 of 4 current subjects has multiple runs** (sub-0002). For single-run subjects the table is a 1-row identity — pure noise.

3. **Subject-level aggregation is a cross-run DAG dependency.** Adding it to neurodags requires a derivative that collects all run-level QCRecords for the same subject before computing. The framework handles per-file derivatives cleanly; this would need a "gather all runs for subject X" node, adding non-trivial complexity.

4. **The `eeg_adhd_epilepsy/` QC layer already implements this correctly** via `write_subject_preproc_qc_report` + `_aggregate_subject_metrics`. If subject-level aggregated reports with Per-Run Summary become a real need (e.g., many multi-run subjects), run that layer on top of the neurodags `.fif` outputs using `collect_existing_preproc_qc_record`. Zero new code required.

**If this becomes needed:** implement as a post-neurodags step using the existing `eeg_adhd_epilepsy` QC layer, not as a new neurodags derivative.

---

## 6. Extensibility and Portability

### 6.1 Adding a new preprocessing step

**Old:** Modify `base.py` (1034 lines), thread new function through the call chain, add CLI arguments to `run_all.py`, manually handle output paths.

**New:** Write one node function in an existing `nodes_*.py` file, add one YAML entry:
```yaml
- id: N
  node: my_new_step
  args:
    mne_object: id.N-1
    my_param: value
```
The framework handles caching, skipping, and dependency graph automatically.

### 6.2 Adding a new feature descriptor

**Old:** Edit `descriptor_qc.py`, add columns to the CSV schema, update `reports/preproc_qc.py`, rerun all subjects.

**New:** Add a node to `step-1_pipeline@extraction.yml`, declare it as a new derivative. Only subjects lacking the new derivative are computed; existing results are untouched.

### 6.3 Porting to a different EEG dataset

| Concern | Old | New |
|---------|-----|-----|
| Dataset paths | Hardcoded in `run_all.py` CLI flags and BIDS discovery | `step-0_dataset.yml` — change 16 lines |
| Condition/annotation names | Hardcoded `BLOCK_*` throughout Python | `annotation_prefix` YAML arg |
| Sampling rate / filter params | Constants in `base.py` header | `resample`, `l_freq`, `h_freq` YAML args |
| Line noise frequency | Hardcoded `60.0` Hz | `line_freq` YAML arg |
| Adding a new modality (MEG, iEEG) | Requires refactor of BIDS I/O and epoch logic in Python | Swap `SourceFile` derivative; node functions reusable |
| Running a subject subset | `--subjects` CLI flag with custom Python logic | Edit `step-0_dataset.yml` or add `skip: true` per entry |
| Multi-dataset runs | New `run_all.py` invocation per dataset | New `step-0_dataset.yml` per dataset; same pipeline YAML |
| Cluster scaling | Custom SLURM scripts per stage in `cluster/compute_canada/` | Single `neurodags run` invocation |

**Summary:** Porting to a new dataset in the old pipeline means Python code changes. In the new pipeline it is a YAML edit. This is the principal architectural advantage.

### 6.4 Collaborative development

**Old:** Multiple contributors edit the same monolithic files. Merge conflicts are frequent and changes to shared functions have wide blast radius.

**New:** Nodes are independent functions in focused files. A new node can be added without touching existing ones. YAML is the integration point — changes are additive and clearly scoped.

---

## 7. Summary Table

| Criterion | Old (`eeg_adhd_epilepsy/preproc/`) | New (`neurodags_pipelines/`) |
|-----------|----------------------------------|------------------------------|
| Preprocessing code size | ~4 570 lines | ~1 700 lines + YAML |
| Incremental re-run | Manual skip-if-exists | Automatic (neurodags cache) |
| Channel positions | Not loaded — NaN — RANSAC/AR silently fail | Loaded from `*_electrodes.tsv` |
| Raw Duration display | Always `0s` | Correct |
| Condition segment retention | Not computed | Computed and shown in report |
| Per-condition pre-values | Missing | Present |
| Multi-run subjects | Merged (incorrect) | Per-run, correctly aggregated |
| Retained Duration accuracy | Inflated by channel-specific BAD_ marks | Correct |
| ICA crash on small channel counts | Crashes | Clamped `min(n_components, len(picks))` |
| Annotations setter | `AttributeError` on BrainVision | Uses `set_annotations` |
| Column label "Mean Dur (s)" | Wrong label | Fixed: "Total Dur (s)" |
| Porting to new dataset | Python code changes | YAML edit only |
| Adding a processing step | 1 000+ line script edits | One function + one YAML entry |
| Cluster scaling | Manual SLURM scripts per stage | `neurodags run` |
| Dataset summary reports | Generated | Not yet implemented |
| Descriptor QC all conditions | All 8 conditions | EO_baseline only (others not yet run) |
| Colleague compatibility | Shared `eeg_adhd_epilepsy/` package | Needs commit 889f77c to be correct |
