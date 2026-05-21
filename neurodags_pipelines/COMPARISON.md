# Original vs neurodags pipeline — detailed comparison

Covers all major design decisions, algorithmic equivalence, and known gaps.  
Files referenced:

| Original | neurodags |
|---|---|
| `eeg_adhd_epilepsy/preproc/base.py` | `neurodags_pipelines/step-0b_preproc_cleaned.yml` + `custom_nodes.py` |
| `eeg_adhd_epilepsy/analysis/extract_descriptors.py` | `neurodags_pipelines/step-1_features.yml` + `custom_nodes.py` |
| `eeg_adhd_epilepsy/io/bids.py` (block windowing) | `neurodags_pipelines/step-0c_conditions.yml` + `custom_nodes.py` |
| `configs/descriptors.yaml` | YAML anchors `&BANDS`, `&CHANNEL_GROUPS` in `step-1_features.yml` |

---

## 1. Overall architecture

| Dimension | Original | neurodags |
|---|---|---|
| Paradigm | Imperative Python scripts, BIDS-aware | Declarative YAML DAGs, file-pattern-aware |
| Invocation | `python base.py --bids_root ... --subjects ...` | `neurodags run step-0b_preproc_cleaned.yml` |
| Subject selection | CLI `--subjects`, BIDS metadata CSV | Dataset YAML file glob (`datasets_raw.yml`) |
| Granularity | Subject × session × run × condition loops in Python | One YAML derivative per logical output; per-file caching in `.nc` / `.fif` |
| Intermediate storage | `.fif` saved per BIDS entity; derivative named with `_desc-base_epo.fif` | `.nc` (NetCDF4) per derivative per source file; `.fif` for epoch derivatives |
| Resume / idempotency | `_SUCCESS` marker per shard; `--overwrite` flag | `overwrite: False` per derivative; re-runs skip existing files |
| Parallelism | `joblib.Parallel` for short files; sequential for long files | Not built in at YAML level; neurodags `run` is sequential per dataset by default |
| Provenance | JSON sidecar `_prov.json` per run | None — derivative filenames + NetCDF4 metadata serve as implicit provenance |
| BIDS compliance | Full BIDS path conventions; `dataset_description.json` | No BIDS awareness; flat `derivatives/preprocessing/` tree |

---

## 2. Preprocessing chain

### 2.1 Step ordering

| Step | Original (`base.py`) | neurodags step-0b | neurodags step-0c |
|---|---|---|---|
| Annotation inflation | `inflate_bad_annotations` | — | — |
| Resample | `raw.resample(target_sfreq)` | `preprocess_raw(resample=256)` ✓ | `preprocess_raw(resample=256)` ✓ |
| Bandpass filter | 0.1–100 Hz | 0.1–100 Hz ✓ | **1–45 Hz** ✗ |
| ZapLine | `ZapLine.fit_transform(raw)` | `zapline_denoise` ✓ | **missing** ✗ |
| RANSAC bad channels | rest-block subset | full recording ✗ | **missing** ✗ |
| CAR | `set_eeg_reference("average")` | `apply_car` ✓ | **missing** ✗ |
| AR — scope | per-condition (all EO blocks merged, all EC blocks merged) | whole-recording ✗ | per-condition ✓ |
| AR — input quality | on ZapLine+RANSAC+CAR cleaned raw | on cleaned raw ✓ | **on raw-only (no ZapLine/RANSAC/CAR)** ✗ |
| AR — chunking | `_iter_autoreject_chunks` (30 min default) | — ✗ | — ✗ |
| AR — per-channel spans | `BAD_{cond}` with `ch_names` tuples | — ✗ | — ✗ |
| AR — epoch label | `BAD_epoch_{condition_name}` | `BAD_epoch` ✗ (minor) | `BAD_epoch` ✗ (minor) |
| AR — CV | `min(10, n_epochs_chunk)` | `min(10, max(2, min_epochs))` = 5 fixed ✗ | `min(10, max(2, n_epochs))` ✓ |
| AR plots | per-chunk PNG (orientation=horizontal, 16×10 in, 150 dpi) | combined per-condition PNG (same orientation/size/dpi) ± | — |
| Output format | annotated Raw + provenance JSON | Epochs (CleanedPrep.fif) | Epochs (ConditionEO/EC.fif) |

### 2.2 Most critical architectural gap: step-0c starts from unclean raw

`base.py` outputs an **annotated Raw** after the full ZapLine→RANSAC→CAR→AR chain.  
Condition epochs are then extracted **from that cleaned annotated Raw** — they benefit from all cleaning steps.

`step-0c` starts fresh from raw VHDR and applies only `preprocess_raw(1–45 Hz)` before AR and condition extraction.  ZapLine, RANSAC, and CAR are absent.  This means condition epochs from step-0c contain line noise, potentially bad channels unreferenced, and no common average reference.

**For exact correspondence**, step-0c should read the intermediate cleaned Raw produced by step-0b (after ZapLine/RANSAC/CAR but before epoching), then do condition-specific AR and extraction from that.  This requires splitting step-0b into two derivatives:
- `CleanedPrepRaw` — annotated Raw (ZapLine→RANSAC→CAR→`autoreject_annotate_raw` whole-recording)
- `CleanedPrep` — Epochs from `CleanedPrepRaw` via `extract_fixed_length_epochs`

And step-0c reads from `datasets_cleaned_raw.yml` pointing to `*@CleanedPrepRaw.fif`.

### 2.3 Filter range mismatch in step-0c

`base.py` filters 0.1–100 Hz (broadband).  Condition epochs are extracted from this broadband signal.  
`step-0c` filters 1–45 Hz before extraction.

For **spectral features** (bandpower, FOOOF) this makes no difference — the Welch PSD is computed in 1–45 Hz range regardless.  
For **complexity features** (entropy, fractal dimension, Kurtosis, RMS) computed on the raw time-domain signal, the filter affects values: a 0.1–100 Hz signal has different temporal structure than a 1–45 Hz signal.  High-frequency content (45–100 Hz) contributes to sample entropy, Higuchi FD, zero-crossings, etc.

Fix: change step-0c `preprocess_raw` to 0.1–100 Hz (match base.py), and ensure downstream feature YAML still limits Welch PSD to 1–45 Hz (already done via `fmin=1.0, fmax=45.0` in `mne_spectrum_array`).

### 2.4 ZapLine

Both use `mne_denoise.zapline.ZapLine`.  Parameters identical (60 Hz, non-adaptive).  No algorithmic difference.

### 2.5 RANSAC

Both call `NoisyChannels(raw, random_state=42).find_bad_by_ransac()`.

`base.py` crops to `bids.collect_baseline_windows(raw)` — baseline/rest block windows — before fitting RANSAC.  This focuses bad-channel detection on intrinsic channel quality rather than task-related amplitude bursts.  `ransac_bad_channels` now supports `block_label` param (e.g., `block_label: EC`), but step-0b does not currently pass it.  Fix: pass `block_label` to the step-0b RANSAC node for the rest condition.

### 2.6 AutoReject details

**`annotate_artifacts_blockwise` (base.py)** vs **`autoreject_annotate` / `autoreject_annotate_raw`** (neurodags):

| Detail | base.py | neurodags |
|---|---|---|
| Event building | `build_block_events_by_condition`: 1 s non-overlapping events per block window | Same logic in `autoreject_annotate_raw` ✓ |
| Condition merging | All EO blocks merged into one epoch set; one AR instance | Same ✓ |
| Chunking | `_iter_autoreject_chunks`: splits if > 30 min per condition | Not implemented ✗ |
| `n_interpolate` | `[0]` — no interpolation | `[0]` ✓ |
| CV | `min(10, n_epochs_chunk)` — per chunk | `autoreject_annotate`: fixed 5; `autoreject_annotate_raw`: adaptive `min(10, n)` ✓ |
| Epoch annotations | `BAD_epoch_{condition_name}` | `BAD_epoch` (no condition label) |
| Channel span annotations | `BAD_{condition_name}` with `ch_names` per consecutive bad-channel run | Not implemented ✗ |
| AR plots | saved PNG per condition per chunk | Not implemented ✗ |

**Per-channel span annotations**: base.py `_reject_log_to_annotations` groups consecutive epochs where the same channel is marked bad (`labels[:, ch_idx] != 0`) into a single `BAD_{condition}` annotation with `ch_names=(ch_name,)`.  This allows downstream analysis to identify which channels were systematically bad during which condition.  Not needed for `reject_by_annotation="omit"` behavior but is richer provenance.

---

## 3. Condition-level epoch extraction

### 3.1 Original flow

`base.py` outputs annotated Raw (`_desc-base_raw.fif`).  `extract_descriptors.py` loads this via `load_eeg_data(..., condition=condition)` which reads BIDS event structure and extracts condition-labeled epochs on the fly.  All conditions come from the **same cleaned Raw**: they share the same ZapLine/RANSAC/CAR/AR preprocessing.

### 3.2 neurodags flow (`step-0c_conditions.yml`)

Separate per-condition `.fif` files: `*@ConditionEO.fif`, `*@ConditionEC.fif`.

Chain: `preprocess_raw(1–45 Hz)` → `autoreject_annotate_raw(condition_name)` → `extract_condition_epochs(reject_by_annotation="omit")`.

The event-building logic in `autoreject_annotate_raw` matches `build_block_events_by_condition`: collects all `BLOCK_{condition}` windows, creates 1 s events within each, merges across windows, runs one AR instance.  Epoch extraction with `reject_by_annotation="omit"` then drops annotated bad segments.

**Key gap**: step-0c starts from unclean raw (see §2.2).  Steps ZapLine / RANSAC / CAR are absent.

---

## 4. Feature extraction

### 4.1 Entry points

| | Original | neurodags |
|---|---|---|
| Entry | `python extract_descriptors.py --bids_root ...` | `neurodags run step-1_features.yml` (compute), then `neurodags dataframe step-1_features.yml` (export) |
| Feature framework | `coco_pipe.descriptors.DescriptorPipeline` | neurodags built-in nodes + custom_nodes.py |
| Config | `configs/descriptors.yaml` (DescriptorConfig pydantic model) | `step-1_features.yml` YAML anchors `&BANDS`, `&CHANNEL_GROUPS` |
| Output format | `.parquet` + `.csv` per subject/session/condition shard | Single flat CSV per dataset (from `neurodags dataframe`) |

### 4.2 Band power variants

Both produce 6 variants: absolute, log-absolute, relative, corrected-absolute, corrected-log-absolute, corrected-relative.  Bands: delta [1–4], theta [4–8], alpha [8–13], beta [13–30], gamma [30–45].

Spectral estimation: `mne_spectrum_array` (Welch, fmin=1, fmax=45, n_fft=512, n_overlap=256) — matches the Welch parameters in the coco-pipe `band_abs` descriptor family.

FOOOF fit: `fooof` node on the Welch spectrum.  `aperiodic_mode=fixed`, `max_n_peaks=6`, `peak_width_limits=[1.0, 12.0]`, `freq_range=[1,45]`.  This is the same configuration as `configs/descriptors.yaml`.

### 4.3 Aggregation statistics

| Family | Original stats | neurodags |
|---|---|---|
| Band power (abs, corrected) | mean only | Mean (saved), IQR derivative excluded from band_summaries |
| Band power (log, rel, corr-log, corr-rel) | mean + median + IQR | `*Mean`, `*Median`, `*IQR` derivatives |
| FOOOF scalars (exponent, offset, R²) | mean + median + IQR | `*Mean`, `*Median`, `*IQR` derivatives |
| FOOOF peaks | mean + median + IQR | `*Mean`, `*Median`, `*IQR` per variable |
| Complexity (antropy, neurokit2, stats) | mean + median + MAD | `*Mean`, `*Median`, `*MAD` derivatives |

Aggregate functions:
- `aggregate_across_dimension(operation: mean/median)` — uses xarray `.mean()` / `.median()`.
- `iqr_across_dimension` — `scipy.stats.iqr` via `xr.apply_ufunc`.
- `mad_across_dimension` — pure numpy median absolute deviation via `xr.apply_ufunc`.

Original: `coco_pipe.io.DataContainer.aggregate_groups(by="recording_id", groups=aggregation_descriptors)` applies multiple stats in one call from config.

### 4.4 FOOOF peak features (point 2)

**Original** (`configs/descriptors.yaml`, `param_summaries`): FOOOF peak features are extracted inside the coco-pipe `fooof` descriptor family.  Variables: dominant peak CF/PW/BW, alpha peak CF/PW/BW, n_peaks.

**neurodags**: `FooofPeaksDs` → `fooof_peaks(fooof_like, alpha_band=[8.0,13.0])` → returns Dataset with 7 vars: `n_peaks`, `dominant_peak_cf/pw/bw`, `alpha_peak_cf/pw/bw`.  Then each var gets Mean / Median / IQR derivatives (21 total consuming `FooofPeaksDs`).

Algorithmic equivalence: the same peak-extraction logic applied per-epoch per-channel.

### 4.5 Subject-level band ratios (point 4)

**Original** (`_build_feature_outputs`, lines 186–202):
```python
for num, den in aggregated_ratio_pairs:
    for p in ["band_abs_", "band_corr_abs_"]:
        for col in base_agg_features.columns:
            ...
            agg_ratio_columns[f"agg_band_ratio_{num}_{den}_{suffix}"] = np.divide(
                n_vals, d_vals, ..., where=d_vals > aggregated_ratio_floor
            )
```
Runs after `container.aggregate(by="recording_id", stats="mean")` — ratio of means, not mean of ratios.  Applied to both `band_abs_*` and `band_corr_abs_*` columns.

**neurodags** (`BandRatiosOnMeans`):
```yaml
BandRatiosOnMeans:
  nodes:
    - id: 0
      derivative: AbsBandPowerAgg.nc   # saved mean across epochs
    - id: 1
      node: band_ratios
      args:
        bandpower_like: id.0
        combinations: [...]            # same 10 pairs
```
`AbsBandPowerAgg` is the per-recording mean absolute band power (dims: `spaces × freqbands`).  `band_ratios` on this produces ratios of means — semantically identical to the original.

**Gap**: original also computes `agg_band_corr_ratio_*` (ratios on corrected absolute band power means).  The neurodags `BandRatiosOnMeans` uses only uncorrected absolute power.  To complete parity, a `CorrectedBandPowerAgg` + `BandRatiosOnCorrectedMeans` pair would be needed.

### 4.6 Complexity features

| Source | Original (coco-pipe) | neurodags |
|---|---|---|
| antropy | sample, approx, perm, SVD, spectral entropy; Hjorth mobility/complexity; LZiv; zero crossings; Higuchi FD; Katz FD; Petrosian FD; DFA | same 14 measures |
| neurokit2 | multiscale entropy, Shannon entropy, fuzzy entropy, dispersion entropy, Hurst exponent | same 5 measures |
| stats | kurtosis, RMS | same 2 |

All 19 measures × 3 stats (mean/median/MAD) = 57 complexity derivatives in both.

### 4.7 Epoch-level vs subject-level output

**Original**: produces two DataFrames per shard:
- `sensor_epoch_features.csv` — one row per epoch, all feature columns
- `sensor_subject_features.csv` — one row per recording (mean across epochs + grouped stats + ratio-of-means columns)

**neurodags**: `neurodags dataframe` produces one flat CSV where each row is a file and columns are all `for_dataframe: True` derivatives collapsed.  There is no epoch-level output from `neurodags dataframe`.  Epoch-level data is accessible via the `.nc` derivative files.

### 4.8 Feature column naming

Original column convention: `band_abs_delta_Fz`, `complexity_sample_entropy_Fz`.  
neurodags: derivative name becomes column prefix, e.g. `AbsBandPowerMean_spaces_delta_Fz`.  Column names are not identical — mapping required for cross-system comparison.

---

## 5. Spatial pooling (point 3)

### Original

`pipeline.pool_channels(sensor_result, channel_groups)` inside `_build_feature_outputs`.  Produces a second `pooled_result` dict.  Pooled epoch and subject DataFrames are saved separately (`pooled_epoch_features.csv`, `pooled_subject_features.csv`).

Channel groups (9 regions, 10-20 system):
```
front_left: [F7, F3], front_midline: [Fz], front_right: [F8, F4]
central_left: [T3, C3], central_midline: [Cz], central_right: [T4, C4]
posterior_left: [T5, P3, O1], posterior_midline: [Pz], posterior_right: [T6, P4, O2]
```

### neurodags

`pool_channels` node on each band power variant, FOOOF scalar, and band ratio derivative:
- `*Pooled` derivative: per-epoch pooled (saved, dims `epochs × regions × freqbands`)
- `*PooledMean` derivative: mean over epochs (unsaved, `for_dataframe: True`, dims `regions × freqbands`)

Same 9 regions defined in `&CHANNEL_GROUPS` anchor (identical to orignal config).  
Absent channels silently skipped; groups with no present channels dropped.

**Behavior on synthetic data**: synthetic EEG uses channels `EEG000–EEG007` — none match 10-20 labels, so all groups are dropped.  `pool_channels` raises `ValueError("None of the channel_groups channels were found...")`.  The neurodags pipeline continues; pooled derivatives are absent from the dataframe for synthetic data.

---

## 6. QC layer

**Original**: full QC pipeline via `eeg_adhd_epilepsy.qc`:
- `run_descriptor_subject_qc` — computes missingness, flag rates, family summary per subject/session/condition
- `write_subject_preproc_qc_report` — HTML report per subject
- QC outputs: `qc/summary_row.csv`, `qc/flags.csv`, `qc/feature_missingness.csv`, `qc/family_summary.csv`, HTML report

**neurodags**: no QC layer.  Failed derivatives (e.g. FOOOF convergence failure) produce `NaN` values in the dataframe.  No structured failure tracking, no per-subject QC report.

This is the largest functional gap.  For production use, a post-processing step would need to compute missingness and flag suspicious subjects.

---

## 7. Dataset / BIDS integration

**Original**: fully BIDS-aware.
- Subject IDs normalized to `sub-XXXX` format.
- Reads `BIDS/derivatives/preproc/sub-*/ses-*/eeg/*_desc-base_epo.fif`.
- Validates coverage via `validate_bids_coverage`.
- Supports `--metadata_row` for SLURM array jobs (row → subject ID mapping).
- Writes `dataset_description.json` and `config_used.yaml` to derivative root.
- Supports run-aware aggregation: `recording_id = subject_ses-{session}_run-{run}`.

**neurodags**: dataset defined by glob pattern in `datasets_*.yml`.  File-level, not BIDS-aware.  Subject/session/run information is implicit in filenames but not structured.  No metadata CSV integration.  No config versioning or compatibility check.

---

## 8. Failure handling

**Original**: per-descriptor try/except inside coco-pipe `DescriptorPipeline.extract`.  Failures logged to `failures.csv` (subject, obs_id, channel, family, exception type, message).  Processing continues for other descriptors/epochs.

**neurodags**: if a node raises, the derivative computation fails for that file.  The dataframe for that file will have `NaN` for all columns coming from the failed derivative chain.  No structured failure log.

---

## 9. Summary table of algorithmic equivalence

| Feature | Equivalent? | Notes |
|---|---|---|
| Bandpass filter | Yes | Same parameters |
| ZapLine | Yes | Same parameters |
| RANSAC bad channels | Partial | neurodags uses full recording; original uses rest-block subset |
| CAR | Yes | Identical call |
| AutoReject | Partial | Original: condition-aware; neurodags: whole-recording |
| Annotation inflation | No | Not implemented in neurodags |
| Condition epoch extraction | Yes | Same logic, different output format |
| Welch PSD | Yes | Same parameters |
| FOOOF fit | Yes | Same config |
| FOOOF peak features | Yes | Same 7 variables |
| Absolute band power | Yes | — |
| Log/rel/corr band power | Yes | — |
| Band ratios (per epoch) | Yes | Same 10 pairs |
| Band ratios on means (abs) | Yes | Semantically identical |
| Band ratios on means (corr) | **No** | Not implemented in neurodags |
| Multi-stat aggregation (IQR/MAD) | Yes | Via custom nodes |
| Spatial pooling | Yes (real data) | Silently absent on synthetic/non-10-20 data |
| antropy complexity (14) | Yes | — |
| neurokit2 complexity (5) | Yes | — |
| Kurtosis + RMS | Yes | — |
| Epoch-level output | Partial | neurodags exposes `.nc` files; no epoch CSV from `dataframe` command |
| Subject-level output | Yes | Both produce one aggregated row per recording |
| QC reports | **No** | Not implemented in neurodags |
| BIDS integration | **No** | neurodags is file-glob based |
| SLURM array support | **No** | Not built in (could wrap `neurodags run` in an array script) |
| Provenance JSON | **No** | Not implemented in neurodags |
| Config versioning | **No** | Not implemented in neurodags |

---

## 9. Summary equivalence table (updated)

| Feature | Equivalent? | Notes |
|---|---|---|
| Bandpass filter (step-0b) | Yes | 0.1–100 Hz ✓ |
| Bandpass filter (step-0c) | **No** | 1–45 Hz vs 0.1–100 Hz — affects complexity features |
| ZapLine (step-0b) | Yes | — |
| ZapLine (step-0c) | **No** | Missing — step-0c starts from unclean raw |
| RANSAC bad channels (step-0b) | Partial | Full recording vs rest-block subset |
| RANSAC bad channels (step-0c) | **No** | Missing |
| CAR (step-0b) | Yes | — |
| CAR (step-0c) | **No** | Missing |
| AR scope (step-0b) | **No** | Whole-recording vs condition-grouped |
| AR input quality (step-0c) | **No** | step-0c AR on un-cleaned raw; base.py AR on ZapLine+RANSAC+CAR cleaned raw |
| AR event building (step-0c) | Yes | Same block-window logic ✓ |
| AR condition merging (step-0c) | Yes | All blocks of same condition merged ✓ |
| AR chunking | **No** | Not implemented in any neurodags AR node |
| AR per-channel span annotations | **No** | `BAD_{cond}` with ch_names not implemented |
| AR epoch label | Partial | `BAD_epoch` vs `BAD_epoch_{cond}` — functionally same for omit |
| AR CV | Partial | `autoreject_annotate`: fixed 5; `autoreject_annotate_raw`: adaptive ✓ |
| Annotation inflation | No | Only matters for recordings with manual BAD_ marks |
| Condition epoch extraction logic | Yes | Same fixed-length tiling within block windows ✓ |
| reject_by_annotation omit | Yes | step-0c uses it ✓ |
| All 6 band power variants | Yes | — |
| FOOOF fit + scalars | Yes | — |
| FOOOF peak features (7 vars) | Yes | — |
| Band ratios per epoch | Yes | 10 pairs ✓ |
| Band ratios on abs means | Yes | `BandRatiosOnMeans` ✓ |
| Band ratios on corr abs means | Yes | `BandRatiosOnCorrectedMeans` ✓ |
| Multi-stat aggregation (IQR/MAD) | Yes | — |
| Spatial pooling | Yes (real data) | Absent on non-10-20 channel names |
| antropy complexity (14) | Yes | — |
| neurokit2 complexity (5) | Yes | — |
| Kurtosis + RMS | Yes | — |
| Epoch-level CSV | Partial | `.nc` files accessible; no epoch CSV from `dataframe` |
| Subject-level CSV | Yes | Both produce one aggregated row per file |
| QC / failure tracking | Partial | `neurodags status` covers done/missing/errored per derivative |
| BIDS output structure | No | Flat derivative tree, no BIDS conventions |
| Provenance JSON | Partial | `_prov.json` covers bad channels + AR stats; no ICA or filter params |
| Config versioning | No | Not implemented |
| AR rejection plots | Partial | Combined per-condition PNG (`@CleanedPrepRaw_ar_plot_{cond}.png`); original saves one per chunk |

---

## 10. Gaps ranked by impact on numerical output

### Tier 1 — affect feature values

**A. step-0c starts from unclean raw — DONE**  
Fixed: `CleanedPrepRaw` derivative added to step-0b (inflate → preprocess → zapline → ransac → car → `autoreject_annotate_blockwise` → annotated Raw).  step-0c reads `datasets_cleaned_raw.yml` and is a pure extraction pipeline — condition epochs now inherit full cleaning.

**B. Filter range in step-0c — DONE**  
Fixed: step-0c no longer filters; reads from `CleanedPrepRaw` which is already 0.1–100 Hz.

**C. AR scope in step-0b — DONE**  
Fixed: `autoreject_annotate_blockwise` replaces `autoreject_annotate`; discovers all BLOCK_* conditions, one AR instance per condition group, adaptive CV per chunk.

**D. RANSAC rest-block subset — DONE**  
Fixed: `block_label: EC` now set in step-0b RANSAC node; `ransac_bad_channels` crops to EC windows before fitting.

### Tier 2 — affect annotation richness, not epoch rejection

**E. Per-channel span annotations — DONE**  
Fixed: `autoreject_annotate_blockwise` adds `BAD_{cond}` annotations with `ch_names` tuples for consecutive per-channel bad spans, mirroring `_reject_log_to_annotations` in base.py.

**F. AR chunking — DONE**  
Fixed: `autoreject_annotate_blockwise` has `ar_max_chunk_minutes` param; chunks long conditions and merges annotations across chunks.

### Tier 3 — cosmetic / infrastructure

**G. Annotation epoch labels — DONE**  
Fixed: `autoreject_annotate_blockwise` uses `BAD_epoch_{cond_name}`, not bare `BAD_epoch`.

**H. AR CV — DONE**  
Fixed: `autoreject_annotate_blockwise` uses `cv=min(10, len(epochs_chunk))` per chunk, identical to base.py.

**I. Annotation inflation — DONE**  
Fixed: `inflate_bad_annotations` node is first step in `CleanedPrepRaw` chain.

**J. AR rejection plots — PARTIAL**  
`autoreject_annotate_blockwise` saves `@CleanedPrepRaw_ar_plot_{cond}.png` per condition alongside the `.fif` — same params as original (orientation=horizontal, 16×10 in, 150 dpi).  
Difference: neurodags combines labels across chunks into **one plot per condition**; base.py saves **one plot per chunk** (e.g. `_autoreject_eo_chunk1.png`).  Combined gives better overview; per-chunk is finer-grained for long recordings.

**K. Provenance JSON — DONE**  
`autoreject_annotate_blockwise` saves `@CleanedPrepRaw_prov.json` alongside the `.fif`: bad channels (from `raw.info["bads"]` at AR input, i.e. after RANSAC), per-condition epoch counts + bad counts + clean fraction, overall clean fraction.

**L. Config provenance — not implemented**  
base.py copies `config_used.yaml` to derivatives and guards against re-running with a different config.  neurodags guard is `overwrite: False` only.

---

## 11. Implementation status

| Gap | Status | Notes |
|-----|--------|-------|
| A. step-0c unclean raw | **DONE** | `CleanedPrepRaw` chain; step-0c reads `datasets_cleaned_raw.yml` |
| B. Filter range step-0c | **DONE** | step-0c inherits 0.1–100 Hz from `CleanedPrepRaw` |
| C. AR scope | **DONE** | `autoreject_annotate_blockwise`: per-condition, adaptive CV |
| D. RANSAC rest-subset | **DONE** | `block_label: EC` set in step-0b |
| E. Per-channel span annotations | **DONE** | `BAD_{cond}` + ch_names in `autoreject_annotate_blockwise` |
| F. AR chunking | **DONE** | `ar_max_chunk_minutes` param in `autoreject_annotate_blockwise` |
| G. Epoch annotation labels | **DONE** | `BAD_epoch_{cond_name}` in `autoreject_annotate_blockwise` |
| H. AR CV | **DONE** | `min(10, len(epochs_chunk))` per chunk |
| I. Annotation inflation | **DONE** | `inflate_bad_annotations` first in `CleanedPrepRaw` chain |
| J. AR rejection plots | **PARTIAL** | Combined per-condition PNG; original is per-chunk (see §10.J) |
| K. Provenance JSON | **DONE** | `@CleanedPrepRaw_prov.json`: bad channels, per-condition AR stats, clean fractions |
| L. Config versioning | open | Not implemented |
