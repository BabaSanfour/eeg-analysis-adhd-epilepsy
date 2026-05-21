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

| Step | Original (`base.py`) | neurodags (`step-0b_preproc_cleaned.yml`) |
|---|---|---|
| Annotation inflation | `inflate_bad_annotations` (major→5 s, common→3 s) | Not implemented |
| Resample | `raw.resample(target_sfreq)` | `preprocess_raw(resample=256)` |
| Bandpass filter | `raw.filter(hp, lp)` (0.1–100 Hz default) | `preprocess_raw(filter_args={l_freq:0.1, h_freq:100})` |
| ZapLine | `ZapLine(sfreq, line_freq, adaptive).fit_transform(raw)` | `zapline_denoise(line_freq=60.0)` — same call |
| RANSAC bad channels | `NoisyChannels(raw_for_ransac, random_state=42).find_bad_by_ransac()` | `ransac_bad_channels` — same call |
| RANSAC data scope | **Rest-block-biased**: crops to baseline/rest windows only | **Whole recording**: runs on full `raw` |
| CAR | `raw.set_eeg_reference("average", projection=False)` | `apply_car` — same call |
| AutoReject | **Condition-aware blockwise** (see §3) | **Whole-recording** fixed-length segments |
| ICA | Not in default chain (separate `correct.py` stage) | `ica_artifact_correction` — optional node, not in step-0b |
| Output | `raw` with BAD_ annotations; saved as `*_desc-base_raw.fif` | `Epochs` (fixed-length 2 s); saved as `*@CleanedPrep.fif` |

### 2.2 ZapLine

Both use `mne_denoise.zapline.ZapLine`.  Parameters are identical (non-adaptive by default, 60 Hz).  
No algorithmic difference.

### 2.3 RANSAC

Both call `pyprep.find_noisy_channels.NoisyChannels(..., random_state=42).find_bad_by_ransac()`.

Key difference: the original crops to rest/baseline blocks (`bids.collect_baseline_windows`) before running RANSAC.  This removes task-related high-amplitude activity so the channel quality estimate is driven by resting-state intrinsics.  The neurodags version uses the full recording; on long task paradigms this could mark fewer channels bad.

### 2.4 AutoReject — most significant preprocessing difference

**Original** (`annotate_artifacts_blockwise`):
- Groups recording into conditions using `BLOCK_*` annotations.
- Runs a separate AutoReject instance per condition (e.g., one for all EO blocks, one for all EC blocks).
- Supports chunking long conditions (`ar_max_chunk_minutes`) to avoid memory issues.
- `AutoReject(n_interpolate=[0], cv=min(10, n_epochs))` — does not interpolate channels, only marks bad epochs and bad channel spans.
- Converts reject log to `BAD_epoch_<condition>` and `BAD_<condition>` (per-channel) annotations on the raw; does NOT remove epochs.
- Outputs: annotated Raw, downstream epoch extraction uses `reject_by_annotation="omit"` per condition.

**neurodags** (`autoreject_annotate`):
- Runs AutoReject on the **whole recording** at once using fixed 1 s segments.
- No condition awareness; all epochs pooled for threshold estimation.
- `BAD_epoch` annotations added (no per-channel span annotations).
- Final `make_fixed_length_epochs(reject_by_annotation="omit")` creates the `CleanedPrep.fif` output.
- No chunking — could fail on very long recordings.

**Consequence**: Original AR thresholds are condition-specific (better for paradigms with distinct SNR between conditions); neurodags uses a single global threshold.

---

## 3. Condition-level epoch extraction

### 3.1 Original flow

`base.py` leaves the raw annotated but un-epoched.  Condition epochs are created later in `extract_descriptors.py` via `load_eeg_data(..., condition=condition)`, which reads `BLOCK_*`-tagged events from the saved `_desc-base_epo.fif` derivatives.

The original epoch file stores **all conditions** in a single MNE Epochs object with `event_id` keys per condition label (e.g., `{"EO_baseline": 1, "EC_baseline": 2}`).  The descriptor script selects per-condition by iterating `sessions × conditions` and loading with a condition filter.

### 3.2 neurodags flow (`step-0c_conditions.yml`)

Produces separate per-condition `.fif` files:
- `*@ConditionEO.fif` — epochs from `BLOCK_EO` windows
- `*@ConditionEC.fif` — epochs from `BLOCK_EC` windows

Each is extracted by `extract_condition_epochs`:
1. Find all annotations matching `BLOCK_<condition>` (strips BrainVision `Comment/` prefix).
2. Crop each window, tile with `make_fixed_length_epochs(duration=2.0, overlap=0.0)`.
3. Concatenate windows.

The original epoch extraction (`build_block_events_by_condition`) works the same way — creates fixed-length events within each BLOCK window, groups by condition label.  The neurodags node is a faithful re-implementation.

**Key difference**: neurodags applies only bandpass + resample (1–45 Hz, 256 Hz) before condition extraction — no artifact cleaning.  The original condition epochs come from the already-AR-annotated recording (BAD_ regions omitted).

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

## 10. Known gaps — status

1. **Corrected-absolute band ratio-of-means** (`agg_band_corr_ratio_*`): **DONE** — `CorrectedBandPowerAgg` + `BandRatiosOnCorrectedMeans` added to `step-1_features.yml`.

2. **Condition-aware AutoReject**: **DONE** — new `autoreject_clean_epochs` node in `custom_nodes.py` takes Epochs directly (not Raw), runs AR, drops bad epochs.  Wired as node id.3 in both `ConditionEO` and `ConditionEC` derivatives in `step-0c_conditions.yml`.  Each condition gets its own AR instance — matches `annotate_artifacts_blockwise` behavior from `base.py`.

3. **RANSAC on block subset**: **DONE** — `ransac_bad_channels` now accepts `block_label: str | None` (e.g., `block_label: EC`).  When set, RANSAC crops to windows matching `BLOCK_{block_label}` before fitting, mirroring the rest-block-biased approach in `base.py`.  `None` = full recording (original behavior preserved).

4. **Annotation inflation**: **not needed for neurodags pipelines**.  `inflate_bad_annotations` assigns fixed durations to point-like *manual* `BAD_` annotations made in BrainVision (e.g., `bad_yawn` → 5 s).  neurodags pipelines do not use manual annotations — all artifact detection is automatic.

5. **QC / failure status**: **covered by `neurodags status`**.  Run `neurodags status step-1_features.yml` to see per-derivative done/missing/errored counts per file.  `--list-errors` prints the errored file paths.  This covers both QC missingness (missing derivative = NaN in CSV) and failure logging (errored derivative = computation failed).

6. **Epoch-level CSV**: **not needed now**.  Create a second YAML where the `for_dataframe: True` derivatives are the per-epoch intermediates (e.g., `AbsBandPower`, `SampleEntropy`) instead of their aggregated equivalents.

7. **Config provenance**: copy the pipeline YAMLs and `custom_nodes.py` to `derivatives/preprocessing/code/` at run time.  Also record the neurodags git commit.  Can be done with a small wrapper script around `neurodags run`.

8. **Run-aware aggregation**: **out of scope for neurodags** — neurodags produces one row per file.  Subject × session × run grouping is post-processing, inferable from the filename using BIDS entity regex on `recording_id`.

9. **Failure logging**: **covered by `neurodags status --list-errors`** (see gap 5).
