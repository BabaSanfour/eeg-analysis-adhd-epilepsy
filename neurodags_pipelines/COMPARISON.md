# Original vs neurodags pipeline — detailed comparison

Covers all major design decisions, algorithmic equivalence, and known gaps.

| Original | neurodags |
|---|---|
| `eeg_adhd_epilepsy/preproc/base.py` | `step-0b_preproc_cleaned.yml` + `custom_nodes.py` |
| `eeg_adhd_epilepsy/analysis/extract_descriptors.py` | `step-1_features.yml` + `custom_nodes.py` |
| `eeg_adhd_epilepsy/io/bids.py` (block windowing) | `step-0c_conditions.yml` + `custom_nodes.py` |
| `configs/descriptors.yaml` | YAML anchors `&BANDS`, `&CHANNEL_GROUPS` in `step-1_features.yml` |

---

## 1. Overall architecture

| Dimension | Original | neurodags |
|---|---|---|
| Paradigm | Imperative Python scripts, BIDS-aware | Declarative YAML DAGs, file-pattern-aware |
| Invocation | `python base.py --bids_root ... --subjects ...` | `neurodags run step-0b_preproc_cleaned.yml` |
| Subject selection | CLI `--subjects`, BIDS metadata CSV | Dataset YAML file glob (`datasets_raw.yml`) |
| Granularity | Subject × session × run × condition loops in Python | One YAML derivative per logical output; per-file caching in `.nc` / `.fif` |
| Intermediate storage | `.fif` per BIDS entity (`_desc-base_raw.fif`) | `.fif` for Raw/Epoch derivatives; `.nc` (NetCDF4) for feature derivatives |
| Resume / idempotency | `_SUCCESS` marker per shard; `--overwrite` flag | `overwrite: False` per derivative; re-runs skip existing files |
| Parallelism | `joblib.Parallel` for short files; sequential (n_jobs per file) for long files | `--n-jobs` flag on `neurodags run`; sequential by default |
| Provenance | Rich per-run JSON (`_prov.json`) + `config_used.yaml` guard | `@CleanedPrepRaw_prov.json` (AR stats) + `derivatives_path/code/` snapshot on every run |
| BIDS compliance | Full BIDS conventions; `dataset_description.json` | No BIDS awareness; flat `derivatives/preprocessing/` tree |

---

## 2. Preprocessing chain

### 2.1 Step ordering (current state)

| Step | Original (`base.py`) | neurodags step-0b | neurodags step-0c |
|---|---|---|---|
| Annotation inflation | `inflate_bad_annotations` | `inflate_bad_annotations` ✓ | — (via CleanedPrepRaw) |
| **Resample** | `raw.resample()` — **before** bandpass | `preprocess_raw(resample=256)` — **after** bandpass ± | — (via CleanedPrepRaw) |
| Bandpass filter | 0.1–100 Hz | 0.1–100 Hz ✓ | — (via CleanedPrepRaw) ✓ |
| ZapLine | `ZapLine.fit_transform(raw)`, adaptive from config | `zapline_denoise(line_freq=60, adaptive=False)` ± | — (via CleanedPrepRaw) ✓ |
| RANSAC bad channels | EC-block subset (`collect_baseline_windows`) | `ransac_bad_channels(block_label="EC")` ✓ | — (via CleanedPrepRaw) ✓ |
| CAR | `set_eeg_reference("average", projection=False)` | `apply_car` ✓ | — (via CleanedPrepRaw) ✓ |
| AR — scope | per-condition (all EO blocks merged, all EC blocks merged) | `autoreject_annotate_blockwise`: per-condition ✓ | pure extraction (no AR) |
| AR — input | ZapLine+RANSAC+CAR cleaned Raw | cleaned Raw (CleanedPrepRaw) ✓ | from CleanedPrepRaw ✓ |
| AR — chunking | `_iter_autoreject_chunks`: `max(1, int(chunk_min*60/seg_dur))` | `max(min_epochs, int(...))` ± | — |
| AR — CV | `min(10, n_epochs_chunk)` per chunk | `min(10, len(epochs_chunk))` ✓ | — |
| AR — epoch label | `BAD_epoch_{condition_name}` | `BAD_epoch_{cond_name}` ✓ | — |
| AR — per-channel spans | `BAD_{cond}` + `ch_names` tuples | `BAD_{cond}` + `ch_names` tuples ✓ | — |
| AR plots | per-chunk PNG (horizontal, 16×10, 150 dpi) | combined per-condition PNG (same params) ± | — |
| Provenance JSON | rich per-run JSON (bads, integrity stats, by-block) | `@CleanedPrepRaw_prov.json` (bads, AR stats) ± | — |
| Output | annotated Raw (`_desc-base_raw.fif`) | `@CleanedPrepRaw.fif` (Raw) + `@CleanedPrep.fif` (Epochs) | `@ConditionEO/EC.fif` (Epochs) |

**Legend:** ✓ = equivalent, ± = minor difference (see §2.x), ✗ = known gap, — = not applicable

### 2.2 Resample order

**base.py** resamples **before** bandpass: `raw.resample(target_sfreq)` → `raw.filter(l_freq, h_freq)`.

**neurodags** `preprocess_raw` does the reverse: `raw.filter(l_freq, h_freq)` → `raw.resample(target_sfreq)`.

Impact: when source recording is already at 256 Hz (the target), no difference — resample is a no-op.
For higher-rate recordings (e.g. 1000 Hz), the order matters numerically:
- base.py: resample to 256 Hz (MNE applies anti-aliasing LP internally), then filter 0.1–100 Hz.
- neurodags: filter 0.1–100 Hz at 1000 Hz, then resample to 256 Hz.

Both are valid; the output is nearly identical in practice.

### 2.3 ZapLine adaptive parameter

base.py reads `adaptive` from config (CLI `--adaptive` flag); default is `False`.
neurodags hardcodes `adaptive=False`. Functionally identical for default usage; a YAML param would be needed to expose this.

### 2.4 AR chunk size minimum

base.py: `max_per_chunk = max(1, int((chunk_minutes * 60.0) / segment_duration))`
neurodags: `n_per_chunk = max(min_epochs, int((ar_max_chunk_minutes * 60.0) / segment_duration))`

Difference: neurodags uses `min_epochs` (default 5) as the floor instead of 1. Affects only very short conditions where chunk limit < min_epochs. Practically identical on normal data.

### 2.5 AR n_jobs

base.py passes `n_jobs` to AutoReject (from CLI `--n-jobs`).
neurodags hardcodes `n_jobs=1`. No impact on results; only on speed for large AR fits.

### 2.6 AR plot granularity

base.py saves one PNG per chunk: `{record_label}_autoreject_{cond}{chunk_suffix}.png` (e.g. `_chunk1.png`).
neurodags saves one PNG per condition (labels concatenated across chunks): `@CleanedPrepRaw_ar_plot_{cond}.png`.
Same visual params: `orientation="horizontal"`, `figsize=(16, 10)`, `dpi=150`.
Neurodags combined view is easier for overview; per-chunk is finer-grained for long recordings.

### 2.7 Provenance JSON richness

**base.py** `_prov.json` fields:
- `subject_id`, `config` (full PreprocConfig), `steps_completed`
- `pipeline_warnings`
- `bad_channels_global` (RANSAC bads)
- `artifact_stats` per condition: `bad_epochs`, `bad_channel_spans`, `artifacts_count`, `by_block`
- `block_stats`
- `integrity_stats`: `clean_duration_s`, `clean_fraction`, `manual_bad_fraction`, `autoreject_bad_fraction`

**neurodags** `@CleanedPrepRaw_prov.json` fields:
- `bad_channels` (from `raw.info["bads"]` at AR input, i.e. after RANSAC)
- `conditions`: per-condition `n_epochs`, `n_bad_epochs`, `clean_fraction`
- `overall_clean_fraction`

Missing vs original: full config dump, pipeline warnings, manual vs autoreject fraction split, by-block detail.
Config is separately captured in `derivatives_path/code/` (see §1).

### 2.8 Float precision on CleanedPrepRaw save/load

step-0b saves `CleanedPrepRaw.fif` as float32 (MNE default). step-0c reloads it.
base.py keeps the annotated Raw in memory and extracts condition epochs without a save/load round-trip.
Numerical effect: float32 precision (~7 significant digits) vs float64 (~15). Negligible for EEG.

---

## 3. Condition-level epoch extraction

### 3.1 Original flow

`base.py` outputs annotated Raw (`_desc-base_raw.fif`). `extract_descriptors.py` loads this via
`load_eeg_data(..., use_derivatives=True, desc="base")` — reads BIDS event structure and extracts
condition-labeled epochs on the fly. All conditions come from the **same in-memory cleaned Raw**.

### 3.2 neurodags flow

step-0c reads `CleanedPrepRaw.fif` (produced by step-0b) via `datasets_cleaned_raw.yml`.
`extract_condition_epochs(condition_name, reject_by_annotation="omit")` tiles fixed-length 2 s epochs
within BLOCK_{cond} windows, dropping BAD_-annotated segments.

**Correspondence:** both extract fixed-length 2 s epochs from BLOCK_* windows of the fully cleaned annotated Raw. The only difference is the float32 round-trip (§2.8).

---

## 4. Feature extraction

### 4.1 Entry points

| | Original | neurodags |
|---|---|---|
| Entry | `python extract_descriptors.py --bids_root ...` | `neurodags run step-1_features.yml` → `neurodags dataframe step-1_features.yml` |
| Feature framework | `coco_pipe.descriptors.DescriptorPipeline` | neurodags built-in nodes + `custom_nodes.py` |
| Config | `configs/descriptors.yaml` (DescriptorConfig pydantic model) | `step-1_features.yml` YAML anchors `&BANDS`, `&CHANNEL_GROUPS` |
| Output format | `.parquet` + `.csv` per subject/session/condition shard | Single flat CSV per dataset |

### 4.2 Band power variants and stats

Both produce 6 variants: absolute, log-absolute, relative, corrected-absolute, corrected-log-absolute, corrected-relative.
Bands: delta [1–4 Hz], theta [4–8], alpha [8–13], beta [13–30], gamma [30–45].
Spectral estimation: Welch, fmin=1, fmax=45, n_fft=512, n_overlap=256 — matches coco-pipe parameters.

| Family | Original stats | neurodags |
|---|---|---|
| Band power abs + corrected-abs | **mean only** | Mean + Median + IQR (more than original) |
| Band power log, rel, corr-log, corr-rel | mean + median + IQR | Mean + Median + IQR ✓ |
| FOOOF scalars (exponent, offset, R²) | mean + median + IQR | Mean + Median + IQR ✓ |
| FOOOF peaks (7 vars) | mean + median + IQR | Mean + Median + IQR ✓ |
| Complexity (antropy, neurokit2, stats) | mean + median + MAD | Mean + Median + MAD ✓ |

Note: neurodags computes **Median and IQR for absolute band power** even though the original only aggregates absolute power by mean. These extra columns are present in the neurodags dataframe but absent in the original.

### 4.3 SpectralEntropy sampling frequency

`antropy.spectral_entropy` requires `sf` (sampling frequency).

~~neurodags **hardcodes `sf=256.0`** in `step-1_features.yml`.~~
**Fixed**: `extract_sfreq_from_xarray` node reads `sfreq` from epoch xarray `attrs["metadata"]` JSON (set by neurodags epoch factory); passed as `sf: id.1` to `antropy_spectral_entropy`.

### 4.4 FOOOF fit

`aperiodic_mode=fixed`, `max_n_peaks=6`, `peak_width_limits=[1.0, 12.0]`, `freq_range=[1, 45]`.
Same config as `configs/descriptors.yaml`. Equivalent.

### 4.5 Band ratios

10 pairs: delta/theta, delta/alpha, delta/beta, delta/gamma, theta/alpha, theta/beta, theta/gamma, alpha/beta, alpha/gamma, beta/gamma.

| Variant | Original | neurodags |
|---|---|---|
| Per-epoch ratios (mean/median/IQR over epochs) | Yes | `BandRatiosMean/Median/IQR` ✓ |
| Ratio-of-means on abs power | Yes — `agg_band_ratio_*` | `BandRatiosOnMeans` ✓ |
| Ratio-of-means on corr-abs power | Yes — `agg_band_corr_ratio_*` | `BandRatiosOnCorrectedMeans` ✓ |

Original applies a `aggregated_ratio_floor` guard (`where=d_vals > floor`) before dividing. neurodags `band_ratios` node should implement the same guard — verify this if NaN/inf values appear in the ratio columns.

### 4.6 Run-aware aggregation

**Original** (`_with_recording_group_columns`): creates `recording_id = subject + "_ses-" + session + "_run-" + run` and aggregates by this key — so multi-run subjects get one aggregated row per run, not one row total.

**neurodags**: `neurodags dataframe` is file-level — one row per source file. Run information is implicit in the filename but not parsed. For multi-run studies, each run file gets its own row; cross-run aggregation must be done post-hoc.

### 4.7 Complexity features

| Source | Original | neurodags |
|---|---|---|
| antropy (14) | SampleEnt(order=2,metric=chebyshev), ApproxEnt, PermEnt(order=3,delay=1,normalize), SVDEnt(order=3,delay=1,normalize), SpectralEnt(**sf=sfreq**,method=welch,nperseg=128,normalize), HjorthMobility, HjorthComplexity, LZiv, NumZeroCross(normalize), HiguchiFD(kmax=10), KatzFD, PetrosianFD, DFA | Same 14, SpectralEnt sf dynamic via `extract_sfreq_from_xarray` ✓ |
| neurokit2 (5) | MultiscaleEntropy, ShannonEntropy, FuzzyEntropy, DispersionEntropy, HurstExponent | Same 5 ✓ |
| stats (2) | Kurtosis, RMS | Same 2 ✓ |

All 21 measures × 3 stats (mean/median/MAD) = 63 complexity aggregation derivatives.

### 4.8 Epoch-level vs subject-level output

**Original**: two DataFrames per shard:
- `sensor_epoch_features.csv` — one row per epoch
- `sensor_subject_features.csv` — one row per recording (mean + ratio-of-means)

**neurodags**: `neurodags dataframe` produces one flat CSV; each row is a source file with all `for_dataframe: True` derivatives. No epoch-level CSV from the `dataframe` command. Epoch data accessible via `.nc` files.

### 4.9 Feature column naming

Original: `band_abs_delta_Fz`, `complexity_sample_entropy_Fz`.
neurodags: derivative name as prefix, e.g. `AbsBandPowerMean_spaces_delta_Fz`. Column names differ — mapping required for cross-system comparison.

---

## 5. Spatial pooling

### Original

`pipeline.pool_channels(sensor_result, channel_groups)` in `_build_feature_outputs`.
9 regions (10-20 system):
```
front_left: [F7, F3]    front_midline: [Fz]      front_right: [F8, F4]
central_left: [T3, C3]  central_midline: [Cz]    central_right: [T4, C4]
posterior_left: [T5, P3, O1]  posterior_midline: [Pz]  posterior_right: [T6, P4, O2]
```
Produces separate `pooled_epoch_features.csv` and `pooled_subject_features.csv`.

### neurodags

`pool_channels` node on each band power, FOOOF scalar, and band ratio derivative.
Same 9 regions via `&CHANNEL_GROUPS` anchor — identical to original config.
Absent channels silently skipped. `*Pooled` saved as `.nc`; `*PooledMean` is `for_dataframe: True`.

**Synthetic data**: channels `EEG000–EEG007` match no 10-20 labels → pooled derivatives absent from dataframe.

---

## 6. QC layer

**Original**: `eeg_adhd_epilepsy.qc` pipeline:
- `run_descriptor_subject_qc` — missingness, flag rates, family summary
- `write_subject_preproc_qc_report` — HTML per subject
- Outputs: `qc/summary_row.csv`, `qc/flags.csv`, `qc/feature_missingness.csv`, `qc/family_summary.csv`, HTML

**neurodags**: no QC layer. `neurodags status` reports done/missing/errored counts per derivative.
Failed derivatives yield `NaN` in the dataframe. No structured failure log, no HTML reports.

Largest functional gap for production use.

---

## 7. Dataset / BIDS integration

**Original**: fully BIDS-aware.
- Subject IDs normalized to `sub-XXXX`.
- Reads `BIDS/derivatives/preproc/sub-*/ses-*/eeg/*_desc-base_epo.fif`.
- Validates coverage via `validate_bids_coverage`.
- `--metadata_row` for SLURM array jobs.
- Writes `dataset_description.json` and `config_used.yaml`.
- `recording_id = subject_ses-{session}_run-{run}` for run-aware aggregation.

**neurodags**: file-glob based. Subject/session/run implicit in filenames. No metadata CSV, no BIDS paths, no config versioning guard. Config snapshot written to `derivatives_path/code/` on every run.

---

## 8. Failure handling

**Original**: per-descriptor try/except in coco-pipe `DescriptorPipeline.extract`.
Failures logged to `failures.csv` (subject, obs_id, channel, family, exception type, message).
Processing continues for other descriptors and epochs.

**neurodags**: if a node raises, the derivative fails for that file. The dataframe has `NaN` for all downstream columns. No structured failure log.

---

## 9. Algorithmic equivalence summary

| Feature | Equivalent? | Notes |
|---|---|---|
| Annotation inflation | Yes | `inflate_bad_annotations` ✓ |
| Resample | Near-equivalent | Filter then resample in neurodags; resample then filter in base.py (§2.2) |
| Bandpass filter 0.1–100 Hz | Yes | Same parameters ✓ |
| ZapLine | Near-equivalent | `adaptive=False` hardcoded in neurodags; configurable in original (§2.3) |
| RANSAC bad channels | Yes | Both use EC-block subset, `random_state=42` ✓ |
| CAR | Yes | `set_eeg_reference("average", projection=False)` ✓ |
| AR scope | Yes | Per-condition, all windows merged, one AR per condition ✓ |
| AR input quality | Yes | Both run AR on ZapLine+RANSAC+CAR cleaned Raw ✓ |
| AR chunking | Near-equivalent | `max(min_epochs, ...)` vs `max(1, ...)` floor (§2.4) ✓ |
| AR CV | Yes | `min(10, len(epochs_chunk))` per chunk ✓ |
| AR epoch label | Yes | `BAD_epoch_{cond_name}` ✓ |
| AR per-channel spans | Yes | `BAD_{cond}` + `ch_names` tuples ✓ |
| AR n_jobs | Partial | Hardcoded `n_jobs=1` in neurodags; no result difference |
| AR rejection plots | Partial | Combined per-condition PNG vs per-chunk in original (§2.6) |
| Condition epoch extraction | Near-equivalent | float32 round-trip via CleanedPrepRaw.fif save/load (§2.8) |
| Provenance JSON | Partial | AR stats + bad channels; missing integrity fractions + config dump (§2.7) |
| Config snapshot | Partial | YAML + code in `derivatives_path/code/`; no re-run guard |
| Welch PSD | Yes | fmin=1, fmax=45, n_fft=512, n_overlap=256 ✓ |
| FOOOF fit | Yes | aperiodic_mode=fixed, max_n_peaks=6, peak_width_limits=[1,12] ✓ |
| FOOOF peak features (7 vars) | Yes | n_peaks, dom/alpha CF/PW/BW ✓ |
| Abs/corr-abs band power stats | **More than original** | neurodags also computes Median+IQR; original mean only (§4.2) |
| Log/rel/corr-log/corr-rel band power stats | Yes | mean+median+IQR ✓ |
| Band ratios per epoch | Yes | 10 pairs ✓ |
| Band ratios on abs means | Yes | `BandRatiosOnMeans` ✓ |
| Band ratios on corr-abs means | Yes | `BandRatiosOnCorrectedMeans` ✓ |
| Multi-stat aggregation (IQR/MAD) | Yes | Via custom nodes ✓ |
| Spatial pooling | Yes (real data) | Absent on non-10-20 channels |
| antropy complexity (14 measures) | Near-equivalent | SpectralEntropy `sf` hardcoded 256.0 (§4.3) |
| neurokit2 complexity (5) | Yes | ✓ |
| Kurtosis + RMS | Yes | ✓ |
| Run-aware aggregation | **No** | neurodags is file-level only; no recording_id grouping (§4.6) |
| Epoch-level CSV | Partial | `.nc` accessible; no epoch CSV from `dataframe` |
| Subject-level CSV | Yes | One aggregated row per file ✓ |
| QC / failure tracking | Partial | `neurodags status` covers done/missing/errored; no HTML reports |
| BIDS output structure | No | Flat derivative tree |
| SLURM array support | No | Wrap `neurodags run` in array script externally |

---

## 10. Remaining gaps

### Tier 1 — affect feature values

~~**M. SpectralEntropy hardcoded sf=256.0** — FIXED~~  
`extract_sfreq_from_xarray` node reads sfreq from epoch xarray attrs; passed as `sf: id.1`.

### Tier 2 — minor numerical differences

**N. Resample order** (§2.2)  
base.py resamples before filtering; neurodags filters then resamples. Affects only recordings where source sfreq > 256 Hz. Practically equivalent on 256 Hz data.

**O. AR chunk size floor** (§2.4)  
`max(min_epochs=5, ...)` vs `max(1, ...)`. Affects only very short conditions (< 5 epochs at chunk boundary). Negligible in practice.

**P. Float32 precision on CleanedPrepRaw** (§2.8)  
`@CleanedPrepRaw.fif` is saved as float32; reloaded for step-0c. base.py keeps Raw in memory as float64. Difference is below noise floor for EEG.

### Tier 3 — cosmetic / infrastructure

**Q. ZapLine adaptive param not exposed** (§2.3)  
Hardcoded `adaptive=False`. Add a YAML param if needed.

**R. AR n_jobs hardcoded 1** (§2.5)  
No result difference. Add `n_jobs` param to `autoreject_annotate_blockwise` if speed matters on large datasets.

**S. Provenance richness** (§2.7)  
Missing: integrity stats (manual vs AR bad fractions), by-block detail, full config dump in JSON.

**T. AR plot per-chunk vs combined** (§2.6)  
Combined is better for overview; original per-chunk useful for very long recordings split into > 1 chunk.

**U. Abs band power extra stats** (§4.2)  
neurodags produces Median+IQR for abs/corr-abs power; original does not. Extra columns in the dataframe — not a gap but an asymmetry to be aware of when comparing outputs.

**V. Run-aware aggregation** (§4.6)  
neurodags is file-level only. Post-hoc grouping by run is needed for multi-run studies.

**W. Band ratio floor guard**  
Verify `band_ratios` node applies the same `aggregated_ratio_floor` guard as the original to prevent division by near-zero denominators.

---

## 11. Implementation status

| Gap | Status | Notes |
|-----|--------|-------|
| A. step-0c unclean raw | **DONE** | `CleanedPrepRaw` chain; step-0c reads `datasets_cleaned_raw.yml` |
| B. Filter range step-0c | **DONE** | step-0c inherits 0.1–100 Hz from `CleanedPrepRaw` |
| C. AR scope | **DONE** | `autoreject_annotate_blockwise`: per-condition, adaptive CV |
| D. RANSAC rest-subset | **DONE** | `block_label: EC` in step-0b |
| E. Per-channel span annotations | **DONE** | `BAD_{cond}` + `ch_names` in `autoreject_annotate_blockwise` |
| F. AR chunking | **DONE** | `ar_max_chunk_minutes` in `autoreject_annotate_blockwise` |
| G. Epoch annotation labels | **DONE** | `BAD_epoch_{cond_name}` |
| H. AR CV | **DONE** | `min(10, len(epochs_chunk))` per chunk |
| I. Annotation inflation | **DONE** | `inflate_bad_annotations` first in `CleanedPrepRaw` chain |
| J. AR rejection plots | **PARTIAL** | Combined per-condition PNG; original per-chunk (§10.T) |
| K. Provenance JSON | **DONE** | `@CleanedPrepRaw_prov.json`: bads + AR stats |
| L. Config versioning | **DONE\*** | Snapshot to `derivatives_path/code/`; no re-run guard |
| M. SpectralEntropy sf | **DONE** | Dynamic via `extract_sfreq_from_xarray` node |
| N. Resample order | **open (minor)** | Filter→resample vs resample→filter; negligible for 256 Hz source |
| O. AR chunk floor | **open (negligible)** | min_epochs vs 1 floor; no practical impact |
| P. Float32 precision | **open (negligible)** | CleanedPrepRaw save/load; below EEG noise floor |
| Q. ZapLine adaptive | **open (minor)** | Add YAML param if needed |
| R. AR n_jobs | **open (perf)** | No result impact; add param for speed |
| S. Provenance richness | **open** | Missing integrity fractions and by-block stats |
| T. AR plot granularity | **open (minor)** | Per-chunk plots for long recordings |
| U. Abs power extra stats | **by design** | neurodags computes more; not a bug |
| V. Run-aware aggregation | **open** | Post-hoc only in neurodags; no recording_id grouping |
| W. Band ratio floor guard | **verify** | Check `band_ratios` node applies denominator floor |
