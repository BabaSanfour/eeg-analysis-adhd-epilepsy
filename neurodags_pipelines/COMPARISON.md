# Pipeline Comparison: original (`preproc/base.py` + `extract_descriptors.py`) vs neurodags

*Last updated: 2026-05-21. All gaps re-audited from source.*

---

## 1. Processing chains side-by-side

### 1.1 Preprocessing

| Step | Original (`base.py`) | neurodags (`step-0b`) | Status |
|------|----------------------|-----------------------|--------|
| Annotation inflation | `inflate_bad_annotations` (point → 3 s; major → 5 s) | `inflate_bad_annotations` id=1 (same defaults) | ✓ |
| Resample | `raw.resample(target_sfreq)` — **before** bandpass | `preprocess_raw(resample_first=True)` — resample before bandpass | ✓ |
| Bandpass | `raw.filter(0.1, min(100, nyquist-0.1))` | `filter_args: {l_freq: 0.1, h_freq: 100.0}` | ✓ |
| Line noise | `ZapLine(sfreq, line_freq=60, adaptive=False).fit_transform(raw)` | `zapline_denoise(line_freq=60.0, adaptive=False)` | ✓ |
| Bad channels | RANSAC on EC windows only (`NoisyChannels`, `random_state=42`) | `ransac_bad_channels(block_label=EC)` | ✓ |
| Reference | `set_eeg_reference("average", projection=False)` | `apply_car` | ✓ |
| AR | per-condition, 1 s epochs, `min(10, n)` CV folds, `n_interpolate=[0]`, chunked at 30 min | `autoreject_annotate_blockwise` same params + `n_jobs=1` (YAML) | ✓ |
| AR annotations | `BAD_epoch_{cond}` (whole-epoch) + `BAD_{cond}` (per-channel ch_names) | same | ✓ |
| Output (continuous) | in-memory `Raw` | `@CleanedPrepRaw.fif` (float32 on disk) | ✓ (±float32) |
| Output (epochs) | condition-specific epoch objects | `@CleanedPrep.fif` (all conditions, BAD omitted) | ✓ |
| AR chunk floor | `max(1, int(...))` | `max(1, int(...))` | ✓ |

### 1.2 Feature extraction

| Step | Original (`extract_descriptors.py`) | neurodags (`step-1_features.yml`) | Status |
|------|-------------------------------------|-----------------------------------|--------|
| Input | per-condition epochs (EO, EC run separately) | `*@BasicPrep.fif` by default; run with `-d datasets_conditions.yml` for per-condition | ⚠ see §3.1 |
| Welch PSD | `fmin=1, fmax=45, n_fft=512, n_overlap=256` | same (`SpectrumWelch`) | ✓ |
| FOOOF | `fixed`, `max_n_peaks=6`, `peak_width_limits=[1,12]`, `freq_range=[1,45]`, `freq_res=0.5` | same (`FooofFit`) | ✓ |
| Bands | delta/theta/alpha/beta/gamma `[1-4/4-8/8-13/13-30/30-45]` | same | ✓ |
| Abs power stats | mean only | mean + median + IQR (more than original) | ➕ extra |
| Log/rel power stats | mean + median + IQR | same | ✓ |
| Corr-abs/log/rel stats | mean + median + IQR | same | ✓ |
| Band ratios (per-epoch) | 10 pairs, mean/median/IQR | same (`BandRatios*`) | ✓ |
| Band ratios on means | `agg_band_ratio_*`, `agg_band_corr_ratio_*` | `BandRatiosOnMeans`, `BandRatiosOnCorrectedMeans` | ✓ |
| Ratio floor guard | `where d > 0.0` (floor=0, NaN if ≤ 0) | `where \|d\| > eps` (machine epsilon, NaN if near-zero) | ✓ |
| Spatial pooling | 9 regions (10-20), per-epoch + mean | `*Pooled`, `*PooledMean` (same 9 regions) | ✓ |
| SpectralEntropy sf | `sf=sfreq` (dynamic from epoch info) | `sf=id.1` via `extract_sfreq_from_xarray` | ✓ |
| antropy (14) | all 14 measures, same params | same | ✓ |
| neurokit2 (5) | all 5 | same (**EntropyMultiscale fails on NumPy 2.0**) | ⚠ |
| stats (2) | kurtosis, RMS | same | ✓ |
| Epoch aggregation | mean, median, IQR (spectral/FOOOF); mean, median, MAD (complexity) | same | ✓ |
| Run-aware aggregation | `recording_id = sub_ses_run`; one row per run per subject | file-level only; no cross-file grouping | ⚠ see §3.2 |
| QC layer | HTML reports + structured CSV (failures, missingness, flags) | `neurodags status` / `.error` markers only | ⚠ see §3.3 |

---

## 2. Remaining open gaps

### 2.1 Condition separation in feature extraction (§3.1)

**Original**: `extract_descriptors.py` runs once per condition (`--conditions EO EC`). Each condition produces its own output directory with `sensor_epoch_features.csv` for that condition's epochs only.

**neurodags default**: `step-1_features.yml` with `datasets: datasets_preprocessed.yml` → `*@BasicPrep.fif` as input → all epochs regardless of condition.

**neurodags per-condition run**:
```bash
neurodags run step-1_features.yml -d neurodags_pipelines/datasets_conditions.yml
```
`datasets_conditions.yml` defines two datasets: `synthetic_eeg_eo` (→ `*@ConditionEO.fif`) and `synthetic_eeg_ec` (→ `*@ConditionEC.fif`), writing features to separate `derivatives/features_conditions_eo/` and `features_conditions_ec/` paths.

**Status**: infrastructure exists; user must explicitly invoke per-condition run. Not yet documented in a single shell command.

### 2.2 Run-aware aggregation

**Original**: `_with_recording_group_columns()` creates `recording_id = subject + "_ses-" + session + "_run-" + run` and aggregates by this key so multi-run subjects get one row per run.

**neurodags**: `neurodags dataframe` assembles one row per source file. For most studies (1 run per subject) this is equivalent. For multi-run studies, aggregation across runs requires post-hoc pandas groupby on the assembled CSV.

**Status**: by-design limitation of the file-level architecture. Document and handle post-hoc.

### 2.3 QC layer

**Original** produces per-condition:
- `sensor_epoch_features.csv` / `pooled_epoch_features.csv` (per-epoch)
- `sensor_subject_features.csv` / `pooled_subject_features.csv` (aggregated)
- `failures.csv` (columns: condition, subject, obs_id, obs_index, channel_index, channel_name, family, exception_type, message)
- `qc/summary_row.csv`, `qc/summary_metrics.csv`, `qc/flags.csv`, `qc/feature_missingness.csv`, `qc/family_summary.csv`
- `_SUCCESS` marker per condition

**neurodags**: `.error` markers per failed file/derivative, `neurodags status --list-errors`. No per-feature missingness tracking, no structured failure rows, no `_SUCCESS` markers.

**Status**: significant gap for production use. Not planned for current scope.

### 2.4 Float32 precision on CleanedPrepRaw

MNE saves `.fif` as float32 by default. `CleanedPrepRaw.fif` → reloaded by `step-0c_conditions.yml` introduces float32 round-trip. Original keeps Raw in memory (float64) for condition extraction.

Effect: ~7 significant digits vs ~15. Below EEG noise floor (~1 µV). **Negligible in practice.**

### 2.5 AR plots granularity

Original: per-chunk AR reject-log plot (`{record_label}_autoreject_{condition}_chunk{N}.png`).
neurodags: combined reject-log across all chunks per condition (`@CleanedPrepRaw_ar_plot_{cond}.png`).

For recordings that require chunking (> 30 min per condition), the combined plot is slightly less interpretable. **Minor.**

### 2.6 Provenance: zapline_stats

Original `prov.json` includes:
```json
"zapline_stats": {"method": "zapline", "line_freq": 60.0, "adaptive": false, "n_removed": N}
```
neurodags `@CleanedPrepRaw_prov.json` does not include `zapline_stats` because ZapLine runs as a separate node (`zapline_denoise`) upstream of the AR node that writes the provenance. `n_removed_` attribute is not captured anywhere.

**Status**: minor — config is in `code/step-0b_preproc_cleaned.yml` snapshot; n_removed not tracked.

### 2.7 Provenance: manual_overlap_pct

Original computes `_compute_artifact_overlap(raw, new_annots)` — percentage of new AR annotations that overlap pre-existing manual BAD spans. Not in neurodags provenance.

**Status**: minor gap.

### 2.8 EntropyMultiscale (NumPy 2.0)

`neurokit_entropy_multiscale` fails with `module 'numpy' has no attribute 'trapz'` — NumPy 2.0 removed `np.trapz` (renamed `np.trapezoid`). neurokit2 hasn't patched this yet.

**Workaround**: pin `numpy<2.0` or wait for neurokit2 fix.

---

## 3. Equivalence table — full feature set

| Feature family | Original name pattern | neurodags derivative(s) | Status |
|---|---|---|---|
| Abs band power (per epoch) | `band_abs_{band}_{ch}` | `AbsBandPower.nc` | ✓ |
| Abs band power (mean) | aggregated mean | `AbsBandPowerMean` | ✓ |
| Abs band power (median/IQR) | *not computed* | `AbsBandPower` (no Med/IQR in orig) | ➕ extra |
| Log band power (epoch→mean/med/IQR) | `band_log_*` | `LogBandPower*` | ✓ |
| Rel band power (epoch→mean/med/IQR) | `band_rel_*` | `RelBandPower*` | ✓ |
| Corr-abs power (epoch→mean) | `band_corr_abs_*` | `CorrectedBandPower*` | ✓ |
| Corr-log power (epoch→mean/med/IQR) | `band_corr_log_*` | `CorrectedLogBandPower*` | ✓ |
| Corr-rel power (epoch→mean/med/IQR) | `band_corr_rel_*` | `CorrectedRelBandPower*` | ✓ |
| Band ratios per epoch | `band_ratio_*` (mean/med/IQR) | `BandRatios*` | ✓ |
| Ratio of abs means | `agg_band_ratio_*` | `BandRatiosOnMeans` | ✓ |
| Ratio of corr-abs means | `agg_band_corr_ratio_*` | `BandRatiosOnCorrectedMeans` | ✓ |
| Pooled band/ratio (mean) | `pooled_*` | `*Pooled`, `*PooledMean` | ✓ |
| FOOOF exponent (mean/med/IQR) | `param_exponent_*` | `FooofExponent*`, `FooofExponentPooled*` | ✓ |
| FOOOF offset (mean/med/IQR) | `param_offset_*` | `FooofOffset*`, `FooofOffsetPooled*` | ✓ |
| FOOOF R² (mean/med/IQR) | `param_r_squared_*` | `FooofRSquared*`, `FooofRSquaredPooled*` | ✓ |
| FOOOF n_peaks (mean/med/IQR) | `param_n_peaks_*` | `FooofNPeaks*` | ✓ |
| FOOOF dominant peak CF/PW/BW | `param_dom_{cf,pw,bw}_*` | `FooofDomCF/PW/BW*` | ✓ |
| FOOOF alpha peak CF/PW/BW | `param_alpha_{cf,pw,bw}_*` | `FooofAlphaCF/PW/BW*` | ✓ |
| SampleEntropy | `complexity_sample_entropy_*` | `SampleEntropy*` | ✓ |
| AppEntropy | `complexity_app_entropy_*` | `AppEntropy*` | ✓ |
| PermEntropy | `complexity_perm_entropy_*` | `PermEntropy*` | ✓ |
| SVDEntropy | `complexity_svd_entropy_*` | `SVDEntropy*` | ✓ |
| SpectralEntropy (sf=sfreq) | `complexity_spectral_entropy_*` | `SpectralEntropy*` | ✓ |
| HjorthMobility | `complexity_hjorth_mobility_*` | `HjorthParams*` (mobility dim) | ✓ |
| HjorthComplexity | `complexity_hjorth_complexity_*` | `HjorthParams*` (complexity dim) | ✓ |
| LZivComplexity | `complexity_lziv_*` | `LZivComplexity*` | ✓ |
| NumZeroCross | `complexity_num_zero_cross_*` | `NumZeroCross*` | ✓ |
| HiguchiFD | `complexity_higuchi_fd_*` | `HiguchiFD*` | ✓ |
| KatzFD | `complexity_katz_fd_*` | `KatzFD*` | ✓ |
| PetrosianFD | `complexity_petrosian_fd_*` | `PetrosianFD*` | ✓ |
| DetrendedFluctuation | `complexity_detrended_fluct_*` | `DetrendedFluctuation*` | ✓ |
| EntropyMultiscale | `complexity_multiscale_*` | `EntropyMultiscale*` | ⚠ NumPy 2.0 |
| EntropyShannon | `complexity_shannon_*` | `EntropyShannon*` | ✓ |
| EntropyFuzzy | `complexity_fuzzy_*` | `EntropyFuzzy*` | ✓ |
| EntropyDispersion | `complexity_dispersion_*` | `EntropyDispersion*` | ✓ |
| FractalHurst | `complexity_hurst_*` | `FractalHurst*` | ✓ |
| Kurtosis | `complexity_kurtosis_*` | `Kurtosis*` | ✓ |
| RMS | `complexity_rms_*` | `RMS*` | ✓ |

---

## 4. Gap status summary

| Gap | Status | Notes |
|-----|--------|-------|
| A. CleanedPrepRaw chain | **DONE** | step-0b produces CleanedPrepRaw → CleanedPrep |
| B. Filter range step-0c | **DONE** | inherited 0.1–100 Hz |
| C. AR scope | **DONE** | per-condition blockwise |
| D. RANSAC rest-subset | **DONE** | `block_label: EC` |
| E. Per-channel span annotations | **DONE** | `BAD_{cond}` with `ch_names` |
| F. AR chunking | **DONE** | `ar_max_chunk_minutes: 30.0` |
| G. Epoch annotation labels | **DONE** | `BAD_epoch_{cond_name}` |
| H. AR CV folds | **DONE** | `min(10, len(epochs_chunk))` |
| I. Annotation inflation | **DONE** | `inflate_bad_annotations` |
| J. AR rejection plots | **PARTIAL** | Combined per-condition; original per-chunk |
| K. Provenance JSON | **DONE** | `artifact_stats` + `integrity_stats` + `config` + `by_block` |
| L. Config versioning | **DONE\*** | Snapshot to `code/`; no re-run guard |
| M. SpectralEntropy sf | **DONE** | Dynamic via `extract_sfreq_from_xarray` |
| N. Resample order | **DONE** | `resample_first: True` |
| O. AR chunk floor | **DONE** | `max(1, ...)` |
| P. Float32 precision | **open (negligible)** | Below EEG noise floor |
| Q. ZapLine adaptive | **DONE** | Explicit `adaptive: False` in step-0b YAML |
| R. AR n_jobs | **DONE** | `n_jobs: 1` in step-0b YAML |
| S. Provenance richness | **DONE** | integrity_stats + by_block + config now in prov JSON |
| T. AR plot granularity | **open (minor)** | Combined vs per-chunk for long recordings |
| U. Abs power extra stats | **by design** | neurodags computes more (Med+IQR); original mean only |
| V. Run-aware aggregation | **open** | No recording_id grouping; post-hoc only |
| W. Band ratio floor guard | **DONE** | Both guard near-zero (eps vs 0.0 floor; equivalent) |
| X. Condition separation | **open (workflow)** | Default run uses all epochs; must pass `-d datasets_conditions.yml` |
| Y. ZapLine n_removed in prov | **open (minor)** | Not tracked; config is in code/ snapshot |
| Z. QC layer | **open** | No structured failure CSV, no feature missingness; out of scope |
