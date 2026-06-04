# Pipeline Comparison: original (`preproc/base.py` + `extract_descriptors.py`) vs neurodags

*Last updated: 2026-05-25. Reflects in-memory epoching: step-0 produces only CleanedPrepRaw; condition epoching happens in-memory in step-1 (save: False).*

---

## 1. Processing chains side-by-side

### 1.1 Preprocessing

| Step | Original (`base.py`) | neurodags (`step-0_pipeline@preprocessing`) | Status |
|------|----------------------|---------------------------------------|--------|
| BLOCK annotation source | `load_bids_raw` → `read_raw_bids` reads `_events.tsv` written by `to_bids.py` → BLOCK_* present on load | `load_meeg` → `read_raw_brainvision` reads `.vmrk` (hardware triggers only) → BLOCK_* absent; `inject_block_annotations` id=0 re-reads `_segments.csv` sidecar | ✓ see §2.9 |
| Annotation inflation | `inflate_bad_annotations` (point → 3 s; major → 5 s) | `inflate_bad_annotations` id=2 (same defaults) | ✓ |
| Resample | `raw.resample(target_sfreq)` — **before** bandpass | `preprocess_raw(resample_first=True)` — resample before bandpass | ✓ |
| Bandpass | `raw.filter(0.1, min(100, nyquist-0.1))` | `filter_args: {l_freq: 0.1, h_freq: 100.0}` | ✓ |
| Line noise | `ZapLine(sfreq, line_freq=60, adaptive=False).fit_transform(raw)` | `zapline_denoise(line_freq=60.0, adaptive=False)` | ✓ |
| Bad channels | RANSAC on EC windows only (`NoisyChannels`, `random_state=42`) | `ransac_bad_channels(block_label=EC)` | ✓ |
| Reference | `set_eeg_reference("average", projection=False)` | `apply_car` | ✓ |
| AR | per-condition, 1 s epochs, `min(10, n)` CV folds, `n_interpolate=[0]`, chunked at 30 min | `autoreject_annotate_blockwise` same params + `n_jobs=1` (YAML) | ✓ |
| AR annotations | `BAD_epoch_{cond}` (whole-epoch) + `BAD_{cond}` (per-channel ch_names) | same | ✓ |
| Output (continuous) | in-memory `Raw` | `@CleanedPrepRaw.fif` (float32 on disk) | ✓ (±float32) |
| Output (epochs) | condition-specific epoch objects | in-memory via `CleanedPrep` in step-1 (`save: False`; no disk file) | ✓ |
| AR chunk floor | `max(1, int(...))` | `max(1, int(...))` | ✓ |

### 1.2 Feature extraction

| Step | Original (`extract_descriptors.py`) | neurodags (`step-1_pipeline@extraction.yml`) | Status |
|------|-------------------------------------|---------------------------------------------|--------|
| Input | per-condition epochs (EO, EC run separately) | `*@CleanedPrepRaw.fif` → `CleanedPrep` in-memory epoching per condition | ✓ see §2.1 |
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

### 2.1 Condition separation in feature extraction

**Original**: `extract_descriptors.py` runs once per condition (`--conditions EO EC`). Each condition produces its own output directory with `sensor_epoch_features.csv` for that condition's epochs only.

**neurodags**: `step-1_pipeline@extraction.yml` defines `CleanedPrep` (`save: False`) that runs `extract_condition_epochs` on `CleanedPrepRaw.fif` in-memory. No per-condition files on disk. The condition name is set via `_condition_name` anchor in the pipeline YAML; the `derivatives_path` is selected via the active entry in `step-1_dataset.yml`.

Workflow (all 8 conditions active — one run covers all):
```bash
neurodags run neurodags_pipelines/step-1_pipeline@extraction.yml
neurodags dataframe neurodags_pipelines/step-1_pipeline@extraction.yml \
    --output results/features_all_conditions.csv
# Split by condition post-hoc on the `dataset` column, or use --datasets
# with a single-condition YAML to restrict to one condition.
```

**Status**: resolved — in-memory epoching, no per-condition disk files.

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

**neurodags coverage by sub-gap**:

| Sub-gap | neurodags coverage | Notes |
|---|---|---|
| Complete derivative failures | ✓ **covered** | `.error` marker + `neurodags status --list-errors` |
| `_SUCCESS` markers | ✓ **equivalent** | `neurodags status` scripting-friendly; no explicit file marker |
| Partial NaN within a successful derivative | ✗ **missing** | `.nc` writes OK but may contain NaN for some epochs/channels (e.g. FOOOF fit failures); post-hoc scan needed |
| Structured per-epoch/channel failure log | ✗ **missing** | No partial-failure ledger; neurodags either writes full derivative or `.error` |

**Status**: complete-failure tracking covered. Intra-file missingness and structured failure rows require a post-hoc script that loads each `.nc` and counts NaN per feature/channel — not planned for current scope.

### 2.9 BLOCK annotation source: BIDS conversion → `_events.tsv` vs `_segments.csv`

**Origin of BLOCK annotations**: `to_bids.py` calls `extract_condition_segments(raw)` to derive
condition windows from the hardware trigger log, then:
1. Injects `BLOCK_{segment_type}` annotations into the raw **before** calling `write_raw_bids`.
   `write_raw_bids` (MNE-BIDS) writes these to `_events.tsv` (BIDS standard).
2. Saves the same segment table as `{stem}_segments.csv` sidecar (custom, non-BIDS).

The `.vmrk` file **only contains hardware triggers** (`Stimulus/S XX`); BLOCK entries never
reach the `.vmrk` because BrainVision format stores them as event types that MNE-BIDS routes
to `_events.tsv` instead.

**Original `base.py`**: uses `load_bids_raw` → `read_raw_bids` (MNE-BIDS), which reads
`_events.tsv` and reconstructs all annotations including `BLOCK_*`. The annotations are
available in the raw before any pipeline step runs. The code checks:
```python
n_blocks = len(bids._collect_block_windows(raw))
if n_blocks == 0:
    LOGGER.warning("No embedded BLOCK_* annotations found; ...")
```

**neurodags**: uses `neurodags.loaders.load_meeg` → `mne.io.read_raw_brainvision`, which reads
only the `.vmrk` file (hardware triggers only). `_events.tsv` is not read. BLOCK annotations
are absent. The `inject_block_annotations` node (step-0_pipeline@preprocessing id=0) bridges this gap by reading the
`_segments.csv` sidecar directly — the same segment data, different file.

**Why `_segments.csv` not `_events.tsv`**: both contain the same BLOCK windows. `_segments.csv`
is simpler to parse (plain CSV with typed columns), already used by `load_segments_for_raw` in
`io/bids.py`, and avoids coupling neurodags to MNE-BIDS. To switch to `_events.tsv`, the
node would need to either call `read_raw_bids` (heavier) or parse the TSV manually.

**Status**: resolved via `inject_block_annotations`. See `nodes_annotations.py`.

---

### 2.10 Condition epoch extraction: in-memory approach

**Original (`epochs.py → make_epochs_from_preproc_raw`)**: `build_block_events_by_condition`
scans `BLOCK_*` annotations and groups by **exact** segment_type name (EO_baseline, HV_EO, etc.).
BAD_ rejection not applied at this stage.

**neurodags `step-1_pipeline@extraction.yml`**: `CleanedPrep` derivative (`save: False`) runs
`extract_condition_epochs(condition_name=<active>, reject_by_annotation="omit")` on `CleanedPrepRaw.fif`.
This applies `BAD_epoch_{cond}` annotations written by `autoreject_annotate_blockwise` in step-0,
dropping bad epochs for the active condition. Runs in-memory; no `.fif` written to disk.

**No cross-condition BAD_ bleeding concern**: `extract_condition_epochs` remaps cross-condition
BAD_ spans to `SKIP_` before epoching — only the active condition's BAD marks are applied.

**Status**: resolved — in-memory, per-segment-type, correct BAD_ scoping.

---

### 2.4 Float32 precision on CleanedPrepRaw

MNE saves `.fif` as float32 by default. `CleanedPrepRaw.fif` (float32 on disk) → reloaded by step-1 introduces a float32 round-trip for epoch extraction. Original keeps Raw in memory (float64). Effect below EEG noise floor. **Negligible.**

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

**Status**: minor — config is in `code/step-0_pipeline@preprocessing.yml` snapshot; n_removed not tracked.

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
| A. CleanedPrepRaw chain | **DONE** | step-0 produces CleanedPrepRaw; step-1 epochs in-memory (CleanedPrep, save: False) |
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
| Q. ZapLine adaptive | **DONE** | Explicit `adaptive: False` in step-0_pipeline@preprocessing.yml |
| R. AR n_jobs | **DONE** | `n_jobs: 1` in step-0_pipeline@preprocessing.yml |
| S. Provenance richness | **DONE** | integrity_stats + by_block + config now in prov JSON |
| T. AR plot granularity | **open (minor)** | Combined vs per-chunk for long recordings |
| U. Abs power extra stats | **by design** | neurodags computes more (Med+IQR); original mean only |
| V. Run-aware aggregation | **open** | No recording_id grouping; post-hoc only |
| W. Band ratio floor guard | **DONE** | Both guard near-zero (eps vs 0.0 floor; equivalent) |
| X. Condition separation | **DONE** | `step-1_pipeline@extraction.yml` in-memory epoching via `CleanedPrep` (save: False) per condition |
| Y. ZapLine n_removed in prov | **open (minor)** | Not tracked; config is in code/ snapshot |
| Z. QC layer | **partial** | Complete failures: covered via `.error` + `neurodags status`. Intra-file NaN tracking + structured failure rows: missing; post-hoc scan needed |
| AA. BLOCK annotation injection | **DONE** | `inject_block_annotations` reads `_segments.csv`; original uses `read_raw_bids` → `_events.tsv` (§2.9) |
| AB. Condition epoch extraction | **DONE** | in-memory `extract_condition_epochs` per condition in step-1; BAD_epoch scoped correctly (§2.10) |
| AC. Cross-condition BAD_ bleeding | **DONE** | `extract_condition_epochs` remaps foreign BAD_ spans to SKIP_ before epoching |
