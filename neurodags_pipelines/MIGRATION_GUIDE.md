# Migration Guide: `base.py` + `extract_descriptors.py` → neurodags

*Last updated: 2026-05-21.*

This guide is for researchers who know the original pipeline and want to understand the neurodags equivalent. For a full equivalence audit see `COMPARISON.md`.

---

## 1. Conceptual mapping

| Original concept | neurodags concept |
|---|---|
| Python script you call directly | YAML pipeline file (`step-X.yml`) |
| In-memory `Raw` / `Epochs` objects | `.fif` derivative files on disk |
| CSV outputs per condition | `.nc` (NetCDF/xarray) per derivative, assembled into CSV via `neurodags dataframe` |
| Script arguments (`--subject`, `--condition`) | `datasets.yml` file (glob patterns) |
| Provenance dict in memory | `@DerivativeName_prov.json` on disk |

The key shift: neurodags is **file-in, file-out**. Each step reads its input derivative from disk and writes its output derivative to disk. The DAG is declared in YAML; Python only defines custom computation nodes.

---

## 2. Run sequence

```bash
# 0. Generate synthetic test data (one-time)
python neurodags_pipelines/generate_synthetic.py

# 1. Preprocessing — equivalent to run_base_pipeline()
neurodags run neurodags_pipelines/step-0b_preproc_cleaned.yml

# 2. Condition epoch extraction — equivalent to loading clean raw and epoching per condition
neurodags run neurodags_pipelines/step-0c_conditions.yml

# 3. Feature extraction — equivalent to extract_descriptors.py
#    Default: all epochs regardless of condition
neurodags run neurodags_pipelines/step-1_features.yml

#    Per-condition (mirrors --conditions EO EC split):
neurodags run neurodags_pipelines/step-1_features.yml \
    -d neurodags_pipelines/step-1_datasets_conditions.yml
```

Steps 1–3 are idempotent (`overwrite: False`). Re-running skips already-computed files.

---

## 3. Output file layout

### 3.1 Preprocessing derivatives

Original: in-memory `Raw` / `Epochs` objects.

neurodags: written to `derivatives/preprocessing/<subject>/eeg/`.

```
derivatives/preprocessing/sub-0/eeg/
  sub-0_run-0_eeg.vhdr@CleanedPrepRaw.fif      ← fully annotated continuous Raw
  sub-0_run-0_eeg.vhdr@CleanedPrepRaw_prov.json ← provenance (AR stats, config)
  sub-0_run-0_eeg.vhdr@CleanedPrepRaw_ar_plot_EO.png
  sub-0_run-0_eeg.vhdr@CleanedPrepRaw_ar_plot_EC.png
  sub-0_run-0_eeg.vhdr@CleanedPrep.fif          ← fixed-length 2 s epochs, all conditions
  sub-0_run-0_eeg.vhdr@ConditionEO.fif          ← 2 s epochs within BLOCK_EO windows
  sub-0_run-0_eeg.vhdr@ConditionEC.fif          ← 2 s epochs within BLOCK_EC windows
```

**Naming convention**: `{source_basename}@{DerivativeName}.{ext}`

The `@` separates the source file identity from the derivative name. This means two runs from the same source file (`sub-0_run-0_eeg.vhdr`) have distinct filenames — no collision.

### 3.2 Feature derivatives

Original: `sensor_epoch_features.csv`, `sensor_subject_features.csv`, one directory per condition.

neurodags: one `.nc` (NetCDF) file per feature family, stored at the **dataset level** (not per subject). Each `.nc` covers all subjects × runs × epochs for that feature.

```
derivatives/features/
  features@AbsBandPower.nc          ← dims: (epochs, channels, freqbands)
  features@AbsBandPowerAgg.nc       ← dims: (channels, freqbands) — epoch-mean
  features@AbsBandPowerPooled.nc    ← dims: (epochs, regions, freqbands)
  features@FooofFit.nc
  features@SampleEntropy.nc
  ...

derivatives/features_conditions_eo/ ← per-condition run (step-1_datasets_conditions.yml)
  features@AbsBandPower.nc          ← EO epochs only
  ...

derivatives/features_conditions_ec/
  features@AbsBandPower.nc          ← EC epochs only
  ...
```

### 3.3 Feature xarray structure

Each `.nc` is an `xr.DataArray`. Typical dims for epoch-level features:

```
AbsBandPower     : (epochs=N, spaces=8_channels, freqbands=5)
AbsBandPowerAgg  : (spaces=8, freqbands=5)          ← epoch aggregate
SampleEntropy    : (epochs=N, spaces=8)
FooofFit         : stored as Dataset with multiple variables
```

Load and inspect:
```python
import xarray as xr
da = xr.open_dataarray("derivatives/features@AbsBandPower.nc")
df = da.to_dataframe(name="value").reset_index()
# columns: epochs, spaces, freqbands, freqband_low, freqband_high, value
```

The `epochs` dimension index does **not** embed subject/run identity by default. Use `neurodags dataframe` (§4) to get subject/run columns.

---

## 4. Assembling the flat CSV

Original: `extract_descriptors.py` writes `sensor_subject_features.csv` directly.

neurodags: assemble post-hoc with `neurodags dataframe`.

```bash
# Wide format — one row per file, one column per feature
neurodags dataframe neurodags_pipelines/step-1_features.yml \
    --format wide \
    --output results/features_wide.csv

# Long format
neurodags dataframe neurodags_pipelines/step-1_features.yml \
    --format long \
    --output results/features_long.csv

# Per-condition
neurodags dataframe neurodags_pipelines/step-1_features.yml \
    -d neurodags_pipelines/step-1_datasets_conditions.yml \
    --format wide \
    --output results/features_eo_wide.csv
```

The assembled CSV has one row per source file (= one run). Subject/run identity comes from the source filename parsed by neurodags's BIDS-aware index.

---

## 5. Step-by-step preprocessing equivalence

### Original `run_base_pipeline()` order

```python
inflate_bad_annotations(raw)
raw.resample(target_sfreq)          # resample FIRST
raw.filter(0.1, min(100, nyquist))
ZapLine(...).fit_transform(raw)
NoisyChannels(raw, ...).find_bad_by_ransac(...)  # EC windows only
raw.set_eeg_reference("average")
annotate_artifacts_blockwise(raw)   # per-condition AR
```

### neurodags `step-0b_preproc_cleaned.yml` order

```yaml
id.1 inflate_bad_annotations
id.2 preprocess_raw          # resample_first: True → resample then bandpass
id.3 zapline_denoise
id.4 ransac_bad_channels     # block_label: EC
id.5 apply_car
id.6 autoreject_annotate_blockwise
→ writes @CleanedPrepRaw.fif + _prov.json + _ar_plot_{cond}.png
```

Identical order. `resample_first: True` matches the original's resample-before-filter order.

---

## 6. Provenance

Original: `prov.json` built in `annotate_artifacts_blockwise`.

neurodags: `@CleanedPrepRaw_prov.json` — same fields, same structure.

```json
{
  "bad_channels": [],
  "config": {
    "annotation_prefix": "BLOCK_",
    "segment_duration": 1.0,
    "n_interpolate": [0],
    "min_epochs": 5,
    "ar_max_chunk_minutes": 30.0,
    "n_jobs": 1
  },
  "artifact_stats": {
    "bad_epochs": 0,
    "bad_channel_spans": 2,
    "artifacts_count": 2,
    "by_block": [
      {"condition": "EO", "n_windows": 2, "n_epochs": 12, "n_bad_epochs": 0,
       "n_bad_channel_spans": 0, "chunks_processed": 1, "clean_fraction": 1.0},
      {"condition": "EC", ...}
    ]
  },
  "integrity_stats": {
    "clean_duration_s": 30.0,
    "clean_fraction": 1.0,
    "manual_bad_fraction": 0.0,
    "autoreject_bad_fraction": 0.0
  },
  "overall_clean_fraction": 1.0
}
```

Also, on every `neurodags run`, the pipeline YAML + `custom_nodes.py` + datasets YAML are snapshotted into `derivatives/<step>/code/` — equivalent to storing the exact config that produced each derivative.

---

## 7. Monitoring pipeline status

```bash
# How many files done / missing / errored per derivative?
neurodags status neurodags_pipelines/step-0b_preproc_cleaned.yml

# Show which files errored
neurodags status neurodags_pipelines/step-1_features.yml --list-errors

# Per derivative
neurodags status neurodags_pipelines/step-1_features.yml \
    --derivative SpectralEntropy --derivative SampleEntropy
```

Errored files get a `.error` marker alongside the expected output path:
```
derivatives/features@EntropyMultiscale.error   ← NumPy 2.0 issue
```

---

## 8. Known gaps vs original

| Gap | Severity | Workaround |
|-----|----------|------------|
| QC CSVs (failures.csv, feature_missingness.csv, flags.csv) | Significant | Use `--list-errors`; add post-hoc checks |
| Per-epoch condition column in default run | Workflow | Use `-d step-1_datasets_conditions.yml` for split output |
| Run-aware aggregation (`recording_id = sub_ses_run`) | Minor | Post-hoc `groupby` on assembled CSV |
| ZapLine `n_removed` not in provenance | Minor | Config snapshot in `code/` has method/params |
| AR plot per chunk (vs combined) | Minor | Combined plot per condition is produced |
| Float32 on CleanedPrepRaw round-trip | Negligible | Below EEG noise floor (~1 µV) |

See `COMPARISON.md` for full details on each gap.

---

## 9. Quick reference: original column → neurodags derivative

| Original CSV column prefix | neurodags derivative | xarray dims |
|---|---|---|
| `band_abs_{band}_{ch}` | `AbsBandPower` | (epochs, spaces, freqbands) |
| `band_log_{band}_{ch}` | `LogBandPower` | (epochs, spaces, freqbands) |
| `band_rel_{band}_{ch}` | `RelBandPower` | (epochs, spaces, freqbands) |
| `band_ratio_{pair}_{ch}` | `BandRatios` | (epochs, spaces, pairs) |
| `agg_band_ratio_{pair}_{ch}` | `BandRatiosOnMeans` | (spaces, pairs) |
| `param_exponent_{ch}` | `FooofExponent` | (epochs, spaces) |
| `param_offset_{ch}` | `FooofOffset` | (epochs, spaces) |
| `complexity_sample_entropy_{ch}` | `SampleEntropy` | (epochs, spaces) |
| `complexity_spectral_entropy_{ch}` | `SpectralEntropy` | (epochs, spaces) |
| `complexity_hjorth_mobility_{ch}` | `HjorthParams` (mobility dim) | (epochs, spaces, params) |
| `pooled_{region}_{feature}` | `*Pooled` derivatives | (epochs, regions, ...) |

For the complete mapping see §3 of `COMPARISON.md`.
