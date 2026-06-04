# EEG Pipeline Ecosystem Report
# Old Script-Based vs neurodags-Powered — Commonalities, Gaps, and Strategic Recommendations

**Date:** 2026-05-28  
**Scope:** preprocessing, feature extraction, ML boundary, portability across datasets/projects  
**Packages reviewed:** `eeg_adhd_epilepsy/`, `neurodags_pipelines/`, `~/code/neurodags`, `~/code/coco-pipe`

---

## 1. What Each Package Does (Layer Map)

```
┌─────────────────────────────────────────────────────────────────┐
│  ML / Analysis                                                  │
│  coco_pipe.decoding    — sklearn + FM (REVE, CbraMod), group CV │
│  coco_pipe.dim_reduction — PCA, manifold, topology              │
├─────────────────────────────────────────────────────────────────┤
│  Feature Extraction                                             │
│  OLD: eeg_adhd_epilepsy/analysis/extract_descriptors.py         │
│       wraps coco_pipe.descriptors (PSD, FOOOF, complexity)      │
│  NEW: neurodags step-1 YAML + built-in nodes                    │
│       neurodags.nodes.spectral / antropy / neurokit / factories  │
│       coco_pipe.descriptors used internally by those nodes       │
├─────────────────────────────────────────────────────────────────┤
│  Preprocessing                                                  │
│  OLD: eeg_adhd_epilepsy/preproc/ (base/correct/denoise, ~2700 L)│
│  NEW: neurodags_pipelines/ YAML + nodes_*.py (~1700 L)          │
├─────────────────────────────────────────────────────────────────┤
│  Orchestration / Framework                                      │
│  OLD: joblib.Parallel + manual skip-if-exists                   │
│  NEW: neurodags DAG (caching, deps, YAML, dataframe assembly)   │
├─────────────────────────────────────────────────────────────────┤
│  QC / Reporting (shared layer)                                  │
│  eeg_adhd_epilepsy/qc/ + eeg_adhd_epilepsy/reports/            │
│  coco_pipe.report (HTML engine, quality checks, provenance)     │
│  neurodags_pipelines/nodes_qc.py calls the above               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Feature Parity: Old vs New

### 2.1 Preprocessing

| Step | Old (`preproc/base.py`) | New (`step-0_pipeline`) | Status |
|------|------------------------|------------------------|--------|
| Block annotation injection | `read_raw_bids` → `_events.tsv` | `inject_block_annotations` → `_segments.csv` | ✓ equivalent |
| Annotation inflation | `inflate_bad_annotations` | same node | ✓ identical |
| Resample → filter | ✓ (resample-first) | ✓ (`resample_first: True`) | ✓ identical |
| ZapLine 60 Hz | ✓ | ✓ | ✓ identical |
| RANSAC bad channels (EC only) | ✓ | ✓ | ✓ identical |
| CAR | ✓ | ✓ | ✓ identical |
| Condition-grouped AutoReject | ✓ (1s epochs, 30-min chunks) | ✓ (same params) | ✓ identical |
| ICA correction | ✓ DSS (EOG/ECG) + MWF (EMG), adaptive | ⚠ basic `find_bads_eog`/`find_bads_ecg`; no EMG; no adaptive tuning | **method differs — see COMPARISON.md §2.11** |
| Wiener residual denoise | ✓ | ✓ | ✓ fixed crash (channel positions) |
| Channel position loading | ✗ NaN — silent failure | ✓ loads `_electrodes.tsv` | **bug fixed** |
| Multi-run per-subject | merged (wrong) | per-run (correct) | **bug fixed** |
| Incremental re-run | manual skip logic | neurodags cache | **improved** |

### 2.2 Feature Extraction

| Feature family | Old (coco-pipe `DescriptorPipeline`) | New (neurodags built-in nodes) | Status |
|----------------|--------------------------------------|-------------------------------|--------|
| Band power (abs/log/rel/corr) | ✓ `BandDescriptorExtractor` | ✓ `spectral.py` nodes | ✓ |
| FOOOF / aperiodic | ✓ `ParametricDescriptorExtractor` | ✓ `spectral.py` FOOOF nodes | ✓ |
| Band ratios | ✓ | ✓ `BandRatios*` | ✓ |
| Spatial pooling (9 regions) | ✓ | ✓ `nodes_spatial.py` | ✓ |
| Entropy (sample/app/perm/SVD/spectral/fuzzy/dispersion/shannon) | ✓ `ComplexityDescriptorExtractor` | ✓ `antropy.py` + `neurokit.py` | ✓ |
| Higuchi/Katz/Petrosian FD | ✓ | ✓ | ✓ |
| Hurst, LZiv, NumZeroCross, RMS, kurtosis | ✓ | ✓ | ✓ |
| EntropyMultiscale | ✓ | ⚠ NumPy 2.0 incompatibility in neurokit2 | known |
| PSD shared across consumers | ✓ `_PSDGroup` shares PSD between `BandDescriptorExtractor` + `ParametricDescriptorExtractor` (band power + FOOOF); `ComplexityDescriptorExtractor` recomputes internally | ✓ `SpectrumWelch` derivative shared between band power + FOOOF consumers; entropy/complexity nodes recompute | ✓ equivalent |
| Output format | `.parquet` + `.csv` per condition | `.nc` (xarray NetCDF) per derivative | different |
| Epoch-level output | ✓ `sensor_epoch_features.csv` | ✓ one row per epoch in `.nc` | ✓ |
| Subject-level aggregation | ✓ in script, mean/median/IQR | ✓ `aggregate_across_dimension` node | ✓ |
| Failures log | ✓ structured `failures.csv` per condition+family | ✓ `neurodags status --list-errors/--list-missing` covers file-level; gap: no intra-file partial NaN log within successful `.nc` | narrow gap |
| `_SUCCESS` checkpoints | ✓ per condition | ✓ equivalent (neurodags cache file = skip) | ✓ |
| Run-aware aggregation | ✓ `recording_id = sub+ses+run` | post-hoc groupby required | minor gap |

### 2.3 ML / Decoding

| Feature | Old (`run_ml_pipe.py` + coco-pipe) | New (none yet) | Status |
|---------|-----------------------------------|----------------|--------|
| Classification / regression | ✓ | — | not ported |
| Group CV (LOGO, LGKO) | ✓ | — | not ported |
| FM hub (REVE, CbraMod) | ✓ | — | not ported |
| Feature selection (ANOVA, RFECV) | ✓ | — | not ported |
| Hyperparameter tuning | ✓ | — | not ported |
| Dim reduction (PCA, UMAP, topology) | ✓ | — | not ported |
| HTML experiment reports | ✓ (`coco_pipe.report`) | — | not ported |

ML is explicitly **out of scope** for neurodags and the neurodags pipeline. It lives in coco-pipe and is fed by the CSV assembled from neurodags output.

### 2.4 QC / Reporting

| Feature | Old | New | Status |
|---------|-----|-----|--------|
| Per-subject HTML report (base/correct/denoise) | ✓ | ✓ | ✓ |
| Per-run reports | one merged report (wrong for multi-run) | one per run (correct) | improved |
| Subject-level aggregated report (Per-Run Summary) | ✓ | not in neurodags (see §5.1 note) | intentional gap |
| Dataset-level summary reports | ✓ | descriptor QC: ✓ via `merge_descriptors.py` (post-pipeline); preprocessing summary: open | partial |
| Condition segment retention | ✗ missing | ✓ | improved |
| Raw Duration display | ✗ always `0s` | ✓ | bug fixed |
| Retained Duration accuracy | ✗ inflated by ch-BAD marks | ✓ | bug fixed |
| Descriptor QC HTML | ✓ all conditions | ✓ all 8 conditions active in `step-1_dataset.yml`; run pipeline once | ✓ |

---

## 3. Advantages and Disadvantages

### Old pipeline (`eeg_adhd_epilepsy/preproc/` + `extract_descriptors.py`)

**Advantages:**
- Single Python package — `pip install .` and every step accessible
- Mature: multi-run aggregation, full QC pipeline, dataset summaries, ML integration all present
- `coco_pipe.descriptors.DescriptorPipeline` (`_PSDGroup`) shares PSD between band power (`BandDescriptorExtractor`) and FOOOF (`ParametricDescriptorExtractor`); complexity recomputes PSD internally (not a `BasePSDDescriptorExtractor`)
- Structured failure logging per condition + feature family
- Run-aware grouping (`recording_id`) built into output CSV schema

**Disadvantages:**
- Monolithic scripts (base.py 1034 L, correct.py 900 L, denoise.py 747 L) — hard to modify without wide blast radius
- No automatic caching — re-runs require manual skip logic or deleting outputs
- Channel positions never loaded → RANSAC/AR silently broken on all subjects (critical, silent)
- `_annotation_intervals` bug collapsed retained_duration to ~3s on affected subjects
- Multi-run subjects merged (wrong)
- Portability requires Python code changes (paths, params all hardcoded)
- Parallelism via embedded `joblib.Parallel` — no cluster integration

### New pipeline (`neurodags_pipelines/` + neurodags)

**Advantages:**
- YAML-driven: portability to new datasets = edit `step-0_dataset.yml`
- Automatic caching and incremental recomputation
- Preprocessing bugs fixed (channel positions, retention, annotations, n_components clamp)
- Per-run reports (correct multi-run handling)
- Node functions are small, focused, independently testable
- `neurodags dataframe` assembles per-file outputs into one CSV in a single command
- xarray output carries coordinates (channel, band, epoch, etc.) — richer than flat CSV
- Adding a new step = one node function + one YAML entry

**Disadvantages:**
- No cross-file operations mid-pipeline — subject-level aggregation (e.g., group ICA) requires post-processing outside the framework
- Caching is existence-based — code changes don't invalidate cache; manual `overwrite: true` required
- No structured failure log for intra-derivative partial NaN (only `.error` on complete failure; `neurodags status --list-errors` covers file-level failures)
- PSD shared via `SpectrumWelch` derivative for band power + FOOOF consumers (equivalent to coco-pipe `_PSDGroup`); entropy/complexity recompute internally in both pipelines
- No ML layer — neurodags stops at features
- Dataset summary reports not yet implemented

---

## 4. Portability Assessment: New Dataset / New Project

### What is dataset-specific vs reusable today

| Component | Dataset-specific? | Cost to port |
|-----------|------------------|-------------|
| `step-0_dataset.yml` | Yes | Edit 16 lines (paths, subjects) |
| `step-0_pipeline@preprocessing.yml` | Mostly no | Adjust `annotation_prefix`, `line_freq`, `resample`, `l_freq`, `h_freq` via YAML args |
| `inject_block_annotations` node | Partially | Hardcoded to `_segments.csv` format; needs adapting for datasets without this sidecar |
| `nodes_qc.py` | No | Fully reusable |
| `nodes_autoreject.py`, `nodes_ica.py`, `nodes_preprocessing.py` | No | Pure MNE, no dataset-specific code |
| `step-1_pipeline@extraction.yml` | Mostly no | Change condition names, descriptor list |
| `eeg_adhd_epilepsy/io/bids.py` | Yes | BIDS helpers tied to this dataset's file layout conventions |
| `eeg_adhd_epilepsy/qc/preproc_qc.py` | No | Fully reusable |
| `coco_pipe.descriptors` | No | Dataset-agnostic |
| `coco_pipe.decoding` | No | Dataset-agnostic |

The bottleneck for porting is `inject_block_annotations` (annotation source varies per dataset) and `eeg_adhd_epilepsy/io/bids.py` (BIDS helpers with project-specific assumptions). Everything else is already generic.

### Recommended template structure for a new cocolab project

```
new-project/
  pipelines/
    step-0_pipeline@preprocessing.yml   # copy + adjust line_freq, resample, annotation_prefix
    step-0_dataset.yml                   # new dataset paths
    step-1_pipeline@extraction.yml       # copy + adjust condition names
    step-1_dataset.yml
    nodes_annotations.py                 # adapt inject_block_annotations for new format
    nodes_preprocessing.py              # copy as-is (no changes needed)
    nodes_autoreject.py                 # copy as-is
    nodes_ica.py                        # copy as-is
    nodes_qc.py                         # copy as-is
  analysis/
    run_ml.py                           # thin script: load CSV → coco-pipe Experiment → save
    ml_config.yml
  configs/
    descriptors.yaml
```

Three files need actual editing: `step-0_dataset.yml`, `nodes_annotations.py` (if annotation format differs), and `ml_config.yml`. Everything else is reused unchanged.

---

## 5. Software Boundary: Preproc+Extraction vs ML — Separate or Unified?

### Option A: Single package (everything together)

```python
from eeg_pipeline import run_all
run_all("dataset/", stages=["preproc", "extract", "ml"])
```

**Pros:**
- One install, one command
- Shared config

**Cons:**
- Heavy preprocessing deps (MNE, RANSAC, AutoReject, ZapLine) forced on ML users
- Heavy ML deps (torch, sklearn, foundation models) forced on preprocessing users
- Preprocessing runs once; ML runs dozens of times with different configs
- Changes to ML invalidate nothing in preprocessing but require shipping the whole package
- Testing is harder — unit-testing ML needs preprocessing infrastructure
- Tight coupling means any bug anywhere breaks everything
- Scaling preprocessing (to 200 subjects on a cluster) and scaling ML (GPU nodes) have completely different resource profiles

**Verdict:** poor fit for this ecosystem.

### Option B: Template script integrating separate packages (recommended)

```bash
# Step 1: preprocess (once, heavy deps, CPU cluster)
neurodags run pipelines/step-0_pipeline@preprocessing.yml

# Step 2: extract features (all conditions in one run, CPU)
neurodags run pipelines/step-1_pipeline@extraction.yml
neurodags dataframe pipelines/step-1_pipeline@extraction.yml --output features/all_conditions.csv

# Step 3: ML (many times, GPU optional, quick iteration)
# Filter to target condition via `dataset` column or --datasets flag
python analysis/run_ml.py --features features/all_conditions.csv --config configs/ml_config.yml
```

Or wrapped as a single Makefile target / shell script for "one command" convenience:
```bash
make run-all DATASET=my_dataset CONDITION=EO_baseline
```

**Pros:**
- Separation of concerns: preprocessing deps never imported in ML environment
- Preprocessing cached by neurodags — ML re-runs don't retrigger preprocessing
- coco-pipe can be updated (new models, metrics) without touching the preprocessing pipeline
- neurodags can be updated (bug fixes) without touching ML
- Each package has its own tests and versioning
- ML iteration is fast (seconds to load CSV and fit a model)
- Preprocessing is parallelized via neurodags; ML is parallelized within coco-pipe
- New projects: copy the project template, change 3 files, done

**Cons:**
- Need to manage `requirements.txt` / `pyproject.toml` listing both packages
- Integration point (CSV format) must stay stable — a schema change in neurodags output breaks `run_ml.py`
- No single version pin guarantees end-to-end reproducibility (need to pin neurodags + coco-pipe versions together)

**Mitigation for the schema stability concern:**
`neurodags dataframe` output schema is stable (BIDS entity columns + one column per descriptor). `coco_pipe.io.load_data` accepts generic CSV. The integration contract is thin and unlikely to break.

### Recommended integration contract

```
neurodags dataframe output:
  subject, session, run, condition, [descriptor columns...]

coco_pipe.io.load_data(path, target_col="label", ...)
  → DataContainer(X, y, groups)

coco_pipe.decoding.Experiment(config).fit(X, y, groups)
```

Three lines of glue code. The boundary is clear and stable.

---

## 6. Recommendation: Best Architecture for cocolab Ecosystem

### Package roles (current + recommended)

| Package | Role | Scope |
|---------|------|-------|
| **neurodags** | Orchestration + built-in EEG nodes | Preprocessing DAG, feature extraction DAG, dataframe assembly |
| **coco-pipe** | Signal processing + ML library | Descriptors, decoding, dim reduction, IO, reports |
| **eeg_adhd_epilepsy** | Project glue + QC | BIDS I/O helpers, QC framework, subject-level reports |
| **project template** | Thin YAML + launch scripts | Dataset-specific config, custom nodes if needed |

### What to keep in eeg_adhd_epilepsy vs generalize

The `eeg_adhd_epilepsy/qc/preproc_qc.py` + `eeg_adhd_epilepsy/reports/preproc_qc.py` QC layer is already nearly dataset-agnostic. The only project-specific parts are BIDS path helpers in `io/bids.py`. **These should be generalized into coco-pipe or a separate `coco-preproc` package** so new projects don't need to depend on this repo.

The `inject_block_annotations` node is the main dataset-specific piece. For the project template, this node should be written with a documented interface: "given raw + a CSV of condition windows with `segment_type`, `t_start`, `t_stop` columns, inject BLOCK_* annotations." Any dataset that provides such a CSV can reuse it.

### Migration path for new projects

1. Copy the `neurodags_pipelines/` directory as a template
2. Edit `step-0_dataset.yml` (16 lines): new BIDS root, new subject list
3. If annotation format differs: adapt `inject_block_annotations` in `nodes_annotations.py`
4. Run `neurodags run step-0_pipeline@preprocessing.yml`
5. Run `neurodags run step-1_pipeline@extraction.yml` (all conditions in one run)
6. Run `neurodags dataframe` to assemble CSV
7. Run `python analysis/run_ml.py --features features.csv --config ml_config.yml`

Steps 4–7 are identical across all projects. Steps 1–3 are the only new-project work.

### One command to run everything?

A Makefile or shell wrapper is the right tool — not a monolithic Python package:

```makefile
# Makefile
CONDITION ?= EO_baseline

run-all:
	neurodags run pipelines/step-0_pipeline@preprocessing.yml
	neurodags run pipelines/step-1_pipeline@extraction.yml
	neurodags dataframe pipelines/step-1_pipeline@extraction.yml \
	    --output features/all_conditions.csv
	python analysis/run_ml.py \
	    --features features/all_conditions.csv \
	    --condition $(CONDITION) \
	    --config configs/ml_config.yml
```

```bash
make run-all CONDITION=EO_baseline
```

This gives "one command runs everything" without collapsing the software boundary.

---

## 7. Summary

| Dimension | Old pipeline | New pipeline | Winner |
|-----------|-------------|-------------|--------|
| Preprocessing correctness | ✗ channel positions NaN, retention wrong | ✓ all bugs fixed | **New** |
| Preprocessing portability | Python code changes | YAML edit | **New** |
| Preprocessing caching | manual | automatic | **New** |
| Feature extraction completeness | ✓ all conditions run | ✓ all 8 active in `step-1_dataset.yml`; one run covers all | ✓ equivalent |
| Feature extraction shared PSD | ✓ `_PSDGroup` (band power + FOOOF only; complexity always recomputes) | ✓ `SpectrumWelch` (band power + FOOOF; complexity recomputes) | ✓ equivalent |
| Failure logging | ✓ structured `failures.csv` | `neurodags status --list-errors`; gap: intra-file partial NaN only | narrow gap |
| ML pipeline | ✓ full (sklearn + FM hub) | ✗ not present | Old (intended) |
| Dataset summary QC reports | ✓ | ✗ not yet | Old |
| Portability to new datasets | hard (Python changes) | easy (YAML edit) | **New** |
| Extensibility (new step) | 1000+ line edits | one function + YAML | **New** |
| Collaboration (merge conflicts) | frequent (monolithic) | rare (independent nodes) | **New** |
| Software boundary (preproc vs ML) | blurred (same package) | clean separation | **New** |
| Cluster scaling | manual SLURM scripts | `neurodags run` | **New** |

**Overall recommendation:** use the neurodags-powered pipeline as the standard for preprocessing and extraction in new cocolab projects. Keep coco-pipe as the ML library. The integration is three lines of glue code (`load_data` → `DataContainer` → `Experiment`). A project template with a Makefile gives "one command" convenience without collapsing the boundary.
