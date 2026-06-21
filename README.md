# EEG Analysis for ADHD, Epilepsy and Medication Effects

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the analysis code used for an EEG study of ADHD, epilepsy, and medication exposure. It brings together data organization, preprocessing, quality control, feature extraction, dimensionality reduction analysis, and visualization in a single codebase.

## Overview

The codebase currently includes:

- metadata and cohort-report tooling for tracking recruitment and clinical groups
- BIDS-oriented data handling, EEG preprocessing, and pre/post-clean QC utilities
- signal quality control and HTML reporting tools
- descriptor-based feature extraction using `coco-pipe.descriptors`
- dimensionality reduction analysis using `coco-pipe.dim_reduction`
- visualization and reporting utilities for exploratory analysis and result inspection using `coco-pipe.viz` and `coco-pipe.report`
- reusable foundation-model embeddings and grouped decoding with CBraMod, LaBraM, REVE, and LUNA

The active pipeline runs in this order: metadata → BIDS conversion/QC → preprocessing → epoching → descriptor extraction → dimensionality reduction. Cluster-ready SLURM scripts for each stage live in [cluster/](cluster/), numbered in pipeline order.

## Foundation Models and Decoding

The foundation and classical decoding entry points rely on `coco-pipe`; the
project only supplies study-specific loading, targets, BIDS paths, and reports:

```bash
eeg-foundation-embeddings --config configs/foundation_embeddings.example.yaml
eeg-decode --config configs/decoding.example.yaml
eeg-foundation-decode --config configs/foundation_decoding.example.yaml
```

Install the project with the `coco-pipe[decoding,foundation]` dependency from
`pyproject.toml`. Foundation backends require PyTorch and Braindecode 1.5 or
newer. Fine-tuning is practical on a CUDA GPU; CPU runs are intended for small
validation jobs. REVE is gated on Hugging Face: accept its model terms and set
`HF_TOKEN` before running it. Saved configs are redacted so tokens are not
written to derivatives or reports.

Each model declares its own EEG window requirements. The example configs use
10-second derivative epochs for CBraMod, REVE, and LUNA. MNE's inclusive final
sample is removed explicitly so these become exact half-open 2000-sample model
windows. LaBraM requires exactly 3000 samples at 200 Hz; because the current
clean cohort derivatives are 10 seconds, the example config explicitly skips
LaBraM cohort runs until 15-second clean derivatives are generated.
`window_mismatch_policy` is one of `error`, `skip`, or `re_epoch`; no workflow
silently pads or arbitrarily crops incompatible windows.

Embedding extraction writes BIDS-shaped derivatives with NPZ data, JSON
sidecars, a run manifest, dataset description, config provenance, and an
offline HTML summary. Classical and foundation decoding write fold predictions,
metrics, selected features or model artifacts, statistics, capability records,
and offline reports. Grouped CV uses `patient_group_id`, and scaling, reduction,
selection, and model fitting occur inside each training fold.

## Metadata Workflow

Metadata currently starts from two CSV files collected by students William and Jeanne: `EEG_Psychostimulants_PatientList_08-2025.csv` and `IRSC_data_final.csv`. The builder in [eeg_adhd_epilepsy/io/patients.py](eeg_adhd_epilepsy/io/patients.py) merges them into one canonical schema, applies the agreed cleaning rules, assigns a generated `patient_group_id` for repeated recordings from the same patient, and writes:

- `patients_metadata.csv`
- `patients_metadata_clean.csv`
- `patients_metadata_removed.json`

Rebuild with:

```bash
eeg-build-patients-metadata
```

The intended downstream entry point is `patients_metadata_clean.csv`.

## Cohort Report Workflow

The cohort report starts from `patients_metadata_clean.csv` and optionally reads `patients_metadata_removed.json` for provenance. The builder in [eeg_adhd_epilepsy/analysis/cohort.py](eeg_adhd_epilepsy/analysis/cohort.py) can:

- build the full clean-cohort report directly
- apply a cohort filter from a YAML file
- optionally add recruitment milestones with `--with_recruitment`

Run with:

```bash
eeg-cohort-report --metadata_csv /path/to/patients_metadata_clean.csv --output_dir /path/to/output
```

The main output is `cohort_report.html`, plus opportunity and recruitment CSVs when enabled.

## BIDS and Raw EEG QC Workflow

The raw EEG to BIDS conversion and QC is done by [eeg_adhd_epilepsy/preproc/to_bids.py](eeg_adhd_epilepsy/preproc/to_bids.py). It is the single stage that should be run before preprocessing (that is, `base.py`) or any work with the dataset, and now handles:

- raw recording discovery
- raw to BIDS conversion
- canonical annotation rewrite
- condition block derivation and sibling `_segments.csv` writing
- optional pre-base EEG reports
- optional pre-base raw-QC reports

Run with:

```bash
.venv/bin/python -m eeg_adhd_epilepsy.preproc.to_bids \
  --raw_root /path/to/raw_data \
  --bids_root /path/to/BIDS \
  --metadata_csv /path/to/patients_metadata_clean.csv \
  --with_eeg_reports \
  --with_raw_qc \
  --raw_qc_analysis_level both \
  --n_jobs 4
```

Use `--overwrite` only when you want to rebuild existing BIDS subject folders. Without `--overwrite`, the script skips existing runs and reconstructs the pre-base EEG-report and raw-QC payloads from the written BIDS files and `_segments.csv` files, so reruns can resume and still regenerate summary reports.

### Outputs

`to_bids.py` writes:

- BIDS EEG files under `BIDS/`
- sibling `_segments.csv` files next to each BIDS run
- reports under a `reports/` directory at the same level as `BIDS`.

Subject-level reports:

- EEG report (descriptive inventory, conditions, annotations, metadata-linked structure):
  - `reports/sub-XXXX/ses-01/eeg_pre_base/sub-XXXX_ses-01_eeg_pre_base_report.html`
- raw QC report (broad usability, noise burden, bad channels, segment-level QC figures):
  - `reports/sub-XXXX/ses-01/raw_qc_pre_base/sub-XXXX_ses-01_raw_qc_pre_base_report.html`

Dataset summary outputs:

- EEG:
  - `reports/summary/eeg_pre_base/`
- raw QC:
  - `reports/summary/raw_qc_pre_base/`

### Split of Responsibilities

- `to_bids.py` handles conversion, canonical annotations, and orchestration. It calls [eeg_adhd_epilepsy/reports/eeg_report.py](eeg_adhd_epilepsy/reports/eeg_report.py) to generate the descriptive EEG report, [eeg_adhd_epilepsy/qc/raw_qc.py](eeg_adhd_epilepsy/qc/raw_qc.py) to compute and aggregate broad raw-QC metrics, and [eeg_adhd_epilepsy/reports/raw_qc.py](eeg_adhd_epilepsy/reports/raw_qc.py) to render the raw-QC report.

## Preprocessing and Post-Clean QC Workflow

Once the BIDS conversion and raw QC are completed, you can run the primary preprocessing pipeline using `base.py`. This stage reads the raw BIDS files, applies automated cleaning, and outputs analysis-ready BIDS derivatives alongside post-preprocessing quality control (QC) reports.

The pipeline performs:
- Bandpass filtering (0.1–99.5 Hz default) and Line Noise removal using Adaptive ZapLine (`mne-denoise`)
- Bad channel detection using RANSAC (`pyprep`)
- Robust re-referencing (Common Average Reference)
- Condition-wise bad segment/epoch detection and annotation using `AutoReject`
- Comprehensive post-clean QC metric extraction and HTML report generation

Run with:

```bash
.venv/bin/python -m eeg_adhd_epilepsy.preproc.base \
  --bids_root /path/to/BIDS \
  --n_jobs 4
```

Like `to_bids.py`, use `--overwrite` to force reprocessing of files that have already been cleaned. Otherwise, the script acts incrementally and resumes work. To target specific subjects you can use the `--subjects` flag.

### Outputs

`base.py` writes:

- Preprocessed continuous EEG files (`_desc-base_eeg.fif`) under `BIDS/derivatives/eeg_adhd_epilepsy/`
- Subject-level post-clean QC reports:
  - `reports/sub-XXXX/ses-01/base_qc/sub-XXXX_ses-01_base_qc_report.html`
- Dataset-level summary outputs:
  - `reports/summary/base_qc/` containing aggregate CSVs, JSON records, and a master `base_qc_dataset_summary.html` report.

### Split of Responsibilities

- `base.py` handles the pipeline orchestration, BIDS loading/saving, and mapping the core denoising algorithms.
- `eeg_adhd_epilepsy/qc/preproc_qc.py` orchestrates the post-preprocessing QC validation. It explicitly delegates condition-level signal quality computation back to `qc/raw_qc.py`, guaranteeing direct comparability between the "Raw" and "Clean" stages.
- `eeg_adhd_epilepsy/reports/preproc_qc.py` is responsible for building the semantic blocks of the post-clean HTML reports, defining tables that compare retention metrics, residual artifact burdens, and Raw vs Cleaned signal quality summaries.
- `eeg_adhd_epilepsy/viz/preproc_qc.py` composites the comparative visual artifacts (grouped histograms, distribution tables, side-by-side Topomaps) that populate the HTML reports.

## Epoch Generation Workflow

Once continuous files are cleaned, the signals must be chunked into uniform epochs bounded to specific functional blocks (conditions). This is dynamically orchestrated by `epochs.py`. 

The script slices the `_desc-base_eeg.fif` files according to embedded condition annotations (e.g., `PHOTO_EC`, `EC_baseline`) and seamlessly outputs separated BIDS-nested derivative components.

Run with:

```bash
.venv/bin/python -m eeg_adhd_epilepsy.preproc.epochs \
  --bids_root /path/to/BIDS \
  --segment_duration 10.0 \
  --ignore_annotations
```

### Outputs
- Condition-specific chunked subsets organically placed alongside continuous parents:
  - `BIDS/derivatives/preproc/sub-XXXX/ses-XX/eeg/sub-XXXX_ses-XX_task-<condition>_run-XX_desc-base_epo.fif`

## Descriptor Extraction Workflow

With data epoched, `extract_descriptors.py` internally tracks the hierarchical `sub-XXXX/ses-XX/eeg` structure natively mapped by `coco-pipe`, computing high-dimensional feature banks with `coco-pipe.descriptors`. 

This pipeline strictly reads the configuration matrix from `configs/descriptors.yaml`, calculating three comprehensive families of signal characteristics:
1. **Spectral Bands:** Broad/narrowband absolute, relative, and log power distributions via Welch's PSD.
2. **Parametric Modeling:** Aperiodic and periodic isolation via SpecParam (formerly FOOOF).
3. **Complexity & Entropy:** Multi-dimensional non-linear dynamics including Permutation Entropy, Lempel-Ziv complexity, and Petrosian/Higuchi fractal dimensions.

Additionally, the pipeline aggregates spatial dimensions using region-based channel pooling mapped in the configuration. The extracted features are then robustly joined with clinical targets from `patients_metadata_clean.csv`.

Run with:

```bash
.venv/bin/python -m eeg_adhd_epilepsy.analysis.extract_descriptors \
  --bids_root /path/to/BIDS \
  --metadata /path/to/patients_metadata_clean.csv \
  --config configs/descriptors.yaml \
  --subject_col study_id \
  --target_col adhd \
  --conditions all
```

### Outputs
- Compiled descriptor shards strictly deposited in:
  - `BIDS/derivatives/signal_features/descriptors/sub-XXXX/ses-XX/eeg/*_desc-descriptors_eeg.parquet`

## Dimensionality Reduction Workflow

The dimensionality reduction entry point is [eeg_adhd_epilepsy/analysis/dimensionality_reduction.py](eeg_adhd_epilepsy/analysis/dimensionality_reduction.py). It is the main analysis script for exploring low-dimensional structure in raw EEG or extracted descriptors. It builds on `coco-pipe.dim_reduction` and supports multiple analysis styles, post-hoc evaluation targets, and report generation in one run.

Run with:

```bash
.venv/bin/python -m eeg_adhd_epilepsy.analysis.dimensionality_reduction \
  --bids_root /path/to/BIDS \
  --config /path/to/config.yaml \
  --metadata /path/to/patients_metadata_clean.csv \
  --input_mode raw \
  --analysis_mode flat \
  --representation subject_flat \
  --output_group example_group \
  --n_jobs 4
```

### Core Concepts

Three arguments define what the script loads and what one “analysis unit” means:

- `--input_mode`
  - `raw`: load EEG from BIDS or saved derivatives
  - `descriptors`: load merged descriptor tables
  - `foundation_embeddings`: load a model-specific embedding derivative
- `--analysis_mode`
  - `flat`: one embedding per condition or pooled dataset
  - `sensor`: one independent analysis per sensor or channel-group
  - `family`: one independent analysis per descriptor family
  - `subfamily`: one independent analysis per descriptor subfamily
  - `sensor_within_family`: one independent sensor analysis inside each family
  - `sensor_within_subfamily`: one sensor analysis inside each subfamily
  - `feature`: one descriptor feature across sensors
  - `feature_within_family`: one descriptor feature inside each family
  - `descriptor`: one descriptor, retaining all of its aggregation statistics
  - `descriptor_sensor`: one descriptor at one sensor
- `--representation`
  - `epoch_flat`: keep each epoch/window as one observation
  - `subject_flat`: average epochs within each subject, then flatten the feature space to `subjects x features`
  - `subject_native`: keep the native tensor layout after subject averaging; this is mainly used for raw sensor-wise analysis

The most common subject-level raw EEG setup is:

- `--input_mode raw`
- `--analysis_mode flat`
- `--representation subject_flat`

This means:

- one row per subject
- the feature space is the joint `sensor x time` space
- the reducers operate across subjects on that flattened subject-level representation

For descriptor analyses:

- `flat` compares the full selected descriptor space
- `sensor` compares sensors or channel-groups against each other
- `family` compares descriptor families against each other
- `subfamily`, `feature`, and `descriptor` expose progressively finer descriptor units
- `sensor_within_family` compares sensors inside each requested family
- `sensor_within_subfamily` and `descriptor_sensor` provide the corresponding fine-grained sensor analyses
- `--representation` is table-driven for descriptors: the script uses the descriptor table stem, such as `sensor_subject_features`, so separate tables get separate output/report variants
- descriptor rows with selected finite feature values above `--descriptor_max_abs_value` are dropped before fitting; the default is `1e12`

### Evals, Selection, Reports and Outputs

- The script reports both fit metrics from the reducer itself (`trustworthiness`, `continuity`, `shepard_correlation`) and post-hoc eval metrics from user-defined specs (e.g. logistic-regression balanced accuracy on clinical/demographic targets like `med_adhd_vs_ctrl`, `sex_separation`, `age_separation`, `condition_separation`).
- Best-fit selection is driven by `selection_metric` (e.g. `separation_logreg_balanced_accuracy`) and, when you want fits ranked on one specific target, `selection_eval_name`. Set both explicitly for any serious comparison study, along with `output_group`, to keep report selection unambiguous and run variants separated on disk.
- Outputs are separated by analysis variant under `BIDS/derivatives/dim_reduction/<output_group>/<dataset_name>/<analysis_mode>_<input_mode>_<representation-label>_cfg-<hash>/`. The configuration hash isolates cohorts, QC, filters, reducers, evaluation targets, and model-specific embedding spaces. Raw labels drop duplicated layout suffixes (`epoch_flat` -> `epoch`, `recording_flat` -> `recording`); `subject_*` variants aggregated by recording add an explicit suffix.
- Each run root contains invocation-scoped `runs/fit_runs.json`, `runs/eval_runs.json`, `runs/run_summary.json`, checkpoint artifacts under `artifacts/fits/` and `artifacts/evals/`, and a terminal marker (`_RUN_SUCCESS`, `_RUN_PARTIAL`, or `_RUN_FAILED`). Matching reports use the same hashed variant under `reports/summary/dim_reduction/.../<dataset_name>/`. Re-running rebuilds the inventories from reusable checkpoint artifacts, preventing stale experiments from entering the current report.

### Parallelism and Practical Notes

- `--n_jobs` controls outer-task parallelism across independent fit/eval tasks (`1` = serial, `>1` = that many workers, `-1` = all CPUs). It does not parallelize inner CV folds, so moderate values (start with `4`–`6`) tend to work best for eval-heavy or large sensor runs — increase only after checking memory/CPU behavior.
- `subject_flat` averages epochs within subject — use it for one observation per subject. PCA-like reducers remain bounded by `min(n_samples, n_features)`, so small subject-level cohorts can't request large `n_components`.
- Raw-sensor topomaps use the standard 10-20 montage and need sensor names that match valid EEG channel labels. For descriptor sensor analyses, "sensor" depends on the table: per-channel tables give true electrodes (`Fz`), pooled tables give grouped regions (`front_left`).

## Installation

Python `3.10+` is required.

Clone the repository:

```bash
git clone https://github.com/your-username/eeg-analysis-adhd-epilepsy.git
cd eeg-analysis-adhd-epilepsy
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the project in editable mode:

```bash
pip install -e .
```

If you are actively editing `coco-pipe` itself, you can override the pinned dependency with a local checkout in the same environment:

```bash
pip install -e '/Users/hamzaabdelhedi/Projects/packages/coco-pipe[descriptors]'
```

## Repository Layout

```text
.
├── eeg_adhd_epilepsy/     # Main Python package (preproc, qc, analysis, viz, reports, io, utils, signal_quality)
├── configs/               # YAML configs for descriptors, annotations, dimensionality reduction
├── cluster/               # Numbered SLURM submission scripts, one per pipeline stage
├── data/                  # Raw and BIDS-converted EEG data (not tracked in git)
├── results/               # Generated analysis outputs (not tracked in git)
├── reports/               # Generated HTML QC and summary reports (not tracked in git)
├── tests/                 # Automated tests
├── pyproject.toml         # Project metadata, dependencies, and CLI entry points
├── LICENSE                # Project license
└── README.md              # Project overview
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
