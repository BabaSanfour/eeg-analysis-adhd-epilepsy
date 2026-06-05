# EEG Analysis for ADHD, Epilepsy and Medication Effects

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the analysis code used for an EEG study of ADHD, epilepsy, and medication exposure. It brings together data organization, preprocessing, quality control, feature extraction, modeling, and visualization in a single codebase.

## Overview

The codebase currently includes:

- BIDS-oriented data handling and EEG preprocessing utilities
- neurodags-based preprocessing and feature extraction pipeline (`neurodags_pipelines/`) — the current recommended approach
- signal quality control and reporting tools
- descriptor-based feature extraction using `coco-pipe.descriptors`
- dimensionality reduction analysis using `coco-pipe.dim_reduction`
- machine learning and deep learning analysis modules using `coco-pipe.decoding`
- visualization utilities for exploratory analysis and result inspection using `coco-pipe.viz` and `coco-pipe.report`

## Metadata Workflow

Metadata currently starts from two CSV files collected by students William and Jeanne: `EEG_Psychostimulants_PatientList_08-2025.csv` and `IRSC_data_final.csv`. The builder in [eeg_adhd_epilepsy/qc/metadata.py](eeg_adhd_epilepsy/qc/metadata.py) merges them into one canonical schema, applies the agreed cleaning rules, assigns a generated `patient_group_id` for repeated recordings from the same patient, and writes:

- `patients_metadata.csv`
- `patients_metadata_clean.csv`
- `patients_metadata_removed.json`

Rebuild with:

```bash
eeg-build-patients-metadata
```

The intended downstream entry point is `patients_metadata_clean.csv`.

## Cohort Report Workflow

The cohort report starts from `patients_metadata_clean.csv` and optionally reads `patients_metadata_removed.json` for provenance. The builder in [eeg_adhd_epilepsy/qc/cohort_report.py](eeg_adhd_epilepsy/qc/cohort_report.py) can:

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

- `to_bids.py` handles conversion, canonical annotations, `_segments.csv`, and orchestration. It calls [eeg_adhd_epilepsy/reports/eeg_report.py](eeg_adhd_epilepsy/reports/eeg_report.py) to generate the descriptive EEG report, [eeg_adhd_epilepsy/qc/raw_metrics.py](eeg_adhd_epilepsy/qc/raw_metrics.py) to compute and aggregate broad raw-QC metrics, and [eeg_adhd_epilepsy/reports/raw_qc.py](eeg_adhd_epilepsy/reports/raw_qc.py) to render the raw-QC report.

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
- `eeg_adhd_epilepsy/qc/preproc_qc.py` orchestrates the post-preprocessing QC validation. It explicitly delegates condition-level signal quality computation back to `qc/raw_metrics.py`, guaranteeing direct comparability between the "Raw" and "Clean" stages.
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

## Neurodags Pipeline (current approach)

The neurodags pipeline replaces the manual combination of `base.py` + `epochs.py` + `extract_descriptors.py` with a YAML-driven DAG. It is the recommended way to run preprocessing and feature extraction on this dataset.

Pipeline files live in `neurodags_pipelines/`. Three YAMLs cover the full workflow:

| File | Purpose |
|---|---|
| `step-0_pipeline@preprocessing.yml` | Per-subject preprocessing: inject annotations → bandpass/resample → ZapLine → RANSAC → CAR → AutoReject → ICA (DSS+MWF) → residual denoise |
| `step-0_pipeline@qc.yml` | Per-subject QC records + HTML reports + dataset-level summary |
| `step-1_pipeline@extraction.yml` | Feature extraction across all 8 conditions (epoching in-memory); writes `.nc` files per feature family |

### Run sequence

```bash
# 1. Preprocessing — writes @CleanedPrepRaw.fif, @CorrectRaw.fif, @DenoiseRaw.fif per subject
neurodags run neurodags_pipelines/step-0_pipeline@preprocessing.yml

# 2. QC reports — per-subject HTML + dataset summary
neurodags run neurodags_pipelines/step-0_pipeline@qc.yml

# 3. Feature extraction — all 8 conditions, one run
neurodags run neurodags_pipelines/step-1_pipeline@extraction.yml

# 4. Assemble flat CSV (one row per source file; `dataset` column = condition)
neurodags dataframe neurodags_pipelines/step-1_pipeline@extraction.yml \
    --output results/features_all_conditions.csv
```

Add `--n-jobs N` to any `run` or `dataframe` call for parallelism. Steps are idempotent — re-running skips already-computed files.

Check status at any point:

```bash
neurodags status neurodags_pipelines/step-0_pipeline@preprocessing.yml
neurodags status neurodags_pipelines/step-1_pipeline@extraction.yml --list-errors
```

### Outputs

Derivatives are written to `/home/yorguin/datasets/eeg-adhd-epilepsy/derivatives/`:

```
preprocessing/sub-*/eeg/
  *@CleanedPrepRaw.fif           annotated + cleaned continuous Raw
  *@CleanedPrepRaw_prov.json     provenance (AR stats, config snapshot)
  *@CorrectRaw.fif               ICA-corrected Raw (DSS+MWF)
  *@DenoiseRaw.fif               residual-denoised Raw
  *@*_qc_report.html             per-subject QC reports (base / correct / denoise)

features_conditions/{condition}/
  features@AbsBandPower.nc       one .nc per feature family, covering all subjects
  features@SampleEntropy.nc
  ...
```

The `dataset` column in the assembled CSV identifies the condition (e.g., `EO_baseline`). Split post-hoc with pandas:

```python
df = pd.read_csv("results/features_all_conditions.csv")
eo = df[df["dataset"] == "EO_baseline"]
```

### Cluster run

For Compute Canada (Narval/Béluga/Graham/Cedar), use the cluster script:

```bash
salloc --time=4:00:00 --cpus-per-task=8 --mem=32G --account=def-<pi>
bash cluster/compute_canada/run_neurodags_pipeline.sh
```

### Further reading

- `neurodags_pipelines/MIGRATION_GUIDE.md` — step-by-step equivalence with old pipeline
- `neurodags_pipelines/COMPARISON.md` — full gap audit (old vs new)
- `neurodags_pipelines/ECOSYSTEM_REPORT.md` — architectural overview and portability notes

## Dimensionality Reduction Workflow

The dimensionality reduction entry point is [eeg_adhd_epilepsy/analysis/dimensionality_reduction.py](eeg_adhd_epilepsy/analysis/dimensionality_reduction.py). It is the main analysis script for exploring low-dimensional structure in:

- raw EEG
- extracted descriptors
- saved embeddings when supported by the current config

It builds on `coco-pipe.dim_reduction` and supports multiple analysis styles, post-hoc evaluation targets, and report generation in one run.

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
  - embedding modes remain config-dependent and should be used only when the saved artifacts exist
- `--analysis_mode`
  - `flat`: one embedding per condition or pooled dataset
  - `sensor`: one independent analysis per sensor or channel-group
  - `family`: one independent analysis per descriptor family
  - `sensor_within_family`: one independent sensor analysis inside each family
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
- `sensor_within_family` compares sensors inside each requested family
- `--representation` is table-driven for descriptors: the script uses the descriptor table stem, such as `sensor_subject_features`, so separate tables get separate output/report variants
- descriptor rows with selected finite feature values above `--descriptor_max_abs_value` are dropped before fitting; the default is `1e12`

### Evals and Selection

The script separates:

- fit metrics from the reducer itself, such as:
  - `trustworthiness`
  - `continuity`
  - `shepard_correlation`
- post-hoc evaluation metrics from user-defined eval specs, such as:
  - logistic-regression balanced accuracy for clinical or demographic targets

Each config can define multiple evals. Typical examples are:

- the main clinical comparison, such as `med_adhd_vs_ctrl`
- additional control analyses, such as `sex_separation`, `age_separation`, or `condition_separation`

Best-fit selection is driven by:

- `selection_metric`
- optionally `selection_eval_name`

Use:

- `selection_metric: separation_logreg_balanced_accuracy`
- `selection_eval_name: <main_eval_name>`

when you want the report to rank fits by separation on one specific analysis target.

The report can still show the additional evals, but the “best” rows and summary choices should usually be tied to one explicit primary eval.

### Reports and Outputs

Outputs are separated by analysis variant. The main derivative root is:

- `BIDS/derivatives/dim_reduction/<output_group>/<dataset_name>/<input_mode>/<analysis_mode>__<representation>/`

This separation is important when you run the same config with:

- different `input_mode`
- different `analysis_mode`
- different `representation`

Reports follow the same variant split under:

- `reports/summary/dim_reduction/<output_group>/<dataset_name>/<input_mode>/<analysis_mode>__<representation>_dataset_summary.html`

At the run root, the script writes:

- fit and eval inventories
- per-fit artifacts
- the dataset summary report
- `run_summary.json`
- one terminal marker:
  - `_RUN_SUCCESS`
  - `_RUN_PARTIAL`
  - `_RUN_FAILED`

This makes it easier to see which variants completed cleanly, which are partial, and which should be rerun.

### Parallelism and Runtime Notes

The script supports outer-task parallelism through `--n_jobs`:

- `1`: fully serial
- `>1`: use that many outer workers
- `-1`: use all available CPUs

Parallelism is applied to:

- independent fit tasks
- independent eval tasks

It does not currently parallelize inner CV folds inside the separation evaluation itself. In practice:

- moderate `n_jobs` values are usually better for eval-heavy runs
- `-1` can be too aggressive for large sensor analyses with many eval targets

Recommended rule of thumb:

- start with `--n_jobs 4` or `--n_jobs 6`
- increase only after confirming memory and CPU behavior are acceptable

### Practical Notes

- `subject_flat` averages epochs within subject. Use this when you want one observation per subject.
- PCA-like reducers are still limited by `min(n_samples, n_features)`, so subject-level runs with small cohorts cannot request very large `n_components`.
- In raw sensor analyses, topomaps use the standard 10-20 montage and work only when sensor names match valid EEG channel labels.
- In descriptor sensor analyses, the meaning of “sensor” depends on the input table:
  - per-channel descriptor tables produce true electrodes like `Fz`
  - pooled descriptor tables produce grouped regions like `front_left`
- For any serious comparison study, explicitly set:
  - `selection_metric`
  - `selection_eval_name`
  - `output_group`

This avoids ambiguous report selection and keeps run variants separated on disk.

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
├── eeg_adhd_epilepsy/     # Main Python package
├── tests/                 # Automated tests
├── pyproject.toml         # Project metadata and dependencies
├── LICENSE                # Project license
└── README.md              # Project overview
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
