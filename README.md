# EEG Analysis for ADHD, Epilepsy and Medication Effects

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the analysis code used for an EEG study of ADHD, epilepsy, and medication exposure. It brings together data organization, preprocessing, quality control, feature extraction, modeling, and visualization in a single codebase.

## Overview

The codebase currently includes:

- BIDS-oriented data handling and EEG preprocessing utilities
- signal quality control and reporting tools
- descriptor-based feature extraction using `coco-pipe.descriptors`
- dimensionality reduction analysis using `coco-pipe.dim_reduction`
- machine learning and deep learning analysis modules using `coco-pipe.decoding`
- visualization utilities for exploratory analysis and result inspection using `coco-pipe.viz` and `coco-pipe.report`

## Metadata Workflow

Metadata currently starts from two CSV files collected by students William and Jeanne: `EEG_Psychostimulants_PatientList_08-2025.csv` and `IRSC_data_03-22-2026.csv`. The builder in [eeg_adhd_epilepsy/qc/metadata.py](eeg_adhd_epilepsy/qc/metadata.py) merges them into one canonical schema, applies the agreed cleaning rules, and writes:

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
