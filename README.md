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

Metadata currently starts from two CSV files collected by students William and Jeanne: `EEG_Psychostimulants_PatientList_08-2025.csv` and `IRSC_data_03-22-2026.csv`. The builder in [eeg_adhd_epilepsy/qc/metadata.py](/Users/hamzaabdelhedi/Projects/research/EEG_psychostim/eeg_analysis_adhd_epilepsy/eeg_adhd_epilepsy/qc/metadata.py) merges them into one canonical schema, applies the agreed cleaning rules, and writes:

- `patients_metadata.csv`
- `patients_metadata_clean.csv`
- `patients_metadata_removed.json`

Rebuild with:

```bash
eeg-build-patients-metadata
```

The intended downstream entry point is `patients_metadata_clean.csv`.

## Cohort Report Workflow

The cohort report starts from `patients_metadata_clean.csv` and optionally reads `patients_metadata_removed.json` for provenance. The builder in [eeg_adhd_epilepsy/qc/cohort_report.py](/Users/hamzaabdelhedi/Projects/research/EEG_psychostim/eeg_analysis_adhd_epilepsy/eeg_adhd_epilepsy/qc/cohort_report.py) can:

- build the full clean-cohort report directly
- apply a cohort filter from a YAML file
- optionally add recruitment milestones with `--with_recruitment`

Run with:

```bash
eeg-cohort-report --metadata_csv /path/to/patients_metadata_clean.csv --output_dir /path/to/output
```

The main output is `cohort_report.html`, plus opportunity and recruitment CSVs when enabled.

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
