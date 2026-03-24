# EEG Analysis for ADHD, Epilepsy and Medication Effects

[![Python Version](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/downloads/release/python-3130/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains a complete framework for analyzing EEG data from patients with ADHD and/or Epilepsy with medication effects. The project includes tools for data preprocessing, descriptor-based feature extraction with `CoCo-Pipe`, machine learning, deep learning models like REVE for embeddings, and various visualization techniques.

## Features

*   End-to-end EEG analysis pipeline from raw data to publication-ready figures.
*   BIDS-compatible data organization.
*   Preprocessing and quality control for EEG signals using MNE.
*   Descriptor-based feature extraction from saved EEG epochs with `CoCo-Pipe`.
*   Machine learning pipelines for classification and regression tasks.
*   Deep learning models for generating EEG embeddings.
*   A comprehensive suite of visualization tools for results, including topographic maps, embeddings, and feature importance plots.
*   Statistical analysis of findings.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/eeg-analysis-adhd-epilepsy.git
    cd eeg-analysis-adhd-epilepsy
    ```

2.  **Create and activate a virtual environment (Python >= 3.10 is required):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    This project uses `pyproject.toml` to manage dependencies. Install the project in editable mode, which will install the study package and the pinned `coco-pipe` branch used for descriptor extraction:
    ```bash
    pip install -e .
    ```

4.  **Optional: override with a local `coco-pipe` checkout during active descriptor development:**
    If you are actively editing `coco-pipe` itself and want to override the pinned branch dependency with a local checkout, reinstall it explicitly into the same environment:
    ```bash
    pip install -e '/Users/hamzaabdelhedi/Projects/packages/coco-pipe[descriptors]'
    ```

## Usage

This project provides several command-line scripts for running different parts of the analysis pipeline.

### Preprocessing and BIDS conversion

The `eeg_adhd_epilepsy/preproc` module contains scripts for converting raw data to BIDS format and for preprocessing the EEG data. These scripts are meant to be run before the main analysis pipelines.

### Feature Extraction

The canonical feature-generation path is now
`eeg_adhd_epilepsy/analysis/extract_descriptors.py`. It loads saved epoched
derivatives, extracts descriptors with `coco-pipe`, and writes checkpointed
per-subject shards.

```bash
eeg-descriptors \
  --bids_root /Users/hamzaabdelhedi/Projects/data/EEG_psychostimulant_data/EEG_psychostimulants_2025-02/BIDS \
  --metadata /Users/hamzaabdelhedi/Projects/data/EEG_psychostimulant_data/EEG_psychostimulants_2025-02/csv/EEG_Psychostimulants_PatientList_08-2025.csv
```

This script expects saved epochs to already exist under the derivatives tree.
Its outputs include:

*   checkpointed per-subject, per-condition shards under `<BIDS>/derivatives/signal_features/descriptors/sub-<subject>/eeg/<condition>/`
*   raw descriptor bundles (`.npz`)
*   epoch-level feature tables (`.parquet` and `.csv`)
*   subject-level aggregated feature tables (`.parquet` and `.csv`)
*   failures tables and feature-column sidecars next to each shard
*   `dataset_description.json` and `config_used.yaml` at the derivative root for provenance and resume safety

Combined tables are built separately with:

```bash
eeg-merge-descriptors \
  --bids_root /Users/hamzaabdelhedi/Projects/data/EEG_psychostimulant_data/EEG_psychostimulants_2025-02/BIDS
```

This merges completed shards into:

*   `<BIDS>/derivatives/signal_features/descriptors/combined/epoch_features.parquet|csv`
*   `<BIDS>/derivatives/signal_features/descriptors/combined/subject_features.parquet|csv`
*   `<BIDS>/derivatives/signal_features/descriptors/combined/failures.csv`

The `eeg_adhd_epilepsy/explore` and `eeg_adhd_epilepsy/features` modules may
still be useful for QC and ad hoc exploration, but they are no longer the
canonical feature-generation path for ML.

### Machine Learning Pipeline

The `eeg_adhd_epilepsy/ml` module contains the machine learning pipeline. You can run the pipeline using the `eeg-ml-run` script with a configuration file.

```bash
# Example of running the ML pipeline for ADHD
eeg-ml-run --config eeg_adhd_epilepsy/ml/config_adhd.yml
```

### Deep Learning Embeddings

The `eeg_adhd_epilepsy/dl` module contains deep learning models. You can generate EEG embeddings using the `eeg-embeddings` script.

```bash
# Example of generating EEG embeddings
eeg-embeddings
```

### Visualization

The `eeg_adhd_epilepsy/viz` module contains scripts for generating plots and visualizations.

```bash
# Example of dimensionality reduction for visualization
eeg-dim-reduce

# Example of plotting embeddings
python -m eeg_adhd_epilepsy.viz.plot_embeddings
```

## Data and Results

The `data/` and `results/` directories are not tracked by Git. You need to create them manually when a given workflow needs them.

*   **`data/`**: This directory can still hold repo-local artifacts, but the descriptor pipeline consumes saved epoched derivatives under the BIDS derivatives tree.
*   **`results/`**: This directory is still useful for downstream reports, figures, and trained models, but descriptor extraction now writes its canonical outputs into `<BIDS>/derivatives/signal_features/descriptors/`.

## Project Structure

```
.
├── data/                  # Raw and processed data (not tracked by Git)
├── eeg_adhd_epilepsy/     # Main Python package
│   ├── dl/                # Deep learning models (e.g., REVE)
│   ├── explore/           # Feature extraction and data exploration
│   ├── io/                # Input/output functions
│   ├── ml/                # Machine learning pipelines (CoCo-Pipe)
│   ├── preproc/           # EEG preprocessing and BIDS conversion
│   ├── utils/             # Utility functions
│   └── viz/               # Visualization scripts
├── results/               # Analysis results and figures (not tracked by Git)
├── tests/                 # Tests for the package
├── .gitignore             # Files and directories to be ignored by Git
├── LICENSE                # Project license
├── pyproject.toml         # Project metadata and dependencies
└── README.md              # This file
```
## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
