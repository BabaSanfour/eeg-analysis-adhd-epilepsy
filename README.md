# EEG Analysis for ADHD, Epilepsy and Medication Effects

[![Python Version](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/downloads/release/python-3130/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains a complete framework for analyzing EEG data from patients with ADHD and/or Epilepsy with medication effects. The project includes tools for data preprocessing, feature extraction, machine learning with `CoCo-Pipe`, deep learning models like REVE for embeddings, and various visualization techniques.

## Features

*   End-to-end EEG analysis pipeline from raw data to publication-ready figures.
*   BIDS-compatible data organization.
*   Preprocessing and quality control for EEG signals using MNE.
*   Feature extraction from EEG data, including spectral and connectivity measures.
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
    This project uses `pyproject.toml` to manage dependencies. Install the project in editable mode, which will install all necessary dependencies:
    ```bash
    pip install -e .
    ```

## Usage

This project provides several command-line scripts for running different parts of the analysis pipeline.

### Preprocessing and BIDS conversion

The `eeg_adhd_epilepsy/preproc` module contains scripts for converting raw data to BIDS format and for preprocessing the EEG data. These scripts are meant to be run before the main analysis pipelines.

### Feature Extraction and Exploration

The `eeg_adhd_epilepsy/explore` module is used for feature extraction and initial data exploration.

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

The `data/` and `results/` directories are not tracked by Git. You need to create them manually.

*   **`data/`**: This directory should contain the raw and processed EEG data. The preprocessing scripts expect the raw data to be in a specific format.
*   **`results/`**: This directory will store the outputs of the analysis, such as figures, tables, and trained models.

You can create these directories with the following command:

```bash
mkdir data results
```

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