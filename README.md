# EEG Analysis for ADHD and Epilepsy

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides a comprehensive framework for analyzing EEG data from patients with ADHD and/or Epilepsy, with a particular focus on the effects of psychostimulant medication. It includes tools for data preprocessing, feature extraction, machine learning, and visualization.

## Features

*   End-to-end pipeline for EEG analysis.
*   Preprocessing and quality control checks for EEG signals.
*   Feature extraction from EEG data.
*   Machine learning pipelines using `CoCo-Pipe`.
*   Foundation models (e.g., REVE).
*   Visualization of results, including topographic maps and embeddings.
*   Statistical analysis of findings.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/eeg-analysis-adhd-epilepsy.git
    cd eeg-analysis-adhd-epilepsy
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    This project uses `pyproject.toml` to manage dependencies. Install the project in editable mode:
    ```bash
    pip install -e .
    ```

## Usage

The main functionalities of this project are organized into modules and can be run as scripts or imported into your own workflows.

### Preprocessing

The `eeg_adhd_epilepsy/preproc` module contains scripts for preprocessing EEG data.

```bash
# Example of running a preprocessing script (hypothetical)
python -m eeg_adhd_epilepsy.preproc.preprocessing --input-dir /path/to/raw/data --output-dir /path/to/processed/data
```

### Feature Extraction

The `eeg_adhd_epilepsy/explore` module is used for feature extraction.

```bash
# Example of running a feature extraction script (hypothetical)
python -m eeg_adhd_epilepsy.explore.features --input-dir /path/to/processed/data --output-file /path/to/features.csv
```

### Machine Learning

The `eeg_adhd_epilepsy/ml` module contains machine learning pipelines. The configuration for these pipelines can be found in the `.yml` files in this directory.

```bash
# Example of running the ML pipeline
python -m eeg_adhd_epilepsy.ml.run_ml_pipe --config eeg_adhd_epilepsy/ml/config_adhd.yml
```

### Visualization

The `eeg_adhd_epilepsy/viz` module contains scripts for generating plots and visualizations.

```bash
# Example of generating a plot (hypothetical)
python -m eeg_adhd_epilepsy.viz.plot_pca --feature-file /path/to/features.csv
```

## Project Structure

```
.
├── data/                  # Raw and processed data
├── eeg_adhd_epilepsy/     # Main Python package
│   ├── dl/                # Deep learning models
│   ├── explore/           # Feature extraction and data exploration
│   ├── io/                # Input/output functions
│   ├── ml/                # Machine learning pipelines
│   ├── preproc/           # EEG preprocessing
│   ├── utils/             # Utility functions
│   └── viz/               # Visualization scripts
├── results/               # Analysis results and figures
├── tests/                 # Tests for the package
└── pyproject.toml         # Project metadata and dependencies
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
