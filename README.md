# EEG Psychostimulant Analysis Repository

This repository contains scripts and data for comparing EEG signals between psychostimulant and non-psychostimulant subjects.

## Directory Structure and Scripts
- **data**: Contains scripts to convert raw EEG files and metadata into BIDS format and performs minimal processing (e.g., filtering, channel removal, montage setting).
- **utils**: Utility scripts.
- **explore**: Data exploration utilities for CSVs and analysis setup.
- **ml_pipelines**: Scripts for building general machine learning pipelines and executing them for various analyses.
- **dl_pipelines**: Scripts for running the reve_model and extracting embeddings.
- **viz**: Visualization tools and scripts.
- **get_umap.ipynb**: Notebook for running experiments and visualizing reve embeddings.

## To-Do (Scripts)
- Add comprehensive documentation for each script.
- Automate ML functions and pipeline steps to support various datasets and CSV files.
- Integrate additional machine learning analyses.
- Merge pipeline components into a single class and consolidate the run_ml_pipe function into one class that returns a unified object. This object will maintain consistent keys (with empty values when results are unavailable) to simplify visualization.
- Enhance the reve_model to automate processing across different datasets and prepare outputs for both visualization and ML pipeline analyses.
- Create a visualzation class for decoding results and another for embeddings (umap, tsne, pca)

## To-Do (Analysis)
- Incorporate additional features.
- Conduct further experiments with the reve_model to diagnose and resolve issues (Test the reve_model on sleep and eyes open/closed EEG data; Fine tune the model on the data?).
- Develop an automated analysis workflow that tests different conditions (e.g., with/without specific factors like ADHD, Epilepsy, TSA), considers age groups, and analyzes different medication types. This may include generating selections via a JSON configuration file.
- Normalize the data before feeding it to REVE!
- Test finetuning strategies (REVE)!

## Quick CSV Exploration

Use the patients CSV explorer to print dataset info, counts, and confusion matrices (e.g., TDAH x Psychostimulant):

```
python -m eeg_adhd_epilepsy_psychostimulant.explore.patients_csv_explorer \
  --csv_file data/csv/EEG_Psychostimulants_PatientList_08-2025.csv --grouped --save
```

Notes:
- The loader auto-detects comma/semicolon separators and drops empty trailing columns.
- If the environment variable `EEG_DATA_DIR` is unset, it defaults to the repo-local `data` directory.
