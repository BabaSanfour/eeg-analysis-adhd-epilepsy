# EEG Analysis for ADHD, Epilepsy and Medication Effects

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Analysis code for an EEG study of ADHD, epilepsy, and psychostimulant exposure.
It covers the full path from raw recordings to results: data organization,
preprocessing, quality control, feature extraction, dimensionality reduction,
foundation-model embeddings, and decoding — built on the
[`coco-pipe`](https://github.com/BabaSanfour/coco-pipe) analysis library.

## Pipeline at a glance

```
raw EEG ─▶ BIDS + raw QC ─▶ preprocess (desc-base) ─▶ epochs
        ─▶ descriptors ─▶ merge ─┬─▶ dimensionality reduction ─▶ reports
                                 ├─▶ classical decoding       ─▶ reports
        ─▶ foundation embeddings ┴─▶ foundation decoding      ─▶ reports
patient CSVs ─▶ metadata ─▶ cohort report
```

| Stage | Console script | Module |
|-------|----------------|--------|
| Build patient metadata | `eeg-build-patients-metadata` | `metadata.patients` |
| Cohort report | `eeg-cohort-report` | `metadata.cohort` |
| Raw → BIDS + raw QC | `eeg-to-bids` | `preproc.to_bids` |
| Preprocess (cleaning + QC) | `eeg-preprocess` | `preproc.base` |
| Epoch | `eeg-save-epochs` | `preproc.epochs` |
| Descriptors | `eeg-descriptors` | `analysis.extract_descriptors` |
| Merge descriptors | `eeg-merge-descriptors` | `analysis.merge_descriptors` |
| Foundation embeddings | `eeg-foundation-embeddings` | `analysis.extract_foundation_embeddings` |
| Dimensionality reduction | `eeg-dim-reduce` | `analysis.dimensionality_reduction` |
| Classical decoding | `eeg-classical-decode` | `analysis.classical_decoding` |
| Foundation decoding | `eeg-foundation-decode` | `analysis.foundation_decoding` |
| **Orchestrate all of the above** | **`eeg-run`** | `run` |

## Quick start

```bash
# 1. Install (Python 3.10+). Re-run after pulling changes that touch entry points.
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. See the stage order and preview the whole chain WITHOUT running anything.
eeg-run --list
eeg-run --dry-run \
  --raw_root /path/to/raw --bids_root /path/to/BIDS --metadata /path/to/patients_metadata_clean.csv \
  --cohort_config configs/cohorts/medicated_adhd_vs_controls/pooled/01_all_subjects/total.yaml \
  --dim_analysis_config configs/analyses/dim_reduction/descriptors.yaml \
  --decode_analysis_config configs/analyses/decoding/classical.yaml

# 3. Run a single stage, a range, or the whole thing (resume-by-default).
eeg-run --from preprocess --to merge --bids_root /path/to/BIDS --metadata /path/to/meta.csv
```

`eeg-run` runs the **local, single-machine** form of each stage and skips a stage
whose output already exists (use `--overwrite` to force). For large jobs use the
numbered SLURM scripts in [cluster/](cluster/) instead (see
[cluster/README.md](cluster/README.md)). The [Makefile](Makefile) wraps the same
commands — `make help` lists targets; set the paths once and run e.g.
`make descriptors BIDS_ROOT=… METADATA=…`.

The individual `eeg-*` scripts still work standalone — `eeg-run` only sequences
them.

### Point it at your data

This repo targets one real dataset. To run it on your own (similar) data:

1. Convert/point to a **BIDS** EEG dataset (`--bids_root`) and a cleaned
   **metadata CSV** (`--metadata`) with the columns the configs reference
   (e.g. `study_id`, `session`, `patient_group_id`, `combined_diagnosis`, `adhd`,
   `psychostimulant`, `age_group`, `sex`). Dataset paths are **CLI/env arguments**,
   never stored in the configs.
2. Copy a cohort config from `configs/cohorts/` and an analysis config from
   `configs/analyses/` and edit them for your question (see below and
   [configs/README.md](configs/README.md)).

## The two-config model (analysis stages)

The analysis stages (`eeg-dim-reduce`, `eeg-classical-decode`,
`eeg-foundation-decode`) take **two** config files, separating *which cohort* from
*which analysis*:

- a **cohort config** (`configs/cohorts/...`) — the dataset selection and clinical
  question: `dataset_name`, `conditions`, `group_filters`,
  `filter_col`/`filter_val`, and `evals` (targets + label maps).
- an **analysis config** (`configs/analyses/<type>/...`) — the method and
  hyperparameters: `analysis_modes` (dim-reduction; each mode names its reducers
  and n_components sweep), `models`/`cv`/
  `feature_selection` (decoding), `models`/`train_modes` (foundation), plus
  input-shaping and run controls.

```bash
eeg-classical-decode \
  --cohort_config   configs/cohorts/medicated_adhd_vs_controls/pooled/01_all_subjects/total.yaml \
  --analysis_config configs/analyses/decoding/classical.yaml \
  --bids_root /path/to/BIDS --metadata /path/to/meta.csv
```

The two files are validated and deep-merged (the analysis config overrides the
cohort on overlap, e.g. `conditions`), so one cohort is reused across analyses and
one analysis across cohorts. A missing/misspelled key raises an actionable error
instead of a deep `KeyError`. See [configs/README.md](configs/README.md).

## Prerequisites

- **Python 3.10+** and the editable install (`pip install -e .`), which pulls
  `coco-pipe[decoding,foundation]` and `mne-denoise`.
- **Foundation models** need PyTorch and Braindecode ≥ 1.5. Fine-tuning/LoRA are
  practical on a CUDA **GPU**; CPU runs are for small validation only.
- **REVE** is a gated Hugging Face model: run `hf auth login` or export `HF_TOKEN`
  before using it, otherwise it is skipped with `authentication_required`. Saved
  configs are redacted so tokens never reach derivatives or reports.
- **Real data**: the pipeline expects a BIDS EEG dataset and a cleaned metadata
  CSV — see "Point it at your data".

## Metadata workflow

Metadata starts from two CSVs collected by students William and Jeanne
(`EEG_Psychostimulants_PatientList_08-2025.csv` and `IRSC_data_final.csv`). The
builder in [metadata/patients.py](eeg_adhd_epilepsy/metadata/patients.py) merges them into one
canonical schema, applies the agreed cleaning rules, assigns a `patient_group_id`
for repeated recordings, and writes `patients_metadata.csv`,
`patients_metadata_clean.csv`, and `patients_metadata_removed.json`.

```bash
eeg-build-patients-metadata \
  --adhd_csv /path/to/EEG_Psychostimulants_PatientList_08-2025.csv \
  --drug_resistant_csv /path/to/IRSC_data_final.csv \
  --output_dir /path/to/csv
```

The downstream entry point is `patients_metadata_clean.csv`.

## Cohort report

```bash
eeg-cohort-report --metadata_csv /path/to/patients_metadata_clean.csv --output_dir /path/to/output
```

Builds `cohort_report.html` (demographics, diagnosis, medication breakdowns,
analysis-opportunity tables), optionally with `--with_recruitment` milestones. It
can also apply a cohort filter from a YAML file. See
[metadata/cohort.py](eeg_adhd_epilepsy/metadata/cohort.py).

The set of *possible studies* — every comparison, the cohort filters it runs
under, and the membership rule for each group — is declared in one place,
[metadata/analysis_opportunities_schema.py](eeg_adhd_epilepsy/metadata/analysis_opportunities_schema.py).
Each constraint and analysis carries an executable predicate, so the schema is
the single source of truth and the engine never re-implements the logic.

## BIDS conversion and raw QC

[preproc/to_bids.py](eeg_adhd_epilepsy/preproc/to_bids.py) is the single stage to
run before preprocessing. It does raw discovery, raw→BIDS conversion, canonical
annotation rewrite, canonical `BLOCK_*` condition annotations, and
optional pre-base EEG and raw-QC reports.

```bash
eeg-to-bids \
  --raw_root /path/to/raw_data \
  --bids_root /path/to/BIDS \
  --metadata_csv /path/to/patients_metadata_clean.csv \
  --with_eeg_reports --with_raw_qc --raw_qc_analysis_level both \
  --n_jobs 4
```

Without `--overwrite` it resumes (skips existing runs, rebuilds summary reports
from written files). Outputs: BIDS EEG under `BIDS/`, embedded canonical block
annotations, and subject/summary reports under a sibling `reports/` directory.

## Preprocessing and post-clean QC

[preproc/base.py](eeg_adhd_epilepsy/preproc/base.py) reads raw BIDS, applies
automated cleaning, and writes analysis-ready derivatives plus QC reports:

- Bandpass filter (0.1–99.5 Hz) + line-noise removal via Adaptive ZapLine (`mne-denoise`)
- Bad-channel detection via RANSAC (`pyprep`)
- Robust re-referencing (common average)
- Condition-wise bad-segment/epoch detection via `AutoReject`
- Post-clean QC metrics + HTML reports

```bash
eeg-preprocess --bids_root /path/to/BIDS --n_jobs 4
```

Incremental by default (`--overwrite` to force, `--subjects` to target). Outputs
`*_desc-base_eeg.fif` under `BIDS/derivatives/preproc/` and `base_qc` reports.

> An experimental "Part 2" artifact-correction pipeline
> (`preproc/{correct,denoise,compare,run_all}.py`, `ARTIFACT_STRATEGIES.md`) is
> **not** part of the canonical run; `base.py` is the production path.

## Epoching

```bash
eeg-save-epochs --bids_root /path/to/BIDS --segment_duration 10.0 --ignore_annotations
```

Slices `*_desc-base_eeg.fif` by condition annotations into
`*_task-<condition>_run-XX_desc-base_epo.fif` derivatives.

## Descriptor extraction and merge

[analysis/extract_descriptors.py](eeg_adhd_epilepsy/analysis/extract_descriptors.py)
computes feature banks with `coco-pipe.descriptors` from `configs/descriptors.yaml`:
spectral bands (Welch PSD), parametric modeling (SpecParam), and complexity/entropy
measures, with region-based channel pooling, then joins clinical targets.

```bash
# All subjects (sequential). For the cluster array form see cluster/05.
eeg-descriptors \
  --bids_root /path/to/BIDS \
  --metadata /path/to/patients_metadata_clean.csv \
  --config configs/descriptors.yaml \
  --subject_col study_id --conditions all

# Merge the per-subject shards into combined tables.
eeg-merge-descriptors --bids_root /path/to/BIDS
```

Per-subject shards land under
`BIDS/derivatives/signal_features/descriptors/sub-*/...`; merge writes
`descriptors/combined/{sensor,pooled}_{epoch,subject}_features.{parquet,csv}`.

## Dimensionality reduction

[analysis/dimensionality_reduction.py](eeg_adhd_epilepsy/analysis/dimensionality_reduction.py)
explores low-dimensional structure in raw EEG, descriptors, or embeddings (builds
on `coco-pipe.dim_reduction`).

```bash
eeg-dim-reduce \
  --bids_root /path/to/BIDS --metadata /path/to/patients_metadata_clean.csv \
  --cohort_config   configs/cohorts/medicated_adhd_vs_controls/pooled/01_all_subjects/total.yaml \
  --analysis_config configs/analyses/dim_reduction/raw.yaml \
  --input_mode raw --analysis_mode flat --representation subject \
  --n_jobs 4
```

### Core concepts

Three arguments define what is loaded and what one “analysis unit” is:

- `--input_mode`: `raw` (BIDS/derivatives), `descriptors` (merged tables), or
  `foundation_embeddings` (a model-specific embedding derivative).
- `--analysis_mode`: `flat`, `sensor`, `family`, `subfamily`,
  `sensor_within_family`, `sensor_within_subfamily`, `feature`,
  `feature_within_family`, `descriptor`, `descriptor_sensor` — progressively finer
  units (most are descriptor-only).
- `--representation`: for `raw` inputs, this is the averaging granularity (`epoch`
  or `subject`). For `descriptors` and `foundation_embeddings`, it dictates which
  pre-computed representation file to load (e.g., `epoch`, `recording`, `subject`).
  It is orthogonal to `--analysis_mode`.

The most common subject-level raw setup is `--input_mode raw --analysis_mode flat
--representation subject`: one row per subject over the joint sensor×time
space. Best-fit selection is driven by `selection_metric` /`selection_eval_name`
(in the analysis config); the default separation ranking uses RF balanced
accuracy first and LR balanced accuracy second. Outputs are separated by a
configuration hash under `BIDS/derivatives/dim_reduction/<dataset_name>/...`.
Re-runs rebuild inventories from reusable checkpoints. `--n_jobs` controls
outer-task parallelism (start with 4–6).

## Foundation models and decoding

```bash
# Dataset-wide embedding extraction (CBraMod, LaBraM, REVE, LUNA).
eeg-foundation-embeddings --config /path/to/foundation_embeddings.yaml

# Decoding (two-config). Classical = descriptors/embeddings; foundation = direct probing/fine-tune/LoRA.
eeg-classical-decode  --cohort_config <cohort.yaml> --analysis_config configs/analyses/decoding/classical.yaml --bids_root … --metadata …
eeg-foundation-decode --cohort_config <cohort.yaml> --analysis_config configs/analyses/decoding/foundation.yaml --bids_root … --metadata …
```

Each model declares its own EEG window requirements; the example configs use
10-second derivative epochs for CBraMod/REVE/LUNA, and re-epoch the cleaned
continuous `desc-base` recording at 15 s for LaBraM (3000 samples at 200 Hz).
`window_mismatch_policy` is `error` | `skip` | `re_epoch` — nothing silently pads
or crops. Grouped CV uses `patient_group_id`; scaling/reduction/selection/fitting
happen inside each training fold. See
[docs/design/foundation_models_and_decoding.md](docs/design/foundation_models_and_decoding.md).

## Cluster (SLURM)

Numbered submission scripts, one per stage, live in [cluster/](cluster/) — see
[cluster/README.md](cluster/README.md) for the order, env vars, and the two-config
pairing used by the array jobs.

## Installation

```bash
git clone <repo-url> && cd eeg_analysis_adhd_epilepsy
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

To develop against a local `coco-pipe` checkout in the same env:

```bash
pip install -e '/Users/hamzaabdelhedi/Projects/packages/coco-pipe[decoding,foundation]'
```

## Module map

```text
eeg_adhd_epilepsy/
├── io/           # BIDS layout/paths + recording-id grouping (bids), raw ingest
│                 # (ingest), analysis-input loading (containers)
├── metadata/     # clinical patient-metadata concern: builder (patients), cohort
│                 # analysis + possible-study enumeration (cohort), the study
│                 # schema (analysis_opportunities_schema), shared constants (schema)
├── preproc/      # to_bids, base (canonical cleaning), epochs  [+ experimental Part-2]
├── analysis/     # the pipeline entry points (descriptors, merge, dim-reduction,
│                 # decoding, foundation) + analysis/utils
├── signal_quality/  # primitive QC metrics (spectral, time-domain)
├── qc/           # stage-level QC orchestration (raw_qc, preproc_qc, descriptor_qc)
├── reports/      # HTML report composition (one per stage)
├── viz/          # figures embedded in the reports
└── utils/        # config (two-config loader), yaml, constants, logging
```

Per-stage QC follows a consistent split: `qc/` computes metrics, `viz/` draws the
figures, `reports/` assembles the HTML.

## Repository layout

```text
.
├── eeg_adhd_epilepsy/   # Main package (io, metadata, preproc, analysis, qc, signal_quality, reports, viz, utils)
├── configs/             # cohorts/ (dataset+question) and analyses/ (method); descriptors.yaml; examples
├── cluster/             # Numbered SLURM scripts, one per stage (see cluster/README.md)
├── scripts/             # One-off maintenance scripts (e.g. split_configs.py)
├── docs/                # Design notes, backlog, and the improvement-plan tracker
├── tests/               # Automated tests
├── Makefile             # Convenience targets over eeg-run / the eeg-* CLIs
├── pyproject.toml       # Metadata, dependencies, console-script entry points
└── README.md
```

BIDS data, derivatives, and generated reports are written under the paths you pass
(`--bids_root` and its `derivatives/`/sibling `reports/`); they are not tracked in
git.

## License

MIT — see [LICENSE](LICENSE).
