# EEG Analysis for ADHD, Epilepsy and Medication Effects

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
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
| Verify descriptors | `eeg-verify-descriptors` | `analysis.verify_descriptors` |
| Merge descriptors | `eeg-merge-descriptors` | `analysis.merge_descriptors` |
| Foundation embeddings | `eeg-foundation-embeddings` | `analysis.extract_foundation_embeddings` |
| Align subject embeddings | `eeg-align-subject-embeddings` | `analysis.align_subject_embeddings` |
| Dimensionality reduction | `eeg-dim-reduce` | `analysis.dimensionality_reduction` |
| Classical decoding | `eeg-classical-decode` | `analysis.classical_decoding` |
| Foundation decoding | `eeg-foundation-decode` | `analysis.foundation_decoding` |

## Quick start

```bash
# 1. Install (Python 3.11+). Re-run after pulling changes that touch entry points.
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. List the direct local stage targets.
make help

# 3. Run one stage directly.
make descriptors BIDS_ROOT=/path/to/BIDS METADATA=/path/to/BIDS/participants.tsv
```

Each [Makefile](Makefile) target calls the owning module directly. `make all`
runs those targets in pipeline order, while individual targets run only the
named stage. For large jobs use the numbered SLURM scripts in
[cluster/](cluster/) instead (see [cluster/README.md](cluster/README.md)).

### Point it at your data

This repo targets one real dataset. To run it on your own (similar) data:

1. Convert/point to a **BIDS** EEG dataset (`--bids_root`) whose `participants.tsv`
   (`--metadata`) has the columns the configs reference (e.g. `patient_group_id`,
   `combined_diagnosis`, `adhd`, `psychostimulant`, `age_group`, `sex`), keyed by
   `participant_id`. Dataset paths are **CLI/env arguments**,
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
  --bids_root /path/to/BIDS --metadata /path/to/BIDS/participants.tsv
```

The two files are validated and deep-merged (the analysis config overrides the
cohort on overlap, e.g. `conditions`), so one cohort is reused across analyses and
one analysis across cohorts. A missing/misspelled key raises an actionable error
instead of a deep `KeyError`. See [configs/README.md](configs/README.md).

## Prerequisites

- **Python 3.11+** (required by `coco-pipe`) and the editable install
  (`pip install -e .`), which pulls `coco-pipe[decoding,descriptors,foundation]`
  and `mne-denoise` from GitHub `main`, and requires MNE ≥ 1.13.
- **Foundation models** run on PyTorch and Braindecode ≥ 1.8.1 (from coco-pipe's
  `foundation` extra). Fine-tuning/LoRA are practical on a CUDA **GPU**; CPU runs
  are for small validation only.
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

`patients_metadata_clean.csv` is the input of the BIDS conversion, which writes its
subject-level columns (all but `study_id`, `patient_id` and `eeg_date`) to
`participants.tsv`. Every analysis reads its labels from that `participants.tsv`.

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

- Bandpass filter (0.1–100 Hz by default) + line-noise removal via Adaptive ZapLine (`mne-denoise`)
- Bad-channel detection via RANSAC (`pyprep`)
- Robust re-referencing (common average)
- Condition-wise bad-segment/epoch detection via `AutoReject`
- Post-clean QC metrics + HTML reports

```bash
eeg-preprocess --bids_root /path/to/BIDS --n_jobs 4
```

Rewrites its outputs on every run (`--subjects` to target). Outputs
`*_desc-base_eeg.fif` under `BIDS/derivatives/preproc/` and `base_qc` reports.

> An experimental "Part 2" artifact-correction pipeline
> (`preproc/{correct,denoise,compare,run_all}.py`) is
> **not** part of the canonical run; `base.py` is the production path.

## Epoching

```bash
eeg-save-epochs --bids_root /path/to/BIDS --segment_duration 10.0 --n_jobs 8
```

Slices each `*_desc-base_eeg.fif` into fixed-length epochs labelled by condition
and writes them next to it as `*_desc-base_epo.fif`, with an `*_epo.json`
provenance sidecar. With `--reject_by_annotation` it drops the epochs overlapping a
`BAD_` annotation and writes `*_desc-baserejected_epo.fif` instead; descriptor extraction
reads either with `--desc`. Every run rewrites its outputs, and the job fails when a
recording fails.

## Descriptor extraction and merge

[analysis/extract_descriptors.py](eeg_adhd_epilepsy/analysis/extract_descriptors.py)
computes feature banks with `coco-pipe.descriptors` from `configs/descriptors.yaml`:
spectral bands (Welch PSD), parametric modeling (specparam) and complexity/entropy
measures, per channel, then joins clinical targets. `--desc` picks the epoch variant
(`base` or `baserejected`); on the cluster every stage from 05 runs once per variant
(`EPOCHS_DESC`, `cluster/README.md`).

```bash
# All subjects (sequential). For the cluster array form see cluster/05.
eeg-descriptors \
  --bids_root /path/to/BIDS \
  --metadata /path/to/BIDS/participants.tsv \
  --config configs/descriptors.yaml \
  --desc baserejected \
  --derivative_root /path/to/shards \
  --conditions all

# Merge the per-subject shards into combined tables.
eeg-merge-descriptors --derivative_root /path/to/shards --output_root /path/to/merged \
  --reports_root /path/to/reports
```

A shard, `<shards>/sub-*/ses-*/eeg/<condition>/`, holds the epoch table
(`sensor_epoch_features.parquet`) and `shard.json` (QC summary, QC flags, failures); each
subject gets one QC report with all its conditions. The merge writes
`<merged>/combined/sensor_{epoch,recording}_features.parquet`, the recording table built
from the merged epochs; each table carries its feature columns in its own metadata.

## Dimensionality reduction

[analysis/dimensionality_reduction.py](eeg_adhd_epilepsy/analysis/dimensionality_reduction.py)
explores low-dimensional structure in raw EEG, descriptors, or embeddings (builds
on `coco-pipe.dim_reduction`).

```bash
eeg-dim-reduce \
  --bids_root /path/to/BIDS --metadata /path/to/BIDS/participants.tsv \
  --cohort_config   configs/cohorts/medicated_adhd_vs_controls/pooled/01_all_subjects/total.yaml \
  --analysis_config configs/analyses/dim_reduction/raw.yaml \
  --representation subject \
  --n_jobs 4
```

### Core concepts

Three settings define what is loaded and what one “analysis unit” is. The first
two live in the analysis config; the third can also be given on the command line:

- `input_mode`: `raw` (BIDS/derivatives), `descriptors` (merged tables), or
  `foundation_embeddings` (a model-specific embedding derivative, chosen with
  `--embedding_derivative_root` and `--embedding_model_key`).
- `analysis_modes`: a mapping of mode → spec (reducers and n_components sweep).
  Modes are `flat`, `sensor`, `family`, `subfamily`, `sensor_within_family`,
  `sensor_within_subfamily`, `feature`, `feature_within_family`, `descriptor`,
  `descriptor_sensor` — progressively finer units (most are descriptor-only).
- `representation` / `--representation`: `epoch`, `recording` or `subject`. For
  `raw` inputs this is the averaging granularity. For `descriptors` and
  `foundation_embeddings`, it selects which pre-computed table to load. It is
  orthogonal to the analysis mode.

The most common subject-level raw setup is `raw.yaml` with the `flat` mode and
`--representation subject`: one row per subject over the joint sensor×time
space. Best-fit selection is driven by `selection_metric` /`selection_eval_name`
(in the analysis config); the default separation ranking uses RF balanced
accuracy first and LR balanced accuracy second. Outputs are separated by a
configuration hash under `BIDS/derivatives/dim_reduction/<dataset_name>/...`.
Re-runs rebuild inventories from reusable checkpoints. `--n_jobs` controls
outer-task parallelism (start with 4–6).
Pass `--derivative_root /scratch/.../BIDS/derivatives/dim_reduction` to keep
checkpointed outputs outside the input BIDS tree without changing run identity.

## Foundation models and decoding

```bash
# Dataset-wide embedding extraction (CBraMod, LaBraM, REVE, LUNA, BIOT, SignalJEPA, EEGPT, BENDR).
eeg-foundation-embeddings --config configs/foundation_extraction.yaml --bids_root … --metadata …

# Decoding (two-config). Classical = descriptors/embeddings; foundation = direct probing/fine-tune/LoRA.
eeg-classical-decode  --cohort_config <cohort.yaml> --analysis_config configs/analyses/decoding/classical.yaml --bids_root … --metadata …
eeg-foundation-decode --cohort_config <cohort.yaml> --analysis_config configs/analyses/decoding/foundation.yaml --bids_root … --metadata …
```

Both decoding commands accept
`--derivative_root /scratch/.../BIDS/derivatives/decoding`. This relocates
checkpoints and aggregate inputs while leaving `--bids_root` pointed at the
source dataset; the output-only path is excluded from the scientific run hash.

Each model declares its own EEG window requirements in the config. Most use the
saved 10-second derivative epochs (`window_source: derivative`); a model that
needs another window, such as LaBraM at 15 s, is re-epoched from the cleaned
continuous `desc-base` recording (`window_source: re_epoch`).
`window_mismatch_policy` is `raise` | `skip` — nothing silently pads or crops.
Grouped CV uses `patient_group_id`; scaling/reduction/selection/fitting happen
inside each training fold.

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
pip install -e '/path/to/coco-pipe[decoding,foundation]'
```

## Module map

```text
eeg_adhd_epilepsy/
├── io/           # BIDS layout/paths + recording-id grouping (bids), raw ingest
│                 # (ingest), table/JSON readers (readers), report directories
│                 # (report_paths)
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
└── utils/        # config (two-config loader), yaml, constants, logs, events,
                  # artifacts, formatting
```

Per-stage QC follows a consistent split: `qc/` computes metrics, `viz/` draws the
figures, `reports/` assembles the HTML.

## Repository layout

```text
.
├── eeg_adhd_epilepsy/   # Main package (io, metadata, preproc, analysis, qc, signal_quality, reports, viz, utils)
├── configs/             # cohorts/ (dataset+question) and analyses/ (method); descriptors.yaml,
│                        # annotations.yaml, foundation_extraction.yaml
├── cluster/             # Numbered SLURM scripts, one per stage (see cluster/README.md)
├── tests/               # Automated tests
├── Makefile             # Direct convenience targets over the stage CLIs
├── pyproject.toml       # Metadata, dependencies, console-script entry points
└── README.md
```

BIDS data, derivatives, and generated reports are written under the paths you pass
(`--bids_root` and its `derivatives/`/sibling `reports/`); they are not tracked in
git.

## License

MIT — see [LICENSE](LICENSE).
