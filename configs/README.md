# Configuration guide

The pipeline separates the population being studied from the method used to
study it. Dataset paths are not stored in either config: pass `--bids_root` and
`--metadata` on the command line, or set the corresponding cluster environment
variables.

## Layout

```text
configs/
├── cohorts/
│   └── medicated_adhd_vs_controls/  # populations and clinical questions
├── analyses/
│   ├── dim_reduction/              # reducers and selection settings
│   ├── decoding/                   # classical models, CV, and inputs
│   └── foundation_decoding/        # foundation models and training modes
├── descriptors.yaml                    # dataset-wide descriptor extraction
├── annotations.yaml                    # annotation normalization
```

A **cohort config** answers "who and which clinical question?" It owns fields
such as `dataset_name`, `output_group`, `conditions`, population filters, and
`evals` (including targets and label maps).

An **analysis config** answers "which method?" It owns reducers or models,
cross-validation, feature selection, tuning, input shaping, and run controls.
If both files define `conditions`, the analysis value overrides the cohort
default. The loader validates each role and then deep-merges the analysis onto
the cohort.

## Which commands use which configs

| Command | Configuration |
|---|---|
| `eeg-dim-reduce` | `--cohort_config` + `--analysis_config` |
| `eeg-classical-decode` | `--cohort_config` + `--analysis_config` |
| `eeg-foundation-decode` | `--cohort_config` + `--analysis_config` |
| `eeg-descriptors` | `--config configs/descriptors.yaml` |
| `eeg-foundation-embeddings` | one dataset-wide `--config` |
| BIDS conversion, preprocessing, and epoching | CLI arguments; no cohort split |

For an `eeg-run` range containing both consumer stages, pass the method files
separately as `--dim_analysis_config` and `--decode_analysis_config`. The generic
`--analysis_config` remains convenient for a range containing only one consumer.

## Create a run configuration

1. Copy the closest file under `cohorts/` and change its dataset name, output
   group, filters, conditions, and evaluations. Keep paths out of the YAML.
2. Copy the closest file under `analyses/<type>/` and change only method,
   input-shaping, and run-control settings.
3. Run the pair explicitly:

```bash
eeg-classical-decode \
  --cohort_config configs/cohorts/medicated_adhd_vs_controls/pooled/01_all_subjects/EO.yaml \
  --analysis_config configs/analyses/decoding/EO.yaml \
  --bids_root /path/to/BIDS \
  --metadata /path/to/patients_metadata_clean.csv
```

For dimensionality reduction, pair any compatible cohort with
`configs/analyses/dim_reduction/default.yaml`. Its `selection_eval_name` must
match the name of an entry in the cohort's `evals`; validation fails early when
they do not match.

Cluster jobs use the same pairing through `COHORT_CONFIG`, `ANALYSIS_CONFIG`,
`BIDS_ROOT`, and `METADATA_PATH`. See [`../cluster/README.md`](../cluster/README.md) for submission order and array-job details.
