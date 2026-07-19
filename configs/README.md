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
│   └── decoding/                   # classical + foundation decoding methods
├── descriptors.yaml                    # dataset-wide descriptor extraction
├── annotations.yaml                    # annotation normalization
```

A **cohort config** answers "who and which clinical question?" It owns fields
such as `dataset_name`, `conditions`, population filters, and
`evals` (including targets and label maps).

An **analysis config** answers "which method?" It owns reducer specs or models,
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

The Makefile keeps method files explicit and separate: `DIM_ANALYSIS` selects
the descriptor dimensionality-reduction config, `ALIGN_DIM_ANALYSIS` selects the
foundation/alignment dimensionality-reduction config, and `DECODE_ANALYSIS`
selects the classical-decoding config.

## Create a run configuration

1. Copy the closest file under `cohorts/` and change its dataset name, output
   group, filters, conditions, and evaluations. Keep paths out of the YAML.
2. Copy the closest file under `analyses/<type>/` and change only method,
   input-shaping, and run-control settings.
3. Run the pair explicitly:

```bash
eeg-classical-decode \
  --cohort_config configs/cohorts/medicated_adhd_vs_controls/pooled/01_all_subjects/total.yaml \
  --analysis_config configs/analyses/decoding/classical.yaml \
  --bids_root /path/to/BIDS \
  --metadata /path/to/patients_metadata_clean.csv
```

For dimensionality reduction, pick the analysis config that matches the input:
`configs/analyses/dim_reduction/{raw,descriptors,foundation}.yaml`. Each one
sweeps its full analysis-mode plan **in-process** (one run loads each condition
once and reduces every mode). The config is **organized around the analysis
mode** — the unit of work, since a mode is loaded once and then swept over
reducers × n_components. `analysis_modes` is a mapping of mode -> spec, where each
spec **fully declares that mode's run**: the `reducers` to fit and the
`n_components` to sweep. There is no global default — each mode is the single
source of truth for its own sweep, so granular modes simply list a smaller range.
Every input uses the same `analysis_modes` mapping. For raw inputs the averaging
granularity is a single top-level `representation: epoch | subject` (orthogonal to
the mode's flat/sensor axis), so it lives outside `analysis_modes`, not per spec.
Descriptor/foundation inputs omit it — their granularity is set by which file is
loaded (optionally labelled for output paths via `granularity_label`).
(This deliberately diverges from decoding, which stays organized around `models`
because a model carries rich per-estimator config; a reducer is just a name, so
the mode owns it.)
The descriptor config also carries the shared `qc` block. Every config's
`selection_eval_name` must match the name of an entry in the cohort's `evals`;
validation fails early when they do not match.

Cluster jobs use the same pairing through `COHORT_CONFIG`, `ANALYSIS_CONFIG`,
`BIDS_ROOT`, and `METADATA_PATH`. See [`../cluster/README.md`](../cluster/README.md) for submission order and array-job details.
