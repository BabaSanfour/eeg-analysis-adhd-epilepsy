# Cluster (SLURM) submission scripts

Scripts are numbered in execution order. Each script is self-contained: it loads modules, activates the venv, sets thread/cache env, validates its inputs, and runs one stage. Override the defaults with environment variables (they all use
`${VAR:-default}`), e.g.:

```bash
sbatch --export=ALL,BIDS_ROOT=/my/BIDS,METADATA_PATH=/my/meta.csv 03_submit_base_cleaning.sh
```

## Order

| # | Script | Stage | Shape |
|---|--------|-------|-------|
| 01 | `01_submit_to_bids.sh` | raw → BIDS + pre-base QC | array (per subject) |
| 02 | `02_tar_raw_data.sh` | archive raw data | single |
| 03 | `03_submit_base_cleaning.sh` | BIDS → `desc-base` derivatives + QC | single (all subjects) |
| 04 | `04_submit_epochs.sh` | `desc-base` → condition epochs | single |
| 05 | `05_submit_extract_descriptors_array.sh` | epochs → descriptor shards | array (per metadata row) |
| 06 | `06_verify_descriptors.sh` | audit shard completeness + QC before merge | single |
| 07 | `07_submit_merge_descriptors.sh` | shards → combined tables | single |
| 08 | `08_submit_foundation_embeddings.sh` | foundation embeddings | single (GPU) |
| 09 | `09_batch_run_dim_reduction.sh` | raw dim-reduction | array (cohorts × modes) |
| 10 | `10_batch_run_dim_reduction_descriptors.sh` | descriptor dim-reduction | array (cohorts × modes) |
| 11 | `11_submit_classical_decode.sh` | classical decoding | single (one cohort × analysis) |
| 12 | `12_submit_foundation_decode.sh` | foundation decoding | single (GPU) |


## Common environment variables

| Variable | Meaning | Default |
|----------|---------|---------|
| `PROJECT_ROOT` | repo checkout on the cluster | `/home/hamza97/EEG_psychostimulant` |
| `BIDS_ROOT` | BIDS dataset root | shared project path |
| `METADATA_PATH` | `patients_metadata_clean.csv` | shared project path |
| `RAW_ROOT` | raw recordings (01 only) | shared project path |
| `VENV_PATH` | virtualenv | `$PROJECT_ROOT/.venv` |
| `OVERWRITE` | `1` to force reprocessing | `0` |

## Two-config analysis stages (09, 10, 11, 12)

Analysis stages take a **cohort** config (`configs/cohorts/...`, the dataset +
clinical question) **and** an **analysis** config (`configs/analyses/<type>/...`,
the method). See `../configs/README.md`.

- **09 / 10 (dim-reduction)** sweep every cohort under `CONFIGS_DIR`
  (default `configs/cohorts`) paired with a single `ANALYSIS_CONFIG`
  (default `configs/analyses/dim_reduction/default.yaml`). The SLURM `--array`
  bound **must equal `cohorts × modes`** — a guard fails the job loudly if it
  doesn't (update the `#SBATCH --array=1-N` line, or set `CONFIGS_DIR` to a
  subtree to narrow the sweep). With 74 cohorts: 09 = `74×2 = 148`,
  10 = `74×10 = 740`.
- **11 / 12 (decoding)** run a single `COHORT_CONFIG` × `ANALYSIS_CONFIG` pair;
  submit several jobs (overriding those vars) to cover a grid.

## Foundation stages (08, 12)

Need a GPU (`--gres=gpu:1`). REVE is a gated Hugging Face model — run
`hf auth login` or export `HF_TOKEN` before submitting, or REVE is skipped with
`authentication_required`.

## Local runs

For a local, single-machine run of the whole chain (no SLURM) use the top-level
`Makefile` / `eeg-run` instead — see the repo README.
