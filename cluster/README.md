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
| 08 | `08_submit_foundation_embeddings.sh` | epochs → embedding shards | array (per metadata row, GPU) |
| 09 | `09_submit_merge_foundation_embeddings.sh` | shards → manifest + report | single |
| 10 | `10_batch_run_dim_reduction.sh` | raw dim-reduction | array (per cohort) |
| 11 | `11_batch_run_dim_reduction_descriptors.sh` | descriptor dim-reduction | array (per cohort) |
| 12 | `12_batch_run_dim_reduction_foundation.sh` | foundation dim-reduction | array (cohorts × models × representations) |
| 13 | `13_submit_compare_foundation_dim_reduction.sh` | per-model dim-reduction runs → cross-model comparison report | single (run after 12) |
| 14 | `14_submit_classical_decode.sh` | classical decoding | single (one cohort × analysis) |
| 15 | `15_submit_foundation_decode.sh` | foundation decoding | single (GPU) |
| 16 | `16_submit_main_dim_reduction.sh` | main-cohort dim-reduction smoke runner | single (raw/descriptors/foundation) |
| 17 | `17_submit_main_decoding.sh` | main-cohort decoding smoke runner | single (epoch + recording, resources via `sbatch`) |


## Common environment variables

| Variable | Meaning | Default |
|----------|---------|---------|
| `PROJECT_ROOT` | repo checkout on the cluster | `/home/hamza97/EEG_psychostimulant` |
| `BIDS_ROOT` | BIDS dataset root | shared project path |
| `METADATA_PATH` | `patients_metadata_clean.csv` | shared project path |
| `RAW_ROOT` | raw recordings (01 only) | shared project path |
| `VENV_PATH` | virtualenv | `$PROJECT_ROOT/.venv` |
| `OVERWRITE` | `1` to force reprocessing | `0` |

## Two-config analysis stages (10–12, 14–15)

Analysis stages take a **cohort** config (`configs/cohorts/...`, the dataset +
clinical question) **and** an **analysis** config (`configs/analyses/<type>/...`,
the method). See `../configs/README.md`.

- **10 / 11 / 12 (dim-reduction)** sweep every cohort under `CONFIGS_DIR`
  (default `configs/cohorts`) paired with the input-specific `ANALYSIS_CONFIG`
  (`configs/analyses/dim_reduction/{raw,descriptors,foundation}.yaml`). The
  analysis config now drives the full mode sweep **in-process** (one process
  loads each condition once and reduces every analysis mode), so the `--array`
  bound is just the cohort count for 10/11, and `cohorts × models ×
  representations` for 12 (epoch/recording/subject, run as separate runs). A
  guard fails the job loudly if `#SBATCH --array=1-N` is stale; set `CONFIGS_DIR`
  to a subtree (or `MODELS` / `REPRESENTATIONS` for 12) to narrow the sweep. Each
  run also emits a cross-mode `rollup_leaderboard.html` for comparing
  representations.
- **13 (foundation comparison)** runs once after 12 (`--dependency=afterok`) and
  gathers every model × representation leaderboard for a cohort into one
  `foundation_model_comparison.html` ranking models on the same axes.
- **14 / 15 (decoding)** run a single `COHORT_CONFIG` × `ANALYSIS_CONFIG` pair;
  submit several jobs (overriding those vars) to cover a grid.
- **16 / 17 (main smoke runners)** run the hardcoded
  `medicated_adhd_vs_controls/pooled/01_all_subjects/total.yaml` cohort
  serially. Script 17 runs epoch- and recording-level decoding by default for
  both descriptor/classical decoding and foundation decoding. Script 17
  intentionally leaves account/time/memory/CPU/GPU resources to the `sbatch`
  command; submit the CPU and GPU branches separately to avoid mixed-allocation
  waste. They are for end-to-end checks of the main cohort, not the full cohort
  grid.

For script 17, submit the CPU and GPU branches separately:

```bash
sbatch --account=rrg-kjerbi --time=24:00:00 --cpus-per-task=16 --mem=128G \
  cluster/17_submit_main_decoding.sh decoding

sbatch --account=def-kjerbi --time=24:00:00 --cpus-per-task=8 --mem=128G \
  --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1 \
  cluster/17_submit_main_decoding.sh foundation
```

## Foundation stages (08, 12, 15, 17)

Need a GPU (`--gres=gpu:1`), except **12** (foundation dim-reduction) which is
CPU-only — it reduces already-extracted embeddings. For **17**, pass `--gres`
only when submitting `foundation` or `all`; use separate `decoding` and
`foundation` submissions to avoid reserving a GPU for classical decoding. REVE
is a gated Hugging Face model — run
`hf auth login` or export `HF_TOKEN` before submitting, or REVE is skipped with
`authentication_required`.

## Local runs

For a local, single-machine run of the whole chain (no SLURM) use the top-level
`Makefile` / `eeg-run` instead — see the repo README.
