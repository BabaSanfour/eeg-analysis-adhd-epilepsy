# Cluster (SLURM) submission scripts

Scripts are numbered in execution order. Each script activates the venv, sets
thread/cache env, validates its inputs, and runs one stage. No script names a
cluster: jobs are submitted with [cluv](https://github.com/mila-iqia/cluv) from
the project root on your own machine.

```bash
uv tool install cluster-uv      # once
cluv login fir trillium-gpu     # opens the SSH connections (asks for MFA)

# CPU stage, on fir
cluv submit fir cluster/07_submit_merge_descriptors.sh

# GPU stage, on fir: the rrg allocation there is CPU only, so def-kjerbi, plus
# the GPU slice, CPUs and memory
cluv submit fir cluster/17_submit_foundation_decode.sh --account=def-kjerbi \
    --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1 --cpus-per-task=1 --mem=32G

# GPU stage, on trillium: one full H100 with 24 cores (from [tool.cluv]), shared
# by several fits at once
cluv submit trillium-gpu cluster/17_submit_foundation_decode.sh -- FOUNDATION_N_JOBS=4

# variables of a script go after `--`; paths are relative to the project root
cluv submit fir cluster/17_submit_foundation_decode.sh --array=1 ... \
    -- MODELS=cbramod COHORT_CONFIG=configs/cohorts/checks/finetune_eo_vs_ec.yaml
```

`cluv submit` needs a clean git tree. It pushes the commit, checks it out in
`$HOME/eeg_analysis_adhd_epilepsy` on the cluster, runs `uv sync` there (the
environment is `uv.lock`), and submits with the account and GPU request of that
cluster from `[tool.cluv]` in `pyproject.toml`. Slurm logs go to
`$SCRATCH/logs/eeg_analysis_adhd_epilepsy/<cluster>_<jobid>/` and are copied back
to `logs/` here by `cluv sync`.

Every cluster has the same layout: inputs under
`/project/rrg-kjerbi/shared/eeg-adhdh-epilepsy` (`BIDS/`, `csv/`), outputs under
`$SCRATCH/eeg-epilepsy-adhd`. trillium-gpu takes GPU stages only (08, 17), and
its compute nodes have no internet. Jobs on every cluster read model weights
from `$HF_HOME` and the report scripts from a cache on scratch; fill both once per
cluster, on a login node:

```bash
cd ~/eeg_analysis_adhd_epilepsy
HF_HOME=$SCRATCH/eeg-epilepsy-adhd/hf_home .venv/bin/python -c "
from coco_pipe.decoding import get_foundation_model_spec
from huggingface_hub import snapshot_download
for key in 'cbramod labram reve luna biot signaljepa eegpt bendr'.split():
    spec = get_foundation_model_spec(key)
    snapshot_download(spec.hub_repo, revision=spec.checkpoint_revision)
snapshot_download('brain-bzh/reve-positions')"
COCO_PIPE_REPORT_ASSET_CACHE=$SCRATCH/eeg-epilepsy-adhd/coco_pipe_report_assets \
    .venv/bin/python -c "from coco_pipe.report._assets import vendor_assets; vendor_assets()"
```

## Order

| # | Script | Stage | Shape |
|---|--------|-------|-------|
| 01 | `01_submit_to_bids.sh` | raw → BIDS + pre-base QC | single (all subjects) |
| 02 | `02_tar_raw_data.sh` | archive raw data | single |
| 03 | `03_submit_base_cleaning.sh` | BIDS → `desc-base` derivatives + QC | single (all subjects) |
| 04 | `04_submit_epochs.sh` | `desc-base` → condition epochs | single |
| 05 | `05_submit_extract_descriptors_array.sh` | epochs → descriptor shards | array (per metadata row) |
| 06 | `06_verify_descriptors.sh` | audit shard completeness + QC before merge | single |
| 07 | `07_submit_merge_descriptors.sh` | shards → combined tables | single |
| 07b | `07b_submit_archive_descriptors.sh` | complete descriptor derivative → validated `.tar.zst` archive | single |
| 08 | `08_submit_foundation_embeddings.sh` | epochs → embedding shards | array (per metadata row, GPU) |
| 09 | `09_submit_align_subject_embeddings.sh` | pooled embeddings + native tokens → aligned variants | array (per model, CPU) |
| 10 | `10_submit_merge_foundation_embeddings.sh` | raw/aligned shards → combined tables, manifest, and report | single |
| 11 | `11_batch_run_dim_reduction.sh` | raw dim-reduction | array (per cohort) |
| 12 | `12_batch_run_dim_reduction_descriptors.sh` | descriptor dim-reduction | array (per cohort) |
| 13 | `13_batch_run_dim_reduction_foundation.sh` | foundation dim-reduction | array (cohorts × raw/aligned spaces × representations) |
| 14 | `14_submit_compare_foundation_dim_reduction.sh` | per-model dim-reduction runs → cross-model comparison report | single (run after 13) |
| 14b | `14b_submit_archive_dim_reduction.sh` | finished dim-reduction run-variant folders → one consolidated `.tar.zst` | single (run per group, after 11-13) |
| 15 | `15_submit_classical_decode.sh` | descriptor + saved-foundation classical decoding | array (descriptor + models × representations) |
| 16 | `16_submit_compare_classical_decode.sh` | aggregate stage-15 decoding results | single (run after stage 15) |
| 17 | `17_submit_foundation_decode.sh` | direct foundation decoding | array (per model, GPU) |
| 18 | `18_submit_compare_foundation_decode.sh` | aggregate stage-17 decoding results | single (run after stage 17) |
| 19 | `19_submit_main_dim_reduction.sh` | main-cohort dim-reduction smoke runner | single (raw/descriptors/foundation) |
| 20 | `20_submit_main_decoding.sh` | stages 15–18 one-cohort integration runner | single (CPU/GPU branches via `sbatch`) |

Stages 01, 03, 04, 07 and 10 end by running the BIDS validator on the dataset they
wrote (`validate_bids` in `env.sh`); the job fails when it reports an error. Each
derivative tree has a `.bidsignore` for the files BIDS does not define.


## Common environment variables

| Variable | Meaning | Default |
|----------|---------|---------|
| `PROJECT_ROOT` | repo checkout on the cluster | the job's working directory |
| `DATA_ROOT` | shared inputs | `/project/rrg-kjerbi/shared/eeg-adhdh-epilepsy` |
| `BIDS_ROOT` | BIDS dataset root | `$DATA_ROOT/BIDS` |
| `METADATA_PATH` | `patients_metadata_clean.csv` | `$DATA_ROOT/csv/...` |
| `RAW_ROOT` | raw recordings (01 only) | `$DATA_ROOT/raw_data` |
| `VENV_PATH` | virtualenv | `$PROJECT_ROOT/.venv` |
| `SCRATCH_ROOT` | scratch root for outputs | `$SCRATCH/eeg-epilepsy-adhd` |
| `HF_HOME` | model weights cache | `$SCRATCH_ROOT/hf_home` |
| `REPORTS_ROOT` | generated reports | `$SCRATCH_ROOT/reports` |
| `DECODING_ROOT` | decoding checkpoints/results | `$SCRATCH_ROOT/BIDS/derivatives/decoding` |
| `DIM_REDUCTION_ROOT` | dim-reduction checkpoints/results | `$SCRATCH_ROOT/BIDS/derivatives/dim_reduction` |
| `OVERWRITE` | `1` to force reprocessing | `0` |

`BIDS_ROOT` remains the input dataset on project storage. The decoding and
dimensionality-reduction roots are independent output locations, so moving those
two derivative trees to scratch does not change run hashes or prevent checkpoint
resume.

Foundation extraction honors `OVERWRITE=0` and resumes complete embedding/token
pairs. Use `OVERWRITE=1` once when migrating an existing token-free derivative
root to a configuration with `store_tokens: true`; subsequent submissions should
return to the default. Script 08 automatically submits only its second extraction
array because both arrays belong to the same stage. Submit stage 09 alignment
and stage 10 merge manually, in that order, after extraction completes.

## Two-config analysis stages (09, 11–13, 15–18)

Analysis stages take a **cohort** config (`configs/cohorts/...`, the dataset +
clinical question) **and** an **analysis** config (`configs/analyses/<type>/...`,
the method). See `../configs/README.md`.

- **09 (subject alignment)** materializes LEACE, EA-CORAL, EA-Mean, and
  token-based RA variants once per foundation model. It runs after extraction;
  its default cohort supplies the clinical diagnostic subset while transforms
  are fitted over the configured global population. Raw and per-transform
  diagnostics are checkpointed immediately in `variance_diagnostics.csv`, with
  atomic `_alignment_<model>_progress.json` state; a compatible resubmission
  skips completed diagnostics and transforms whose recorded artifacts still
  exist.
- **10 (foundation merge)** scans both raw and aligned embedding derivatives,
  then writes their combined tables, manifest, status, and report.
- **11 / 12 / 13 (dim-reduction)** sweep every cohort under `CONFIGS_DIR`
  (default `configs/cohorts`) paired with the input-specific `ANALYSIS_CONFIG`
  (`configs/analyses/dim_reduction/{raw,descriptors,foundation}.yaml`). The
  analysis config now drives the full mode sweep **in-process** (one process
  loads each condition once and reduces every analysis mode), so the `--array`
  bound is just the cohort count for 11/12, and `cohorts × representation spaces
  × representations` for 13 (epoch/recording/subject, run separately). Each base
  model contributes `none`, `leace`, `ea_coral`, `ea_mean`, and `ra`; explicit
  pooling variants such as `reve_pool-attention` are raw-only. A
  guard fails the job loudly if `#SBATCH --array=1-N` is stale; set `CONFIGS_DIR`
  to a subtree (or `BASE_MODELS` / `ALIGNMENT_TRANSFORMS` /
  `RAW_ONLY_MODELS` / `REPRESENTATIONS` for 13) to narrow the sweep. Each
  run also emits a cross-mode `rollup_leaderboard.html` for comparing
  representations.
- **14 (foundation comparison)** runs once after 13 (`--dependency=afterok`) and
  automatically gathers every raw/aligned model × representation leaderboard.
  It writes `foundation_model_comparison.html` and
  `foundation_model_comparison.csv` for every discovered cohort, or only
  `DATASET_NAME` when that environment variable is set.
- **14b (dim-reduction archive)** consolidates a *finished* group of 11-13
  run-variant folders (e.g. every `foundation_*` or `raw_*` folder for one
  cohort) into a single `.tar.zst`, the same non-destructive tar+zstd pattern
  as `07b`. `DIRS_FILE` names a text file listing the folders to bundle (one
  relative name per line, relative to `$BASE`, default the dataset's folder
  under `DIM_REDUCTION_ROOT`); `ARCHIVE_NAME` names the output. It refuses to
  run if any listed folder is missing or the archive already exists, and
  validates the archive by decompressing and listing it back before the final
  `mv`. It never deletes the source folders — only delete them yourself once
  the reported entry count looks right, and never for a group still being
  written by a live 11/12/13 job (per-fit `_SUCCESS` resumability depends on
  the live directory, not the archive). Example:
  ```bash
  printf '%s\n' foundation_bendr_flat_epoch_cfg-... foundation_biot_flat_epoch_cfg-... \
      > /home/hamza97/foundation_all.txt
  sbatch --export=ALL,DIRS_FILE=/home/hamza97/foundation_all.txt,ARCHIVE_NAME=foundation_all \
      cluster/14b_submit_archive_dim_reduction.sh
  ```
- **17 (direct foundation decoding)** trains on epochs and reports both levels
  from one run: every metric per epoch, and as `<metric>_by_recording` on the mean
  probability of each recording's test epochs.
- **15 (classical decoding)** runs one descriptor baseline plus every configured
  base foundation model × representation. Each foundation task evaluates raw,
  fold-local LEACE, EA-CORAL, EA-Mean, and RA in the same run, so transforms use
  identical folds and are not an array dimension. Array tasks suppress the
  shared report to avoid concurrent writes.
- **16 (classical comparison)** reads every completed stage-15 result and writes
  the shared head-to-head and foundation-transform reports.
- **17 (foundation decoding)** runs one GPU task per configured model using raw
  EEG epochs. Each task evaluates linear probing, LoRA, and full fine-tuning,
  while suppressing the shared report to avoid concurrent writes.
- **18 (foundation comparison)** reads every completed stage-17 result and adds
  direct linear probes to the shared decoding comparison. It separately writes
  `foundation_decoding_comparison.html` and CSV outputs containing linear-probe,
  full-fine-tuning, LoRA, and capability results for every model.
- **19 / 20 (main smoke runners)** run the hardcoded
  `medicated_adhd_vs_controls/pooled/01_all_subjects/total.yaml` cohort
  serially. Script 19 covers raw and descriptor reduction at epoch, recording,
  and subject level; its foundation branch covers all alignment transforms and
  the raw-only pooling path before running the stage-14 comparison. Script 20
  covers descriptor decoding, saved-embedding transforms at all three
  granularities, direct epoch-level linear/full/LoRA decoding, and both shared
  comparison stages. Representative defaults (`cbramod`, plus
  `reve_pool-attention` in stage 19) keep these integration runs bounded; expand
  `BASE_MODELS`, `SAVED_MODELS`, or `DIRECT_MODELS` for backend-wide coverage.
  Script 20
  intentionally leaves account/time/memory/CPU/GPU resources to the `sbatch`
  command; submit the CPU and GPU branches separately to avoid mixed-allocation
  waste. They are for end-to-end checks of the main cohort, not the full cohort
  grid. Stage 20 exposes descriptor and saved-embedding classical decoding only
  through the independent `descriptors` and `embeddings` modes.

For script 20, submit the CPU and GPU branches separately:

```bash
sbatch --account=rrg-kjerbi --time=24:00:00 --cpus-per-task=16 --mem=128G \
  cluster/20_submit_main_decoding.sh descriptors

sbatch --account=rrg-kjerbi --time=24:00:00 --cpus-per-task=16 --mem=128G \
  cluster/20_submit_main_decoding.sh embeddings

sbatch --account=def-kjerbi --time=24:00:00 --cpus-per-task=8 --mem=128G \
  --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1 \
  cluster/20_submit_main_decoding.sh foundation
```

## Foundation stages (08–10, 13, 15–18, 20)

Stages **08**, **17**, and the foundation branch of **20** need a GPU
(`--gres=gpu:1`). Stages **09** (alignment), **10** (merge), **13**
(foundation dim-reduction), **15** (classical decoding over saved embeddings),
**16** (classical comparison), and **18** (foundation comparison) are CPU-only.
For **20**, pass `--gres`
only when submitting `foundation` or `all`; use separate `descriptors`,
`embeddings` and `foundation` submissions to avoid reserving a GPU for classical
decoding. REVE
is a gated Hugging Face model — run
`hf auth login` or export `HF_TOKEN` before submitting, or REVE is skipped with
`authentication_required`.

Submit stage 15 and its comparison dependency explicitly:

```bash
stage15_job=$(sbatch --parsable cluster/15_submit_classical_decode.sh)
sbatch --dependency=afterok:"$stage15_job" \
  cluster/16_submit_compare_classical_decode.sh
```

Submit stage 17 and its comparison dependency the same way:

```bash
stage17_job=$(sbatch --parsable cluster/17_submit_foundation_decode.sh)
sbatch --dependency=afterok:"$stage17_job" \
  cluster/18_submit_compare_foundation_decode.sh
```

## Local runs

For a local, single-machine run (no SLURM), use the direct stage targets in the
top-level `Makefile` — see the repo README.
