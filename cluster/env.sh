#!/usr/bin/env bash
# Shared environment + helpers for the EEG cluster scripts.
#
# This file is *sourced*, never submitted. Each numbered SLURM script keeps its
# own `#SBATCH` header for what the stage needs (time, CPUs, memory, array), and
# everything below `set -euo pipefail` -- path defaults, venv activation,
# BLAS/cache pinning, and the small guard helpers -- lives here.
#
# Nothing here or in the headers names a cluster. Jobs are submitted with
# `cluv submit <cluster> cluster/<script>` from the project root: cluv syncs the
# code, builds the venv with `uv sync`, and adds what differs per cluster
# (account, GPU request, log folder, working directory) from `[tool.cluv]` in
# pyproject.toml. The job starts in the project directory, hence `$PWD` below.
#
# The same layout is used on every cluster:
#   /project/rrg-kjerbi/shared/eeg-adhdh-epilepsy   inputs (BIDS, csv)
#   $SCRATCH/eeg-epilepsy-adhd                      outputs, reports, model cache

# --- Job variables -----------------------------------------------------------
# `cluv submit <cluster> <script> -- MODELS=cbramod COHORT_CONFIG=...` passes
# variables as script arguments; export them before the defaults are read.
for _assignment in "$@"; do
    export "$_assignment"
done

# --- Shared path defaults --------------------------------------------------
# `:=` only assigns when unset, so `--export=ALL,VAR=...` (or an exported
# environment variable) still overrides any of these.
: "${PROJECT_ROOT:=$PWD}"
: "${VENV_PATH:=$PROJECT_ROOT/.venv}"
: "${DATA_ROOT:=/project/rrg-kjerbi/shared/eeg-adhdh-epilepsy}"
: "${BIDS_ROOT:=$DATA_ROOT/BIDS}"
# Subject labels of every analysis stage; stage 01 writes them from PATIENTS_CSV.
: "${METADATA_PATH:=$BIDS_ROOT/participants.tsv}"
: "${PATIENTS_CSV:=$DATA_ROOT/csv/patients_metadata_clean.csv}"
: "${RAW_ROOT:=$DATA_ROOT/raw_data}"
: "${SCRATCH_ROOT:=$SCRATCH/eeg-epilepsy-adhd}"
: "${REPORTS_ROOT:=$SCRATCH_ROOT/reports}"
# A script that submits a follow-up job reuses the account it runs under.
if [ -n "${SLURM_JOB_ACCOUNT:-}" ]; then
    export SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-$SLURM_JOB_ACCOUNT}"
fi
# Model weights are downloaded once on a login node (cluster/README.md); jobs
# read that cache and never the network, which some compute nodes do not have.
export HF_HOME="${HF_HOME:-$SCRATCH_ROOT/hf_home}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# Report scripts (plotly, ...) are cached like the weights: home is read-only on
# some compute nodes.
export COCO_PIPE_REPORT_ASSET_CACHE="${COCO_PIPE_REPORT_ASSET_CACHE:-$SCRATCH_ROOT/coco_pipe_report_assets}"

# --- Epoch variant -----------------------------------------------------------
# Every stage from 05 on runs once per epoch variant written by stage 04:
# EPOCHS_DESC=base (every epoch) or baserejected (the epochs overlapping a BAD_
# annotation dropped), e.g. `cluv submit fir cluster/05_... -- EPOCHS_DESC=base`.
# Each variant has its own outputs and reports, so the two never mix. Descriptor
# shards are per subject on scratch; the merged tables and the shard archive go to
# the project (07, 07b).
dra_epochs_variant() {
    case "${EPOCHS_DESC:-}" in
        base|baserejected) ;;
        *) echo "Set EPOCHS_DESC=base or EPOCHS_DESC=baserejected (after --)." >&2; exit 1 ;;
    esac
    export EPOCHS_DESC
    : "${DESCRIPTOR_SHARDS_ROOT:=$SCRATCH_ROOT/BIDS/derivatives/signal_features/descriptors_$EPOCHS_DESC}"
    : "${DESCRIPTORS_ROOT:=$BIDS_ROOT/derivatives/signal_features/descriptors_$EPOCHS_DESC}"
    : "${EMBEDDING_ROOT:=$SCRATCH_ROOT/BIDS/derivatives/eeg_foundation_embeddings_$EPOCHS_DESC}"
    : "${DIM_REDUCTION_ROOT:=$SCRATCH_ROOT/BIDS/derivatives/dim_reduction_$EPOCHS_DESC}"
    : "${DECODING_ROOT:=$SCRATCH_ROOT/BIDS/derivatives/decoding_$EPOCHS_DESC}"
    case "$REPORTS_ROOT" in
        */"$EPOCHS_DESC") ;;
        *) REPORTS_ROOT="$REPORTS_ROOT/$EPOCHS_DESC" ;;
    esac
}

# --- venv + interpreter env + scratch caches -------------------------------
# cd into the project, activate the venv, and point numba/MNE/matplotlib caches
# at node-local scratch so they neither pollute nor contend on $HOME.
dra_activate() {
    cd "$PROJECT_ROOT"
    # shellcheck disable=SC1091
    source "$VENV_PATH/bin/activate"
    export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
    export PYTHONNOUSERSITE=1
    export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/numba_cache"
    export MNE_HOME="${SLURM_TMPDIR:-/tmp}/mne_home"
    export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl_config"
    export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/triton"
    mkdir -p "$NUMBA_CACHE_DIR" "$MNE_HOME" "$MPLCONFIGDIR" "$TRITON_CACHE_DIR"
}

# --- Native-library thread pinning -----------------------------------------
# Pin BLAS/NumExpr/Numba to $1 (default 1). Pin to 1 when joblib owns the outer
# task parallelism; pass the core count for one task that wants native threads.
dra_pin_threads() {
    local n="${1:-1}"
    export OMP_NUM_THREADS="$n" MKL_NUM_THREADS="$n" \
           OPENBLAS_NUM_THREADS="$n" NUMEXPR_NUM_THREADS="$n" \
           NUMBA_NUM_THREADS="$n"
}

# --- Guards ----------------------------------------------------------------
require_dir()  { [ -d "$1" ] || { echo "Directory not found: $1" >&2; exit 1; }; }
require_file() { [ -f "$1" ] || { echo "File not found: $1" >&2; exit 1; }; }

# Fail the job when the BIDS validator finds an error in the dataset at $1. Without
# `--max-rows -1` it reads only part of participants.tsv and reports a false
# PARTICIPANT_ID_MISMATCH. Call it after `dra_activate`: the validator is in the venv.
validate_bids() {
    bids-validator-deno "$1" --max-rows -1
}

# Fail fast when a stale `#SBATCH --array` bound != the real task count ($1),
# which would otherwise silently drop the trailing tasks.
guard_array_size() {
    local expected="$1"
    if [ -n "${SLURM_ARRAY_TASK_COUNT:-}" ] && [ "$SLURM_ARRAY_TASK_COUNT" -ne "$expected" ]; then
        echo "ERROR: array size $SLURM_ARRAY_TASK_COUNT != expected $expected." >&2
        echo "Update '#SBATCH --array=1-$expected' (or narrow the inputs)." >&2
        exit 1
    fi
}
