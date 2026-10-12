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
: "${METADATA_PATH:=$DATA_ROOT/csv/patients_metadata_clean.csv}"
: "${RAW_ROOT:=$DATA_ROOT/raw_data}"
: "${SCRATCH_ROOT:=$SCRATCH/eeg-epilepsy-adhd}"
: "${REPORTS_ROOT:=$SCRATCH_ROOT/reports}"
: "${DECODING_ROOT:=$SCRATCH_ROOT/BIDS/derivatives/decoding}"
: "${DIM_REDUCTION_ROOT:=$SCRATCH_ROOT/BIDS/derivatives/dim_reduction}"
# Model weights are downloaded once on a login node; compute nodes may be offline.
export HF_HOME="${HF_HOME:-$SCRATCH_ROOT/hf_home}"

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
    mkdir -p "$NUMBA_CACHE_DIR" "$MNE_HOME" "$MPLCONFIGDIR"
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
