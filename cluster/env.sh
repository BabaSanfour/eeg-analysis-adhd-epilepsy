#!/usr/bin/env bash
# Shared environment + helpers for the EEG cluster scripts.
#
# This file is *sourced*, never submitted. Each numbered SLURM script keeps its
# own `#SBATCH` header (those must be literal at the top of the submitted file),
# but everything below `set -euo pipefail` -- module load, path defaults, venv
# activation, BLAS/cache pinning, and the small guard helpers -- lives here so a
# change lands in one place instead of ~15 copies.
#
# SLURM copies the batch script to a spool dir, so `$0` at run time is not the
# cluster/ path. Each script therefore sets PROJECT_ROOT (its one bootstrap line)
# and then `source "$PROJECT_ROOT/cluster/env.sh"` to locate this file robustly.

# --- Shared path defaults --------------------------------------------------
# `:=` only assigns when unset, so `sbatch --export=...,VAR=...` (or an exported
# environment variable) still overrides any of these.
: "${PROJECT_ROOT:=/home/hamza97/EEG_psychostimulant}"
: "${VENV_PATH:=$PROJECT_ROOT/.venv}"
: "${BIDS_ROOT:=/home/hamza97/projects/rrg-kjerbi/shared/eeg-adhdh-epilepsy/BIDS}"
: "${METADATA_PATH:=/home/hamza97/projects/rrg-kjerbi/shared/eeg-adhdh-epilepsy/csv/patients_metadata_clean.csv}"
: "${RAW_ROOT:=/home/hamza97/projects/rrg-kjerbi/shared/eeg-adhdh-epilepsy/raw_data}"
: "${SCRATCH_ROOT:=/home/hamza97/scratch/eeg-epilepsy-adhd}"
: "${REPORTS_ROOT:=$SCRATCH_ROOT/reports}"
: "${DECODING_ROOT:=$SCRATCH_ROOT/BIDS/derivatives/decoding}"
: "${DIM_REDUCTION_ROOT:=$SCRATCH_ROOT/BIDS/derivatives/dim_reduction}"

# --- Cluster modules -------------------------------------------------------
dra_load_modules() {
    module purge
    module load gcc arrow/23.0.1 python/3.11
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
    mkdir -p "$NUMBA_CACHE_DIR" "$MNE_HOME" "$MPLCONFIGDIR"
}

# --- BLAS thread pinning ---------------------------------------------------
# Pin OMP/MKL/OpenBLAS/NumExpr to $1 (default 1). Pin to 1 when parallelism is at
# the process level (joblib over subjects); pass the core count for a single
# process that wants threaded BLAS (merges, per-recording descriptor extraction).
dra_pin_threads() {
    local n="${1:-1}"
    export OMP_NUM_THREADS="$n" MKL_NUM_THREADS="$n" \
           OPENBLAS_NUM_THREADS="$n" NUMEXPR_NUM_THREADS="$n"
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
