#!/usr/bin/env bash
#SBATCH --job-name=eeg_dimred_desc
#SBATCH --account=rrg-kjerbi
#SBATCH --output=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%A_%a.out
#SBATCH --error=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --array=1-74
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

set -euo pipefail

# 1. Load Cluster Modules
module purge
module load gcc arrow/23.0.1 python/3.11

# 2. Path Configuration
PROJECT_ROOT=${PROJECT_ROOT:-/home/hamza97/EEG_psychostimulant}
BIDS_ROOT=${BIDS_ROOT:-/home/hamza97/projects/rrg-kjerbi/shared/eeg-adhdh-epilepsy/BIDS}
METADATA_PATH=${METADATA_PATH:-/home/hamza97/projects/rrg-kjerbi/shared/eeg-adhdh-epilepsy/csv/patients_metadata_clean.csv}
VENV_PATH=${VENV_PATH:-$PROJECT_ROOT/.venv}
# One array task = one cohort config. The analysis config sweeps every descriptor
# analysis mode in-process (loading each condition once), so --array equals the
# cohort count (guarded below) instead of cohorts x modes.
CONFIGS_DIR=${CONFIGS_DIR:-$PROJECT_ROOT/configs/cohorts}
ANALYSIS_CONFIG=${ANALYSIS_CONFIG:-$PROJECT_ROOT/configs/analyses/dim_reduction/descriptors.yaml}
SCRATCH_ROOT=${SCRATCH_ROOT:-/home/hamza97/scratch/eeg-epilepsy-adhd}
REPORTS_ROOT="$SCRATCH_ROOT/reports"
OVERWRITE=${OVERWRITE:-0}

# Descriptor Data Paths. Recording-level table (one row per recording) — was
# historically misnamed 'sensor_subject_features'; the merge now writes the honest
# name. Override to 'sensor_subject_features' for the true subject-pooled level.
DESC_ROOT="$BIDS_ROOT/derivatives/signal_features/descriptors/combined"
TABLE_PATH="$DESC_ROOT/sensor_recording_features.parquet"
COLUMNS_PATH="$DESC_ROOT/sensor_recording_features_feature_columns.json"

# 3. Environment Setup
cd "$PROJECT_ROOT"
source "$VENV_PATH/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTHONNOUSERSITE=1
THREADS=${SLURM_CPUS_PER_TASK:-16}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/numba_cache"
export MNE_HOME="${SLURM_TMPDIR:-/tmp}/mne_home"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl_config"
mkdir -p "$NUMBA_CACHE_DIR" "$MNE_HOME" "$MPLCONFIGDIR"

[ -d "$BIDS_ROOT" ] || { echo "BIDS root not found: $BIDS_ROOT"; exit 1; }
[ -f "$METADATA_PATH" ] || { echo "Metadata CSV not found: $METADATA_PATH"; exit 1; }
[ -d "$CONFIGS_DIR" ] || { echo "Config directory not found: $CONFIGS_DIR"; exit 1; }
[ -f "$ANALYSIS_CONFIG" ] || { echo "Analysis config not found: $ANALYSIS_CONFIG"; exit 1; }
[ -f "$TABLE_PATH" ] || { echo "Descriptor table not found: $TABLE_PATH"; exit 1; }
[ -f "$COLUMNS_PATH" ] || { echo "Descriptor feature columns not found: $COLUMNS_PATH"; exit 1; }

# 4. Map this array task to one cohort config
mapfile -t CONFIGS < <(find "$CONFIGS_DIR" -name "*.yaml" | sort)
CONFIG_COUNT=${#CONFIGS[@]}
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}

# Guard: a stale #SBATCH --array bound silently drops the trailing tasks.
if [ -n "${SLURM_ARRAY_TASK_COUNT:-}" ] && [ "$SLURM_ARRAY_TASK_COUNT" -ne "$CONFIG_COUNT" ]; then
    echo "ERROR: array size $SLURM_ARRAY_TASK_COUNT != cohort count $CONFIG_COUNT." >&2
    echo "Update '#SBATCH --array=1-$CONFIG_COUNT' (or set CONFIGS_DIR to a subtree)." >&2
    exit 1
fi

if (( TASK_ID < 1 || TASK_ID > CONFIG_COUNT )); then
    echo "Array task $TASK_ID is outside valid task range 1-$CONFIG_COUNT; nothing to do."
    exit 0
fi

config="${CONFIGS[$((TASK_ID - 1))]}"

echo "================================================================================"
echo "DESCRIPTOR DIM REDUCTION ARRAY TASK $TASK_ID / $CONFIG_COUNT"
echo "Config:   $config"
echo "Analysis: $ANALYSIS_CONFIG (analysis_modes sweep in-process)"
echo "Table:    $TABLE_PATH"
echo "================================================================================"

cmd=(
    python -m eeg_adhd_epilepsy.analysis.dimensionality_reduction
    --bids_root "$BIDS_ROOT"
    --reports_root "$REPORTS_ROOT"
    --metadata "$METADATA_PATH"
    --cohort_config "$config"
    --analysis_config "$ANALYSIS_CONFIG"
    --descriptor_table_path "$TABLE_PATH"
    --descriptor_feature_columns_path "$COLUMNS_PATH"
    --n_jobs "$THREADS"
)

if [ "$OVERWRITE" = "1" ]; then
    cmd+=(--overwrite)
fi

"${cmd[@]}"
