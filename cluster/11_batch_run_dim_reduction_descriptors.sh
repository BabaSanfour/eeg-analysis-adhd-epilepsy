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
PROJECT_ROOT=${PROJECT_ROOT:-/home/hamza97/EEG_psychostimulant}
source "$PROJECT_ROOT/cluster/env.sh"
dra_load_modules

# One array task = one cohort config. The analysis config sweeps every descriptor
# analysis mode in-process (loading each condition once), so --array equals the
# cohort count (guarded below) instead of cohorts x modes.
CONFIGS_DIR=${CONFIGS_DIR:-$PROJECT_ROOT/configs/cohorts}
ANALYSIS_CONFIG=${ANALYSIS_CONFIG:-$PROJECT_ROOT/configs/analyses/dim_reduction/descriptors.yaml}
OVERWRITE=${OVERWRITE:-0}

# Descriptor Data Paths. Recording-level table (one row per recording) — was
# historically misnamed 'sensor_subject_features'; the merge now writes the honest
# name. Override to 'sensor_subject_features' for the true subject-pooled level.
DESC_ROOT="$BIDS_ROOT/derivatives/signal_features/descriptors/combined"
TABLE_PATH="$DESC_ROOT/sensor_recording_features.parquet"
COLUMNS_PATH="$DESC_ROOT/sensor_recording_features_feature_columns.json"

require_dir "$BIDS_ROOT"
require_file "$METADATA_PATH"
require_dir "$CONFIGS_DIR"
require_file "$ANALYSIS_CONFIG"
require_file "$TABLE_PATH"
require_file "$COLUMNS_PATH"

dra_activate
dra_pin_threads 1
THREADS=${SLURM_CPUS_PER_TASK:-16}

# Map this array task to one cohort config.
mapfile -t CONFIGS < <(find "$CONFIGS_DIR" -name "*.yaml" | sort)
CONFIG_COUNT=${#CONFIGS[@]}
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
guard_array_size "$CONFIG_COUNT"

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
