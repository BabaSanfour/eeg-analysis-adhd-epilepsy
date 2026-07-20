#!/usr/bin/env bash
#SBATCH --job-name=eeg_dimred_batch
#SBATCH --account=rrg-kjerbi
#SBATCH --output=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%A_%a.out
#SBATCH --error=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --array=1-74
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-/home/hamza97/EEG_psychostimulant}
source "$PROJECT_ROOT/cluster/env.sh"
dra_load_modules

# One array task = one cohort config. The analysis config drives the raw
# (analysis_mode, representation) plan in-process via `analysis_modes` (each raw
# mode pins its representation), so there is no bash mode fan-out and --array
# equals the cohort count (guarded below).
CONFIGS_DIR=${CONFIGS_DIR:-$PROJECT_ROOT/configs/cohorts}
ANALYSIS_CONFIG=${ANALYSIS_CONFIG:-$PROJECT_ROOT/configs/analyses/dim_reduction/raw.yaml}

require_dir "$BIDS_ROOT"
require_file "$METADATA_PATH"
require_dir "$CONFIGS_DIR"
require_file "$ANALYSIS_CONFIG"

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
echo "RAW DIM REDUCTION ARRAY TASK $TASK_ID / $CONFIG_COUNT"
echo "Config:   $config"
echo "Analysis: $ANALYSIS_CONFIG (analysis_modes drives the in-process sweep)"
echo "================================================================================"

python -m eeg_adhd_epilepsy.analysis.dimensionality_reduction \
    --bids_root "$BIDS_ROOT" \
    --derivative_root "$DIM_REDUCTION_ROOT" \
    --reports_root "$REPORTS_ROOT" \
    --metadata "$METADATA_PATH" \
    --cohort_config "$config" \
    --analysis_config "$ANALYSIS_CONFIG" \
    --n_jobs "$THREADS"
