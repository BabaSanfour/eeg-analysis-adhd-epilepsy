#!/usr/bin/env bash
#SBATCH --job-name=eeg_dimred_desc
#SBATCH --output=slurm-%x-%A.out
#SBATCH --error=slurm-%x-%A.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

set -uo pipefail

# 1. Load Cluster Modules
module purge
module load gcc arrow/23.0.1 python/3.11

# 2. Path Configuration
PROJECT_ROOT=${PROJECT_ROOT:-/home/h/hamza97/links/eeg-analysis-adhd-epilepsy}
BIDS_ROOT=${BIDS_ROOT:-/home/h/hamza97/links/scratch/eeg-epilepsy-adhd/BIDS}
METADATA_PATH=${METADATA_PATH:-/home/h/hamza97/links/projects/aip-kjerbi/shared/eeg-epilepsy-adhd/csv/patients_metadata_clean.csv}
VENV_PATH=${VENV_PATH:-$PROJECT_ROOT/.venv}
CONFIGS_DIR="$PROJECT_ROOT/configs/medicated_adhd_vs_controls"

# Descriptor Data Paths
DESC_ROOT="$BIDS_ROOT/derivatives/signal_features/descriptors/combined"
TABLE_PATH="$DESC_ROOT/sensor_subject_features.parquet"
COLUMNS_PATH="$DESC_ROOT/sensor_subject_features_feature_columns.json"

# 3. Environment Setup
cd "$PROJECT_ROOT"
source "$VENV_PATH/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTHONNOUSERSITE=1
THREADS=${SLURM_CPUS_PER_TASK:-16}
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"

# 4. Tracking
FAILED_CONFIGS=()
TOTAL_RUNS=0
SUCCESSFUL_RUNS=0

run_analysis() {
    local mode=$1
    local config=$2
    local representation="subject_flat"
    
    echo "--------------------------------------------------------------------------------"
    echo "RUNNING: Mode=$mode | Config=$(basename "$config")"
    echo "--------------------------------------------------------------------------------"
    
    TOTAL_RUNS=$((TOTAL_RUNS + 1))
    
    if python -m eeg_adhd_epilepsy.analysis.dimensionality_reduction \
        --bids_root "$BIDS_ROOT" \
        --metadata "$METADATA_PATH" \
        --config "$config" \
        --input_mode descriptors \
        --descriptor_table_path "$TABLE_PATH" \
        --descriptor_feature_columns_path "$COLUMNS_PATH" \
        --analysis_mode "$mode" \
        --representation "$representation" \
        --n_jobs "$THREADS"; then
        echo "SUCCESS: $mode - $(basename "$config")"
        SUCCESSFUL_RUNS=$((SUCCESSFUL_RUNS + 1))
    else
        echo "FAILED: $mode - $(basename "$config")" >&2
        FAILED_CONFIGS+=("$mode: $(basename "$config")")
    fi
}

# 5. Iteration over Analysis Modes
MODES=("flat" "sensor" "family" "sensor_within_family")

for mode in "${MODES[@]}"; do
    echo "=== STARTING $mode MODE ANALYSES ==="
    for conf in $(find "$CONFIGS_DIR" -name "*.yaml" | sort); do
        run_analysis "$mode" "$conf"
    done
done

# 6. Final Summary
echo ""
echo "================================================================================"
echo "FINAL BATCH SUMMARY (DESCRIPTORS)"
echo "================================================================================"
echo "Total attempted: $TOTAL_RUNS"
echo "Successful:      $SUCCESSFUL_RUNS"
echo "Failed:          ${#FAILED_CONFIGS[@]}"

if [ ${#FAILED_CONFIGS[@]} -ne 0 ]; then
    echo ""
    echo "LIST OF FAILED CONFIGURATIONS:"
    for failed in "${FAILED_CONFIGS[@]}"; do
        echo "  - $failed"
    done
    exit 1
else
    echo ""
    echo "All descriptor analyses completed successfully."
    exit 0
fi
