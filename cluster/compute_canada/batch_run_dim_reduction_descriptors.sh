#!/usr/bin/env bash
#SBATCH --job-name=eeg_dimred_desc_resume
#SBATCH --output=slurm-%x-%A.out
#SBATCH --error=slurm-%x-%A.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

set -euo pipefail

# 1. Load Cluster Modules
module purge
module load gcc arrow/23.0.1 python/3.11

# 2. Path Configuration
PROJECT_ROOT=${PROJECT_ROOT:-/home/h/hamza97/links/eeg-analysis-adhd-epilepsy}
BIDS_ROOT=${BIDS_ROOT:-/home/h/hamza97/links/scratch/eeg-epilepsy-adhd/BIDS}
METADATA_PATH=${METADATA_PATH:-/home/h/hamza97/links/projects/aip-kjerbi/shared/eeg-epilepsy-adhd/csv/patients_metadata_clean.csv}
VENV_PATH=${VENV_PATH:-$PROJECT_ROOT/.venv}
CONFIGS_DIR="$PROJECT_ROOT/configs/medicated_adhd_vs_controls"
REPORTS_ROOT="${BIDS_ROOT%/*}/reports"

# Descriptor Data Paths
DESC_ROOT="$BIDS_ROOT/derivatives/signal_features/descriptors/combined"
TABLE_PATH="$DESC_ROOT/sensor_subject_features.csv"
COLUMNS_PATH="$DESC_ROOT/sensor_subject_features_feature_columns.json"

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
[ -f "$TABLE_PATH" ] || { echo "Descriptor table not found: $TABLE_PATH"; exit 1; }
[ -f "$COLUMNS_PATH" ] || { echo "Descriptor feature columns not found: $COLUMNS_PATH"; exit 1; }

# 4. Tracking
FAILED_CONFIGS=()
SKIPPED_RUNS=0
TOTAL_RUNS=0
SUCCESSFUL_RUNS=0

run_analysis() {
    local mode=$1
    local config=$2
    local input_mode="descriptors"
    # Representation is auto-set to the stem of TABLE_PATH in dimensionality_reduction.py
    local representation=$(basename "$TABLE_PATH")
    representation="${representation%.*}"
    
    # Extract dataset info from config to check for existing report
    local ds_name=$(grep "dataset_name:" "$config" | awk '{print $2}')
    local out_grp=$(grep "output_group:" "$config" | awk '{print $2}')
    local report_repr="${representation//_/-}"
    local report_path="$REPORTS_ROOT/summary/dim_reduction/$out_grp/$ds_name/$input_mode/dataset_summary_mode-${mode}_repr-${report_repr}.html"
    
    if [[ -f "$report_path" ]]; then
        echo "SKIPPING: Mode=$mode | Config=$(basename "$config") (Report already exists: $report_path)"
        SKIPPED_RUNS=$((SKIPPED_RUNS + 1))
        return 0
    fi

    echo "--------------------------------------------------------------------------------"
    echo "RUNNING: Mode=$mode | Config=$(basename "$config")"
    echo "--------------------------------------------------------------------------------"
    
    TOTAL_RUNS=$((TOTAL_RUNS + 1))
    
    # We still keep --overwrite for the runs that ARE executed, 
    # but the bash skip check above prevents re-running successful ones.
    if python -m eeg_adhd_epilepsy.analysis.dimensionality_reduction \
        --bids_root "$BIDS_ROOT" \
        --metadata "$METADATA_PATH" \
        --config "$config" \
        --input_mode "$input_mode" \
        --descriptor_table_path "$TABLE_PATH" \
        --descriptor_feature_columns_path "$COLUMNS_PATH" \
        --analysis_mode "$mode" \
        --n_jobs "$THREADS" \
        --overwrite; then
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
echo "FINAL BATCH SUMMARY (DESCRIPTORS RESUME)"
echo "================================================================================"
echo "Total attempted: $TOTAL_RUNS"
echo "Skipped:         $SKIPPED_RUNS"
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
