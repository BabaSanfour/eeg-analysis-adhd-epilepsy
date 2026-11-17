#!/usr/bin/env bash
#SBATCH --job-name=eeg_dimred_batch
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --array=1-210
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-$PWD}
source "$PROJECT_ROOT/cluster/env.sh"
dra_epochs_variant

# One array task = one cohort config x one scope (SCOPES, in env.sh); the analysis
# modes run in-process, at each representation. --array = configs x scopes.
CONFIGS_DIR=${CONFIGS_DIR:-$PROJECT_ROOT/configs/cohorts}
ANALYSIS_CONFIG=${ANALYSIS_CONFIG:-$PROJECT_ROOT/configs/analyses/dim_reduction/raw.yaml}
REPRESENTATIONS=(${REPRESENTATIONS:-epoch})

require_dir "$BIDS_ROOT"
require_file "$METADATA_PATH"
require_dir "$CONFIGS_DIR"
require_file "$ANALYSIS_CONFIG"

dra_activate
dra_pin_threads 1
THREADS=${SLURM_CPUS_PER_TASK:-16}

mapfile -t CONFIGS < <(find "$CONFIGS_DIR" -name "*.yaml" | sort)
TOTAL_TASKS=$((${#CONFIGS[@]} * ${#SCOPES[@]}))
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
guard_array_size "$TOTAL_TASKS"
if (( TASK_ID < 1 || TASK_ID > TOTAL_TASKS )); then
    echo "Array task $TASK_ID is outside valid task range 1-$TOTAL_TASKS; nothing to do."
    exit 0
fi
config="${CONFIGS[$(((TASK_ID - 1) / ${#SCOPES[@]}))]}"
scope="${SCOPES[$(((TASK_ID - 1) % ${#SCOPES[@]}))]}"
dra_scope_args "$scope"

echo "================================================================================"
echo "RAW DIM REDUCTION ARRAY TASK $TASK_ID / $TOTAL_TASKS"
echo "Config:   $config"
echo "Scope:    $scope"
echo "Analysis: $ANALYSIS_CONFIG"
echo "================================================================================"

for rep in "${REPRESENTATIONS[@]}"; do
    python -m eeg_adhd_epilepsy.analysis.dimensionality_reduction \
        --bids_root "$BIDS_ROOT" \
        --desc "$EPOCHS_DESC" \
        --derivative_root "$DIM_REDUCTION_ROOT" \
        --reports_root "$REPORTS_ROOT" \
        --metadata "$METADATA_PATH" \
        --cohort_config "$config" "${MATCHED_ARGS[@]}" \
        --analysis_config "$ANALYSIS_CONFIG" \
        --representation "$rep" "${SCOPE_ARGS[@]}" \
        --n_jobs "$THREADS"
done
