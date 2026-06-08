#!/usr/bin/env bash
#SBATCH --job-name=eeg_dimred_batch
#SBATCH --account=rrg-kjerbi
#SBATCH --output=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%A_%a.out
#SBATCH --error=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --array=1-142
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
CONFIGS_DIR=${CONFIGS_DIR:-$PROJECT_ROOT/configs/medicated_adhd_vs_controls}
REPORTS_ROOT="${BIDS_ROOT%/*}/reports"

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

# 4. Map this array task to one config/mode pair
mapfile -t CONFIGS < <(find "$CONFIGS_DIR" -name "*.yaml" | sort)
MODES=("flat:recording_flat" "sensor:recording_native")
CONFIG_COUNT=${#CONFIGS[@]}
MODE_COUNT=${#MODES[@]}
TOTAL_TASKS=$((CONFIG_COUNT * MODE_COUNT))
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}

if (( TASK_ID < 1 || TASK_ID > TOTAL_TASKS )); then
    echo "Array task $TASK_ID is outside valid task range 1-$TOTAL_TASKS; nothing to do."
    exit 0
fi

task_index=$((TASK_ID - 1))
mode_index=$((task_index / CONFIG_COUNT))
config_index=$((task_index % CONFIG_COUNT))
mode_spec="${MODES[$mode_index]}"
mode="${mode_spec%%:*}"
representation="${mode_spec##*:}"
config="${CONFIGS[$config_index]}"
input_mode="raw"
aggregation_unit="${AGGREGATION_UNIT:-recording}"

ds_name=$(grep "dataset_name:" "$config" | awk '{print $2}')
out_grp=$(grep "output_group:" "$config" | awk '{print $2}')
report_repr="${representation//_/-}"
report_path="$REPORTS_ROOT/summary/dim_reduction/$out_grp/$ds_name/$input_mode/dataset_summary_mode-${mode}_unit-${aggregation_unit}_repr-${report_repr}.html"

echo "================================================================================"
echo "DIM REDUCTION ARRAY TASK $TASK_ID / $TOTAL_TASKS"
echo "Config:         $config"
echo "Mode:           $mode"
echo "Representation: $representation"
echo "Report:         $report_path"
echo "================================================================================"

if [[ -f "$report_path" ]]; then
    echo "SKIPPING: report already exists."
    exit 0
fi

python -m eeg_adhd_epilepsy.analysis.dimensionality_reduction \
    --bids_root "$BIDS_ROOT" \
    --metadata "$METADATA_PATH" \
    --config "$config" \
    --input_mode "$input_mode" \
    --analysis_mode "$mode" \
    --representation "$representation" \
    --aggregation_unit "$aggregation_unit" \
    --n_jobs "$THREADS"
