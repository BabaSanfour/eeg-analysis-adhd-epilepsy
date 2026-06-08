#!/usr/bin/env bash
#SBATCH --job-name=eeg_desc
#SBATCH --account=rrg-kjerbi
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --time=02:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --array=1-1000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

set -euo pipefail

module purge
module load gcc arrow/23.0.1 python/3.11

PROJECT_ROOT=${PROJECT_ROOT:-/home/hamza97/EEG_psychostimulant}
BIDS_ROOT=${BIDS_ROOT:-/home/hamza97/projects/rrg-kjerbi/shared/eeg-adhdh-epilepsy/BIDS}
SCRATCH_ROOT=${SCRATCH_ROOT:-/home/hamza97/scratch/eeg-epilepsy-adhd}
METADATA_PATH=${METADATA_PATH:-/home/hamza97/projects/rrg-kjerbi/shared/eeg-adhdh-epilepsy/csv/patients_metadata_clean.csv}
CONFIG_PATH=${CONFIG_PATH:-$PROJECT_ROOT/configs/descriptors.yaml}
VENV_PATH=${VENV_PATH:-$PROJECT_ROOT/.venv}
SUBMIT_STATE_DIR=${SUBMIT_STATE_DIR:-$PROJECT_ROOT/cluster/.descriptor_array_state}
AUTO_SUBMIT_NEXT=${AUTO_SUBMIT_NEXT:-0}
FIRST_BATCH_SIZE=${FIRST_BATCH_SIZE:-1000}
SECOND_BATCH_SIZE=${SECOND_BATCH_SIZE:-241}

THREADS=${SLURM_CPUS_PER_TASK:-16}
ROW_OFFSET=${ROW_OFFSET:-0}
METADATA_ROW=$((SLURM_ARRAY_TASK_ID + ROW_OFFSET))

[ -d "$PROJECT_ROOT" ] || { echo "Project root not found: $PROJECT_ROOT"; exit 1; }
[ -d "$BIDS_ROOT" ] || { echo "BIDS root not found: $BIDS_ROOT"; exit 1; }
[ -f "$METADATA_PATH" ] || { echo "Metadata CSV not found: $METADATA_PATH"; exit 1; }
[ -f "$CONFIG_PATH" ] || { echo "Descriptor config not found: $CONFIG_PATH"; exit 1; }
[ -d "$VENV_PATH" ] || { echo "Virtual environment not found: $VENV_PATH"; exit 1; }

cd "$PROJECT_ROOT"
source "$VENV_PATH/bin/activate"

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/numba_cache"
export MNE_HOME="${SLURM_TMPDIR:-/tmp}/mne_home"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl_config"
mkdir -p "$NUMBA_CACHE_DIR" "$MNE_HOME" "$MPLCONFIGDIR"

python -m eeg_adhd_epilepsy.analysis.extract_descriptors \
  --bids_root "$BIDS_ROOT" \
  --derivative_root "$SCRATCH_ROOT/BIDS/derivatives/signal_features/descriptors" \
  --reports_root "$SCRATCH_ROOT/reports" \
  --metadata "$METADATA_PATH" \
  --config "$CONFIG_PATH" \
  --subject_col study_id \
  --metadata_row "$METADATA_ROW" \
  --conditions all

if [[ "$AUTO_SUBMIT_NEXT" == "1" && "$ROW_OFFSET" == "0" ]]; then
  batch_state_dir="$SUBMIT_STATE_DIR/${SLURM_ARRAY_JOB_ID:-manual}"
  mkdir -p "$batch_state_dir"
  touch "$batch_state_dir/${SLURM_ARRAY_TASK_ID}.done"

  done_count=$(find "$batch_state_dir" -maxdepth 1 -name '*.done' | wc -l | tr -d ' ')
  if [[ "$done_count" -ge "$FIRST_BATCH_SIZE" ]]; then
    if mkdir "$batch_state_dir/submit_second.lock" 2>/dev/null; then
      sbatch \
        --array=1-"$SECOND_BATCH_SIZE" \
        --export=ALL,ROW_OFFSET="$FIRST_BATCH_SIZE",AUTO_SUBMIT_NEXT=0 \
        "$0"
    fi
  fi
fi
