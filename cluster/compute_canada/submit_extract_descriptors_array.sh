#!/usr/bin/env bash
#SBATCH --job-name=eeg_desc
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --array=1-1000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

set -euo pipefail

module purge
module load gcc arrow/23.0.1 python/3.11

PROJECT_ROOT=${PROJECT_ROOT:-/home/h/hamza97/links/eeg-analysis-adhd-epilepsy}
BIDS_ROOT=${BIDS_ROOT:-/home/h/hamza97/links/projects/aip-kjerbi/shared/eeg-epilepsy-adhd/BIDS}
METADATA_PATH=${METADATA_PATH:-/home/h/hamza97/links/projects/aip-kjerbi/shared/eeg-epilepsy-adhd/csv/EEG_Psychostimulants_PatientList_08-2025.csv}
CONFIG_PATH=${CONFIG_PATH:-$PROJECT_ROOT/configs/descriptors.yaml}
VENV_PATH=${VENV_PATH:-$PROJECT_ROOT/.venv}

SUBJECT_ID=$(printf "%04d" "$SLURM_ARRAY_TASK_ID")
THREADS=${SLURM_CPUS_PER_TASK:-16}

[ -d "$PROJECT_ROOT" ] || { echo "Project root not found: $PROJECT_ROOT"; exit 1; }
[ -d "$BIDS_ROOT" ] || { echo "BIDS root not found: $BIDS_ROOT"; exit 1; }
[ -f "$METADATA_PATH" ] || { echo "Metadata CSV not found: $METADATA_PATH"; exit 1; }
[ -f "$CONFIG_PATH" ] || { echo "Descriptor config not found: $CONFIG_PATH"; exit 1; }
[ -d "$VENV_PATH" ] || { echo "Virtual environment not found: $VENV_PATH"; exit 1; }

cd "$PROJECT_ROOT"
source "$VENV_PATH/bin/activate"

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
  --metadata "$METADATA_PATH" \
  --config "$CONFIG_PATH" \
  --subjects "$SUBJECT_ID"
