#!/usr/bin/env bash
#SBATCH --job-name=eeg_epochs
#SBATCH --account=rrg-kjerbi
#SBATCH --output=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%A.out
#SBATCH --error=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%A.err
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

set -euo pipefail

module purge
module load gcc arrow/23.0.1 python/3.11

PROJECT_ROOT=${PROJECT_ROOT:-/home/hamza97/EEG_psychostimulant}
BIDS_ROOT=${BIDS_ROOT:-/home/hamza97/projects/rrg-kjerbi/shared/eeg-adhdh-epilepsy/BIDS}
VENV_PATH=${VENV_PATH:-$PROJECT_ROOT/.venv}
SEGMENT_DURATION=${SEGMENT_DURATION:-10.0}
OVERLAP=${OVERLAP:-0.0}
OVERWRITE=${OVERWRITE:-0}

[ -d "$PROJECT_ROOT" ] || { echo "Project root not found: $PROJECT_ROOT"; exit 1; }
[ -d "$BIDS_ROOT" ] || { echo "BIDS root not found: $BIDS_ROOT"; exit 1; }
[ -d "$VENV_PATH" ] || { echo "Virtual environment not found: $VENV_PATH"; exit 1; }

BASE_COUNT=$(find "$BIDS_ROOT/derivatives/preproc" -name '*desc-base_eeg.fif' -type f | wc -l)
[ "$BASE_COUNT" -gt 0 ] || { echo "No base-cleaned FIF files found under: $BIDS_ROOT/derivatives/preproc"; exit 1; }
echo "Found $BASE_COUNT base-cleaned FIF files for epoching."

cd "$PROJECT_ROOT"
source "$VENV_PATH/bin/activate"

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/numba_cache"
export MNE_HOME="${SLURM_TMPDIR:-/tmp}/mne_home"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl_config"
mkdir -p "$NUMBA_CACHE_DIR" "$MNE_HOME" "$MPLCONFIGDIR"

cmd=(
  python -m eeg_adhd_epilepsy.preproc.epochs
  --bids_root "$BIDS_ROOT"
  --desc base
  --segment_duration "$SEGMENT_DURATION"
  --overlap "$OVERLAP"
  --ignore_annotations
)

if [ "$OVERWRITE" = "1" ]; then
  cmd+=(--overwrite)
fi

"${cmd[@]}"
