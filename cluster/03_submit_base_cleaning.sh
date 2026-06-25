#!/usr/bin/env bash
#SBATCH --job-name=base_clean
#SBATCH --account=rrg-kjerbi
#SBATCH --output=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%A.out
#SBATCH --error=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%A.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

set -euo pipefail

module purge
module load gcc arrow/23.0.1 python/3.11

PROJECT_ROOT=${PROJECT_ROOT:-/home/hamza97/EEG_psychostimulant}
BIDS_ROOT=${BIDS_ROOT:-/home/hamza97/projects/rrg-kjerbi/shared/eeg-adhdh-epilepsy/BIDS}
SCRATCH_ROOT=${SCRATCH_ROOT:-/home/hamza97/scratch/eeg-epilepsy-adhd}
VENV_PATH=${VENV_PATH:-$PROJECT_ROOT/.venv}
OVERWRITE=${OVERWRITE:-0}

THREADS=${SLURM_CPUS_PER_TASK:-32}

[ -d "$PROJECT_ROOT" ] || { echo "Project root not found: $PROJECT_ROOT"; exit 1; }
[ -d "$BIDS_ROOT" ] || { echo "BIDS root not found: $BIDS_ROOT"; exit 1; }
[ -d "$VENV_PATH" ] || { echo "Virtual environment not found: $VENV_PATH"; exit 1; }

VHDR_COUNT=$(find "$BIDS_ROOT" -path '*/eeg/*_eeg.vhdr' -type f | wc -l)
[ "$VHDR_COUNT" -gt 0 ] || { echo "No BIDS .vhdr EEG files found under: $BIDS_ROOT"; exit 1; }
echo "Found $VHDR_COUNT BIDS EEG recordings for base cleaning."

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
  python -m eeg_adhd_epilepsy.preproc.base
  --bids_root "$BIDS_ROOT"
  --reports_root "$SCRATCH_ROOT/reports"
  --n_jobs "$THREADS"
)

if [ "$OVERWRITE" = "1" ]; then
  cmd+=(--overwrite)
fi

"${cmd[@]}"
