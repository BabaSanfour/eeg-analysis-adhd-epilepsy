#!/usr/bin/env bash
#SBATCH --job-name=to_bids
#SBATCH --account=rrg-kjerbi
#SBATCH --output=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%A_%a.out
#SBATCH --error=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

set -euo pipefail

module purge
module load gcc arrow/23.0.1 python/3.11

PROJECT_ROOT=${PROJECT_ROOT:-/home/hamza97/EEG_psychostimulant}
BIDS_ROOT=${BIDS_ROOT:-/home/hamza97/projects/rrg-kjerbi/shared/eeg-adhdh-epilepsy/BIDS}
METADATA_PATH=${METADATA_PATH:-/home/hamza97/projects/rrg-kjerbi/shared/eeg-adhdh-epilepsy/csv/patients_metadata_clean.csv}
RAW_ROOT=${RAW_ROOT:-/home/hamza97/projects/rrg-kjerbi/shared/eeg-adhdh-epilepsy/raw_data}
SCRATCH_ROOT=${SCRATCH_ROOT:-/home/hamza97/scratch/eeg-epilepsy-adhd}
VENV_PATH=${VENV_PATH:-$PROJECT_ROOT/.venv}
OVERWRITE=${OVERWRITE:-1}

THREADS=${SLURM_CPUS_PER_TASK:-32}

[ -d "$PROJECT_ROOT" ] || { echo "Project root not found: $PROJECT_ROOT"; exit 1; }
[ -d "$RAW_ROOT" ] || { echo "RAW root not found: $RAW_ROOT"; exit 1; }
[ -f "$METADATA_PATH" ] || { echo "Metadata CSV not found: $METADATA_PATH"; exit 1; }
[ -d "$VENV_PATH" ] || { echo "Virtual environment not found: $VENV_PATH"; exit 1; }

mkdir -p "$BIDS_ROOT"

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
  python -m eeg_adhd_epilepsy.preproc.to_bids
  --raw_root "$RAW_ROOT"
  --bids_root "$BIDS_ROOT"
  --metadata_csv "$METADATA_PATH"
  --reports_root "$SCRATCH_ROOT/reports"
  --with_eeg_reports
  --with_raw_qc
  --raw_qc_analysis_level both
  --n_jobs "$THREADS"
)

if [ "$OVERWRITE" = "1" ]; then
  cmd+=(--overwrite)
fi

"${cmd[@]}"
