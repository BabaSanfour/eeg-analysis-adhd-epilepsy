#!/usr/bin/env bash
#SBATCH --job-name=eeg_merge_desc
#SBATCH --account=rrg-kjerbi
#SBATCH --output=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%A.out
#SBATCH --error=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%A.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

set -euo pipefail

module purge
module load gcc arrow/23.0.1 python/3.11

PROJECT_ROOT=${PROJECT_ROOT:-/home/hamza97/EEG_psychostimulant}
BIDS_ROOT=${BIDS_ROOT:-/home/hamza97/projects/rrg-kjerbi/shared/eeg-adhdh-epilepsy/BIDS}
VENV_PATH=${VENV_PATH:-$PROJECT_ROOT/.venv}

[ -d "$PROJECT_ROOT" ] || { echo "Project root not found: $PROJECT_ROOT"; exit 1; }
[ -d "$BIDS_ROOT" ] || { echo "BIDS root not found: $BIDS_ROOT"; exit 1; }
[ -d "$VENV_PATH" ] || { echo "Virtual environment not found: $VENV_PATH"; exit 1; }

SHARD_COUNT=$(find "$BIDS_ROOT/derivatives/signal_features/descriptors" -name '_SUCCESS' 2>/dev/null | wc -l)
[ "$SHARD_COUNT" -gt 0 ] || { echo "No completed descriptor shards under $BIDS_ROOT"; exit 1; }
echo "Found $SHARD_COUNT completed descriptor shards to merge."

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

python -m eeg_adhd_epilepsy.analysis.merge_descriptors --bids_root "$BIDS_ROOT"
