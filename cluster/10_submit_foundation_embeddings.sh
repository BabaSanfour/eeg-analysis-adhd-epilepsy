#!/usr/bin/env bash
#SBATCH --job-name=eeg_foundation_emb
#SBATCH --account=rrg-kjerbi
#SBATCH --output=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%A.out
#SBATCH --error=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%A.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

set -euo pipefail

module purge
module load gcc arrow/23.0.1 python/3.11

# Dataset-wide producer: one config (configs/foundation_embeddings.example.yaml,
# or your edited copy). REVE is gated — set HF_TOKEN / `hf auth login` first.
PROJECT_ROOT=${PROJECT_ROOT:-/home/hamza97/EEG_psychostimulant}
VENV_PATH=${VENV_PATH:-$PROJECT_ROOT/.venv}
FOUNDATION_CONFIG=${FOUNDATION_CONFIG:-$PROJECT_ROOT/configs/foundation_embeddings.example.yaml}

[ -d "$PROJECT_ROOT" ] || { echo "Project root not found: $PROJECT_ROOT"; exit 1; }
[ -f "$FOUNDATION_CONFIG" ] || { echo "Foundation config not found: $FOUNDATION_CONFIG"; exit 1; }
[ -d "$VENV_PATH" ] || { echo "Virtual environment not found: $VENV_PATH"; exit 1; }

cd "$PROJECT_ROOT"
source "$VENV_PATH/bin/activate"

export PYTHONNOUSERSITE=1
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/numba_cache"
export MNE_HOME="${SLURM_TMPDIR:-/tmp}/mne_home"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl_config"
mkdir -p "$NUMBA_CACHE_DIR" "$MNE_HOME" "$MPLCONFIGDIR"

python -m eeg_adhd_epilepsy.analysis.extract_foundation_embeddings --config "$FOUNDATION_CONFIG"
