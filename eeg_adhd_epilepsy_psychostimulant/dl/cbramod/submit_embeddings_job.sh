#!/usr/bin/env bash
#SBATCH --job-name=embed_cbramod
#SBATCH --output=/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy_psychostimulant/data/results/dl/logs/cbramod/embed_%j.out
#SBATCH --error=/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy_psychostimulant/data/results/dl/logs/cbramod/embed_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:1
# #SBATCH --account=def-kjerbi

set -euo pipefail

# 1. Environment Setup
source /home/mat/projects/EEG_psychostimulant/.venv/bin/activate

# Define Project Root ensuring absolute paths
PROJECT_ROOT="/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy_psychostimulant"

# Create directories
mkdir -p "$PROJECT_ROOT/data/results/dl/logs/cbramod"
mkdir -p "$PROJECT_ROOT/data/results/dl/embeddings/cbramod"

# 2. Configuration
# Input: Preprocessed baseline data
DERIV_ROOT=${DERIV_ROOT:-"/home/mat/scratch/preproc/baseline/"}

# Output: HDF5 file
# Default location
OUT_FILE=${OUT_FILE:-"$PROJECT_ROOT/data/results/dl/embeddings/cbramod/cbramod_embeddings.h5"}

# Model Weights
WEIGHTS=${WEIGHTS:-"/home/mat/CBraMod/pretrained_weights/pretrained_weights.pth"}

DEVICE=${DEVICE:-"cuda"}

# Hardcoded settings
POINTS_PER_PATCH=200
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-/tmp/numba_cache}
# Add project root to PYTHONPATH so models.cbramod can be imported if needed, 
# though CBraMod seems to be installed or available in venv/user modules.
# Just in case local modules are needed:
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

echo "Starting embedding generation via SLURM..."
echo "  - Script: $PROJECT_ROOT/dl/cbramod/make_embeddings.py"
echo "  - Input: $DERIV_ROOT"
echo "  - Output: $OUT_FILE"
echo "  - Device: $DEVICE"

# 3. Build Command
# Using absolute path to the script
cmd=("python" "$PROJECT_ROOT/dl/cbramod/make_embeddings.py" \
  --deriv-root "$DERIV_ROOT" \
  --out-file "$OUT_FILE" \
  --weights "$WEIGHTS" \
  --device "$DEVICE" \
  --points-per-patch "$POINTS_PER_PATCH")

# 4. Execute
"${cmd[@]}"
