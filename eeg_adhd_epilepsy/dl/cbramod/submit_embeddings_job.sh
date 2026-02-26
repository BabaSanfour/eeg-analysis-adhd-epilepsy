#!/usr/bin/env bash
#SBATCH --job-name=embed_cbramod
#SBATCH --output=/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy_psychostimulant/data/results/dl/logs/cbramod/embed_%j.out
#SBATCH --error=/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy_psychostimulant/data/results/dl/logs/cbramod/embed_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
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
# Input: Preprocessed data root
DERIV_PROC_ROOT=${DERIV_PROC_ROOT:-"/home/mat/scratch/preproc/"}

# Output: Directory for .npy files (one per subject)
# Default location
# Output: Directory for .npy files (one per subject)
# Default location
OUT_FILE=${OUT_FILE:-"/home/mat/scratch/extracted_embeddings/cbramod"}

# Model Weights
WEIGHTS=${WEIGHTS:-"/home/mat/CBraMod/pretrained_weights/pretrained_weights.pth"}

DEVICE=${DEVICE:-"cuda"}
STAGE=${STAGE:-"base"}

# Hardcoded settings
POINTS_PER_PATCH=200
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-/tmp/numba_cache}
# Add project root to PYTHONPATH so models.cbramod can be imported if needed, 
# though CBraMod seems to be installed or available in venv/user modules.
# Just in case local modules are needed:
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

echo "Starting embedding generation via SLURM..."
echo "  - Script: $PROJECT_ROOT/dl/cbramod/make_embeddings.py"
echo "  - Input: $DERIV_PROC_ROOT"
echo "  - Output: $OUT_FILE"
echo "  - Stage: $STAGE"
echo "  - Device: $DEVICE"

# 3. Build Command
# Using absolute path to the script
cmd=("python" "$PROJECT_ROOT/dl/cbramod/make_embeddings.py" \
  --deriv-proc-root "$DERIV_PROC_ROOT" \
  --out-file "$OUT_FILE" \
  --weights "$WEIGHTS" \
  --device "$DEVICE" \
  --stage "$STAGE" \
  --points-per-patch "$POINTS_PER_PATCH")

# 4. Execute
"${cmd[@]}"
