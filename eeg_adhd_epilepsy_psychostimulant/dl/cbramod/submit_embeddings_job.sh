#!/usr/bin/env bash
#SBATCH --job-name=embed_cbramod
#SBATCH --output=data/results/dl/logs/embed_%j.out
#SBATCH --error=data/results/dl/logs/embed_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --account=def-kjerbi

set -euo pipefail

# 1. Environment Setup
source /home/mat/projects/EEG_psychostimulant/.venv/bin/activate
mkdir -p data/results/dl/logs

# 2. Configuration
DERIV_ROOT=${DERIV_ROOT:-"/home/mat/scratch/derivatives/"}
OUT_CSV=${OUT_CSV:-"data/results/dl/embeddings/cbramod_embeddings.csv"}
WEIGHTS=${WEIGHTS:-"/home/mat/CBraMod/pretrained_weights/pretrained_weights.pth"}
DEVICE=${DEVICE:-"cuda"}
SES=${SES:-"ses-01"}
MAX_SUBJECTS=${MAX_SUBJECTS:-}

# Hardcoded settings
POINTS_PER_PATCH=200
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-/tmp/numba_cache}

echo "Starting embedding generation via SLURM..."
echo "  - Script: make_embeddings.py"
echo "  - Output: $OUT_CSV"
echo "  - Device: $DEVICE"

# 3. Build Command
# We use 'python' directly since the environment is activated
cmd=("python" "dl/cbramod/make_embeddings.py" \
  --deriv-root "$DERIV_ROOT" \
  --out-csv "$OUT_CSV" \
  --weights "$WEIGHTS" \
  --device "$DEVICE" \
  --ses "$SES" \
  --points-per-patch "$POINTS_PER_PATCH")

if [ -n "$MAX_SUBJECTS" ]; then
  cmd+=(--max-subjects "$MAX_SUBJECTS")
fi

# 4. Execute
"${cmd[@]}"
