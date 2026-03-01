#!/usr/bin/env bash
#SBATCH --job-name=embed_reve
#SBATCH --output=/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/data/results/dl/logs/reve/embed_%j.out
#SBATCH --error=/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/data/results/dl/logs/reve/embed_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:1

set -euo pipefail

# 1. Environment Setup
module load cuda # Ensure CUDA libraries are available
source /home/mat/projects/EEG_psychostimulant/.venv/bin/activate

# Ensure Hugging Face token is available for gated models
if [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
fi

# Limit threading to prevent resource issues
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

# Define directories
LOG_DIR="/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/data/results/dl/logs/reve"
OUT_DIR="/home/mat/scratch/extracted_embeddings/reve"

mkdir -p "$LOG_DIR"
mkdir -p "$OUT_DIR"

# 2. Configuration
# We assume the project root is accessible or we are in it.
# Let's set PYTHONPATH or navigate to root
cd /home/mat/projects/EEG_psychostimulant

DATA_ROOT=${DATA_ROOT:-"/home/mat/scratch/preproc/"} # Adjust this default
DEVICE="cuda"
SUBJECT=${SUBJECT:-""}  # Optional: specify a single subject to test

echo "Starting REVE embedding generation..."
echo "  - Script: eeg_adhd_epilepsy/dl/reve/reve_extract.py"
echo "  - Data Root: $DATA_ROOT"
echo "  - Output Dir: $OUT_DIR"

# 3. Build command
STAGE=${STAGE:-"baseline"}
MODEL_SIZE=${MODEL_SIZE:-"base"}
echo "  - Stage: $STAGE"
echo "  - Model Size: $MODEL_SIZE"
LIMIT=${LIMIT:-""}
echo "  - Limit: $LIMIT"
NO_POOL=${NO_POOL:-""}
echo "  - No Pool: $NO_POOL"

cmd=(python eeg_adhd_epilepsy/dl/reve/reve_extract.py \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUT_DIR" \
  --stage "$STAGE" \
  --model-size "$MODEL_SIZE" \
  --device "$DEVICE")

# Add subject filter if specified
if [ -n "$SUBJECT" ]; then
  cmd+=(--subject "$SUBJECT")
fi

# Add limit if specified
if [ -n "$LIMIT" ]; then
  cmd+=(--limit "$LIMIT")
fi

# Disable pooling if specified
if [ -n "$NO_POOL" ]; then
  cmd+=(--no-pool)
fi

# 4. Execute
"${cmd[@]}"

echo "Job finished."
