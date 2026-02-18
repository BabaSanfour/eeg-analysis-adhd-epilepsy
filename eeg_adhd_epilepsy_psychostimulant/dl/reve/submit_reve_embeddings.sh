#!/usr/bin/env bash
#SBATCH --job-name=embed_reve
#SBATCH --output=/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy_psychostimulant/data/results/dl/logs/reve/embed_%j.out
#SBATCH --error=/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy_psychostimulant/data/results/dl/logs/reve/embed_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:1

set -euo pipefail

# 1. Environment Setup
module load cuda # Ensure CUDA libraries are available
source /home/mat/projects/EEG_psychostimulant/.venv/bin/activate

# Limit threading to prevent resource issues
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

# Define directories
LOG_DIR="/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy_psychostimulant/data/results/dl/logs/reve"
OUT_DIR="/home/mat/scratch/extracted_embeddings/reve"

mkdir -p "$LOG_DIR"
mkdir -p "$OUT_DIR"

# 2. Configuration
# We assume the project root is accessible or we are in it.
# Let's set PYTHONPATH or navigate to root
cd /home/mat/projects/EEG_psychostimulant

DATA_ROOT=${DATA_ROOT:-"/home/mat/scratch/preproc/"} # Adjust this default
CSV_PATH="data/results/dl/subjects.csv"
DEVICE="cuda"
SUBJECT=${SUBJECT:-""}  # Optional: specify a single subject to test

echo "Starting REVE embedding generation..."
echo "  - Script: eeg_adhd_epilepsy_psychostimulant/dl/reve/reve_extract.py"
echo "  - Data Root: $DATA_ROOT"
echo "  - Output Dir: $OUT_DIR"

# 3. Build command
STAGE=${STAGE:-"base"}
echo "  - Stage: $STAGE"

cmd=(python eeg_adhd_epilepsy_psychostimulant/dl/reve/reve_extract.py \
  --data-root "$DATA_ROOT" \
  --csv-path "$CSV_PATH" \
  --output-dir "$OUT_DIR" \
  --stage "$STAGE" \
  --device "$DEVICE")

# Add subject filter if specified
if [ -n "$SUBJECT" ]; then
  cmd+=(--subject "$SUBJECT")
fi

# 4. Execute
"${cmd[@]}"

echo "Job finished."
