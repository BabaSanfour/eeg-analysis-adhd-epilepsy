#!/usr/bin/env bash
# Run inside an interactive allocation, e.g.:
# salloc --time=5:00:00 \
#        --cpus-per-task=16 \
#        --mem=32G \
#        --gres=gpu:1 \
#        --account=def-kjerbi

# (If you prefer A100 + 128G, use: salloc --time=05:00:00 --cpus-per-task=16 --mem=128G --gres=gpu:a100:1 --account=def-kjerbi)

set -euo pipefail

# Path configuration
DERIV_ROOT=${DERIV_ROOT:-/home/mat/scratch/motor}
# Default output filename (1s segments, CAR+optional zapline)
OUT_CSV=${OUT_CSV:-embeddings_motor.csv}
WEIGHTS=${WEIGHTS:-/home/mat/CBraMod/pretrained_weights/pretrained_weights.pth}
DEVICE=${DEVICE:-cuda}
SES=${SES:-ses-01}
APPLY_CAR=${APPLY_CAR:-true}
APPLY_ZAPLINE=${APPLY_ZAPLINE:-false}
ZAPLINE_LINE_FREQ=${ZAPLINE_LINE_FREQ:-60}
ZAPLINE_CHUNK_LEN=${ZAPLINE_CHUNK_LEN:-30}

# Use 1-second patches (200 samples @ 200Hz)
POINTS_PER_PATCH=200
MAX_SUBJECTS=${MAX_SUBJECTS:-}

PYTHON_BIN=${PYTHON_BIN:-/home/mat/ep/bin/python}
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-/tmp/numba_cache}

cmd=("$PYTHON_BIN" "$(dirname "$0")/make_embeddings_motor.py" \
  --deriv-root "$DERIV_ROOT" \
  --out-csv "$OUT_CSV" \
  --weights "$WEIGHTS" \
  --device "$DEVICE" \
  ${MAX_SUBJECTS:+--max-subjects "$MAX_SUBJECTS"})

cmd+=(--points-per-patch "$POINTS_PER_PATCH")
if [ "$APPLY_CAR" = "true" ]; then
  cmd+=(--car)
fi
if [ "$APPLY_ZAPLINE" = "true" ]; then
  cmd+=(--zapline --zapline_line_freq "$ZAPLINE_LINE_FREQ" --zapline_chunk_length "$ZAPLINE_CHUNK_LEN")
fi

echo "Starting extraction with 1s (200-sample) patches... CAR=$APPLY_CAR, ZAPLINE=$APPLY_ZAPLINE"
"${cmd[@]}"
