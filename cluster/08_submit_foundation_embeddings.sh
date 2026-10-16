#!/usr/bin/env bash
#SBATCH --job-name=eeg_foundation_emb
#SBATCH --time=12:00:00
#SBATCH --array=1-305
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$PWD}
source "$PROJECT_ROOT/cluster/env.sh"
dra_epochs_variant

FOUNDATION_CONFIG=${FOUNDATION_CONFIG:-$PROJECT_ROOT/configs/foundation_extraction.yaml}
METADATA=${METADATA:-$METADATA_PATH}
OVERWRITE=${OVERWRITE:-0}

# ROWS_PER_TASK participants.tsv rows per task, loaded together so each model runs
# once over them: 4 rows x 305 tasks cover the 1,218 rows, within trillium's 500
# queued jobs per user (fir takes array indices up to 9,999).
ROWS_PER_TASK=${ROWS_PER_TASK:-4}
first_row=$(((SLURM_ARRAY_TASK_ID - 1) * ROWS_PER_TASK + 1))
METADATA_ROWS=$(seq "$first_row" $((first_row + ROWS_PER_TASK - 1)))

require_file "$FOUNDATION_CONFIG"
require_dir "$BIDS_ROOT"
require_file "$METADATA"
require_dir "$VENV_PATH"

# GPU job (embedding extraction runs on-device); no BLAS pinning. Add a
# HuggingFace cache on node-local scratch on top of the standard caches.
dra_activate

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARN: HF_TOKEN is unset; REVE (gated) will be skipped." >&2
fi

cmd=(
  python -m eeg_adhd_epilepsy.analysis.extract_foundation_embeddings
  --config "$FOUNDATION_CONFIG"
  --bids_root "$BIDS_ROOT"
  --desc "$EPOCHS_DESC"
  --metadata "$METADATA"
  --derivative_root "$EMBEDDING_ROOT"
  --metadata_rows $METADATA_ROWS
)
if [[ "$OVERWRITE" == "1" ]]; then
  cmd+=(--overwrite)
fi
"${cmd[@]}"
