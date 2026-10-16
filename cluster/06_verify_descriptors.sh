#!/usr/bin/env bash
#SBATCH --job-name=eeg_verify_desc
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$PWD}
source "$PROJECT_ROOT/cluster/env.sh"
dra_epochs_variant

ROWS=${ROWS:-}
STRICT=${STRICT:-0}

require_dir "$DESCRIPTOR_SHARDS_ROOT"
require_dir "$VENV_PATH"

# CPU-light verification; no BLAS pinning.
dra_activate

# Every participants.tsv row is expected to have a shard, unless ROWS narrows it.
cmd=(
  python -m eeg_adhd_epilepsy.analysis.verify_descriptors
  --derivative_root "$DESCRIPTOR_SHARDS_ROOT"
  --reports_root "$REPORTS_ROOT"
  --metadata "$METADATA_PATH"
)
if [[ -n "$ROWS" ]]; then
  cmd+=(--rows "$ROWS")
fi
if [[ "$STRICT" == "1" ]]; then
  cmd+=(--strict)
fi

"${cmd[@]}"
