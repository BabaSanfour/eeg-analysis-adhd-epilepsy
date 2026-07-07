#!/usr/bin/env bash
#SBATCH --job-name=eeg_verify_desc
#SBATCH --account=rrg-kjerbi
#SBATCH --output=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%j.out
#SBATCH --error=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/hamza97/EEG_psychostimulant}
source "$PROJECT_ROOT/cluster/env.sh"
dra_load_modules

DERIVATIVE_ROOT=${DERIVATIVE_ROOT:-$SCRATCH_ROOT/BIDS/derivatives/signal_features/descriptors}
ROWS=${ROWS:-}
STRICT=${STRICT:-0}

require_dir "$DERIVATIVE_ROOT"
require_dir "$VENV_PATH"

# CPU-light verification; no BLAS pinning.
dra_activate

cmd=(
  python -m eeg_adhd_epilepsy.analysis.verify_descriptors
  --derivative_root "$DERIVATIVE_ROOT"
  --reports_root "$REPORTS_ROOT"
)
if [[ -n "$ROWS" ]]; then
  cmd+=(--metadata "$METADATA_PATH" --rows "$ROWS")
fi
if [[ "$STRICT" == "1" ]]; then
  cmd+=(--strict)
fi

"${cmd[@]}"
