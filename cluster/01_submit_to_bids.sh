#!/usr/bin/env bash
#SBATCH --job-name=to_bids
#SBATCH --account=rrg-kjerbi
#SBATCH --output=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%A_%a.out
#SBATCH --error=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-/home/hamza97/EEG_psychostimulant}
source "$PROJECT_ROOT/cluster/env.sh"
dra_load_modules

OVERWRITE=${OVERWRITE:-1}

require_dir "$RAW_ROOT"
require_file "$METADATA_PATH"
require_dir "$VENV_PATH"

mkdir -p "$BIDS_ROOT"

dra_activate
dra_pin_threads 1
THREADS=${SLURM_CPUS_PER_TASK:-32}

cmd=(
  python -m eeg_adhd_epilepsy.preproc.to_bids
  --raw_root "$RAW_ROOT"
  --bids_root "$BIDS_ROOT"
  --metadata_csv "$METADATA_PATH"
  --reports_root "$SCRATCH_ROOT/reports"
  --with_eeg_reports
  --with_raw_qc
  --raw_qc_analysis_level both
  --n_jobs "$THREADS"
)

if [ "$OVERWRITE" = "1" ]; then
  cmd+=(--overwrite)
fi

"${cmd[@]}"
