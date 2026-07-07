#!/usr/bin/env bash
#SBATCH --job-name=base_clean
#SBATCH --account=rrg-kjerbi
#SBATCH --output=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%A.out
#SBATCH --error=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%A.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/hamza97/EEG_psychostimulant}
source "$PROJECT_ROOT/cluster/env.sh"
dra_load_modules

OVERWRITE=${OVERWRITE:-0}
THREADS=${SLURM_CPUS_PER_TASK:-32}

# Memory-aware concurrency: cap simultaneous subjects to the RAM budget, keeping
# ~15% headroom for the main process, report writing, and the OS. Tune
# MEM_PER_SUBJECT_GB from data with: python -m eeg_adhd_epilepsy.preproc.base
# ... --measure-peak-rss --max-mem-gb <budget>
MEM_PER_SUBJECT_GB=${MEM_PER_SUBJECT_GB:-4}
SLURM_MEM_MB=${SLURM_MEM_PER_NODE:-196608}
MAX_MEM_GB=$(( SLURM_MEM_MB / 1024 * 85 / 100 ))

require_dir "$BIDS_ROOT"
require_dir "$VENV_PATH"

VHDR_COUNT=$(find "$BIDS_ROOT" -path '*/eeg/*_eeg.vhdr' -type f | wc -l)
[ "$VHDR_COUNT" -gt 0 ] || { echo "No BIDS .vhdr EEG files found under: $BIDS_ROOT"; exit 1; }
echo "Found $VHDR_COUNT BIDS EEG recordings for base cleaning."

dra_activate
dra_pin_threads 1

echo "Memory budget: ${MAX_MEM_GB} GB usable (~${MEM_PER_SUBJECT_GB} GB/subject) across ${THREADS} cores."

cmd=(
  python -m eeg_adhd_epilepsy.preproc.base
  --bids_root "$BIDS_ROOT"
  --reports_root "$SCRATCH_ROOT/reports"
  --n_jobs "$THREADS"
  --max-mem-gb "$MAX_MEM_GB"
  --mem-per-subject-gb "$MEM_PER_SUBJECT_GB"
)

if [ "$OVERWRITE" = "1" ]; then
  cmd+=(--overwrite)
fi

"${cmd[@]}"
