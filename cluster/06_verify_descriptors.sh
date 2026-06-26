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

module purge
module load gcc arrow/23.0.1 python/3.11

PROJECT_ROOT=${PROJECT_ROOT:-/home/hamza97/EEG_psychostimulant}
SCRATCH_ROOT=${SCRATCH_ROOT:-/home/hamza97/scratch/eeg-epilepsy-adhd}
DERIVATIVE_ROOT=${DERIVATIVE_ROOT:-$SCRATCH_ROOT/BIDS/derivatives/signal_features/descriptors}
REPORTS_ROOT=${REPORTS_ROOT:-$SCRATCH_ROOT/reports}
METADATA_PATH=${METADATA_PATH:-/home/hamza97/projects/rrg-kjerbi/shared/eeg-adhdh-epilepsy/csv/patients_metadata_clean.csv}
VENV_PATH=${VENV_PATH:-$PROJECT_ROOT/.venv}
ROWS=${ROWS:-}
STRICT=${STRICT:-0}

[ -d "$PROJECT_ROOT" ] || { echo "Project root not found: $PROJECT_ROOT"; exit 1; }
[ -d "$DERIVATIVE_ROOT" ] || { echo "Derivative root not found: $DERIVATIVE_ROOT"; exit 1; }
[ -d "$VENV_PATH" ] || { echo "Virtual environment not found: $VENV_PATH"; exit 1; }

cd "$PROJECT_ROOT"
source "$VENV_PATH/bin/activate"
export PYTHONNOUSERSITE=1

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
