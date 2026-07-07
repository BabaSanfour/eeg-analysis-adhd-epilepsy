#!/usr/bin/env bash
#SBATCH --job-name=tar_raw_data
#SBATCH --account=rrg-kjerbi
#SBATCH --output=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%A_%a.out
#SBATCH --error=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

set -euo pipefail
# No venv/module load: this is a pure tar+zstd job. Source env.sh only for the
# shared RAW_ROOT default and the require_dir helper.
PROJECT_ROOT=${PROJECT_ROOT:-/home/hamza97/EEG_psychostimulant}
source "$PROJECT_ROOT/cluster/env.sh"

SHARED_ROOT=$(dirname "$RAW_ROOT")
require_dir "$RAW_ROOT"

cd "$SHARED_ROOT"

echo "Starting high-speed compression of raw_data..."
tar -I "zstd -T$SLURM_CPUS_PER_TASK -1" -cf raw_data_archive.tar.zst raw_data/
echo "Done! The folder has been archived to raw_data_archive.tar.zst"
