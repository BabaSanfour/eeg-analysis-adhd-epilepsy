#!/usr/bin/env bash
#SBATCH --job-name=tar_raw_data
#SBATCH --account=rrg-kjerbi
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

set -euo pipefail

RAW_ROOT=${RAW_ROOT:-/home/hamza97/projects/rrg-kjerbi/shared/eeg-adhdh-epilepsy/raw_data}
SHARED_ROOT=$(dirname "$RAW_ROOT")

[ -d "$RAW_ROOT" ] || { echo "RAW root not found: $RAW_ROOT"; exit 1; }

cd "$SHARED_ROOT"

echo "Starting high-speed compression of raw_data..."
tar -I "zstd -T$SLURM_CPUS_PER_TASK -1" -cf raw_data_archive.tar.zst raw_data/
echo "Done! The folder has been archived to raw_data_archive.tar.zst"
