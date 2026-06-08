#!/usr/bin/env bash
#SBATCH --job-name=tar_scratch
#SBATCH --account=rrg-kjerbi
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

set -euo pipefail

# Replace this with the exact name of the folder you want to compress
TARGET_DIR="/home/hamza97/scratch/data/DDM/source_rec"
TARGET_FOLDER="source_rec"

[ -d "$TARGET_DIR" ] || { echo "Directory not found: $TARGET_DIR"; exit 1; }

cd "/home/hamza97/scratch/data/DDM/"

echo "Starting blazing fast compression of $TARGET_FOLDER..."
echo "Using $SLURM_CPUS_PER_TASK CPUs for parallel zstd compression."

# -T$SLURM_CPUS_PER_TASK uses all 32 allocated cores
# -1 uses the absolute fastest compression ratio to prioritize speed over size
tar -I "zstd -T$SLURM_CPUS_PER_TASK -1" -cf "${TARGET_FOLDER}_archive.tar.zst" "$TARGET_FOLDER/"

echo "Compression complete!"
echo "Archive saved to: /home/hamza97/scratch/data/DDM/${TARGET_FOLDER}_archive.tar.zst"
echo "You can now safely delete the original folder to free up ~600k inodes."
