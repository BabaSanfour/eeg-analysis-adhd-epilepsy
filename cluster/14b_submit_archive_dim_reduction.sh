#!/usr/bin/env bash
#SBATCH --job-name=eeg_archive_dimred_group
#SBATCH --account=rrg-kjerbi
#SBATCH --output=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%j.out
#SBATCH --error=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

# Archive a named GROUP of dim_reduction run-variant folders (source dirs listed
# in $DIRS_FILE, one relative folder name per line, relative to the dataset
# folder under DIM_REDUCTION_ROOT) into one .tar.zst. Sources are retained --
# this is a non-destructive consolidation, same pattern as
# 07b_submit_archive_descriptors.sh, just fanned out over multiple directories
# so a whole finished sweep (e.g. every foundation_* or raw_* run-variant) can
# be collapsed into a single archive in one pass. Delete the source folders
# yourself once the archive's reported entry count looks right -- this script
# never deletes anything.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/hamza97/EEG_psychostimulant}
source "$PROJECT_ROOT/cluster/env.sh"
dra_load_modules

BASE=${BASE:-$DIM_REDUCTION_ROOT/pooled_01_all_subjects_total}
DIRS_FILE=${DIRS_FILE:?Set DIRS_FILE to a file listing folder names, one per line}
ARCHIVE_NAME=${ARCHIVE_NAME:?Set ARCHIVE_NAME (e.g. foundation_all)}
ARCHIVE_PATH="$BASE/${ARCHIVE_NAME}.tar.zst"

command -v tar >/dev/null 2>&1
command -v zstd >/dev/null 2>&1

if [[ -e "$ARCHIVE_PATH" ]]; then
    echo "Archive already exists: $ARCHIVE_PATH" >&2
    exit 1
fi

mapfile -t DIRS < "$DIRS_FILE"
for d in "${DIRS[@]}"; do
    if [[ ! -d "$BASE/$d" ]]; then
        echo "Missing source directory: $BASE/$d" >&2
        exit 1
    fi
done

file_count=0
for d in "${DIRS[@]}"; do
    c=$(find "$BASE/$d" -type f | wc -l)
    file_count=$((file_count + c))
done
source_size=$(du -sch "${DIRS[@]/#/$BASE/}" 2>/dev/null | tail -1 | awk '{print $1}')

echo "================================================================================"
echo "DIM_REDUCTION GROUP ARCHIVE: $ARCHIVE_NAME"
echo "Folders:     ${#DIRS[@]}"
echo "Files:       $file_count"
echo "Total size:  $source_size"
echo "Archive:     $ARCHIVE_PATH"
echo "================================================================================"

temporary_archive="$BASE/.${ARCHIVE_NAME}.partial.${SLURM_JOB_ID:-$$}.tar.zst"
trap 'rm -f "$temporary_archive"' EXIT

(
    cd "$BASE"
    tar -cf - "${DIRS[@]}" |
        zstd -T"${SLURM_CPUS_PER_TASK:-1}" -1 -q -o "$temporary_archive"
)

echo "Validating archive integrity..."
archive_entries=$(zstd -q -dc "$temporary_archive" | tar -tf - | wc -l)
archive_size=$(du -h "$temporary_archive" | awk '{print $1}')

mv -f "$temporary_archive" "$ARCHIVE_PATH"
trap - EXIT

echo "================================================================================"
echo "ARCHIVE COMPLETE: $ARCHIVE_NAME"
echo "Archive: $ARCHIVE_PATH"
echo "Size:    $archive_size"
echo "Entries: $archive_entries"
echo "Source directories retained; nothing was deleted."
echo "================================================================================"
