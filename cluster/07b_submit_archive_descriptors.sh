#!/usr/bin/env bash
#SBATCH --job-name=eeg_archive_descriptors
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=8G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

# Move the per-subject descriptor shards of one epoch variant from scratch into a
# single archive next to the merged tables on the project (07), so the project
# holds two files per variant instead of one per shard. The scratch shards are
# deleted once the archive lists every one of their files and folders.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$PWD}
source "$PROJECT_ROOT/cluster/env.sh"
dra_epochs_variant

ARCHIVE_PATH="$DESCRIPTORS_ROOT/shards.tar.zst"

require_dir "$DESCRIPTOR_SHARDS_ROOT"
require_dir "$DESCRIPTORS_ROOT"
command -v tar >/dev/null 2>&1 || {
    echo "Required command not found: tar" >&2
    exit 1
}
command -v zstd >/dev/null 2>&1 || {
    echo "Required command not found: zstd" >&2
    exit 1
}

source_parent=$(dirname "$DESCRIPTOR_SHARDS_ROOT")
source_name=$(basename "$DESCRIPTOR_SHARDS_ROOT")
temporary_archive="$DESCRIPTORS_ROOT/.shards.partial.${SLURM_JOB_ID:-$$}.tar.zst"
trap 'rm -f "$temporary_archive"' EXIT

file_count=$(find "$DESCRIPTOR_SHARDS_ROOT" -type f | wc -l)
directory_count=$(find "$DESCRIPTOR_SHARDS_ROOT" -type d | wc -l)
source_size=$(du -sh "$DESCRIPTOR_SHARDS_ROOT" | awk '{print $1}')

echo "================================================================================"
echo "DESCRIPTOR SHARD ARCHIVE ($EPOCHS_DESC)"
echo "Source:      $DESCRIPTOR_SHARDS_ROOT"
echo "Archive:     $ARCHIVE_PATH"
echo "Files:       $file_count"
echo "Directories: $directory_count"
echo "Source size: $source_size"
echo "================================================================================"

# Stream tar directly into fast, multithreaded Zstandard compression. Parquet is
# already compressed, so level 1 avoids wasting CPU while consolidating the many
# small files.
(
    cd "$source_parent"
    tar -cf - "$source_name" |
        zstd -T"${SLURM_CPUS_PER_TASK:-1}" -1 -q -o "$temporary_archive"
)

echo "Validating archive integrity..."
archive_entries=$(zstd -q -dc "$temporary_archive" | tar -tf - | wc -l)
if [ "$archive_entries" -ne $((file_count + directory_count)) ]; then
    echo "Archive lists $archive_entries entries, expected $((file_count + directory_count))." >&2
    exit 1
fi
archive_size=$(du -h "$temporary_archive" | awk '{print $1}')

mv -f "$temporary_archive" "$ARCHIVE_PATH"
trap - EXIT
rm -rf "$DESCRIPTOR_SHARDS_ROOT"

echo "================================================================================"
echo "ARCHIVE COMPLETE"
echo "Archive: $ARCHIVE_PATH"
echo "Size:    $archive_size"
echo "Entries: $archive_entries"
echo "Scratch shards deleted: $DESCRIPTOR_SHARDS_ROOT"
echo "================================================================================"
