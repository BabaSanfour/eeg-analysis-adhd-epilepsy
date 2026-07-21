#!/usr/bin/env bash
#SBATCH --job-name=eeg_archive_descriptors
#SBATCH --account=rrg-kjerbi
#SBATCH --output=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%j.out
#SBATCH --error=/home/hamza97/EEG_psychostimulant/cluster/logs/slurm-%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=8G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

# Archive the complete descriptor derivative, including per-subject shards and
# combined tables. The source directory is intentionally retained because
# downstream dimensionality-reduction and decoding jobs may still be using it.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/hamza97/EEG_psychostimulant}
source "$PROJECT_ROOT/cluster/env.sh"
dra_load_modules

DESCRIPTORS_DIR=${DESCRIPTORS_DIR:-$SCRATCH_ROOT/BIDS/derivatives/signal_features/descriptors}
ARCHIVE_PATH=${ARCHIVE_PATH:-${DESCRIPTORS_DIR}.tar.zst}
OVERWRITE=${OVERWRITE:-0}

require_dir "$DESCRIPTORS_DIR"
command -v tar >/dev/null 2>&1 || {
    echo "Required command not found: tar" >&2
    exit 1
}
command -v zstd >/dev/null 2>&1 || {
    echo "Required command not found: zstd" >&2
    exit 1
}

if [[ "$ARCHIVE_PATH" != *.tar.zst ]]; then
    echo "ARCHIVE_PATH must end in .tar.zst: $ARCHIVE_PATH" >&2
    exit 1
fi
if [[ -e "$ARCHIVE_PATH" && "$OVERWRITE" != "1" ]]; then
    echo "Archive already exists: $ARCHIVE_PATH" >&2
    echo "Set OVERWRITE=1 to replace it." >&2
    exit 1
fi

archive_parent=$(dirname "$ARCHIVE_PATH")
archive_name=$(basename "$ARCHIVE_PATH" .tar.zst)
source_parent=$(dirname "$DESCRIPTORS_DIR")
source_name=$(basename "$DESCRIPTORS_DIR")
temporary_archive="$archive_parent/.${archive_name}.partial.${SLURM_JOB_ID:-$$}.tar.zst"

mkdir -p "$archive_parent"
trap 'rm -f "$temporary_archive"' EXIT

file_count=$(find "$DESCRIPTORS_DIR" -type f | wc -l)
directory_count=$(find "$DESCRIPTORS_DIR" -type d | wc -l)
source_size=$(du -sh "$DESCRIPTORS_DIR" | awk '{print $1}')

echo "================================================================================"
echo "DESCRIPTOR ARCHIVE"
echo "Source:      $DESCRIPTORS_DIR"
echo "Archive:     $ARCHIVE_PATH"
echo "Files:       $file_count"
echo "Directories: $directory_count"
echo "Source size: $source_size"
echo "================================================================================"

# Stream tar directly into fast, multithreaded Zstandard compression. Parquet
# and many model artifacts are already compressed, so level 1 avoids wasting
# CPU while consolidating the large number of small files.
(
    cd "$source_parent"
    tar -cf - "$source_name" |
        zstd -T"${SLURM_CPUS_PER_TASK:-1}" -1 -q -o "$temporary_archive"
)

echo "Validating archive integrity..."
archive_entries=$(zstd -q -dc "$temporary_archive" | tar -tf - | wc -l)
archive_size=$(du -h "$temporary_archive" | awk '{print $1}')

mv -f "$temporary_archive" "$ARCHIVE_PATH"
trap - EXIT

echo "================================================================================"
echo "ARCHIVE COMPLETE"
echo "Archive: $ARCHIVE_PATH"
echo "Size:    $archive_size"
echo "Entries: $archive_entries"
echo "Source directory retained; nothing was deleted."
echo "================================================================================"
