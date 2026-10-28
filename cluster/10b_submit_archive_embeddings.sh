#!/usr/bin/env bash
#SBATCH --job-name=eeg_archive_embeddings
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca

# Move one model's per-recording embedding files (stage 08) from scratch into one
# archive per label on the project, desc-<label>_shards.tar.zst next to the label's
# all-recordings files (10), as 07b does for the descriptor shards. A label is archived
# only once 10 has written its window file to the project; its scratch files are
# deleted once the archive lists every one of them.
#   cluv submit fir cluster/10b_submit_archive_embeddings.sh -- MODEL=reve

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$PWD}
source "$PROJECT_ROOT/cluster/env.sh"

: "${MODEL:?Set MODEL=<model key> (after --).}"
SOURCE=$EMBEDDING_SCRATCH_ROOT/coco-pipe-$MODEL
TARGET=$EMBEDDING_PROJECT_ROOT/coco-pipe-$MODEL
require_dir "$SOURCE"
require_dir "$TARGET"

list=$(mktemp)
trap 'rm -f "$list" "$TARGET"/.*.partial.tar.zst' EXIT
for manifest in "$SOURCE"/desc-*_manifest.json; do
    label=${manifest##*/desc-}
    label=${label%_manifest.json}
    archive=$TARGET/desc-${label}_shards.tar.zst
    if [ ! -f "$TARGET/task-clinical_desc-${label}_windowembeddings.nc" ]; then
        echo "REFUSE $label: no all-recordings window file on the project; run 10 first." >&2
        exit 1
    fi
    (cd "$SOURCE" && find sub-* -type f -name "*_desc-${label}_*" | sort) > "$list"
    n_files=$(wc -l < "$list")
    if [ "$n_files" -eq 0 ]; then
        echo "$label: no per-recording file left on scratch"
        continue
    fi
    [ -e "$archive" ] && { echo "REFUSE $label: $archive exists" >&2; exit 1; }
    partial=$TARGET/.${label}.partial.tar.zst
    tar -C "$SOURCE" -cf - -T "$list" | zstd -T"${SLURM_CPUS_PER_TASK:-1}" -1 -q -o "$partial"
    n_archived=$(zstd -q -dc "$partial" | tar -tf - | wc -l)
    if [ "$n_archived" -ne "$n_files" ]; then
        echo "$label: archive lists $n_archived files, expected $n_files." >&2
        exit 1
    fi
    mv "$partial" "$archive"
    (cd "$SOURCE" && xargs -a "$list" rm -f --)
    echo "$label: $n_files files -> $archive ($(du -h "$archive" | cut -f1))"
done
find "$SOURCE" -mindepth 1 -type d -empty -delete
echo "Left on scratch: $(find "$SOURCE" -type f | wc -l) files (manifests and dataset files)"
