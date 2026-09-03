#!/bin/bash

# Usage:
#   bash wrapper_compare_wes_wgs_overlap.sh <WES_ROOT> <WGS_ROOT> <MASK_REGIONS> [WGS_COLUMN]
#
# This wrapper matches WES and WGS sample folders by normalized sample name, then
# submits a separate overlap job for each sample. It expects processed bin-level
# tables, not raw BAM files. The WGS value column is usually wgs_tumor_depth,
# but can be overridden with the optional 4th argument.

WES_ROOT="${1:-/home/sspandau/CCLE_WXS/WES2WGS_CCLE}"
WGS_ROOT="${2:-/home/sspandau/CCLE_WGS}"
MASK_REGIONS="${3:-}"
WGS_COLUMN="${4:-wgs_tumor_depth}"

normalize_name() {
    printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_+|_+$//g; s/_+/_/g'
}

find_matching_wgs_file() {
    local sample="$1"
    local norm_sample
    norm_sample="$(normalize_name "$sample")"

    if [ ! -d "$WGS_ROOT" ]; then
        return 1
    fi

    while IFS= read -r dir; do
        [ -n "$dir" ] || continue
        local dir_name
        dir_name="$(basename "$dir")"
        if [ "$(normalize_name "$dir_name")" = "$norm_sample" ]; then
            local candidate
            candidate=$(find "$dir" -type f \( -name "*.csv" -o -name "*.tsv" -o -name "*.txt" \) 2>/dev/null | head -n 1)
            if [ -n "$candidate" ] && [ -f "$candidate" ]; then
                printf '%s\n' "$candidate"
                return 0
            fi
        fi
    done < <(find "$WGS_ROOT" -type d 2>/dev/null)

    return 1
}

mkdir -p logs

for SAMPLE_DIR in "$WES_ROOT"/*/; do
    if [ ! -d "$SAMPLE_DIR" ]; then
        continue
    fi

    SAMPLE=$(basename "$SAMPLE_DIR")
    WES_INPUT=$(find "$SAMPLE_DIR" -maxdepth 1 -type f \( -name "*_off_target_copy_ratios.tsv" -o -name "*.tsv" -o -name "*.csv" \) | head -n 1)
    if [ -z "$WES_INPUT" ] || [ ! -f "$WES_INPUT" ]; then
        continue
    fi

    WGS_INPUT=""
    if [ -d "$WGS_ROOT" ]; then
        WGS_INPUT="$(find_matching_wgs_file "$SAMPLE")" || WGS_INPUT=""
    fi

    if [ -z "$WGS_INPUT" ] || [ ! -f "$WGS_INPUT" ]; then
        echo "[warn] no matched WGS processed file found for sample '${SAMPLE}' under ${WGS_ROOT}; skipping."
        continue
    fi

    OUTDIR="${SAMPLE_DIR}/wes_wgs_overlap_output"
    echo "Submitting overlap job for sample: $SAMPLE"
    sbatch /home/sspandau/WES2WGS_CNscaling/scripts/compare_wes_wgs_overlap_slurm.sh \
        "$WES_INPUT" \
        "$WGS_INPUT" \
        "$MASK_REGIONS" \
        "$WGS_COLUMN" \
        "$OUTDIR"
done
