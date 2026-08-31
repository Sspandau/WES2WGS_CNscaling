#!/bin/bash

# Usage:
#   bash wrapper_compare_wes_wgs_overlap.sh <WES_ROOT> <WGS_ROOT> <MASK_REGIONS> [WGS_COLUMN]
#
# This wrapper matches WES and WGS by sample folder name, then submits a separate
# overlap job for each sample. It does not run the clustering script.

WES_ROOT="${1:-/home/sspandau/CCLE_WXS/WES2WGS_CCLE}"
WGS_ROOT="${2:-/home/sspandau/CCLE_WGS}"
MASK_REGIONS="${3:-}"
WGS_COLUMN="${4:-predicted_loess_upscale_depth}"

mkdir -p logs

for SAMPLE_DIR in "$WES_ROOT"/*/; do
    if [ ! -d "$SAMPLE_DIR" ]; then
        continue
    fi

    SAMPLE=$(basename "$SAMPLE_DIR")
    WES_INPUT=$(find "$SAMPLE_DIR" -maxdepth 1 -type f -name "*_off_target_copy_ratios.tsv" | head -n 1)
    if [ -z "$WES_INPUT" ] || [ ! -f "$WES_INPUT" ]; then
        continue
    fi

    WGS_SAMPLE_DIR="$WGS_ROOT/$SAMPLE"
    WGS_INPUT=""
    if [ -d "$WGS_SAMPLE_DIR" ]; then
        WGS_INPUT=$(find "$WGS_SAMPLE_DIR" -type f \( -name "*.csv" -o -name "*.tsv" -o -name "*.txt" \) | head -n 1)
    fi

    if [ -z "$WGS_INPUT" ] || [ ! -f "$WGS_INPUT" ]; then
        echo "[warn] no matched WGS file found for sample '${SAMPLE}' under ${WGS_ROOT}; skipping."
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
