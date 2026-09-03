#!/bin/bash

# Usage:
#   bash wrapper_ccle_genomic_cluster.sh <WES_ROOT> <WGS_ROOT> <MASK_REGIONS> [WGS_COLUMN]
#
# The sample match is based on the directory name (folder name), not on the BAM
# file name. Each WES sample is expected to live in a folder like: WES_ROOT/SAMPLE/
# and the matching WGS data is expected under WGS_ROOT/SAMPLE/ (or a file inside
# that folder that shares the same sample name).

WES_ROOT="${1:-/home/sspandau/CCLE_WXS/WES2WGS_CCLE}"
WGS_ROOT="${2:-/home/sspandau/CCLE_WGS}"
MASK_REGIONS="${3:-/home/sspandau/CCLE_WXS/WES2WGS_CCLE/recurrent_amplification_v3_optimization/best_combo_recurrent_orange_overlap.csv}"
WGS_COLUMN="${4:-predicted_loess_upscale_depth}"

mkdir -p logs

# Iterate WES sample folders, not BAM names.
for SAMPLE_DIR in "$WES_ROOT"/*/; do
    if [ ! -d "$SAMPLE_DIR" ]; then
        continue
    fi

    SAMPLE=$(basename "$SAMPLE_DIR")
    WGS_SAMPLE_DIR="$WGS_ROOT/$SAMPLE"
    MATCHED_WGS_FILE=""

    if [ -d "$WGS_SAMPLE_DIR" ]; then
        MATCHED_WGS_FILE=$(find "$WGS_SAMPLE_DIR" -type f \( -name "*.csv" -o -name "*.tsv" -o -name "*.txt" \) | head -n 1)
    fi

    if [ -z "$MATCHED_WGS_FILE" ]; then
        echo "[warn] no matched WGS file found for sample '${SAMPLE}' under ${WGS_ROOT}; skipping WGS comparison for this run."
    fi

    TSV_FILE=$(find "$SAMPLE_DIR" -maxdepth 1 -type f -name "*_off_target_copy_ratios.tsv" | head -n 1)
    if [ -z "$TSV_FILE" ] || [ ! -f "$TSV_FILE" ]; then
        continue
    fi

    echo "Submitting clustering job for sample: $SAMPLE"
    sbatch /home/sspandau/WES2WGS_CNscaling/scripts/genomic_cluster_CCLE_slurm.sh \
        "$TSV_FILE" \
        "$MASK_REGIONS" \
        "$MATCHED_WGS_FILE" \
        "$WGS_COLUMN"
done
