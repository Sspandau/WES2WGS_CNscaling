#!/bin/bash

# Submit one WGS clustering job per sample, using the WGS column of each
# sample's existing *_off_target_copy_ratios.tsv (no masking).
#
# Usage:
#   bash wrapper_genomic_cluster_WGS.sh [WES_ROOT] [WGS_COLUMN] [REBIN_TO|none]

ROOT="${1:-/home/sspandau/CCLE_WXS/WES2WGS_CCLE}"
WGS_COLUMN="${2:-wgs_tumor_depth}"
REBIN_TO="${3:-25000}"
SLURM_SCRIPT=/home/sspandau/WES2WGS_CNscaling/scripts/genomic_cluster_WGS_slurm.sh

mkdir -p /home/sspandau/logs

for SAMPLE_DIR in "$ROOT"/*/; do
    [ -d "$SAMPLE_DIR" ] || continue
    SAMPLE=$(basename "$SAMPLE_DIR")

    INPUT=$(find "$SAMPLE_DIR" -maxdepth 1 -type f -name "*_off_target_copy_ratios.tsv" | head -n 1)
    if [ -z "$INPUT" ]; then
        echo "[skip] no off-target copy-ratio file for '${SAMPLE}'"
        continue
    fi

    # Skip samples whose file has no WGS column
    if ! head -n 1 "$INPUT" | tr '\t' '\n' | grep -qx "$WGS_COLUMN"; then
        echo "[skip] '${SAMPLE}': column '${WGS_COLUMN}' not in header"
        continue
    fi

    echo "Submitting WGS clustering for: $SAMPLE"
    sbatch "$SLURM_SCRIPT" "$INPUT" "$WGS_COLUMN" "$REBIN_TO"
done