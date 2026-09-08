#!/bin/bash
#SBATCH --job-name=wes_wgs_overlap
#SBATCH --output=/home/sspandau/logs/%x_%A_%a.out
#SBATCH --error=/home/sspandau/logs/%x_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G

WES_INPUT="${1:-}"
WGS_INPUT="${2:-}"
MASK_REGIONS="${3:-/home/sspandau/CCLE_WXS/WES2WGS_CCLE/recurrent_amplification_v3_optimization/best_combo_recurrent_orange_overlap.csv}"
WES_COLUMN="${4:-predicted_loess_upscale_depth}"
WGS_COLUMN="${5:-wgs_tumor_depth}"
OUTDIR="${6:-}"

if [ -z "$WES_INPUT" ] || [ -z "$WGS_INPUT" ]; then
    echo "Usage: sbatch compare_wes_wgs_overlap_slurm.sh WES_INPUT WGS_INPUT [MASK_REGIONS] [WES_COLUMN] [WGS_COLUMN] [OUTDIR]" >&2
    exit 1
fi

if [ -z "$OUTDIR" ]; then
    OUTDIR="$(dirname "$WES_INPUT")/wes_wgs_overlap_output"
fi

source /home/sspandau/miniconda3/etc/profile.d/conda.sh
conda activate cfamp

ARGS=(
    --wes-input "$WES_INPUT"
    --wes-column "$WES_COLUMN"
    --wgs-input "$WGS_INPUT"
    --wgs-column "$WGS_COLUMN"
    --rebin-to 25000
    --smooth-window 5
    --wes-quantile 0.90
    --wgs-quantile 0.90
    --cn-floor 3.0
    --outdir "$OUTDIR"
)

if [ -n "$MASK_REGIONS" ] && [ -f "$MASK_REGIONS" ]; then
    ARGS+=(--mask-regions "$MASK_REGIONS")
fi

python3 /home/sspandau/WES2WGS_CNscaling/scripts/compare_wes_wgs_overlap.py "${ARGS[@]}"
