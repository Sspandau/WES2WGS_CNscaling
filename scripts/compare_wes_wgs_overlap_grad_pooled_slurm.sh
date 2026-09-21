#!/bin/bash
#SBATCH --job-name=wes_wgs_pooled_grad
#SBATCH --output=/home/sspandau/logs/%x_%j.out
#SBATCH --error=/home/sspandau/logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Usage:
#   sbatch compare_wes_wgs_overlap_pooled_grad_slurm.sh MANIFEST OUTDIR [MAX_FRAC] [MIN_CN_LIKE] [AGGREGATE] [MASK_REGIONS] [WES_COLUMN] [WGS_COLUMN]

MANIFEST="${1:-}"
OUTDIR="${2:-}"
MAX_FRAC="${3:-0.01}"
MIN_CN_LIKE="${4:-4.0}"
AGGREGATE="${5:-micro}"
MASK_REGIONS="${6:-/home/sspandau/CCLE_WXS/WES2WGS_CCLE/recurrent_amplification_v3_optimization/best_combo_recurrent_orange_overlap.csv}"
WES_COLUMN="${7:-predicted_loess_upscale_depth}"
WGS_COLUMN="${8:-wgs_tumor_depth}"

if [ -z "$MANIFEST" ] || [ -z "$OUTDIR" ]; then
    echo "Usage: sbatch $0 MANIFEST OUTDIR [MAX_FRAC] [MIN_CN_LIKE] [AGGREGATE] [MASK_REGIONS] [WES_COLUMN] [WGS_COLUMN]" >&2
    exit 1
fi

source /home/sspandau/miniconda3/etc/profile.d/conda.sh
conda activate wes2wgs

ARGS=(
    --manifest "$MANIFEST"
    --wes-column "$WES_COLUMN"
    --wgs-column "$WGS_COLUMN"
    --target-bin-size 25000
    --metric both
    --aggregate "$AGGREGATE"
    --max-frac "$MAX_FRAC"
    --min-cn-like "$MIN_CN_LIKE"
    --n-jobs "${SLURM_CPUS_PER_TASK:-1}"
    --outdir "$OUTDIR"
)

if [ -n "$MASK_REGIONS" ] && [ -f "$MASK_REGIONS" ]; then
    ARGS+=(--mask-regions "$MASK_REGIONS")
fi

python3 /home/sspandau/WES2WGS_CNscaling/scripts/compare_wes_wgs_overlap_grad_pooled.py "${ARGS[@]}"