#!/bin/bash
#SBATCH --job-name=wes_wgs_cnlike_shared_pooled
#SBATCH --output=/home/sspandau/logs/%x_%j.out
#SBATCH --error=/home/sspandau/logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Usage:
#   sbatch run_pooled_cnlike_shared_slurm.sh MANIFEST OUTDIR [MAX_FRAC] [MIN_CN_LIKE] [AGGREGATE] [MASK_REGIONS] [ONTARGET_WES_COL] [ONTARGET_WGS_COL] [OFFTARGET_WES_COL] [OFFTARGET_WGS_COL] [POOLED_GRAD_SCRIPT_PATH]
#
# MANIFEST columns: sample, ontarget_input, offtarget_input

MANIFEST="${1:-}"
OUTDIR="${2:-}"
MAX_FRAC="${3:-0.01}"
MIN_CN_LIKE="${4:-4.0}"
AGGREGATE="${5:-micro}"
MASK_REGIONS="${6:-/home/sspandau/CCLE_WXS/WES2WGS_CCLE/recurrent_amplification_v3_optimization/best_combo_recurrent_orange_overlap.csv}"
ONTARGET_WES_COL="${7:-raw_wes_depth}"
ONTARGET_WGS_COL="${8:-wgs_tumor_depth}"
OFFTARGET_WES_COL="${9:-predicted_loess_upscale_depth}"
OFFTARGET_WGS_COL="${10:-wgs_tumor_depth}"
POOLED_GRAD_SCRIPT_PATH="${11:-/home/sspandau/WES2WGS_CNscaling/scripts/compare_wes_wgs_overlap_grad_pooled.py}"

if [ -z "$MANIFEST" ] || [ -z "$OUTDIR" ]; then
    echo "Usage: sbatch $0 MANIFEST OUTDIR [MAX_FRAC] [MIN_CN_LIKE] [AGGREGATE] [MASK_REGIONS] [ONTARGET_WES_COL] [ONTARGET_WGS_COL] [OFFTARGET_WES_COL] [OFFTARGET_WGS_COL] [POOLED_GRAD_SCRIPT_PATH]" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEARCH_SCRIPT="${SCRIPT_DIR}/compare_wes_wgs_overlap_grad_pooled_on_off.py"
for f in "$SEARCH_SCRIPT" "$POOLED_GRAD_SCRIPT_PATH"; do
    if [ ! -f "$f" ]; then
        echo "[-] Error: file not found: $f" >&2
        exit 1
    fi
done

source /home/sspandau/miniconda3/etc/profile.d/conda.sh
conda activate wes_cnv

ARGS=(
    --manifest "$MANIFEST"
    --pooled-grad-script-path "$POOLED_GRAD_SCRIPT_PATH"
    --ontarget-wes-column "$ONTARGET_WES_COL"
    --ontarget-wgs-column "$ONTARGET_WGS_COL"
    --offtarget-wes-column "$OFFTARGET_WES_COL"
    --offtarget-wgs-column "$OFFTARGET_WGS_COL"
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

python3 "$SEARCH_SCRIPT" "${ARGS[@]}"