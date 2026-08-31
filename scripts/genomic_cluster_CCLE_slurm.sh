#!/bin/bash
#SBATCH --job-name=genomic_cluster
#SBATCH --output=/home/sspandau/logs/%x_%A_%a.out
#SBATCH --error=/home/sspandau/logs/%x_%A_%a.err
#SBATCH --time=04:00:00        # Adjust time as needed
#SBATCH --cpus-per-task=2      # Adjust CPUs as needed
#SBATCH --mem=8G               # Adjust memory as needed

INPUT_TSV="${1:-}"
RECURRENT_MASK_REGIONS="${2:-}"
WGS_INPUT="${3:-}"
WGS_COLUMN="${4:-predicted_loess_upscale_depth}"

if [ -z "$INPUT_TSV" ]; then
    echo "Usage: sbatch genomic_cluster_CCLE_slurm.sh INPUT_TSV [MASK_REGIONS] [WGS_INPUT] [WGS_COLUMN]" >&2
    exit 1
fi

# Dynamically construct the output directory based on the input file
# e.g., turns /.../NCIH889_LUNG/SRR8618966_off_target_copy_ratios.tsv
# into /.../NCIH889_LUNG/SRR8618966_cluster_results
BASE_DIR=$(dirname "$INPUT_TSV")
FILENAME=$(basename "$INPUT_TSV")
PREFIX="${FILENAME%_off_target_copy_ratios.tsv}"
OUTDIR="${BASE_DIR}/${PREFIX}_cluster_results"

mkdir -p "$OUTDIR"

source /home/sspandau/miniconda3/etc/profile.d/conda.sh
conda activate cfamp

echo "Starting clustering for: $FILENAME"
echo "Outputting to: $OUTDIR"

ARGS=(
    --input "$INPUT_TSV"
    --column predicted_loess_upscale_depth
    --rebin-to 25000
    --max-gap auto
    --outdir "$OUTDIR"
)

if [ -n "$RECURRENT_MASK_REGIONS" ] && [ -f "$RECURRENT_MASK_REGIONS" ]; then
    ARGS+=(--mask-regions "$RECURRENT_MASK_REGIONS")
fi

if [ -n "$WGS_INPUT" ] && [ -f "$WGS_INPUT" ]; then
    ARGS+=(
        --compare-wgs-input "$WGS_INPUT"
        --compare-wgs-column "$WGS_COLUMN"
        --compare-wes-quantile 0.90
        --compare-wgs-quantile 0.90
        --compare-rebin-to 25000
    )
fi

python3 /home/sspandau/WES2WGS_CNscaling/scripts/genomic_cluster_upscaleddepth.py "${ARGS[@]}"

echo "Finished clustering for: $FILENAME"
