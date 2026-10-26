#!/bin/bash
#SBATCH --job-name=genomic_cluster_wgs
#SBATCH --output=/home/sspandau/logs/%x_%A_%a.out
#SBATCH --error=/home/sspandau/logs/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# Clustering on the WGS tumor depth column of the same per-sample file used for WES.
# No mask regions, no WES/WGS comparison.
#
# Usage: sbatch genomic_cluster_WGS_slurm.sh INPUT_TSV [WGS_COLUMN] [REBIN_TO|none] [MAX_GAP] [MIN_REL_CN|none]
#
#   MAX_GAP     fixed number of non-high bins allowed inside a cluster (default 2), or 'auto'
#   MIN_REL_CN  absolute floor for 'high', in multiples of the sample's mean depth (default 3)

set -euo pipefail

INPUT_TSV="${1:-}"
WGS_COLUMN="${2:-wgs_tumor_depth}"
REBIN_TO="${3:-25000}"
MAX_GAP="${4:-2}"
MIN_REL_CN="${5:-3}"

if [ -z "$INPUT_TSV" ]; then
    echo "Usage: sbatch genomic_cluster_WGS_slurm.sh INPUT_TSV [WGS_COLUMN] [REBIN_TO|none] [MAX_GAP] [MIN_REL_CN|none]" >&2
    exit 1
fi

# e.g. /.../NCIH889_LUNG/SRR8618966_off_target_copy_ratios.tsv
#   -> /.../NCIH889_LUNG/SRR8618966_wgs_cluster_results
BASE_DIR=$(dirname "$INPUT_TSV")
FILENAME=$(basename "$INPUT_TSV")
PREFIX="${FILENAME%_off_target_copy_ratios.tsv}"
OUTDIR="${BASE_DIR}/${PREFIX}_wgs_cluster_results"
mkdir -p "$OUTDIR"

# conda activate hooks (e.g. openjdk's JAVA_HOME) reference unset vars, so relax -u here
set +u
source /home/sspandau/miniconda3/etc/profile.d/conda.sh
conda activate cfamp
set -u

echo "Starting WGS clustering for: $FILENAME (column: $WGS_COLUMN, max-gap: $MAX_GAP, min-rel-cn: $MIN_REL_CN)"
echo "Outputting to: $OUTDIR"

ARGS=(
    --input "$INPUT_TSV"
    --column "$WGS_COLUMN"
    --max-gap "$MAX_GAP"
    --outdir "$OUTDIR"
)
if [ "$REBIN_TO" != "none" ]; then
    ARGS+=(--rebin-to "$REBIN_TO")
fi
if [ "$MIN_REL_CN" != "none" ]; then
    ARGS+=(--min-rel-cn "$MIN_REL_CN")
fi

python3 /home/sspandau/WES2WGS_CNscaling/scripts/genomic_cluster_upscaleddepth.py "${ARGS[@]}"

echo "Finished WGS clustering for: $FILENAME"