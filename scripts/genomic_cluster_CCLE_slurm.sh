#!/bin/bash
#SBATCH --job-name=genomic_cluster
#SBATCH --output=/home/sspandau/logs/%x_%A_%a.out
#SBATCH --error=/home/sspandau/logs/%x_%A_%a.err
#SBATCH --time=04:00:00        # Adjust time as needed
#SBATCH --cpus-per-task=2      # Adjust CPUs as needed
#SBATCH --mem=8G               # Adjust memory as needed

INPUT_TSV="$1"
RECURRENT_MASK_REGIONS="$2"

# Dynamically construct the output directory based on the input file
# e.g., turns /.../NCIH889_LUNG/SRR8618966_off_target_copy_ratios.tsv 
# into /.../NCIH889_LUNG/SRR8618966_cluster_results
BASE_DIR=$(dirname "$INPUT_TSV")
FILENAME=$(basename "$INPUT_TSV")
PREFIX="${FILENAME%_off_target_copy_ratios.tsv}"
OUTDIR="${BASE_DIR}/${PREFIX}_cluster_results"

# Ensure the output directory exists
mkdir -p "$OUTDIR"

# Initialize conda and activate the cfamp environment
source /home/sspandau/miniconda3/etc/profile.d/conda.sh
conda activate cfamp

echo "Starting clustering for: $FILENAME"
echo "Outputting to: $OUTDIR"

# Execute the python script
python3 /home/sspandau/WES2WGS_CNscaling/scripts/genomic_cluster_upscaleddepth.py \
    --input "$INPUT_TSV" \
    --column predicted_loess_upscale_depth \
    --rebin-to 25000 \
    --mask-regions "$RECURRENT_MASK_REGIONS" \
    --max-gap auto \
    --outdir "$OUTDIR"

echo "Finished clustering for: $FILENAME"
