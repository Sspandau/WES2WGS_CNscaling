#!/bin/bash
#SBATCH --job-name=wes2wgs_scale
#SBATCH --output=/home/sspandau/logs/ccle_scale_wes_%j.out
#SBATCH --error=/home/sspandau/logs/ccle_scale_wes_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sspandau@ucsd.edu

SAMPLE="$1"

# Define dynamic paths based on the sample name
WES_DIR="/pedigree2/cui/CCLE_WXS/${SAMPLE}/"
WGS_BAM="/nucleus/projects/cancer_cell_lines/WGS/ccle_bams/${SAMPLE}/${SAMPLE}.cs.rmdup.bam"
OUTPUT_DIR="/home/sspandau/CCLE_WXS/WES2WGS_CCLE/${SAMPLE}"
TEMP_DIR="./"

# Ensure the output directory exists
mkdir -p "$OUTPUT_DIR"

# Conda setup
source /home/sspandau/miniconda3/etc/profile.d/conda.sh
conda activate wes2wgs

# Execute the python script
echo "Starting processing for sample: $SAMPLE"

python3 /home/sspandau/WES2WGS_CNscaling/scripts/scale_WES_tracks_withmatchedWGS_080226.py \
    --offtarget_windows /home/sspandau/CCLE_WXS/WES2WGS_CCLE/v5_offtargets.bed \
    --pon_tsv /home/sspandau/CCLE_WXS/WES2WGS_CCLE/PoN_1000genomes_wgs_normals_noautosome.tsv \
    --wes_dir "$WES_DIR" \
    --wgs_bam "$WGS_BAM" \
    --output_dir "$OUTPUT_DIR" \
    --temp_dir "$TEMP_DIR"

echo "Finished processing sample: $SAMPLE"
