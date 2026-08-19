#!/bin/bash

# Base directory containing the WES samples
WES_BASE_DIR="/pedigree2/cui/CCLE_WXS"

# Ensure the logs directory exists for Slurm output
mkdir -p logs

# Loop through all directories in the base folder
for dir in "$WES_BASE_DIR"/*/; do
    
    # Extract just the directory name (removes the path and trailing slash)
    SAMPLE=$(basename "$dir")

    # Exclude specific directories
    if [[ "$SAMPLE" == "1000genomes_highcov_WGS" || "$SAMPLE" == "GRCh37" || "$SAMPLE" == "logs" ]]; then
        echo "Skipping excluded directory: $SAMPLE"
        continue
    fi

    # Submit the Slurm job, passing the sample name as the first argument
    echo "Submitting job for sample: $SAMPLE"
    sbatch scaling_WES_CCLE_slurm.sh "$SAMPLE"

done
