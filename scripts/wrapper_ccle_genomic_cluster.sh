#!/bin/bash

# Base directory containing the outputs from the previous step
RESULTS_DIR="/home/sspandau/CCLE_WXS/WES2WGS_CCLE"

# Ensure the logs directory exists for Slurm output
mkdir -p logs

# Find all matching TSV files in the sample subdirectories
for TSV_FILE in "$RESULTS_DIR"/*/*_off_target_copy_ratios.tsv; do
    
    # Failsafe in case no files are found (glob returns the literal string)
    if [ ! -f "$TSV_FILE" ]; then
        echo "No off-target copy ratio files found in $RESULTS_DIR"
        exit 0
    fi

    # Extract sample name for the terminal output
    SAMPLE_DIR=$(basename $(dirname "$TSV_FILE"))
    FILE_NAME=$(basename "$TSV_FILE")
    
    echo "Submitting clustering job for $SAMPLE_DIR -> $FILE_NAME"
    
    # Submit the Slurm job, passing the full path to the TSV file
    sbatch genomic_cluster_CCLE_slurm.sh "$TSV_FILE"

done
