#!/bin/bash
#SBATCH --job-name=myc_bins
#SBATCH --output=/home/sspandau/logs/%x_%j.out
#SBATCH --error=/home/sspandau/logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
set -euo pipefail

source /home/sspandau/miniconda3/etc/profile.d/conda.sh
conda activate wes2wgs

S=/home/sspandau/WES2WGS_CNscaling/scripts
R=/metropolis/projects/sspandau/depmap/pooled_grad_onoff_clustered_experiments_macro_tuning_10726
BEDS=/metropolis/projects/sspandau/depmap/recurrent_bins_inputs/classification_bed_merged   # *_intervals.bed, WES names
export PYTHONPATH="$S${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1

[[ -d "$BEDS" ]] || { echo "ERROR: classification BED dir not found: $BEDS" >&2; exit 1; }

python3 "$S/myc_amplicon_bins.py" \
  --search-script "$S/compare_wes_wgs_overlap_grad_pooled_on_off_tuning.py" \
  --manifest "$R/manifests/pooled.all.tsv" \
  --classification-bed-dir "$BEDS" \
  --mask-regions /metropolis/projects/sspandau/depmap/best_combo_recurrent_orange_overlap_093026.csv \
  --gene MYC --gene-locus chr8:127735434-127742951 --window-mb 2.5 \
  --n-jobs "${SLURM_CPUS_PER_TASK:-8}" \
  --outdir "$R/figures_myc"
