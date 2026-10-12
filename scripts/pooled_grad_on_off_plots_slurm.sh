#!/bin/bash
#SBATCH --job-name=viz_pooled
#SBATCH --output=/home/sspandau/logs/%x_%j.out
#SBATCH --error=/home/sspandau/logs/%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail

source /home/sspandau/miniconda3/etc/profile.d/conda.sh
conda activate wes2wgs

S=/home/sspandau/WES2WGS_CNscaling/scripts
R=/metropolis/projects/sspandau/depmap/pooled_grad_onoff_clustered_experiments_macro_tuning_10726
export PYTHONPATH="$S${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1

# --requested-only: just the figures in <run>/figures/requested/ (drop it to also make the general diagnostics)
python3 "$S/pooled_grad_on_off_plots.py" --run-dir "$R" \
  --search-script "$S/compare_wes_wgs_overlap_grad_pooled_on_off_tuning.py" \
  --mask-regions /metropolis/projects/sspandau/depmap/best_combo_recurrent_orange_overlap_093026.csv \
  --compare-mask /metropolis/projects/sspandau/depmap/recurrent_bins_search/best_combo_recurrent_orange_overlap.csv \
  --classification-tsv /metropolis/projects/sspandau/depmap/recurrent_bins_inputs/amplicon_classification_profiles.merged.tsv \
  --n-jobs "${SLURM_CPUS_PER_TASK:-8}"
