#!/bin/bash
#SBATCH --job-name=pooled_onoff_experiments
#SBATCH --output=/home/sspandau/logs/%x_%j.out
#SBATCH --error=/home/sspandau/logs/%x_%j.err
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Pooled on+off-target WES-vs-WGS overlap experiments:
#   - CN-like computed per region (on / off target) on ALL bins first
#   - recurrent bins masked, then off-target bins outside each sample's clusters dropped
#   - Exp 1: separate on/off WES thresholds + one WGS threshold, raw AND CN-like
#   - Exp 2: one shared CN-like WES threshold + one CN-like WGS threshold
#   - each for jaccard and overlap; fit on a random 60% of samples, the other
#     40% held out and scored at the thresholds found on the 60%
#
# Submit from anywhere, after the clustering jobs finish:
#   sbatch ~/WES2WGS_CNscaling/scripts/run_pooled_onoff_experiments_slurm.sh
#   (or chain it -- see the --dependency note at the bottom)

set -euo pipefail

source /home/sspandau/miniconda3/etc/profile.d/conda.sh
conda activate wes2wgs

# ------------------------------ CONFIG ------------------------------------
PROJECT_DIR="/home/sspandau/WES2WGS_CNscaling"
SCRIPTS_DIR="${PROJECT_DIR}/scripts"
SEARCH_SCRIPT="${SCRIPTS_DIR}/compare_wes_wgs_overlap_grad_pooled_on_off.py"
MANIFEST_BUILDER="${SCRIPTS_DIR}/make_pooled_depmap_manifest.sh"

ONTARGET_ROOT="/metropolis/projects/sspandau/depmap/wes/ontarget_output"     # <S>/ontarget_depth_output/<S>_on_target_depth.tsv
OFFTARGET_ROOT="/metropolis/projects/sspandau/depmap/wes/scaled_output"      # <S>/offtarget_scaled_output/<S>_off_target_copy_ratios.tsv
                                                                              # + <S>_cluster_results/ next to it
MASK_REGIONS="/metropolis/projects/sspandau/depmap/best_combo_recurrent_orange_overlap_093026.csv"
# 1 = mask only recurrent bins that do NOT overlap an ecDNA/BFB/CNC (orange)
#     location, so known amplicon regions stay in the analysis; 0 = mask all rows
MASK_EXCLUDE_ORANGE_OVERLAP=0

TRAIN_FRAC=0.6            # fraction of samples the search is fit on; the rest are held out
SPLIT_SEED=0              # change for a different (but reproducible) split

OUTDIR="/metropolis/projects/sspandau/depmap/pooled_grad_onoff_clustered_experiments_macro_oldmask"
MANIFEST_PREFIX="${OUTDIR}/manifests/pooled"
N_JOBS="${SLURM_CPUS_PER_TASK:-1}"
# --------------------------------------------------------------------------

export PYTHONPATH="${SCRIPTS_DIR}${PYTHONPATH:+:${PYTHONPATH}}"   # for 'from compare_wes_wgs_overlap import ...'
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1  # parallelism comes from --n-jobs
export PYTHONUNBUFFERED=1

cd "${PROJECT_DIR}"
echo "== Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"

for f in "$SEARCH_SCRIPT" "$MANIFEST_BUILDER" "$MASK_REGIONS" \
         "${SCRIPTS_DIR}/compare_wes_wgs_overlap.py"; do
    [[ -f "$f" ]] || { echo "ERROR: missing $f" >&2; exit 1; }
done
mkdir -p "$OUTDIR" "$(dirname "$MANIFEST_PREFIX")"

# ---------------- step 1: manifests + 60/40 split -------------------------
echo "== Building manifests (train ${TRAIN_FRAC}, seed ${SPLIT_SEED})"
python3 "$MANIFEST_BUILDER" \
  --ontarget-root "$ONTARGET_ROOT" \
  --offtarget-root "$OFFTARGET_ROOT" \
  --train-frac "$TRAIN_FRAC" \
  --seed "$SPLIT_SEED" \
  --out-prefix "$MANIFEST_PREFIX"

# ---------------- step 2: mask pre-flight ---------------------------------
MASK_USED="$MASK_REGIONS"
if [[ "$MASK_EXCLUDE_ORANGE_OVERLAP" == "1" ]]; then
    MASK_USED="${OUTDIR}/recurrent_mask_no_orange_overlap.csv"
    python3 - "$MASK_REGIONS" "$MASK_USED" <<'EOF'
import sys, pandas as pd
m = pd.read_csv(sys.argv[1])
keep = ~m["overlaps_orange"].astype(str).str.lower().eq("true")
m[keep].to_csv(sys.argv[2], index=False)
print(f"== Mask: kept {int(keep.sum())}/{len(m)} recurrent bins (dropped {int((~keep).sum())} overlapping orange)")
EOF
fi
python3 - "$MASK_USED" <<'EOF'
import sys
from compare_wes_wgs_overlap import load_mask_regions
m = load_mask_regions(sys.argv[1])
n = len(m) if m is not None else 0
print(f"== Mask loads OK: {n} regions from {sys.argv[1]}")
if n == 0:
    sys.exit("ERROR: mask loaded 0 regions -- check the file format load_mask_regions expects")
EOF

# ---------------- step 3: experiments on train, score holdout -------------
echo "== Starting experiments at $(date)"
python3 "$SEARCH_SCRIPT" \
  --manifest "${MANIFEST_PREFIX}.train.tsv" \
  --holdout-manifest "${MANIFEST_PREFIX}.holdout.tsv" \
  --mask-regions "$MASK_USED" \
  --cluster-exclude-labels "-1,noise" \
  --missing-clusters skip \
  --ontarget-wes-column raw_wes_depth \
  --ontarget-wgs-column wgs_tumor_depth \
  --offtarget-wes-column predicted_loess_upscale_depth \
  --offtarget-wgs-column wgs_tumor_depth \
  --target-bin-size 25000 \
  --experiments exp1_raw,exp1_cnlike,exp2_cnlike \
  --metrics jaccard,overlap \
  --aggregate macro \
  --grid-size 100 \
  --min-frac 0.0005 --max-frac 0.01 --min-bins 1000 --min-samples 10 --min-cn-like 4.0 \
  --n-jobs "$N_JOBS" \
  --outdir "$OUTDIR"

echo "== Done at $(date). Results in ${OUTDIR}"
echo "   summary.tsv: one row per experiment x metric (thresholds, train + holdout scores)"

# --------------------------------------------------------------------------
# Chain after the clustering jobs (job name 'genomic_cluster'):
#   ids=$(squeue -u "$USER" -h -n genomic_cluster -o %i | paste -sd:)
#   sbatch --dependency=afterany:${ids} ~/WES2WGS_CNscaling/scripts/run_pooled_onoff_experiments_slurm.sh
# --------------------------------------------------------------------------
