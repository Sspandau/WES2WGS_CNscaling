#!/bin/bash
#SBATCH --job-name=pooled_onoff_experiments
#SBATCH --output=/home/sspandau/logs/%x_%j.out
#SBATCH --error=/home/sspandau/logs/%x_%j.err
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Pooled on+off-target WES-vs-WGS overlap experiments, GRADIENT-DESCENT search, train / eval / test:
#   - CN-like per region (off-target mean over bins > 0, on-target mean over all bins)
#   - ALL recurrent bins removed (KEEP_MASKED_IN_CLUSTERS=off, also inside clusters), then
#     off-target bins outside each sample's clusters dropped
#   - CN-like floor tuned over 3, 4 and 5
#   - Exp 1 (separate on/off WES thresholds, raw + CN-like) and Exp 2 (shared CN-like WES threshold)
#   - thresholds found by Adam on a smooth surrogate (best = highest hard score among eligible steps)
#   - CN-like floor (--min-cn-like, tuned via TUNE): CN-like thresholds >= floor; in the RAW experiment
#     a bin can only be called positive if its CN-like value exceeds the floor (raw thresholds are what
#     is optimised). Add --no-raw-cn-floor to GD_ARGS to turn the raw gate off.
#   - TRAIN (60%): thresholds searched here
#   - EVAL  (20%): picks the best constraint combo (TUNE)
#   - TEST  (20%): scored once at the final choice -- report this number
#
#   sbatch ~/WES2WGS_CNscaling/scripts/compare_wes_wgs_overlap_grad_pooled_on_off_tuning_fullmask_cn3to5_slurm.sh

set -euo pipefail

source /home/sspandau/miniconda3/etc/profile.d/conda.sh
conda activate wes2wgs

# ------------------------------ CONFIG ------------------------------------
PROJECT_DIR="/home/sspandau/WES2WGS_CNscaling"
SCRIPTS_DIR="${PROJECT_DIR}/scripts"
SEARCH_SCRIPT="${SCRIPTS_DIR}/compare_wes_wgs_overlap_grad_pooled_on_off_tuning.py"
MANIFEST_BUILDER="${SCRIPTS_DIR}/build_pooled_manifest_depmap_tuning.sh"      # python script (run with python3)

ONTARGET_ROOT="/metropolis/projects/sspandau/depmap/wes/ontarget_output"
OFFTARGET_ROOT="/metropolis/projects/sspandau/depmap/wes/scaled_output"

# Recurrent-bin mask
MASK_REGIONS="/metropolis/projects/sspandau/depmap/best_combo_recurrent_orange_overlap_093026.csv"
# Masked bins that lie inside the sample's clusters are KEPT:
#   both      = on- and off-target bins overlapping a cluster interval escape the mask
#   offtarget = only off-target bins (= no mask for the scored off-target bins)
#   off       = mask everything (previous behaviour)
KEEP_MASKED_IN_CLUSTERS="off"         # mask every recurrent bin, inside clusters too

TRAIN_FRAC=0.6
EVAL_FRAC=0.2             # test = 1 - TRAIN_FRAC - EVAL_FRAC
SPLIT_SEED=0

AGGREGATE="macro"
# Base constraints (used for anything not listed in TUNE)
BASE_CONSTRAINTS="--min-frac 0.0005 --max-frac 0.01 --min-bins 1000 --min-samples 10 --min-cn-like 4.0"
# Gradient-descent settings (same meaning as compare_wes_wgs_overlap_grad_pooled.py; --temp is now
# relative to the fraction of bins currently called positive -- see the script docstring)
GD_ARGS="--n-steps 100 --n-starts 5 --lr 0.2 --temp 0.1 --seed 0 --count-penalty 1.0"
# Values to tune on the eval split. EVERY combination is searched on train and scored on eval.
#   constraints : max_frac, min_bins, min_samples, min_cn_like (min_frac also allowed)
#   optimizer   : lr, temp, n_steps, n_starts
# Runtime grows with (number of combos) x (n_starts x n_steps): one search at 5 starts x 100 steps
# takes ~1 min on the ~440k-bin training pool, and every combo is searched for all 6
# experiment x metric pairs. The setting below is 432 combos -> roughly 55-65 h. Each extra value
# multiplies that, so keep --time in mind (96 h here); n_steps=200 or n_starts=10 doubles a search.
TUNE="max_frac=0.01,0.05,0.10 min_bins=100,1000 min_samples=10,50 min_cn_like=3,4,5 lr=0.1,0.2 temp=0.05,0.1,0.25 n_steps=100 n_starts=5,10"

OUTDIR="/metropolis/projects/sspandau/depmap/pooled_grad_onoff_clustered_experiments_macro_tuning_fullmask_cn3to5"
MANIFEST_PREFIX="${OUTDIR}/manifests/pooled"
N_JOBS="${SLURM_CPUS_PER_TASK:-1}"
# --------------------------------------------------------------------------

export PYTHONPATH="${SCRIPTS_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "${PROJECT_DIR}"
echo "== Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
for f in "$SEARCH_SCRIPT" "$MANIFEST_BUILDER" "$MASK_REGIONS" "${SCRIPTS_DIR}/compare_wes_wgs_overlap.py"; do
    [[ -f "$f" ]] || { echo "ERROR: missing $f" >&2; exit 1; }
done
grep -q "class GradSearch" "$SEARCH_SCRIPT" || { echo "ERROR: $SEARCH_SCRIPT is not the gradient-descent version" >&2; exit 1; }
grep -q "eval-manifest" "$SEARCH_SCRIPT"   || { echo "ERROR: $SEARCH_SCRIPT has no --eval-manifest" >&2; exit 1; }
grep -q "raw-cn-floor" "$SEARCH_SCRIPT"    || { echo "ERROR: $SEARCH_SCRIPT is missing the raw CN-like floor (old version)" >&2; exit 1; }
grep -q "keep-masked-in-clusters" "$SEARCH_SCRIPT" || { echo "ERROR: $SEARCH_SCRIPT has no --keep-masked-in-clusters (old version)" >&2; exit 1; }
grep -q "eval-frac" "$MANIFEST_BUILDER"    || { echo "ERROR: $MANIFEST_BUILDER is an old version (no --eval-frac)" >&2; exit 1; }
mkdir -p "$OUTDIR" "$(dirname "$MANIFEST_PREFIX")"

# ---------------- step 1: manifests, 60 / 20 / 20 -------------------------
echo "== Building manifests (train ${TRAIN_FRAC}, eval ${EVAL_FRAC}, seed ${SPLIT_SEED})"
python3 "$MANIFEST_BUILDER" \
  --ontarget-root "$ONTARGET_ROOT" --offtarget-root "$OFFTARGET_ROOT" \
  --train-frac "$TRAIN_FRAC" --eval-frac "$EVAL_FRAC" --seed "$SPLIT_SEED" \
  --out-prefix "$MANIFEST_PREFIX"

# ---------------- step 2: search on train, tune on eval, score test ------
echo "== Mask file: $MASK_REGIONS ($(stat -c '%y' "$MASK_REGIONS" | cut -d. -f1), $(($(wc -l < "$MASK_REGIONS") - 1)) recurrent bins)"
python3 - "$TUNE" <<'EOF'
import sys, math
vals = [len(t.split("=", 1)[1].split(",")) for t in sys.argv[1].split()]
cost = 1.0
for t in sys.argv[1].split():
    k, v = t.split("=", 1)
    if k in ("n_steps", "n_starts"):
        xs = [float(x) for x in v.split(",")]
        cost *= sum(xs) / len(xs) / (100.0 if k == "n_steps" else 5.0)
n = math.prod(vals)
print(f"== Tuning {n} combos x 6 experiment/metric pairs = {6 * n} searches "
      f"(~{6 * n * cost / 60:.0f} h at ~1 min per 5x100-step search)")
EOF
echo "== Starting search at $(date)"
# shellcheck disable=SC2086
python3 "$SEARCH_SCRIPT" \
  --manifest "${MANIFEST_PREFIX}.train.tsv" \
  --eval-manifest "${MANIFEST_PREFIX}.eval.tsv" \
  --test-manifest "${MANIFEST_PREFIX}.test.tsv" \
  --mask-regions "$MASK_REGIONS" \
  --keep-masked-in-clusters "$KEEP_MASKED_IN_CLUSTERS" \
  --cluster-exclude-labels "-1,noise" --missing-clusters skip \
  --ontarget-wes-column raw_wes_depth --ontarget-wgs-column wgs_tumor_depth \
  --offtarget-wes-column predicted_loess_upscale_depth --offtarget-wgs-column wgs_tumor_depth \
  --target-bin-size 25000 \
  --experiments exp1_raw,exp1_cnlike,exp2_cnlike --metrics jaccard,overlap \
  --aggregate "$AGGREGATE" \
  $GD_ARGS \
  $BASE_CONSTRAINTS \
  --tune $TUNE \
  --n-jobs "$N_JOBS" --cache-dir "${OUTDIR}/cache" \
  --outdir "$OUTDIR"

# ---------------- step 3: final table (chosen on eval; test is the unbiased score) ----
python3 - "$OUTDIR" "$AGGREGATE" <<'EOF'
import sys, pandas as pd
outdir, agg = sys.argv[1], sys.argv[2]
d = pd.read_csv(f"{outdir}/summary.tsv", sep="\t")
d = d[d["status"] == "ok"].copy()
d["eval_score"] = [r[f"eval_{agg}_{r['metric']}"] for _, r in d.iterrows()]
d["test_score"] = [r[f"test_{agg}_{r['metric']}"] for _, r in d.iterrows()]
cols = ["experiment", "metric"] + [c for c in d.columns if c.startswith("chosen_")] + \
       [c for c in d.columns if c.startswith("wes_threshold_")] + ["wgs_threshold", "train_score", "eval_score", "test_score"]
d[cols].to_csv(f"{outdir}/final_selection.tsv", sep="\t", index=False)
print("\n== FINAL (constraints chosen on eval; test is the unbiased score)")
print(d[["experiment", "metric", "train_score", "eval_score", "test_score"]].round(4).to_string(index=False))
EOF

echo "== Done at $(date). Results in ${OUTDIR}"
echo "   final_selection.tsv                  chosen constraints + thresholds per experiment x metric, with TEST score"
echo "   summary.tsv                          full detail (all train/eval/test micro + macro scores)"
echo "   <exp>_<metric>_tuning.tsv            every constraint combo: train + eval scores"
echo "   <exp>_<metric>_search_history.tsv    every Adam step for the chosen combo"
