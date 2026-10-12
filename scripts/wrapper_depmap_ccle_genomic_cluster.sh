#!/bin/bash
# ==========================================================================
# wrapper_depmap_ccle_genomic_cluster.sh
#
# Submits one genomic_cluster_CCLE_slurm.sh job per WES sample.
#
# The DepMap off-target copy-ratio TSVs already carry the matched WGS depth
# (column wgs_tumor_depth) next to predicted_loess_upscale_depth, so each
# sample's OWN TSV is passed as the WGS comparison file -- WES and WGS values
# come from the same rows, no cross-directory sample matching needed.
#
# Usage:
#   bash wrapper_depmap_ccle_genomic_cluster.sh [MASK_REGIONS] [WGS_COLUMN]
#   DRY_RUN=1 bash wrapper_depmap_ccle_genomic_cluster.sh     # preview, submit nothing
# ==========================================================================
set -euo pipefail

# ------------------------------ CONFIG ------------------------------------
WES_ROOT="/metropolis/projects/sspandau/depmap/wes/scaled_output"
WES_PATTERN="*_off_target_copy_ratios.tsv"        # found anywhere below each sample dir

MASK_REGIONS="${1:-/metropolis/projects/sspandau/depmap/recurrent_bins_search/best_combo_recurrent_orange_overlap.csv}"
WGS_COLUMN="${2:-wgs_tumor_depth}"                # WGS column inside the same TSV

CLUSTER_SCRIPT="/home/sspandau/WES2WGS_CNscaling/scripts/genomic_cluster_CCLE_slurm.sh"
MANIFEST="${MANIFEST:-/metropolis/projects/sspandau/depmap/genomic_cluster_manifest.tsv}"
DRY_RUN="${DRY_RUN:-0}"
# --------------------------------------------------------------------------

[[ -d "$WES_ROOT" ]]       || { echo "ERROR: WES_ROOT not found: $WES_ROOT" >&2; exit 1; }
[[ -f "$CLUSTER_SCRIPT" ]] || { echo "ERROR: cluster script not found: $CLUSTER_SCRIPT" >&2; exit 1; }
[[ -f "$MASK_REGIONS" ]]   || { echo "ERROR: MASK_REGIONS not found: $MASK_REGIONS" >&2; exit 1; }

mkdir -p "$(dirname "$MANIFEST")"

# ------------- find each sample's TSV and check it has the WGS column -------------
export WES_ROOT WES_PATTERN WGS_COLUMN MANIFEST
python3 - <<'EOF'
import os, sys
from collections import Counter
from pathlib import Path

wes_root, pattern = Path(os.environ["WES_ROOT"]), os.environ["WES_PATTERN"]
wgs_col, manifest = os.environ["WGS_COLUMN"], Path(os.environ["MANIFEST"])

rows, counts = [], Counter()
for d in sorted(p for p in wes_root.iterdir() if p.is_dir() and not p.name.startswith(".")):
    hits = sorted(d.rglob(pattern), key=lambda p: (len(p.relative_to(d).parts), str(p)))
    if not hits:
        status, tsv = "no_wes_tsv", "-"
    else:
        tsv = str(hits[0])
        with open(tsv) as fh:
            cols = fh.readline().rstrip("\n").split("\t")
        status = "ok" if wgs_col in cols else "no_wgs_column"
    counts[status] += 1
    rows.append((d.name, tsv, status))

with open(manifest, "w") as fh:
    fh.write("wes_sample\twes_tsv\tstatus\n")
    fh.writelines("\t".join(r) + "\n" for r in rows)

print(f"   {len(rows)} samples: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
print(f"   manifest: {manifest}")
bad = [r[0] for r in rows if r[2] == "no_wgs_column"]
if bad:
    print(f"   [warn] no '{wgs_col}' column (will run without WGS comparison): {bad[:10]}")
if not counts["ok"]:
    sys.exit(f"ERROR: no TSV under {wes_root} has a '{wgs_col}' column")
EOF

# ------------------------------ submit ------------------------------------
n_sub=0; n_skip=0
while IFS=$'\t' read -r SAMPLE TSV_FILE STATUS; do
    if [[ "$STATUS" == "no_wes_tsv" ]]; then
        n_skip=$((n_skip + 1)); continue
    fi
    WGS_FILE="$TSV_FILE"
    [[ "$STATUS" == "no_wgs_column" ]] && WGS_FILE=""

    if [[ "$DRY_RUN" == "1" ]]; then
        if [[ -n "$WGS_FILE" ]]; then wgs_note="same file, column ${WGS_COLUMN}"; else wgs_note="none"; fi
        echo "[dry-run] ${SAMPLE}: ${TSV_FILE} (WGS: ${wgs_note})"
    else
        echo "Submitting clustering job for sample: ${SAMPLE}"
        sbatch "$CLUSTER_SCRIPT" "$TSV_FILE" "$MASK_REGIONS" "$WGS_FILE" "$WGS_COLUMN"
    fi
    n_sub=$((n_sub + 1))
done < <(tail -n +2 "$MANIFEST")

if [[ "$DRY_RUN" == "1" ]]; then
    echo "== DRY RUN: would submit ${n_sub} jobs (${n_skip} samples skipped: no copy-ratio TSV). Nothing submitted."
else
    echo "== Submitted ${n_sub} jobs (${n_skip} samples skipped: no copy-ratio TSV)."
fi
