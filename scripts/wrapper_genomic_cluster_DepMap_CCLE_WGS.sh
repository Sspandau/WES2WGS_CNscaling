#!/bin/bash
# ==========================================================================
# wrapper_depmap_ccle_genomic_cluster_WGS.sh
#
# Submits one genomic_cluster_WGS_slurm.sh job per sample, clustering on the
# WGS tumor depth (column wgs_tumor_depth) that lives in each sample's own
# DepMap off-target copy-ratio TSV. No mask regions and no WES/WGS
# comparison. Samples whose TSV lacks the WGS column are skipped.
#
# Results go next to each TSV in <SRR>_wgs_cluster_results/, so they do not
# overwrite the WES <SRR>_cluster_results/.
#
# Usage:
#   bash wrapper_depmap_ccle_genomic_cluster_WGS.sh [WGS_COLUMN] [REBIN_TO|none] [MAX_GAP] [MIN_REL_CN|none]
#   DRY_RUN=1 bash wrapper_depmap_ccle_genomic_cluster_WGS.sh   # preview, submit nothing
# ==========================================================================
set -euo pipefail

# ------------------------------ CONFIG ------------------------------------
WES_ROOT="/metropolis/projects/sspandau/depmap/wes/scaled_output"
WES_PATTERN="*_off_target_copy_ratios.tsv"        # found anywhere below each sample dir

WGS_COLUMN="${1:-wgs_tumor_depth}"                # WGS column inside the same TSV
REBIN_TO="${2:-25000}"                            # "none" to skip rebinning
MAX_GAP="${3:-2}"                                 # fixed gap in bins, or "auto"
MIN_REL_CN="${4:-3}"                              # floor for "high" (x mean depth), or "none"

CLUSTER_SCRIPT="/home/sspandau/WES2WGS_CNscaling/scripts/genomic_cluster_WGS_slurm.sh"
MANIFEST="${MANIFEST:-/metropolis/projects/sspandau/depmap/genomic_cluster_WGS_manifest.tsv}"
DRY_RUN="${DRY_RUN:-0}"
# --------------------------------------------------------------------------

[[ -d "$WES_ROOT" ]]       || { echo "ERROR: WES_ROOT not found: $WES_ROOT" >&2; exit 1; }
[[ -f "$CLUSTER_SCRIPT" ]] || { echo "ERROR: cluster script not found: $CLUSTER_SCRIPT" >&2; exit 1; }

mkdir -p "$(dirname "$MANIFEST")" /home/sspandau/logs

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
    fh.write("sample\ttsv\tstatus\n")
    fh.writelines("\t".join(r) + "\n" for r in rows)

print(f"   {len(rows)} samples: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
print(f"   manifest: {manifest}")
bad = [r[0] for r in rows if r[2] == "no_wgs_column"]
if bad:
    print(f"   [warn] no '{wgs_col}' column (skipped): {bad[:10]}{' ...' if len(bad) > 10 else ''}")
if not counts["ok"]:
    sys.exit(f"ERROR: no TSV under {wes_root} has a '{wgs_col}' column")
EOF

# ------------------------------ submit ------------------------------------
n_sub=0; n_no_tsv=0; n_no_wgs=0
while IFS=$'\t' read -r SAMPLE TSV_FILE STATUS; do
    case "$STATUS" in
        no_wes_tsv)    n_no_tsv=$((n_no_tsv + 1)); continue ;;
        no_wgs_column) n_no_wgs=$((n_no_wgs + 1)); continue ;;
    esac

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[dry-run] ${SAMPLE}: ${TSV_FILE} (column ${WGS_COLUMN}, rebin ${REBIN_TO}, max-gap ${MAX_GAP}, min-rel-cn ${MIN_REL_CN})"
    else
        echo "Submitting WGS clustering job for sample: ${SAMPLE}"
        sbatch "$CLUSTER_SCRIPT" "$TSV_FILE" "$WGS_COLUMN" "$REBIN_TO" "$MAX_GAP" "$MIN_REL_CN"
    fi
    n_sub=$((n_sub + 1))
done < <(tail -n +2 "$MANIFEST")

summary="${n_sub} jobs (skipped: ${n_no_tsv} no copy-ratio TSV, ${n_no_wgs} no '${WGS_COLUMN}' column)"
if [[ "$DRY_RUN" == "1" ]]; then
    echo "== DRY RUN: would submit ${summary}. Nothing submitted."
else
    echo "== Submitted ${summary}."
fi