#!/bin/bash

# Usage:
#   bash wrapper_extract_ontarget_depth.sh <WES_WGS_MANIFEST> <ONTARGET_WINDOWS_AGILENT> <ONTARGET_WINDOWS_ILLUMINA> [OUTDIR_ROOT]
#
# Submits one on-target depth extraction job per sample, reading WES_WGS_MANIFEST
# (the merged wes_wgs_scaling_manifest_final.tsv from build_wes_wgs_scaling_manifest.py
# + merge_wgs_sources.py) for each sample's already-resolved wes_path/wgs_path and
# wes_baits, rather than re-deriving them here -- your actual WES/WGS BAMs sit as flat
# CDS-ID-named files (not one directory per sample), so a fresh per-sample-directory
# scan here would not find them; the manifest already solved that correctly.
#
# Rows with both_found=False or both_indexed=False (no usable WGS from either source)
# are skipped, same as wrapper_run_wes_wgs_scaling.sh.

WES_WGS_MANIFEST="${1:-}"
ONTARGET_WINDOWS_AGILENT="${2:-}"
ONTARGET_WINDOWS_ILLUMINA="${3:-}"
OUTDIR_ROOT="${4:-}"

declare -A BAITS_TO_WINDOWS

if [ -z "$WES_WGS_MANIFEST" ] || [ -z "$ONTARGET_WINDOWS_AGILENT" ] || [ -z "$ONTARGET_WINDOWS_ILLUMINA" ]; then
    echo "Usage: bash $0 WES_WGS_MANIFEST ONTARGET_WINDOWS_AGILENT ONTARGET_WINDOWS_ILLUMINA [OUTDIR_ROOT]" >&2
    exit 1
fi
if [ ! -f "$WES_WGS_MANIFEST" ]; then
    echo "[-] Error: manifest not found: $WES_WGS_MANIFEST" >&2
    exit 1
fi
for f in "$ONTARGET_WINDOWS_AGILENT" "$ONTARGET_WINDOWS_ILLUMINA"; do
    if [ ! -f "$f" ]; then
        echo "[-] Error: on-target windows BED not found: $f" >&2
        exit 1
    fi
done
BAITS_TO_WINDOWS["AGILENT"]="$ONTARGET_WINDOWS_AGILENT"
BAITS_TO_WINDOWS["ICE"]="$ONTARGET_WINDOWS_ILLUMINA"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SLURM="${SCRIPT_DIR}/extract_ontarget_depth_slurm.sh"
if [ ! -f "$RUN_SLURM" ]; then
    echo "[-] Error: extract_ontarget_depth_slurm.sh not found next to this script "
    echo "    (expected: $RUN_SLURM)" >&2
    exit 1
fi

HEADER=$(head -n 1 "$WES_WGS_MANIFEST")
IFS=$'\x1f' read -r -a COLS <<< "$(printf '%s' "$HEADER" | tr '\t' '\037')"
col_index() {
    local target="$1"
    for i in "${!COLS[@]}"; do
        if [ "${COLS[$i]}" = "$target" ]; then
            echo "$i"
            return 0
        fi
    done
    echo "-1"
}
IDX_ACH=$(col_index "ach_id")
IDX_CELL=$(col_index "cell_line")
IDX_BAITS=$(col_index "wes_baits")
IDX_WES=$(col_index "wes_path")
IDX_WGS=$(col_index "wgs_path")
IDX_BOTHFOUND=$(col_index "both_found")
IDX_BOTHIDX=$(col_index "both_indexed")
for idx_name in IDX_ACH IDX_CELL IDX_BAITS IDX_WES IDX_WGS IDX_BOTHFOUND IDX_BOTHIDX; do
    if [ "${!idx_name}" = "-1" ]; then
        echo "[-] Error: manifest is missing an expected column (checked: $idx_name)" >&2
        exit 1
    fi
done

mkdir -p logs

n_submitted=0
n_failed_submit=0
n_skipped_notusable=0
n_skipped_unmapped=0

while IFS=$'\x1f' read -r -a ROW; do
    ACH_ID="${ROW[$IDX_ACH]}"
    CELL_LINE="${ROW[$IDX_CELL]}"
    BAITS="${ROW[$IDX_BAITS]}"
    WES="${ROW[$IDX_WES]}"
    WGS="${ROW[$IDX_WGS]}"
    BOTH_FOUND="${ROW[$IDX_BOTHFOUND]}"
    BOTH_IDX="${ROW[$IDX_BOTHIDX]}"

    if [ "$BOTH_FOUND" != "True" ] || [ "$BOTH_IDX" != "True" ]; then
        echo "[skip] $ACH_ID / $CELL_LINE: not usable (both_found=$BOTH_FOUND, both_indexed=$BOTH_IDX)"
        n_skipped_notusable=$((n_skipped_notusable + 1))
        continue
    fi

    ONTARGET_WINDOWS="${BAITS_TO_WINDOWS[$BAITS]:-}"
    if [ -z "$ONTARGET_WINDOWS" ]; then
        echo "[skip] $ACH_ID / $CELL_LINE: unmapped WES_Baits value '$BAITS'"
        n_skipped_unmapped=$((n_skipped_unmapped + 1))
        continue
    fi

    OUTDIR="${OUTDIR_ROOT:-.}/${CELL_LINE}/ontarget_depth_output"
    echo "Submitting on-target depth job for sample: $ACH_ID / $CELL_LINE (baits='$BAITS')"
    echo "  WES: $WES"
    echo "  WGS: $WGS"
    if sbatch "$RUN_SLURM" "$WES" "$WGS" "$ONTARGET_WINDOWS" "$OUTDIR" "$CELL_LINE"; then
        n_submitted=$((n_submitted + 1))
    else
        echo "  [!] sbatch FAILED for $ACH_ID / $CELL_LINE -- not counted as submitted" >&2
        n_failed_submit=$((n_failed_submit + 1))
    fi
done < <(tail -n +2 "$WES_WGS_MANIFEST" | tr '\t' '\037')

echo ""
echo "Actually submitted: $n_submitted   sbatch failures: $n_failed_submit   Not usable: $n_skipped_notusable   Skipped (unmapped baits): $n_skipped_unmapped"