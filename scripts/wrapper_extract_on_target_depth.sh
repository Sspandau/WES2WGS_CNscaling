#!/bin/bash

# Usage:
#   bash wrapper_extract_cnvkit_on_off_target.sh <WES_CNVKIT_ROOT> <WGS_CNVKIT_ROOT> [OUTDIR_ROOT]
#
# Runs extract_cnvkit_on_off_target.py once per sample, matching sample names
# between WES_CNVKIT_ROOT and WGS_CNVKIT_ROOT the same way
# wrapper_compare_wes_wgs_overlap.sh matches sample directories -- by normalized
# directory name. If a matched WGS .cnr isn't found, the sample is still
# processed WES-only (wgs_* columns will be NaN).
#
# Assumes one directory per sample, with exactly one *.cnr file directly inside
# it (CNVkit's usual `cnvkit.py batch` output layout). Adjust the `find`
# patterns below if your layout differs.

WES_CNVKIT_ROOT="${1:-}"
WGS_CNVKIT_ROOT="${2:-}"
OUTDIR_ROOT="${3:-}"

if [ -z "$WES_CNVKIT_ROOT" ] || [ -z "$WGS_CNVKIT_ROOT" ]; then
    echo "Usage: bash $0 WES_CNVKIT_ROOT WGS_CNVKIT_ROOT [OUTDIR_ROOT]" >&2
    exit 1
fi
if [ -z "$OUTDIR_ROOT" ]; then
    OUTDIR_ROOT="$WES_CNVKIT_ROOT"
fi

normalize_name() {
    printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_+|_+$//g; s/_+/_/g'
}

find_matching_wgs_cnr() {
    local sample="$1"
    local norm_sample
    norm_sample="$(normalize_name "$sample")"

    [ -d "$WGS_CNVKIT_ROOT" ] || return 1

    while IFS= read -r dir; do
        [ -n "$dir" ] || continue
        if [ "$(normalize_name "$(basename "$dir")")" = "$norm_sample" ]; then
            local candidate
            candidate=$(find "$dir" -maxdepth 1 -type f -name "*.cnr" | head -n 1)
            if [ -n "$candidate" ] && [ -f "$candidate" ]; then
                printf '%s\n' "$candidate"
                return 0
            fi
        fi
    done < <(find "$WGS_CNVKIT_ROOT" -type d 2>/dev/null)

    return 1
}

for SAMPLE_DIR in "$WES_CNVKIT_ROOT"/*/; do
    [ -d "$SAMPLE_DIR" ] || continue
    SAMPLE=$(basename "$SAMPLE_DIR")

    WES_CNR=$(find "$SAMPLE_DIR" -maxdepth 1 -type f -name "*.cnr" | head -n 1)
    if [ -z "$WES_CNR" ] || [ ! -f "$WES_CNR" ]; then
        echo "[skip] no WES .cnr found for sample '${SAMPLE}'; skipping."
        continue
    fi

    WGS_CNR="$(find_matching_wgs_cnr "$SAMPLE")" || WGS_CNR=""
    if [ -z "$WGS_CNR" ]; then
        echo "[warn] no matched WGS .cnr found for sample '${SAMPLE}'; proceeding WES-only "
        echo "       (wgs_* columns will be NaN)."
    fi

    OUTDIR="${OUTDIR_ROOT}/${SAMPLE}/cnvkit_extracted"
    echo "Extracting CNVkit tracks for sample: $SAMPLE"
    echo "  WES .cnr: $WES_CNR"
    echo "  WGS .cnr: ${WGS_CNR:-<none>}"

    ARGS=(
        --wes_cnr "$WES_CNR"
        --sample_name "$SAMPLE"
        --output_dir "$OUTDIR"
    )
    if [ -n "$WGS_CNR" ]; then
        ARGS+=(--wgs_cnr "$WGS_CNR")
    fi

    python3 /home/sspandau/WES2WGS_CNscaling/scripts/extract_cnvkit_on_off_target.py "${ARGS[@]}"
done