#!/bin/bash
# Build the sample manifest for the pooled WES/WGS overlap analysis.
#
# Usage:
#   bash make_overlap_manifest.sh [WES_ROOT] [WGS_ROOT] [MANIFEST_OUT]
#
# Same sample/file matching logic as the per-sample wrapper, but it writes a TSV
# (sample, wes_input, wgs_input) instead of submitting one job per sample.

WES_ROOT="${1:-/home/sspandau/CCLE_WXS/WES2WGS_CCLE}"
WGS_ROOT="${2:-/home/sspandau/CCLE_WGS}"
MANIFEST="${3:-overlap_manifest.tsv}"

normalize_name() {
    printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_+|_+$//g; s/_+/_/g'
}

find_matching_wgs_file() {
    local sample="$1"
    local norm_sample
    norm_sample="$(normalize_name "$sample")"
    [ -d "$WGS_ROOT" ] || return 1
    while IFS= read -r dir; do
        [ -n "$dir" ] || continue
        if [ "$(normalize_name "$(basename "$dir")")" = "$norm_sample" ]; then
            local candidate
            candidate=$(find "$dir" -type f -name "*_off_target_copy_ratios.tsv" 2>/dev/null | head -n 1)
            if [ -n "$candidate" ] && [ -f "$candidate" ]; then
                printf '%s\n' "$candidate"
                return 0
            fi
        fi
    done < <(find "$WGS_ROOT" -type d 2>/dev/null)
    return 1
}

printf 'sample\twes_input\twgs_input\n' > "$MANIFEST"
n=0
for SAMPLE_DIR in "$WES_ROOT"/*/; do
    [ -d "$SAMPLE_DIR" ] || continue
    SAMPLE=$(basename "$SAMPLE_DIR")
    WES_INPUT=$(find "$SAMPLE_DIR" -maxdepth 1 -type f -name "*_off_target_copy_ratios.tsv" | head -n 1)
    if [ -z "$WES_INPUT" ] || [ ! -f "$WES_INPUT" ]; then
        echo "[skip] no off-target copy-ratio file for sample '${SAMPLE}'" >&2
        continue
    fi
    WGS_INPUT="$WES_INPUT"
    if [ -d "$WGS_ROOT" ]; then
        WGS_MATCH="$(find_matching_wgs_file "$SAMPLE")" || WGS_MATCH=""
        if [ -n "$WGS_MATCH" ] && [ -f "$WGS_MATCH" ]; then
            WGS_INPUT="$WGS_MATCH"
        fi
    fi
    printf '%s\t%s\t%s\n' "$SAMPLE" "$WES_INPUT" "$WGS_INPUT" >> "$MANIFEST"
    n=$((n + 1))
done
echo "Wrote $n samples to $MANIFEST"