#!/bin/bash
#SBATCH --job-name=ontarget_depth
#SBATCH --output=/home/sspandau/logs/%x_%A_%a.out
#SBATCH --error=/home/sspandau/logs/%x_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# Usage:
#   sbatch extract_ontarget_depth_slurm.sh WES_BAM WGS_BAM ONTARGET_WINDOWS OUTDIR [SAMPLE_NAME]

WES_BAM="${1:-}"
WGS_BAM="${2:-}"
ONTARGET_WINDOWS="${3:-}"
OUTDIR="${4:-}"
SAMPLE_NAME="${5:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRACT_SCRIPT="${SCRIPT_DIR}/extract_ontarget_depth.py"

if [ -z "$WES_BAM" ] || [ -z "$ONTARGET_WINDOWS" ] || [ -z "$OUTDIR" ]; then
    echo "Usage: sbatch $0 WES_BAM WGS_BAM ONTARGET_WINDOWS OUTDIR [SAMPLE_NAME]" >&2
    exit 1
fi
for f in "$WES_BAM" "$ONTARGET_WINDOWS" "$EXTRACT_SCRIPT"; do
    if [ ! -f "$f" ]; then
        echo "[-] Error: file not found: $f" >&2
        exit 1
    fi
done

source /home/sspandau/miniconda3/etc/profile.d/conda.sh
conda activate wes_cnv

TMP_DIR="${OUTDIR}/tmp"
mkdir -p "$TMP_DIR"

ARGS=(
    --ontarget_windows "$ONTARGET_WINDOWS"
    --wes_bam "$WES_BAM"
    -t "${SLURM_CPUS_PER_TASK:-4}"
    --output_dir "$OUTDIR"
    --temp_dir "$TMP_DIR"
)

if [ -n "$WGS_BAM" ] && [ -f "$WGS_BAM" ]; then
    ARGS+=(--wgs_bam "$WGS_BAM")
fi
if [ -n "$SAMPLE_NAME" ]; then
    ARGS+=(--sample_name "$SAMPLE_NAME")
fi

python3 "$EXTRACT_SCRIPT" "${ARGS[@]}"