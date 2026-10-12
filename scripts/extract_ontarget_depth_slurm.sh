#!/bin/bash
#SBATCH --job-name=ontarget_depth
#SBATCH --output=/home/sspandau/logs/%x_%j.out
#SBATCH --error=/home/sspandau/logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# Usage:
#   sbatch extract_ontarget_depth_slurm.sh WES_BAM WGS_BAM ONTARGET_WINDOWS OUTDIR [SAMPLE_NAME]
#
# CRAM inputs are decoded with REFERENCE below (override with: sbatch --export=ALL,CRAM_REFERENCE=/path/ref.fa ...).
# It must be the FASTA the CRAMs were encoded against -- the DepMap WGS CRAMs are GRCh38
# no-alt (chr-prefixed, e.g. chr1 M5:6aef897c3d6ff0c78aff06ac189178dd).

WES_BAM="${1:-}"
WGS_BAM="${2:-}"
ONTARGET_WINDOWS="${3:-}"
OUTDIR="${4:-}"
SAMPLE_NAME="${5:-}"

# NOTE: intentionally a hardcoded absolute path, not derived from ${BASH_SOURCE[0]} --
# sbatch copies the submitted script into a per-job spool directory and runs it from there.
EXTRACT_SCRIPT="/home/sspandau/WES2WGS_CNscaling/scripts/extract_ontarget_depth.py"
REFERENCE="${CRAM_REFERENCE:-/nucleus/libs/AmpliconArchitect/release/indexed/GRCh38/GCA_000001405.15_GRCh38_no_alt_analysis_set.fa}"

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
    --reference "$REFERENCE"
)

if [ -n "$WGS_BAM" ] && [ -f "$WGS_BAM" ]; then
    ARGS+=(--wgs_bam "$WGS_BAM")
fi
if [ -n "$SAMPLE_NAME" ]; then
    ARGS+=(--sample_name "$SAMPLE_NAME")
fi

python3 "$EXTRACT_SCRIPT" "${ARGS[@]}"
