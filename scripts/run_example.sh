#!/usr/bin/env bash
set -euo pipefail
python scripts/genomic_cluster_upscaleddepth.py \
  --input scripts/example_data/predictions_example.csv \
  --column value \
  --mask-regions scripts/example_data/mask_regions_example.csv \
  --outdir scripts/example_output
