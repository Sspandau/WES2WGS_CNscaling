import os
import sys
import argparse
import pickle
import subprocess
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from pygam import LinearGAM, s

'''
Trains ONE pooled bias-correction model across your full matched WGS/WES
cohort, replacing the per-sample LOESS(GC) step in scale_WES_tracks.py.

Why pool instead of fitting per-sample:
  - The GC/mappability/distance-to-target bias in off-target WES depth is
    mostly a property of the capture chemistry + off-target bleed physics,
    not the individual tumor. Fitting it fresh per-sample (as LOESS does
    today) throws away 33 other samples' worth of signal at every locus,
    and is especially unstable in extreme-GC / low-mappability windows.
  - With matched ground truth (real WGS depth per sample, not just PoN
    median), you can regress WES bias directly against known-true depth,
    then apply the pooled fit to future WES-only tumors with no matched
    WGS at all.

Model:
  y = log2(raw_WES_depth / matched_WGS_depth), per window per sample,
      after per-sample median-depth normalization (removes pure sequencing-
      depth scale differences between samples so the GAM only has to learn
      the shape of the bias, not each sample's overall depth level)

  y ~ s(gc_pct) + s(mappability) + s(dist_to_target)

Validation:
  Grouped K-fold CV, grouped by sample_id (never split a sample's windows
  across train/test) -- this tells you how well the correction generalizes
  to a NEW tumor, which is the only validation that matters here. Reports
  R^2 and MAD of residuals on held-out samples, plus for comparison, what
  a per-sample LOESS(GC)-only model would have achieved on those same
  held-out samples (via leave-one-out simulation), so you can see the
  actual improvement from pooling + added covariates.

python3 train_pooled_bias_model.py \
  --manifest matched_cohort_manifest.tsv \
  --windows_bed v5_offtargets.bed \
  --covariates v5_offtargets.covariates.tsv \
  --temp_dir ./tmp_train \
  --model_output pooled_bias_model.pkl \
  --folds 5

Manifest TSV format (tab-separated, header required):
  sample_id    wes_bam                       wgs_bam
  TCGA-XX-01   /path/TCGA-XX-01.wes.bam      /path/TCGA-XX-01.wgs.bam
  ...
'''

def run_mosdepth(bam_path, bed_path, output_prefix, threads=4):
    cmd = [
        "mosdepth", "--threads", str(threads), "--by", bed_path,
        "--mapq", "20", "--flag", "3844", "--no-per-base",
        output_prefix, bam_path,
    ]
    subprocess.run(cmd, check=True)

def parse_mosdepth_regions(output_prefix):
    regions_file = f"{output_prefix}.regions.bed.gz"
    df = pd.read_csv(regions_file, sep='\t', compression='gzip',
                     header=None, names=['chrom', 'start', 'end', 'window_id', 'depth'])
    return df[['window_id', 'depth']]

# (file truncated in archive copy for brevity)
