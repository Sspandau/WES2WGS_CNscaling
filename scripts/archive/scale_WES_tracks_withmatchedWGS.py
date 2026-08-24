import os
import argparse
import glob
import subprocess
import pandas as pd
import numpy as np
import pybedtools
import statsmodels.api as sm
from pygam import LinearGAM, s

'''
Script that scales WES off-target depth to WGS-comparable copy ratios using
a WGS Panel of Normals (PoN) as the reference baseline.

Normalization strategy:
- Raw WES off-target depth is compared directly to the PoN median at each window.
- GC bias is estimated and corrected from the raw WES/PoN ratios themselves,
  avoiding any use of on-target WES depth which may be inflated by focal CN gains.
- Output log2 copy ratios are directly comparable to WGS copy ratios.
- If a matched tumor WGS BAM is provided, its raw window depth is also included.

python3 scale_WES_tracks.py --offtarget_windows ../CCLE_WXS/WES2WGS_CCLE/v5_offtargets.bed \
--pon_tsv ../CCLE_WXS/1000genomes_highcov_WGS/PoN_1000genomes_wgs_normals_noautosome.tsv \
--wes_dir /pedigree2/cui/CCLE_WXS/SW579_THYROID/ \
--wgs_bam /pedigree2/cui/CCLE_WGS/SW579.wgs.bam \
--output_dir ../CCLE_WXS/WES2WGS_CCLE/ --temp_dir ./
'''

def run_mosdepth(bam_path, bed_path, output_prefix, threads=4):
    """
    Runs mosdepth to pull clean window read counts.
    Filters: MAPQ >= 20, excludes duplicates, supplementary, non-primary, QC-fail reads.
    """
    cmd = [
        "mosdepth",
        "--threads", str(threads),
        "--by", bed_path,
        "--mapq", "20",
        "--flag", "3844",
        "--no-per-base",
        output_prefix,
        bam_path
    ]
    subprocess.run(cmd, check=True)

def parse_mosdepth_regions(output_prefix):
    """Loads mosdepth compressed BED output into a pandas DataFrame."""
    regions_file = f"{output_prefix}.regions.bed.gz"
    df = pd.read_csv(regions_file, sep='\t', compression='gzip',
                     header=None, names=['chrom', 'start', 'end', 'name', 'depth'])
    df['chrom'] = df['chrom'].astype(str)
    df['start'] = df['start'].astype(int)
    df['end'] = df['end'].astype(int)
    return df

def apply_pon_loess_gc_correction(raw_wes_depth, pon_median, offtarget_gc):
    """
    Computes true absolute scale factors by comparing raw WES depth 
    directly to the absolute raw depth profile of the WGS PoN.
    """
    safe_pon_median = np.where(pon_median <= 0, 1e-4, pon_median)
    raw_ratio = raw_wes_depth / safe_pon_median

    valid_mask = (
        (raw_wes_depth > 0) &
        (~np.isnan(offtarget_gc)) &
        (~np.isnan(raw_ratio)) &
        (~np.isinf(raw_ratio))
    )

    if valid_mask.sum() < 50:
        return raw_ratio

    train_gc = offtarget_gc[valid_mask]
    train_ratio = raw_ratio[valid_mask]

    loess_fit = sm.nonparametric.lowess(
        endog=train_ratio, exog=train_gc, frac=0.1, it=3, return_sorted=False
    )

    sort_idx = np.argsort(train_gc)
    fitted_all = np.interp(offtarget_gc, train_gc[sort_idx], loess_fit[sort_idx])
    fitted_all = np.clip(fitted_all, a_min=1e-4, a_max=None)

    depth_scaled = raw_ratio / fitted_all
    return depth_scaled
