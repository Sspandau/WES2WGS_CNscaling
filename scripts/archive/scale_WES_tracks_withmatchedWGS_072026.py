import os
import sys
import argparse
import glob
import subprocess
import pandas as pd
import numpy as np
import pybedtools
import statsmodels.api as sm
from scipy.spatial import KDTree
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import cross_validate, GroupKFold, cross_val_predict
from lightgbm import LGBMRegressor

'''
Script that scales WES off-target depth to WGS-comparable copy ratios using
a WGS Panel of Normals (PoN) as the reference baseline.

Normalization strategy:
- Raw WES off-target depth is compared directly to the PoN median at each window.
- GC bias (and secondary distance/mappability biases) are estimated and corrected 
  using a robust 2D LOESS local linear regression directly from the raw ratios.
- Output log2 copy ratios are directly comparable to WGS copy ratios.
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


def apply_pon_multivariate_loess_correction(raw_wes_depth, pon_median, offtarget_gc,
                                             dist_to_target=None, mappability=None, frac=0.25):
    """
    Robust local linear regression (LOESS) using whichever covariates are
    actually valid: GC is always included; dist_to_target and mappability
    are added if present, non-constant, and not all-NaN. Falls back to
    GC-only 1D LOESS if neither secondary covariate is usable.
    """
    safe_pon_median = np.where(pon_median <= 0, 1e-4, pon_median)
    raw_ratio = raw_wes_depth / safe_pon_median

    df_fit = pd.DataFrame({'y': raw_ratio, 'gc': offtarget_gc})
    covariate_cols = ['gc']

    for name, values in (('dist', dist_to_target), ('mappability', mappability)):
        if values is None:
            continue
        s = pd.Series(values)
        if not s.isna().all() and s.nunique() > 1:
            df_fit[name] = values
            covariate_cols.append(name)

    valid_mask = (raw_wes_depth > 0) & (~np.isinf(df_fit['y'])) & (~np.isnan(df_fit['y']))
    for col in covariate_cols:
        valid_mask &= ~df_fit[col].isna()

    total_bins = len(df_fit)
    min_bins = min(500, max(50, int(total_bins * 0.1)))
    if valid_mask.sum() < min_bins:
        print(f"[-] Warning: Too few training bins ({valid_mask.sum()} < {min_bins}). "
              f"Falling back to uncorrected raw ratio.", file=sys.stderr)
        return raw_ratio

    # --- Fast path: GC only, use statsmodels 1D lowess ---
    if covariate_cols == ['gc']:
        print(f"    [+] Running 1D LOESS (GC only) on {valid_mask.sum()} training bins...")
        train_gc = df_fit.loc[valid_mask, 'gc'].values
        train_y = df_fit.loc[valid_mask, 'y'].values
        all_gc = df_fit['gc'].values
        try:
            fitted = sm.nonparametric.lowess(endog=train_y, exog=train_gc, frac=frac,
                                              it=3, return_sorted=False)
            sort_idx = np.argsort(train_gc)
            predicted_bias = np.interp(all_gc, train_gc[sort_idx], fitted[sort_idx])
        except Exception as e:
            print(f"[-] Warning: 1D LOESS fitting failed ({e}). Falling back to raw ratio.",
                  file=sys.stderr)
            return raw_ratio
        predicted_bias = np.clip(predicted_bias, a_min=1e-4, a_max=None)
        return raw_ratio / predicted_bias

    # --- General path: 2D or 3D local linear regression via KDTree ---
    ndim = len(covariate_cols)
    print(f"    [+] Running {ndim}D LOESS ({' + '.join(covariate_cols)}) "
          f"on {valid_mask.sum()} training bins...")

    means, stds = {}, {}
    z_train_cols, z_all_cols = [], []
    for col in covariate_cols:
        train_vals = df_fit.loc[valid_mask, col].values
        all_vals = df_fit[col].values
        m, s = np.mean(train_vals), np.std(train_vals)
        s = 1.0 if s == 0 else s
        means[col], stds[col] = m, s
        z_train_cols.append((train_vals - m) / s)
        z_all_cols.append((all_vals - m) / s)

    train_pts = np.column_stack(z_train_cols)
    all_pts = np.column_stack(z_all_cols)
    train_y = df_fit.loc[valid_mask, 'y'].values

    tree = KDTree(train_pts)
    n_train = len(train_y)
    k = max(20, int(n_train * frac))
    k = min(k, 400, n_train)

    distances, indices = tree.query(all_pts, k=k)
    d_max = distances[:, -1][:, np.newaxis]
    d_max = np.where(d_max <= 0, 1e-5, d_max)
    u = np.clip(distances / d_max, 0.0, 1.0)
    weights = (1.0 - u**3)**3

    predicted_bias = np.zeros(len(df_fit))
    for i in range(len(df_fit)):
        idx = indices[i]
        w = weights[i]
        X = np.column_stack([np.ones(k)] + [col[idx] for col in z_train_cols])
        point = np.array([1.0] + [col[i] for col in z_all_cols])

        X_w = X * w[:, np.newaxis]
        A = X.T @ X_w
        A.flat[::A.shape[0] + 1] += 1e-4
        b = X.T @ (w * train_y[idx])
        try:
            beta = np.linalg.solve(A, b)
            predicted_bias[i] = np.dot(point, beta)
        except np.linalg.LinAlgError:
            predicted_bias[i] = np.sum(w * train_y[idx]) / np.maximum(np.sum(w), 1e-5)

    predicted_bias = np.clip(predicted_bias, a_min=1e-4, a_max=None)
    return raw_ratio / predicted_bias
