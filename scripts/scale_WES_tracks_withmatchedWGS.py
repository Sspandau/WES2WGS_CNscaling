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
- GC bias (and secondary distance-to-target bias) is estimated and corrected 
  using a robust 2D LOESS local linear regression directly from the raw ratios.
- Output log2 copy ratios are directly comparable to WGS copy ratios.
'''

def run_mosdepth(bam_path, bed_path, output_prefix, threads=4, min_mapq=15):
    """
    Runs mosdepth to pull clean window read counts.
    Filters: MAPQ >= min_mapq (excludes individual reads with low mapping
    quality, i.e. low-mappability reads, before depth is ever computed),
    excludes duplicates, supplementary, non-primary, and QC-fail reads.
    """
    cmd = [
        "mosdepth",
        "--threads", str(threads),
        "--by", bed_path,
        "--mapq", str(min_mapq),
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
                                             dist_to_target=None, frac=0.25):
    """
    Robust local linear regression (LOESS) using whichever covariates are
    actually valid: GC is always included; dist_to_target is added if
    present, non-constant, and not all-NaN. Falls back to GC-only 1D LOESS
    if dist_to_target isn't usable.
    """
    safe_pon_median = np.where(pon_median <= 0, 1e-4, pon_median)
    raw_ratio = raw_wes_depth / safe_pon_median

    df_fit = pd.DataFrame({'y': raw_ratio, 'gc': offtarget_gc})
    covariate_cols = ['gc']

    if dist_to_target is not None:
        s = pd.Series(dist_to_target)
        if not s.isna().all() and s.nunique() > 1:
            df_fit['dist'] = dist_to_target
            covariate_cols.append('dist')

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


def apply_pon_multivariate_lr(raw_wes_depth, pon_median, wgs_tumor_depth, offtarget_gc, chr, dist_to_target):
    """
    Stacking regressor to predict wgs depth from wes depth, pon median, offtarget_gc and distance to target
    """

    #define base models for stack
    base_models = [
        ('lgbm', LGBMRegressor(n_estimators=100, random_state=42)),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('svr', SVR(kernel='rbf', C=1.0))]
    
    #define meta model for stack
    meta_model = Ridge()

    #stacked model with 5k CV
    stack_pipeline = StackingRegressor(
        estimators=base_models, final_estimator=meta_model, cv=5, n_jobs=-1)
    
    #initialize and filter x and y
    df_fit = pd.DataFrame({"y": wgs_tumor_depth,
                           "wes_depth" : raw_wes_depth, "pon_depth" : pon_median, "gc" : offtarget_gc, 
                           "chrom" : chr, "dist" : dist_to_target
    })

    valid_mask = (
            (df_fit['wes_depth'] > 0) & 
            (~df_fit['gc'].isna()) & 
            (~df_fit['dist'].isna()) #may need to add more filtering
        )
    
    y = df_fit.loc[valid_mask, 'y'].values
    wes_train = df_fit.loc[valid_mask, 'wes_depth'].values
    pon_train = df_fit.loc[valid_mask, 'pon_depth'].values
    gc_train = df_fit.loc[valid_mask, 'gc'].values
    dist_train = df_fit.loc[valid_mask, 'dist'].values
    chr_train = df_fit.loc[valid_mask, 'chrom'].values 

    X = pd.DataFrame({
        "wes_depth" : wes_train, "pon_depth": pon_train, "gc" : gc_train, "dist": dist_train
    })

    #cross validation by chromosome
    outer_cv = GroupKFold(n_splits=5)
    cv_results = cross_validate(
    stack_pipeline, 
    X, y, 
    groups=chr_train, #split by chromosome to emulate fitting to independent regions (as seen in LOESS)
    cv=outer_cv, 
    scoring=['r2', 'neg_mean_absolute_error'],
    n_jobs=-1)

    predictions = cross_val_predict(stack_pipeline, X, y, groups=chr_train, cv=outer_cv, n_jobs=-1)

    full_predictions = np.full(shape=len(df_fit), fill_value=np.nan)
    full_predictions[valid_mask] = predictions

    return cv_results, full_predictions
    



def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--offtarget_windows", metavar="FILE", required=True,
                   help="Off-target genomic windows BED from Step 1")
    p.add_argument("--pon_tsv", metavar="FILE", required=True,
                   help="PoN TSV from Step 4 (WGS2PoN.py)")
    p.add_argument("--wes_dir", required=True,
                   help="Directory containing tumor WES BAMs")
    p.add_argument("--wgs_bam", default=None,
                   help="Path to the matched tumor WGS BAM. "
                        "If omitted, wgs_tumor_depth will be filled with NaN.")
    p.add_argument("-t", metavar="INT", type=int,
                   default=4, help="Threads per mosdepth run")
    p.add_argument("--min_mapq", metavar="INT", type=int, default=15,
                   help="Minimum per-read mapping quality (MAPQ) passed to mosdepth. "
                        "Reads below this are excluded from depth counting entirely, "
                        "before any window-level aggregation. Default: 15")
    p.add_argument("--output_dir", required=True,
                   help="Output directory for per-sample copy ratio TSVs")
    p.add_argument("--temp_dir", required=True,
                   help="Temp directory for mosdepth intermediary files")
    args = p.parse_args()

    OFFTARGET_BED  = args.offtarget_windows
    PON_TSV        = args.pon_tsv
    TUMOR_BAM_DIR  = args.wes_dir
    WGS_BAM        = args.wgs_bam
    THREADS        = args.t
    MIN_MAPQ       = args.min_mapq
    OUTPUT_DIR     = args.output_dir
    TMP_DIR        = args.temp_dir

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)

    tumor_bams = glob.glob(os.path.join(TUMOR_BAM_DIR, "*.bam"))
    if not tumor_bams:
        print(f"[-] Error: No tumor BAM files found in: {TUMOR_BAM_DIR}")
        return

    if not os.path.exists(PON_TSV):
        print(f"[-] Error: PoN file missing at {PON_TSV}. Run WGS2PoN.py first.")
        return


    print(f"[+] Loading WGS Panel of Normals: {PON_TSV}")
    df_pon = pd.read_csv(PON_TSV, sep='\t', index_col=0)

    pon_median     = df_pon['pon_median'].values
    pon_variance   = df_pon['pon_variance'].values
    
    gc_col = 'gc_pct' if 'gc_pct' in df_pon.columns else ('gc' if 'gc' in df_pon.columns else None)
    offtarget_gc = df_pon[gc_col].values if gc_col else np.full(len(df_pon), np.nan)
    dist_to_target = df_pon['dist_to_target'].values if 'dist_to_target' in df_pon.columns else None
    chromosomes = df_pon['chrom'].values if 'chrom' in df_pon.columns else None

    print(f"[+] PoN loaded: {len(df_pon)} windows.")
    print(f"[+] Found {len(tumor_bams)} tumor WES BAMs to process.")

    for bam_path in tumor_bams:
        sample_name = os.path.basename(bam_path).split('.')[0]
        print(f"\n" + "="*55)
        print(f"[+] Processing tumor: {sample_name}")
        print("="*55)

        prefix_off = os.path.join(TMP_DIR, f"{sample_name}_off_target")
        print(f"    -> Running mosdepth on off-target windows (WES), "
              f"excluding reads with MAPQ < {MIN_MAPQ}...")
        run_mosdepth(bam_path, OFFTARGET_BED, prefix_off, threads=THREADS, min_mapq=MIN_MAPQ)
        df_off = parse_mosdepth_regions(prefix_off)

        if len(df_off) != len(df_pon):
            print(f"    [-] Error: Window count mismatch — "
                  f"mosdepth={len(df_off)}, PoN={len(df_pon)}. Skipping sample.")
            continue

        raw_wes_depth = df_off['depth'].values.astype(float)


        if WGS_BAM:
            if not os.path.exists(WGS_BAM):
                print(f"    [-] Warning: WGS BAM not found at {WGS_BAM}. "
                      f"wgs_tumor_depth will be NaN.")
                wgs_tumor_depth = np.full(len(df_pon), np.nan)
            else:
                prefix_wgs = os.path.join(TMP_DIR, f"{sample_name}_wgs_tumor")
                print(f"    -> Running mosdepth on matched tumor WGS BAM: {os.path.basename(WGS_BAM)} "
                      f"(excluding reads with MAPQ < {MIN_MAPQ})...")
                run_mosdepth(WGS_BAM, OFFTARGET_BED, prefix_wgs, threads=THREADS, min_mapq=MIN_MAPQ)
                df_wgs = parse_mosdepth_regions(prefix_wgs)
                if len(df_wgs) != len(df_pon):
                    print(f"    [-] Warning: WGS window count mismatch — "
                          f"mosdepth={len(df_wgs)}, PoN={len(df_pon)}. "
                          f"wgs_tumor_depth will be NaN.")
                    wgs_tumor_depth = np.full(len(df_pon), np.nan)
                else:
                    wgs_tumor_depth = df_wgs['depth'].values.astype(float)
        else:
            wgs_tumor_depth = np.full(len(df_pon), np.nan)

        print(f"    -> Computing LOESS correction for depth scaling...")
        depth_scaled = apply_pon_multivariate_loess_correction(
            raw_wes_depth, pon_median, offtarget_gc, dist_to_target)

  #      if dist_to_target is not None:
  #          print(f"    -> Computing LR between raw WES depth and tumor WGS...")
 #           cv_results, predicted_wgs_depth = apply_pon_multivariate_lr(raw_wes_depth, pon_median, wgs_tumor_depth, offtarget_gc, chromosomes, dist_to_target)
#        else:
 #           print("    [-] Warning: dist_to_target not available. Skipping multivariate LR for WGS prediction.")


        print(f"    -> Applying quality filters...")
        pon_sd = np.sqrt(np.maximum(pon_variance, 0))
        safe_pon_median = np.where(pon_median > 0, pon_median, 1.0)
        pon_cv = np.where(pon_median > 0, pon_sd / safe_pon_median, 0)
        
        flag_high_cv = (pon_cv > 0.30).astype(int)
        flag_zero_wes = (raw_wes_depth == 0).astype(int)

        flag_unstable_pon = (pon_median < 0.01).astype(int)
        flag_numerical_instability = (depth_scaled > 1000).astype(int)
        flag_extreme_scaling = (flag_unstable_pon | flag_numerical_instability).astype(int)

        if 'is_high_variance' in df_pon.columns:
            flag_pon_variance = df_pon['is_high_variance'].values
        else:
            variance_threshold = np.percentile(pon_variance, 99)
            flag_pon_variance = (pon_variance > variance_threshold).astype(int)

        master_mask = (
            flag_high_cv |
            flag_zero_wes |
            flag_extreme_scaling |
            flag_pon_variance
        ).astype(int)


        df_output = df_pon[['chrom', 'start', 'end']].copy()
        df_output['gc_pct'] = offtarget_gc
        df_output['dist_to_target'] = dist_to_target if dist_to_target is not None else np.nan
        df_output['raw_wes_depth']      = raw_wes_depth
        df_output['pon_median_wgs']     = pon_median 
        df_output['depth_scaled_ratio'] = depth_scaled
        df_output['pon_cv']             = pon_cv

        df_output['predicted_loess_upscale_depth'] = pon_median * depth_scaled
#        df_output['predicted_lr_upscale_depth'] = predicted_wgs_depth if dist_to_target is not None else np.nan

 #       max_theoretical_depth = pon_median * 200.0
#        df_output['predicted_loess_upscale_depth'] = np.minimum(df_output['predicted_upscale_depth'], max_theoretical_depth)
        df_output['wgs_tumor_depth'] = wgs_tumor_depth

        df_output['flag_high_cv']         = flag_high_cv
        df_output['flag_zero_wes']        = flag_zero_wes
        df_output['flag_extreme_scaling'] = flag_extreme_scaling
        df_output['flag_pon_variance']    = flag_pon_variance
        df_output['mask_rejected']        = master_mask

        df_output['log2_ratio'] = np.log2(np.clip(depth_scaled, a_min=1e-3, a_max=None))
        df_output.loc[master_mask == 1, 'log2_ratio'] = np.nan

        output_path = os.path.join(OUTPUT_DIR, f"{sample_name}_off_target_copy_ratios.tsv")
        df_output.to_csv(output_path, sep='\t', index=True)

        n_usable  = (master_mask == 0).sum()
        n_total   = len(df_output)
        wgs_covered = (~np.isnan(wgs_tumor_depth)).sum()
        
        print(f"    [-] Saved: {output_path}")
        print(f"    -> Usable windows:                  {n_usable} / {n_total}")
        print(f"    -> Dropped (zero WES depth):        {flag_zero_wes.sum()}")
        print(f"    -> Dropped (high PoN CV > 0.3):     {flag_high_cv.sum()}")
        print(f"    -> Dropped (extreme scaling):       {flag_extreme_scaling.sum()}")
        print(f"    -> Dropped (high PoN variance):     {flag_pon_variance.sum()}")
        print(f"    -> WGS tumor windows with depth:    {wgs_covered} / {n_total}")

    print("\n" + "="*55)
    print("[+] scale_WES_tracks.py COMPLETE")
    print("    All tumor samples normalized to WGS PoN scale with robust LOESS.")
