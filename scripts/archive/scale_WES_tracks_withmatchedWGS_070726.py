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


def apply_pon_multivariate_loess_correction(raw_wes_depth, pon_median, offtarget_gc, dist_to_target=None, frac=0.25):
    """
    Replaces multivariate GAM with a robust 2D LOESS local linear regression.
    
    If dist_to_target (or another secondary track) is absent, uniform, or invalid, 
    this falls back automatically to 1D LOESS using GC content only.
    """
    safe_pon_median = np.where(pon_median <= 0, 1e-4, pon_median)
    raw_ratio = raw_wes_depth / safe_pon_median

    df_fit = pd.DataFrame({
        'y': raw_ratio,
        'gc': offtarget_gc
    })
    
    # Verify the secondary feature track is valid and non-uniform
    has_dist = False
    if dist_to_target is not None:
        dist_series = pd.Series(dist_to_target)
        if not dist_series.isna().all() and dist_series.nunique() > 1:
            df_fit['dist'] = dist_to_target
            has_dist = True

    # Establish validation mask for stable regression model training
    if has_dist:
        valid_mask = (
            (raw_wes_depth > 0) & 
            (~df_fit['gc'].isna()) & 
            (~df_fit['dist'].isna()) &
            (~np.isinf(df_fit['y'])) &
            (~np.isnan(df_fit['y']))
        )
    else:
        valid_mask = (
            (raw_wes_depth > 0) & 
            (~df_fit['gc'].isna()) &
            (~np.isinf(df_fit['y'])) &
            (~np.isnan(df_fit['y']))
        )

    # Protect small cohorts from overfitting
    total_bins = len(df_fit)
    min_bins = min(500, max(50, int(total_bins * 0.1)))
    
    if valid_mask.sum() < min_bins:
        print(f"[-] Warning: Too few training bins ({valid_mask.sum()} < {min_bins}). "
              f"Falling back to uncorrected raw ratio.", file=sys.stderr)
        return raw_ratio


    if has_dist:
        print(f"    [+] Running robust 2D LOESS (GC + secondary covariate) on {valid_mask.sum()} training bins...")
        
        # Pull parameters
        train_gc = df_fit.loc[valid_mask, 'gc'].values
        train_dist = df_fit.loc[valid_mask, 'dist'].values
        train_y = df_fit.loc[valid_mask, 'y'].values
        
        all_gc = df_fit['gc'].values
        all_dist = df_fit['dist'].values
        
        # Standardize features for isotropic distance calculations
        gc_mean, gc_std = np.mean(train_gc), np.std(train_gc)
        dist_mean, dist_std = np.mean(train_dist), np.std(train_dist)
        
        gc_std = 1.0 if gc_std == 0 else gc_std
        dist_std = 1.0 if dist_std == 0 else dist_std
        
        z_gc_train = (train_gc - gc_mean) / gc_std
        z_dist_train = (train_dist - dist_mean) / dist_std
        
        z_gc_all = (all_gc - gc_mean) / gc_std
        z_dist_all = (all_dist - dist_mean) / dist_std
        
        train_pts = np.column_stack((z_gc_train, z_dist_train))
        all_pts = np.column_stack((z_gc_all, z_dist_all))
        
        # Use KDTree for neighborhood queries
        tree = KDTree(train_pts)
        n_train = len(train_y)
        k = max(20, int(n_train * frac))
        k = min(k, 400)  # Local regression neighborhood cap for performance
        k = min(k, n_train)
        
        distances, indices = tree.query(all_pts, k=k)
        
        # Cleveland's tricube weight function: w = (1 - (d / d_max)^3)^3
        d_max = distances[:, -1][:, np.newaxis]
        d_max = np.where(d_max <= 0, 1e-5, d_max)
        
        u = distances / d_max
        u = np.clip(u, 0.0, 1.0)
        weights = (1.0 - u**3)**3


        predicted_bias = np.zeros(len(df_fit))
        for i in range(len(df_fit)):
            idx = indices[i]
            w = weights[i]
            
            # Neighborhood arrays
            z_gc_neigh = z_gc_train[idx]
            z_dist_neigh = z_dist_train[idx]
            y_neigh = train_y[idx]
            
            # Local Linear Design Matrix
            X = np.column_stack((np.ones(k), z_gc_neigh, z_dist_neigh))
            point = np.array([1.0, z_gc_all[i], z_dist_all[i]])
            
            # Regularized WLS Direct Solve (extremely fast)
            X_w = X * w[:, np.newaxis]
            A = X.T @ X_w
            A.flat[::A.shape[0] + 1] += 1e-4  # Ridge diagonal stabilization
            b = X.T @ (w * y_neigh)
            
            try:
                beta = np.linalg.solve(A, b)
                predicted_bias[i] = np.dot(point, beta)
            except np.linalg.LinAlgError:
                predicted_bias[i] = np.sum(w * y_neigh) / np.maximum(np.sum(w), 1e-5)
                
        predicted_bias = np.clip(predicted_bias, a_min=1e-4, a_max=None)
        depth_scaled = raw_ratio / predicted_bias
        return depth_scaled


    else:
        print(f"    [+] Running robust 1D LOESS (GC only) using statsmodels on {valid_mask.sum()} training bins...")
        train_gc = df_fit.loc[valid_mask, 'gc'].values
        train_y = df_fit.loc[valid_mask, 'y'].values
        all_gc = df_fit['gc'].values
        
        try:
            loess_fit = sm.nonparametric.lowess(
                endog=train_y, exog=train_gc, frac=frac, it=3, return_sorted=False
            )
            sort_idx = np.argsort(train_gc)
            predicted_bias = np.interp(all_gc, train_gc[sort_idx], loess_fit[sort_idx])
        except Exception as e:
            print(f"[-] Warning: 1D LOESS fitting failed ({str(e)}). "
                  f"Falling back to uncorrected raw ratio.", file=sys.stderr)
            return raw_ratio
            
        predicted_bias = np.clip(predicted_bias, a_min=1e-4, a_max=None)
        depth_scaled = raw_ratio / predicted_bias
        return depth_scaled



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

    print(f"[+] PoN loaded: {len(df_pon)} windows.")
    print(f"[+] Found {len(tumor_bams)} tumor WES BAMs to process.")

    for bam_path in tumor_bams:
        sample_name = os.path.basename(bam_path).split('.')[0]
        print(f"\n" + "="*55)
        print(f"[+] Processing tumor: {sample_name}")
        print("="*55)

        prefix_off = os.path.join(TMP_DIR, f"{sample_name}_off_target")
        print(f"    -> Running mosdepth on off-target windows (WES)...")
        run_mosdepth(bam_path, OFFTARGET_BED, prefix_off, threads=THREADS)
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
                print(f"    -> Running mosdepth on matched tumor WGS BAM: {os.path.basename(WGS_BAM)}")
                run_mosdepth(WGS_BAM, OFFTARGET_BED, prefix_wgs, threads=THREADS)
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

        print(f"    -> Computing PoN-anchored ratios and fitting LOESS multi-variable correction...")
        depth_scaled = apply_pon_multivariate_loess_correction(raw_wes_depth, pon_median, offtarget_gc, dist_to_target)

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

        df_output['predicted_upscale_depth'] = pon_median * depth_scaled

        max_theoretical_depth = pon_median * 200.0
        df_output['predicted_upscale_depth'] = np.minimum(df_output['predicted_upscale_depth'], max_theoretical_depth)
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
    print("="*55)


if __name__ == "__main__":
    main()
