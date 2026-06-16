import os
import argparse
import glob
import subprocess
import pandas as pd
import numpy as np
import pybedtools
import statsmodels.api as sm

'''
Script that scales WES off-target depth to WGS-comparable copy ratios using
a WGS Panel of Normals (PoN) as the reference baseline.

Normalization strategy:
- Raw WES off-target depth is compared directly to the PoN median at each window.
- GC bias is estimated and corrected from the raw WES/PoN ratios themselves,
  avoiding any use of on-target WES depth which may be inflated by focal CN gains.
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

def apply_pon_loess_gc_correction(raw_wes_depth, pon_median, offtarget_gc):
    """
    Computes WES/PoN depth ratios and fits a LOESS GC bias curve directly
    on those ratios. Corrects and returns the GC-corrected copy ratios.

    By anchoring entirely to the PoN median, this avoids any CN inflation
    from using on-target WES depth as a normalizer.

    Args:
        raw_wes_depth: numpy array of raw off-target WES depths per window
        pon_median:    numpy array of PoN median WGS depths per window
        offtarget_gc:  numpy array of GC fraction per window

    Returns:
        depth_scaled: numpy array of GC-corrected WES/PoN copy ratios
    """
    # Protect against zero PoN median (unmappable or blacklisted windows)
    safe_pon_median = np.where(pon_median <= 0, 1e-4, pon_median)

    # Step 1: Compute raw ratio of WES depth to WGS PoN baseline
    raw_ratio = raw_wes_depth / safe_pon_median

    # Step 2: Fit LOESS GC bias curve on the raw ratios
    # Only use windows with nonzero WES depth and valid GC values for training
    valid_mask = (
        (raw_wes_depth > 0) &
        (~np.isnan(offtarget_gc)) &
        (~np.isnan(raw_ratio)) &
        (~np.isinf(raw_ratio))
    )

    if valid_mask.sum() < 50:
        print("    [!] Warning: Too few valid windows to fit LOESS GC model. Returning uncorrected ratios.")
        return raw_ratio

    train_gc = offtarget_gc[valid_mask]
    train_ratio = raw_ratio[valid_mask]

    loess_fit = sm.nonparametric.lowess(
        endog=train_ratio,
        exog=train_gc,
        frac=0.1,
        it=3,
        return_sorted=False
    )

    # Step 3: Interpolate fitted curve to all windows (including zero-depth ones)
    sort_idx = np.argsort(train_gc)
    fitted_all = np.interp(offtarget_gc, train_gc[sort_idx], loess_fit[sort_idx])
    fitted_all = np.clip(fitted_all, a_min=1e-4, a_max=None)

    # Step 4: Divide raw ratios by the fitted GC curve to remove bias
    depth_scaled = raw_ratio / fitted_all

    return depth_scaled


def main():

    # 1. PARAMETERS & INPUT PATHS

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
    THREADS        = args.t
    OUTPUT_DIR     = args.output_dir
    TMP_DIR        = args.temp_dir

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)


    # 2. FILE VERIFICATION & BACKGROUND LOADING

    tumor_bams = glob.glob(os.path.join(TUMOR_BAM_DIR, "*.bam"))
    if not tumor_bams:
        print(f"[-] Error: No tumor BAM files found in: {TUMOR_BAM_DIR}")
        return

    if not os.path.exists(PON_TSV):
        print(f"[-] Error: PoN file missing at {PON_TSV}. Run WGS2PoN.py first.")
        return

    print(f"[+] Loading WGS Panel of Normals: {PON_TSV}")
    df_pon = pd.read_csv(PON_TSV, sep='\t', index_col=0)

    # Extract PoN arrays used throughout
    pon_median    = df_pon['pon_median'].values
    pon_variance  = df_pon['pon_variance'].values
    offtarget_gc  = df_pon['gc_pct'].values

    print(f"[+] PoN loaded: {len(df_pon)} windows.")
    print(f"[+] Found {len(tumor_bams)} tumor WES BAMs to process.")


    # 3. PER-SAMPLE PROCESSING LOOP

    for bam_path in tumor_bams:
        sample_name = os.path.basename(bam_path).split('.')[0]
        print(f"\n" + "="*55)
        print(f"[+] Processing tumor: {sample_name}")
        print("="*55)

        # Step A: Count off-target depth with mosdepth
        prefix_off = os.path.join(TMP_DIR, f"{sample_name}_off_target")
        print(f"    -> Running mosdepth on off-target windows...")
        run_mosdepth(bam_path, OFFTARGET_BED, prefix_off, threads=THREADS)
        df_off = parse_mosdepth_regions(prefix_off)

        # Sanity check: window count must match PoN
        if len(df_off) != len(df_pon):
            print(f"    [-] Error: Window count mismatch — "
                  f"mosdepth={len(df_off)}, PoN={len(df_pon)}. Skipping sample.")
            continue

        raw_wes_depth = df_off['depth'].values.astype(float)

        # Step B: Compute PoN-anchored ratios with LOESS GC correction
        # GC bias is estimated directly from WES/PoN ratios — no on-target
        # WES depth is used, avoiding CN inflation from focal amplifications.
        print(f"    -> Computing PoN-anchored ratios and fitting LOESS GC correction...")
        depth_scaled = apply_pon_loess_gc_correction(raw_wes_depth, pon_median, offtarget_gc)


        # 4. QUALITY FILTERING & FLAGGING (Step 6)

        print(f"    -> Applying quality filters...")

        # Filter 1: High coefficient of variation in PoN normals (noisy window)
        pon_sd = np.sqrt(pon_variance)
        pon_cv = np.where(pon_median > 0, pon_sd / pon_median, 0)
        flag_high_cv = (pon_cv > 0.30).astype(int)

        # Filter 2: Zero WES depth (no reads observed in this window)
        flag_zero_wes = (raw_wes_depth == 0).astype(int)

        # Filter 3: Extreme scaling values (implausible CN — likely mapping artifact)
        flag_extreme_scaling = ((depth_scaled > 20.0) | (depth_scaled < 0.05)).astype(int)

        # Filter 4: Pre-flagged high-variance PoN windows from Step 4
        flag_pon_variance = df_pon['is_high_variance'].values

        # Master rejection mask: 0 = high-confidence, 1 = rejected
        master_mask = (
            flag_high_cv |
            flag_zero_wes |
            flag_extreme_scaling |
            flag_pon_variance
        ).astype(int)


        # 5. ASSEMBLE AND EXPORT OUTPUT TRACK

        df_output = df_pon[['chrom', 'start', 'end', 'gc_pct']].copy()
        df_output['raw_wes_depth']      = raw_wes_depth
        df_output['pon_median_wgs']     = pon_median
        df_output['depth_scaled_ratio'] = depth_scaled
        df_output['pon_cv']             = pon_cv

        # Quality flags
        df_output['flag_high_cv']         = flag_high_cv
        df_output['flag_zero_wes']        = flag_zero_wes
        df_output['flag_extreme_scaling'] = flag_extreme_scaling
        df_output['flag_pon_variance']    = flag_pon_variance
        df_output['mask_rejected']        = master_mask

        # Log2 copy ratio — NaN over masked windows so segmenters ignore them
        df_output['log2_ratio'] = np.log2(np.clip(depth_scaled, a_min=1e-3, a_max=None))
        df_output.loc[master_mask == 1, 'log2_ratio'] = np.nan

        output_path = os.path.join(OUTPUT_DIR, f"{sample_name}_off_target_copy_ratios.tsv")
        df_output.to_csv(output_path, sep='\t', index=True)

        n_usable  = (master_mask == 0).sum()
        n_total   = len(df_output)
        print(f"    [+] Saved: {output_path}")
        print(f"    -> Usable windows:                  {n_usable} / {n_total}")
        print(f"    -> Dropped (zero WES depth):        {flag_zero_wes.sum()}")
        print(f"    -> Dropped (high PoN CV > 0.3):     {flag_high_cv.sum()}")
        print(f"    -> Dropped (extreme scaling):       {flag_extreme_scaling.sum()}")
        print(f"    -> Dropped (high PoN variance):     {flag_pon_variance.sum()}")

    print("\n" + "="*55)
    print("[+] scale_WES_tracks.py COMPLETE")
    print("    All tumor samples normalized to WGS PoN scale.")
    print("="*55)


if __name__ == "__main__":
    main()