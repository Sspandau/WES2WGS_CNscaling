import os
import argparse
import glob
import subprocess
import pandas as pd
import numpy as np
import pybedtools
import statsmodels.api as sm

def run_mosdepth(bam_path, bed_path, output_prefix, threads=4):
    """ Runs mosdepth to pull clean window read counts (MAPQ>=20, strips duplicates/supplementary) """
    cmd = [
        "mosdepth",
        "--threads", str(threads),
        "--by", bed_path,
        "--mapq", "20",
        "--flag", "3844", # Excludes duplicates (1024), supplementary (2048), non-primary (256), QC-fail (512)
        "--no-per-base",
        output_prefix,
        bam_path
    ]
    subprocess.run(cmd, check=True)

def parse_mosdepth_regions(output_prefix):
    """ Loads mosdepth compressed bed file into a pandas dataframe """
    regions_file = f"{output_prefix}.regions.bed.gz"
    return pd.read_csv(regions_file, sep='\t', compression='gzip', 
                       header=None, names=['chrom', 'start', 'end', 'depth'])

def compute_gc_content(windows_bed_path, reference_fasta_path):
    """ Computes GC% fractional metrics for arbitrary BED entries via bedtools nuc """
    windows_bed = pybedtools.BedTool(windows_bed_path)
    nuc_bed = windows_bed.nucleotide_content(fi=reference_fasta_path)
    df_nuc = nuc_bed.to_dataframe(header=None)
    return df_nuc[5].astype(float).values # Col index 5 is %GC

def apply_sample_loess_correction(raw_offtarget_depths, offtarget_gc, raw_ontarget_depths, ontarget_gc):
    """
    Fits a LOESS curve to the tumor sample's ON-TARGET windows where depth data is robust.
    Then, projects and corrects the sparse/noisy OFF-TARGET windows using that curve.
    """
    # Filter on-target data for training (drop zeros/NaNs to protect regression models)
    valid_mask = (raw_ontarget_depths > 0) & (~np.isnan(ontarget_gc)) & (~np.isnan(raw_ontarget_depths))
    train_gc = ontarget_gc[valid_mask]
    train_depth = raw_ontarget_depths[valid_mask]
    
    if len(train_gc) < 50:
        print("    [!] Warning: Too few valid on-target probes to fit LOESS. Skipping GC correction.")
        return raw_offtarget_depths

    # Fit LOESS model to the sample's own robust on-target data
    loess_fit = sm.nonparametric.lowess(
        endog=train_depth, 
        exog=train_gc, 
        frac=0.1, 
        it=3, 
        return_sorted=False
    )
    
    # Linearly extrapolate/interpolate the fitted curve out to the off-target GC grid points
    sort_idx = np.argsort(train_gc)
    fitted_values_offtarget = np.interp(offtarget_gc, train_gc[sort_idx], loess_fit[sort_idx])
    fitted_values_offtarget = np.clip(fitted_values_offtarget, a_min=1e-4, a_max=None)
    
    # Apply fitted pattern to correct the off-target space
    return raw_offtarget_depths / fitted_values_offtarget

def main():
    # ----------------------------------------------------
    # 1. PARAMETERS & INPUT PATHS
    # ----------------------------------------------------
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--targets", metavar="FILE", required=True,
                   help="Targets bed file")
    p.add_argument("--offtarget_windows", metavar="FILE", required=True,
                   help="Off target genomic windows from Step 1")
    p.add_argument("--reference", metavar="FILE", required=True,
                   help="reference fasta")
    p.add_argument("--pon_tsv", metavar="FILE", required=True,
                   help="PoN tsv from Step 4")
    p.add_argument("--wes_dir", required=True,
                   help="Directory path where WES tumor bams are located")
    p.add_argument("-t", metavar="INT", type=int,
                   default=4, help="threads")
    p.add_argument("--output_dir", required=True,
                   help="Output directory path")
    p.add_argument("--temp_dir", required=True,
                   help="Temp directory path")
    args = p.parse_args()
    # Genomic infrastructure files
    ONTARGET_BED = args.targets
    OFFTARGET_BED = args.offtarget_windows
    REF_FASTA = args.reference
    PON_TSV = args.pon_tsv
    
    # Tumor input directory configuration
    TUMOR_BAM_DIR = args.wes_dir
    BAM_PATTERN = os.path.join(TUMOR_BAM_DIR, "*.bam")
    
    THREADS = 4
    OUTPUT_DIR = args.output_dir
    TMP_DIR = args.temp_dir
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)


    # 2. FILE SYSTEM VERIFICATION & BACKGROUND LOADING

    tumor_bams = glob.glob(BAM_PATTERN)
    if not tumor_bams:
        print(f"[-] Error: No tumor BAM tracks found via pattern: {BAM_PATTERN}")
        return
    if not os.path.exists(PON_TSV):
        print(f"[-] Error: Panel of Normals file missing at {PON_TSV}. Run Step 4 first.")
        return

    print(f"[+] Loading SVD Panel of Normals track baseline data...")
    df_pon = pd.read_csv(PON_TSV, sep='\t', index_col=0)
    
    print("[+] Calculating structural GC mappings for genomic windows...")
    offtarget_gc = df_pon['gc_pct'].values # Already calculated in PoN matrix
    ontarget_gc = compute_gc_content(ONTARGET_BED, REF_FASTA)

    print(f"[+] Found {len(tumor_bams)} tumor WES files to cross-normalize.")


    # 3. PROCESSING LOOP PER TUMOR SAMPLE (STEP 5)

    for bam_path in tumor_bams:
        sample_name = os.path.basename(bam_path).split('.')[0]
        print(f"\n" + "="*50 + f"\n[+] PROCESSING TUMOR: {sample_name}\n" + "="*50)
        
        # Part A: Run Dual Count Windows
        prefix_on = os.path.join(TMP_DIR, f"{sample_name}_on_target")
        prefix_off = os.path.join(TMP_DIR, f"{sample_name}_off_target")
        
        print("    -> Quantifying read layers across On-Target capture tracks...")
        run_mosdepth(bam_path, ONTARGET_BED, prefix_on, threads=THREADS)
        df_on = parse_mosdepth_regions(prefix_on)
        
        print("    -> Quantifying read layers across Off-Target background tracks...")
        run_mosdepth(bam_path, OFFTARGET_BED, prefix_off, threads=THREADS)
        df_off = parse_mosdepth_regions(prefix_off)
        
        # Verify indices align flawlessly with our background PoN matrix
        if len(df_off) != len(df_pon):
            print(f"    [-] Error: Count dimensions ({len(df_off)}) disagree with PoN tracker dimensions ({len(df_pon)}). Skipping.")
            continue
            
        # Part B: Autosomal Median Normalization using ON-TARGET regions exclusively
        # Identify autosomes within the on-target frame to evade gender/sex karyotype skews
        autosome_on_mask = df_on['chrom'].str.startswith(('chrX', 'chrY', 'chrM', 'X', 'Y', 'MT')) == False
        if not np.any(autosome_on_mask):
            autosome_on_mask = np.ones(len(df_on), dtype=bool)
            
        wes_library_normalizer = np.median(df_on.loc[autosome_on_mask, 'depth'])
        print(f"    -> Scaled target sequencing capture normalization depth: {wes_library_normalizer:.2f} reads")
        
        if wes_library_normalizer == 0:
            print("    [!] Error: On-target median library estimation returned zero reads. Skipping file.")
            continue
            
        # Convert depth vectors to normalized scale metrics
        norm_on_depths = df_on['depth'].values / wes_library_normalizer
        norm_off_depths = df_off['depth'].values / wes_library_normalizer
        
        # Part C: Project GC Curve Shape from On-Target to Off-Target
        print("    -> Transferring fitted on-target LOESS GC curves over to off-target coordinates...")
        gc_corrected_off_depths = apply_sample_loess_correction(
            raw_offtarget_depths=norm_off_depths, 
            offtarget_gc=offtarget_gc, 
            raw_ontarget_depths=norm_on_depths, 
            ontarget_gc=ontarget_gc
        )
        
        # Part D: Convert to WGS Copy Ratio Scaling
        # Formula: depth_scaled(w) = depth_WES_offtarget_GCcorrected(w) / pon_median(w)
        # (Assuming your input pon_median is already centered to an autosomal baseline of ~1.0)
        # Avoid zero division errors with low coverage or blacklisted windows
        safe_pon_median = np.where(df_pon['pon_median'] <= 0, 1e-4, df_pon['pon_median'])
        depth_scaled = gc_corrected_off_depths / safe_pon_median

  
        # 4. STEP 6: QUALITY FILTERING & FLAGGING UNRELIABLE WINDOWS

        print("    -> Analyzing window architecture and flagging low-confidence intervals...")
        
        # Filter 1: Coefficient of Variation (CV) tracking check across normal cohort
        # CV = standard_deviation / mean = sqrt(variance) / median_baseline
        pon_sd = np.sqrt(df_pon['pon_variance'].values)
        pon_cv = np.where(df_pon['pon_median'] > 0, pon_sd / df_pon['pon_median'], 0)
        flag_high_cv = (pon_cv > 0.30).astype(int)
        
        # Filter 2: Drop structural Dropouts (raw WES depth == 0)
        flag_zero_wes = (df_off['depth'].values == 0).astype(int)
        
        # Filter 3: Track and flag extreme multi-copy inflation or completely unmappable structural limits
        # Flag if scaled factor behaves abnormally (e.g., > 20x or < 0.05x normal values)
        flag_extreme_scaling = ((depth_scaled > 20.0) | (depth_scaled < 0.05)).astype(int)
        
        # Filter 4: SVD Panel Variance Mask Pre-check (Calculated in Step 4)
        flag_pon_variance = df_pon['is_high_variance'].values
        
        # Consolidate all filtering requirements into a clean downstream rejection master bitmask
        # 0 = Perfect high-confidence track window; 1 = Low-confidence masked window
        master_mask = (flag_high_cv | flag_zero_wes | flag_extreme_scaling | flag_pon_variance)
        

        # 5. EXPORT FINALIZED SCALED TARGET TRACK

        df_output = df_pon[['chrom', 'start', 'end', 'gc_pct']].copy()
        df_output['raw_wes_depth'] = df_off['depth'].values
        df_output['pon_median_wgs'] = df_pon['pon_median'].values
        df_output['depth_scaled_ratio'] = depth_scaled
        df_output['pon_cv'] = pon_cv
        
        # Logging flags
        df_output['flag_high_cv'] = flag_high_cv
        df_output['flag_zero_wes'] = flag_zero_wes
        df_output['flag_extreme_scaling'] = flag_extreme_scaling
        df_output['mask_rejected'] = master_mask
        
        # Generate clean filtered copy log2 ratio column for instant plotting imports
        # Protect log calculations against zeroes via a low clip baseline floor
        df_output['log2_ratio'] = np.log2(np.clip(depth_scaled, a_min=1e-3, a_max=None))
        # Enforce NaN over masked exclusions so segmentation algorithms ignore them
        df_output.loc[master_mask == 1, 'log2_ratio'] = np.nan

        output_file_name = os.path.join(OUTPUT_DIR, f"{sample_name}_off_target_copy_ratios.tsv")
        df_output.to_csv(output_file_name, sep='\t', index=True)
        
        print(f"    [+] SAVED: {output_file_name}")
        print(f"        -> Usable Windows: {len(df_output[df_output['mask_rejected'] == 0])} / {len(df_output)}")
        print(f"        -> Dropped due to Zero WES Reads: {flag_zero_wes.sum()}")
        print(f"        -> Dropped due to High Panel CV (>0.3): {flag_high_cv.sum()}")
        print(f"        -> Dropped due to Extreme Scaling Thresholds: {flag_extreme_scaling.sum()}")

    print("\n" + "="*60 + "\n[+] PIPELINE PIPING PIPELINE COMPLETE!\nAll tumor lines are normalized on WGS scales.\n" + "="*60)

if __name__ == "__main__":
    main()