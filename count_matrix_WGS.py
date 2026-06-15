import os
import glob
import argparse
import subprocess
import pandas as pd
import numpy as np
import pybedtools
import statsmodels.api as sm

'''
Script that creates matrix with read counts per genome window (step 1) from normal WGS bams
To add:
log file instead of print statements
'''

def run_mosdepth(bam_path, bed_path, output_prefix, threads=4):
    """
    Executes mosdepth to calculate read counts across targeted BED windows.
    Filters: MAPQ >= 20, excludes duplicates, secondary, and supplementary alignments.
    """
    cmd = [
        "mosdepth",
        "--threads", str(threads),
        "--by", bed_path,
        "--mapq", "20",
        "--flag", "3844",  # Excludes duplicates (1024), supplementary (2048), non-primary (256), QC-fail (512)
        "--no-per-base",
        output_prefix,
        bam_path
    ]
    subprocess.run(cmd, check=True)

def parse_mosdepth_regions(output_prefix):
    """Parses compressed mosdepth region output into a clean DataFrame."""
    regions_file = f"{output_prefix}.regions.bed.gz"
    df = pd.read_csv(regions_file, sep='\t', compression='gzip', 
                     header=None, names=['chrom', 'start', 'end', 'depth'])
    return df

def compute_gc_content(windows_bed_path, reference_fasta_path):
    """
    Uses bedtools nuc to compute GC percentage for each window.
    Returns a pandas Series of GC fractions matching the window index order.
    """
    print(f"[+] Computing GC% tracking metric via bedtools nuc using reference: {reference_fasta_path}")
    windows_bed = pybedtools.BedTool(windows_bed_path)
    
    # Run bedtools nuc
    nuc_bed = windows_bed.nucleotide_content(fi=reference_fasta_path)
    
    # bedtools nuc appends calculation tracking columns to the end of the original BED file
    # For a 4-column input BED (chrom, start, end, name), column 5 is %AT and column 6 is %GC
    df_nuc = nuc_bed.to_dataframe(header=None)
    
    # Extract the %GC column (index 5)
    gc_series = df_nuc[5]
    gc_series = gc_series[gc_series != '6_pct_gc'].astype(float).reset_index(drop=True)
    return gc_series

def apply_loess_gc_correction(depth_series, gc_series, autosome_indices):
    """
    Fits a LOESS curve of normalized_depth ~ GC% using autosomal windows,
    then predicts and normalizes across all genomic windows.
    """
    # Create clean training vectors strictly from valid autosomal windows
    # Reject windows with zero depth or NaN/inf errors to protect regression stability
    valid_mask = autosome_indices & (depth_series > 0) & (~np.isnan(gc_series)) & (~np.isnan(depth_series))
    
    train_gc = gc_series[valid_mask].values
    train_depth = depth_series[valid_mask].values
    
    if len(train_gc) < 10:
        print("    [!] Error: Too few valid autosomal windows to safely fit LOESS model.")
        return depth_series

    # Fit LOESS model: depth ~ GC content
    # frac=0.1 means local regression spans a rolling 10% smoothing window neighborhood
    # iters=3 applies robust biweight updates to downweight structural outlier artifacts
    loess_fit = sm.nonparametric.lowess(
        endog=train_depth, 
        exog=train_gc, 
        frac=0.1, 
        it=3, 
        return_sorted=False
    )
    
    # Map the non-linear fitted trendline values to all windows across the entire genome
    # We use numpy interpolation to infer fitted scaling adjustments for non-autosomal bins
    sort_idx = np.argsort(train_gc)
    fitted_values_all = np.interp(gc_series.values, train_gc[sort_idx], loess_fit[sort_idx])
    
    # Avoid zero-division errors if the trend drops out completely in highly extreme GC fringes
    fitted_values_all = np.clip(fitted_values_all, a_min=1e-4, a_max=None)
    
    # Execute correction formula: depth_corrected = depth_normalized / LOESS(GC)
    corrected_depths = depth_series / fitted_values_all
    return corrected_depths

def main():

    # 1. PARAMETERS & INPUT PATHS

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--windows_bed", metavar="FILE", required=True,
                   help="Genomic windows bed file generated in step 1")
    p.add_argument("--reference", metavar="FILE", required=True,
                   help="Fasta reference")
    p.add_argument("--wgs_bams_dir", metavar="FILE", required=True,
                   help="Directory to normal WGS bams")
    p.add_argument("--matrix", metavar="FILE", required=True,
                   help="Output matrix TSV")
    p.add_argument("--temp_dir", metavar="FILE", required=True,
                   help="Temp directory")
    p.add_argument("-t", metavar="INT", type=int,
                   default=4, help="Threads per bam")
    
    
    args = p.parse_args()

    # Path to the actionable windows BED file generated in Step 1
    WINDOWS_BED = args.windows_bed

    # Path to reference fasta
    REF_FASTA = args.reference
    
    # Directory containing your control/normal WGS BAM files
    WGS_BAM_DIR = args.wgs_bams_dir
    BAM_PATTERN = os.path.join(WGS_BAM_DIR, "*.bam")
    
    # Processing settings
    THREADS_PER_BAM = args.t
    
    # Final matrix configuration output
    MATRIX_OUTPUT_TSV = args.matrix
    
    # Workspace directory for mosdepth intermediary outputs
    TMP_DIR = args.temp_dir
    os.makedirs(TMP_DIR, exist_ok=True)


    # 2. FILE VERIFICATION & GC RUN

    bam_files = glob.glob(BAM_PATTERN)
    if not bam_files:
        print(f"[-] Error: No BAM files found matching pattern: {BAM_PATTERN}")
        return
    if not os.path.exists(REF_FASTA):
        print(f"[-] Error: Reference FASTA missing at {REF_FASTA}. Required for GC calculations.")
        return

    print(f"[+] Found {len(bam_files)} WGS BAM files to process.")
    
    # Compute base pair GC metrics for our window universe
    gc_content_series = compute_gc_content(WINDOWS_BED, REF_FASTA)
    
    depth_matrix_dict = {}
    window_coordinates_df = None


    # 3. MOSDEPTH LOGIC (STEP 2 COUNTS)

    for bam_path in bam_files:
        sample_name = os.path.basename(bam_path).split('.')[0]
        print(f"\n[+] Processing Sample: {sample_name}")
        
        output_prefix = os.path.join(TMP_DIR, f"{sample_name}_mosdepth")
        run_mosdepth(bam_path, WINDOWS_BED, output_prefix, threads=THREADS_PER_BAM)
        sample_depths_df = parse_mosdepth_regions(output_prefix)
        
        if window_coordinates_df is None:
            window_coordinates_df = sample_depths_df[['chrom', 'start', 'end']].copy()
            window_coordinates_df['window_id'] = (
                window_coordinates_df['chrom'] + ":" + 
                window_coordinates_df['start'].astype(str) + "-" + 
                window_coordinates_df['end'].astype(str)
            )

        depth_matrix_dict[sample_name] = sample_depths_df['depth'].values

    # Construct dataframe matrix
    matrix_W = pd.DataFrame(depth_matrix_dict, index=window_coordinates_df['window_id'])


    # 4. NORMALIZATION & LOESS GC CORRECTION (STEPS 2 & 3)

    # Separate sex chromosomes from autosomes to prevent karyotypic biases
    autosome_indices = matrix_W.index.str.startswith(('chrX', 'chrY', 'chrM', 'X', 'Y', 'MT')) == False
    if not np.any(autosome_indices):
        autosome_indices = np.ones(len(matrix_W), dtype=bool)

    for sample in matrix_W.columns:
        raw_depths = matrix_W[sample].copy()
        
        # Part A: Median Autosomal Normalization (Step 2)
        median_autosomal_depth = np.median(raw_depths[autosome_indices])
        if median_autosomal_depth == 0:
            print(f"    [!] Warning: Sample {sample} has an autosomal median depth of 0. Skipping.")
            continue
            
        normalized_depths = raw_depths / median_autosomal_depth
        print(f"    -> Normalized {sample} against autosomal median: {median_autosomal_depth:.2f}")
        
        # Part B: Robust LOESS GC Adjustment (Step 3)
        print(f"    -> Fitting local LOESS regression models to smooth GC bias curve...")
        gc_corrected_depths = apply_loess_gc_correction(normalized_depths, gc_content_series, autosome_indices)
        
        # Re-center the corrected data so the global autosomal median returns exactly to 1.0
        final_recenter_factor = np.median(gc_corrected_depths[autosome_indices])
        matrix_W[sample] = gc_corrected_depths / final_recenter_factor


    # 5. MATRIX EXPORT

    final_output_df = window_coordinates_df[['chrom', 'start', 'end']].copy()
    final_output_df['gc_pct'] = gc_content_series.values
    final_output_df = final_output_df.join(matrix_W)
    
    final_output_df.to_csv(MATRIX_OUTPUT_TSV, sep='\t', index=True)
    
    print("\n" + "="*60)
    print(f"[+] STEP 2 & 3 SUCCESSFUL!")
    print(f"    -> GC-Corrected Matrix shape: {matrix_W.shape} (Windows x Samples)")
    print(f"    -> Fully normalized baseline matrix exported to: {MATRIX_OUTPUT_TSV}")
    print("="*60)

if __name__ == "__main__":
    main()