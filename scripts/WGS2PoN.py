import os
import argparse
import sys
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

'''
Script that computes PoN using SVD and WGS matrix from prior step.
Safely integrates genomic covariates (mappability, target distance) if provided.
'''

def main():
    # STREAMING_CHUNK: Parsing command line arguments...
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--wgs_matrix", metavar="FILE", required=True,
                   help="GC corrected WGS counts per genomic window matrix")
    p.add_argument("-k", metavar="INT", type=int,
                   default=5, help="K determines top components to strip in SVD")
    p.add_argument("--output", metavar="FILE", required=True,
                   help="PoN from WGS samples using SVD (GATK style)")
    p.add_argument("--covariates", metavar="FILE", required=False, default=None,
                   help="Covariates TSV from compute_window_covariates.py (same windows as PoN)")
    
    args = p.parse_args()

    INPUT_MATRIX_TSV = args.wgs_matrix
    SCALES_FILE = INPUT_MATRIX_TSV + ".scales" # Auto-detect companion file
    N_COMPONENTS_TO_REMOVE = args.k
    PON_OUTPUT_TSV = args.output
    COVARIATES_FILE = args.covariates

    if not os.path.exists(INPUT_MATRIX_TSV):
        print(f"[-] Error: Input matrix file '{INPUT_MATRIX_TSV}' not found.", file=sys.stderr)
        sys.exit(1)

    # STREAMING_CHUNK: Loading and preparing data matrix...
    print(f"[+] Loading GC-corrected WGS matrix: {INPUT_MATRIX_TSV}")
    df_matrix = pd.read_csv(INPUT_MATRIX_TSV, sep='\t', index_col=0)
    
    meta_cols = ['chrom', 'start', 'end', 'gc_pct']
    sample_cols = [col for col in df_matrix.columns if col not in meta_cols]
    
    W = df_matrix[sample_cols].values  # Shape: (n_windows, n_samples)
    n_windows, n_samples = W.shape
    
    print(f"    -> Matrix dimensions: {n_windows} windows x {n_samples} samples.")
    
    if n_samples < 2:
        print("[-] Error: SVD denoising requires at least 2 or more sample columns to model covariance.", file=sys.stderr)
        sys.exit(1)
        
    if N_COMPONENTS_TO_REMOVE >= n_samples:
        N_COMPONENTS_TO_REMOVE = max(1, n_samples - 1)
        print(f"[!] Warning: K components cannot exceed sample size. Auto-adjusting K to: {N_COMPONENTS_TO_REMOVE}")

    # STREAMING_CHUNK: Centering matrix and running SVD...
    print("[+] Centering data: Computing and subtracting per-window mean values...")
    window_means = np.mean(W, axis=1, keepdims=True)
    W_centered = W - window_means

    print(f"[+] Computing Singular Value Decomposition (SVD) to isolate top {N_COMPONENTS_TO_REMOVE} batch components...")
    svd = TruncatedSVD(n_components=N_COMPONENTS_TO_REMOVE, random_state=42)
    U_sigma = svd.fit_transform(W_centered) # Shape: (n_windows, K)
    V_T = svd.components_                   # Shape: (K, n_samples)
    
    explained_variance_pct = np.sum(svd.explained_variance_ratio_) * 100
    print(f"    -> Top {N_COMPONENTS_TO_REMOVE} components explain {explained_variance_pct:.2f}% of systematic variance.")

    # STREAMING_CHUNK: Removing systematic biases and scaling raw depth medians...
    W_bias = np.dot(U_sigma, V_T)
    W_denoised_centered = W_centered - W_bias
    W_denoised = W_denoised_centered + window_means
    
    if os.path.exists(SCALES_FILE):
        print(f"[+] Found raw depth scales profile: {SCALES_FILE}")
        scales_df = pd.read_csv(SCALES_FILE, sep='\t', header=None, index_col=0)
        raw_medians = [scales_df.loc[col].values[0] for col in sample_cols]
        raw_medians = np.array(raw_medians)
        W_denoised_raw = W_denoised * raw_medians
    else:
        print(f"[!] Warning: Scales file missing at {SCALES_FILE}. Defaulting to absolute 30x fallback.")
        W_denoised_raw = W_denoised * 30.0

    # STREAMING_CHUNK: Computing descriptive PoN metrics...
    print("[+] Generating Panel of Normals (PoN) absolute tracking statistics...")
    pon_median = np.median(W_denoised_raw, axis=1)
    pon_variance = np.var(W_denoised_raw, axis=1)

    print("[+] Merging statistics into final track table layout...")
    pon_df = df_matrix[meta_cols].copy()
    pon_df['pon_median'] = pon_median
    pon_df['pon_variance'] = pon_variance
    
    # STREAMING_CHUNK: Integrating optional external covariates safely using robust indexing...
    if COVARIATES_FILE and os.path.exists(COVARIATES_FILE):
        print(f"    [+] Found covariates template. Mapping columns securely...")
        # Read without preset index to avoid setting "chrom" (col 0) as index
        cov_df = pd.read_csv(COVARIATES_FILE, sep='\t')
        
        # Ensure we set the index to the unique 'window_id' values
        if 'window_id' in cov_df.columns:
            cov_df = cov_df.set_index('window_id')
        else:
            print("    [!] Warning: 'window_id' column not found in covariates file. Using first column.", file=sys.stderr)
            cov_df = cov_df.set_index(cov_df.columns[0])
        
        map_dict = cov_df['mappability'].to_dict()
        dist_dict = cov_df['dist_to_target'].to_dict()
        
        pon_df['mappability'] = pon_df.index.map(map_dict)
        pon_df['dist_to_target'] = pon_df.index.map(dist_dict)
    else:
        print("    [!] Warning: No covariates template file provided or file path is invalid.")
        print("        Mappability and dist_to_target will be filled with NaNs.")
        pon_df['mappability'] = np.nan
        pon_df['dist_to_target'] = np.nan

    variance_threshold = np.percentile(pon_variance, 99)
    pon_df['is_high_variance'] = (pon_variance > variance_threshold).astype(int)
    
    print(f"    -> Flagged {pon_df['is_high_variance'].sum()} windows exceeding 99th percentile variance threshold ({variance_threshold:.4e})")

    # Export
    pon_df.to_csv(PON_OUTPUT_TSV, sep='\t', index=True)
    
    print("\n" + "="*60)
    print(f"[+] STEP 4 COMPLETE!")
    print(f"    -> SVD Denoised PoN saved to: {PON_OUTPUT_TSV}")
    print("="*60)

if __name__ == "__main__":
    main()
