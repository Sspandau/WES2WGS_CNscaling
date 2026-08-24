import os
import argparse
import glob
import pickle
import subprocess
import pandas as pd
import numpy as np

'''
Drop-in replacement for the correction step in scale_WES_tracks.py.
Everything else in your pipeline (Step 1 windows, Step 4 SVD PoN) stays the
same -- only the GC-only, per-sample LOESS fit is replaced with the pooled,
multi-covariate GAM trained in train_pooled_bias_model.py.

Where the old script computed:
    raw_ratio   = raw_wes_depth / pon_median
    fitted_all  = LOESS(raw_ratio ~ gc_pct)      # refit every sample
    depth_scaled = raw_ratio / fitted_all

this script computes:
    raw_ratio    = raw_wes_depth / pon_median
    predicted_bias = 2 ** GAM.predict([gc_pct, mappability, dist_to_target])
                                                  # pre-trained, pooled across cohort
    depth_scaled = raw_ratio / predicted_bias

The GAM predicts log2(bias) directly (that's how it was trained), so we
exponentiate before dividing.

python3 apply_pooled_bias_model.py \
  --offtarget_windows v5_offtargets.bed \
  --covariates v5_offtargets.covariates.tsv \
  --pon_tsv PoN_1000genomes_wgs_normals.tsv \
  --bias_model pooled_bias_model.pkl \
  --wes_dir /pedigree2/cui/CCLE_WXS/SW579_THYROID/ \
  --output_dir ../CCLE_WXS/WES2WGS_CCLE/ --temp_dir ./
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
                     header=None, names=['chrom', 'start', 'end', 'name', 'depth'])
    df['chrom'] = df['chrom'].astype(str)
    return df

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--offtarget_windows", metavar="FILE", required=True)
    p.add_argument("--covariates", metavar="FILE", required=True,
                   help="Covariates TSV from compute_window_covariates.py (same windows as PoN)")
    p.add_argument("--pon_tsv", metavar="FILE", required=True,
                   help="PoN TSV from WGS2PoN.py (unchanged)")
    p.add_argument("--bias_model", metavar="FILE", required=True,
                   help="Pickled pooled GAM from train_pooled_bias_model.py")
    p.add_argument("--wes_dir", required=True)
    p.add_argument("-t", metavar="INT", type=int, default=4)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--temp_dir", required=True)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)

    tumor_bams = glob.glob(os.path.join(args.wes_dir, "*.bam"))
    if not tumor_bams:
        print(f"[-] Error: No tumor BAM files found in: {args.wes_dir}")
        return

    print(f"[+] Loading WGS Panel of Normals: {args.pon_tsv}")
    df_pon = pd.read_csv(args.pon_tsv, sep='\t', index_col=0)

    print(f"[+] Loading window covariates: {args.covariates}")
    df_cov = pd.read_csv(args.covariates, sep='\t').set_index('window_id')
    df_cov = df_cov.reindex(df_pon.index)
    missing_cov = df_cov[['gc_pct', 'mappability', 'dist_to_target']].isna().any(axis=1).sum()
    if missing_cov > 0:
        print(f"    [!] Warning: {missing_cov} PoN windows missing covariates "
              f"(check windows_bed consistency); filling with column medians.")
        df_cov = df_cov.fillna(df_cov.median(numeric_only=True))

    print(f"[+] Loading pooled bias model: {args.bias_model}")
    with open(args.bias_model, 'rb') as f:
        bundle = pickle.load(f)
    gam = bundle['model']
    feature_order = bundle['feature_order']
    print(f"    -> Model trained on {bundle['n_training_samples']} samples, "
          f"{bundle['n_training_windows']} windows.")

    X_cov = df_cov[feature_order].values
    predicted_log2_bias = gam.predict(X_cov)
    predicted_bias = np.clip(2 ** predicted_log2_bias, 1e-3, None)

    pon_median = df_pon['pon_median'].values
    pon_variance = df_pon['pon_variance'].values

    print(f"[+] Found {len(tumor_bams)} tumor WES BAMs to process.")

    for bam_path in tumor_bams:
        sample_name = os.path.basename(bam_path).split('.')[0]
        print(f"\n" + "=" * 55)
        print(f"[+] Processing tumor: {sample_name}")
        print("=" * 55)

        prefix_off = os.path.join(args.temp_dir, f"{sample_name}_off_target")
        print(f"    -> Running mosdepth on off-target windows...")
        run_mosdepth(bam_path, args.offtarget_windows, prefix_off, threads=args.t)
        df_off = parse_mosdepth_regions(prefix_off)

        if len(df_off) != len(df_pon):
            print(f"    [-] Error: Window count mismatch — "
                  f"mosdepth={len(df_off)}, PoN={len(df_pon)}. Skipping sample.")
            continue

        raw_wes_depth = df_off['depth'].values.astype(float)

        # PoN-anchored raw ratio (same as before)
        safe_pon_median = np.where(pon_median <= 0, 1e-4, pon_median)
        raw_ratio = raw_wes_depth / safe_pon_median

        # Bias correction now comes from the pooled GAM instead of per-sample LOESS
        depth_scaled = raw_ratio / predicted_bias

        # Quality filters (same logic as scale_WES_tracks.py)
        pon_sd = np.sqrt(pon_variance)
        pon_cv = np.where(pon_median > 0, pon_sd / pon_median, 0)
        flag_high_cv = (pon_cv > 0.30).astype(int)
        flag_zero_wes = (raw_wes_depth == 0).astype(int)
        flag_unstable_pon = (pon_median < 0.01).astype(int)
        flag_numerical_instability = (depth_scaled > 1000).astype(int)
        flag_extreme_scaling = (flag_unstable_pon | flag_numerical_instability).astype(int)
        flag_pon_variance = df_pon['is_high_variance'].values

        master_mask = (flag_high_cv | flag_zero_wes | flag_extreme_scaling | flag_pon_variance).astype(int)

        df_output = df_pon[['chrom', 'start', 'end', 'gc_pct']].copy()
        df_output['mappability'] = df_cov['mappability'].values
        df_output['dist_to_target'] = df_cov['dist_to_target'].values
        df_output['raw_wes_depth'] = raw_wes_depth
        df_output['pon_median_wgs'] = pon_median
        df_output['predicted_bias'] = predicted_bias
        df_output['depth_scaled_ratio'] = depth_scaled
        df_output['pon_cv'] = pon_cv

    print("\n" + "=" * 55)
    print("[+] apply_pooled_bias_model.py COMPLETE")
    print("    All tumor samples normalized using the pooled cohort bias model.")
    print("=" * 55)

if __name__ == "__main__":
    main()
