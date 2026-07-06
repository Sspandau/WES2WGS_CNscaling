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

def build_training_table(manifest_df, windows_bed, covariates_df, temp_dir, threads):
    """
    For every matched pair, runs mosdepth on both BAMs over the shared
    off-target windows, median-normalizes each sample independently, and
    returns one long-format dataframe: sample_id, window_id, log2_ratio,
    plus the merged covariates.
    """
    os.makedirs(temp_dir, exist_ok=True)
    long_frames = []

    for _, row in manifest_df.iterrows():
        sample_id = row['sample_id']
        print(f"\n[+] Building training rows for sample: {sample_id}")

        wes_prefix = os.path.join(temp_dir, f"{sample_id}_wes")
        wgs_prefix = os.path.join(temp_dir, f"{sample_id}_wgs")

        run_mosdepth(row['wes_bam'], windows_bed, wes_prefix, threads=threads)
        run_mosdepth(row['wgs_bam'], windows_bed, wgs_prefix, threads=threads)

        wes_df = parse_mosdepth_regions(wes_prefix).rename(columns={'depth': 'wes_depth'})
        wgs_df = parse_mosdepth_regions(wgs_prefix).rename(columns={'depth': 'wgs_depth'})

        merged = wes_df.merge(wgs_df, on='window_id')

        # Per-sample median normalization: puts both tracks on a comparable
        # scale before computing the ratio, so the GAM learns bias SHAPE,
        # not each sample's absolute depth level.
        wes_median = merged.loc[merged['wes_depth'] > 0, 'wes_depth'].median()
        wgs_median = merged.loc[merged['wgs_depth'] > 0, 'wgs_depth'].median()
        if wes_median == 0 or wgs_median == 0 or pd.isna(wes_median) or pd.isna(wgs_median):
            print(f"    [!] Warning: {sample_id} has zero/NaN median depth on one track. Skipping.")
            continue

        merged['wes_norm'] = merged['wes_depth'] / wes_median
        merged['wgs_norm'] = merged['wgs_depth'] / wgs_median

        valid = (merged['wes_depth'] > 0) & (merged['wgs_depth'] > 0)
        merged = merged[valid].copy()
        merged['log2_ratio'] = np.log2(merged['wes_norm'] / merged['wgs_norm'])
        merged['sample_id'] = sample_id

        long_frames.append(merged[['sample_id', 'window_id', 'log2_ratio']])
        print(f"    -> {len(merged)} usable windows (wes_median={wes_median:.2f}, "
              f"wgs_median={wgs_median:.2f})")

    training_df = pd.concat(long_frames, ignore_index=True)
    training_df = training_df.merge(covariates_df, on='window_id', how='inner')

    n_dropped = len(training_df) - training_df.dropna(
        subset=['gc_pct', 'mappability', 'dist_to_target', 'log2_ratio']).shape[0]
    if n_dropped > 0:
        print(f"[!] Dropping {n_dropped} rows with missing covariates/response.")
    training_df = training_df.dropna(
        subset=['gc_pct', 'mappability', 'dist_to_target', 'log2_ratio'])

    # Clip extreme log2 ratios (likely mapping artifacts / focal CN in the
    # tumor itself) so they don't dominate the smooth-term fit.
    lo, hi = training_df['log2_ratio'].quantile([0.001, 0.999])
    training_df = training_df[(training_df['log2_ratio'] >= lo) & (training_df['log2_ratio'] <= hi)]

    return training_df

def fit_gam(df):
    X = df[['gc_pct', 'mappability', 'dist_to_target']].values
    y = df['log2_ratio'].values
    gam = LinearGAM(
        s(0, n_splines=20, lam=0.6) +   # gc_pct
        s(1, n_splines=15, lam=0.6) +   # mappability
        s(2, n_splines=15, lam=0.6)     # dist_to_target
    )
    gam.fit(X, y)
    return gam

def evaluate(gam, df):
    X = df[['gc_pct', 'mappability', 'dist_to_target']].values
    y = df['log2_ratio'].values
    pred = gam.predict(X)
    resid = y - pred
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    mad = np.median(np.abs(resid))
    return r2, mad

def run_grouped_cv(training_df, n_folds, max_rows_per_fold=200_000):
    print(f"\n[+] Running {n_folds}-fold grouped CV (grouped by sample_id)...")
    groups = training_df['sample_id'].values
    n_samples = training_df['sample_id'].nunique()
    if n_folds > n_samples:
        n_folds = n_samples
        print(f"    [!] Reducing folds to {n_folds} (cannot exceed sample count).")

    gkf = GroupKFold(n_splits=n_folds)
    fold_metrics = []

    for i, (train_idx, test_idx) in enumerate(gkf.split(training_df, groups=groups), start=1):
        train_fold = training_df.iloc[train_idx]
        test_fold = training_df.iloc[test_idx]

        if len(train_fold) > max_rows_per_fold:
            train_fold = train_fold.sample(max_rows_per_fold, random_state=42)

        held_out_samples = sorted(test_fold['sample_id'].unique())
        gam_fold = fit_gam(train_fold)
        r2, mad = evaluate(gam_fold, test_fold)
        fold_metrics.append({'fold': i, 'held_out_samples': held_out_samples, 'r2': r2, 'mad': mad})
        print(f"    Fold {i}: held out {held_out_samples} -> "
              f"held-out R^2={r2:.3f}, held-out MAD(log2 resid)={mad:.4f}")

    metrics_df = pd.DataFrame(fold_metrics)
    print(f"\n    -> Mean held-out R^2:  {metrics_df['r2'].mean():.3f}")
    print(f"    -> Mean held-out MAD:  {metrics_df['mad'].mean():.4f}")
    return metrics_df

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", metavar="FILE", required=True,
                   help="TSV: sample_id, wes_bam, wgs_bam columns")
    p.add_argument("--windows_bed", metavar="FILE", required=True,
                   help="Off-target windows BED from Step 1")
    p.add_argument("--covariates", metavar="FILE", required=True,
                   help="Covariates TSV from compute_window_covariates.py")
    p.add_argument("--temp_dir", metavar="FILE", required=True,
                   help="Temp directory for mosdepth intermediates")
    p.add_argument("--model_output", metavar="FILE", required=True,
                   help="Output path for the pickled pooled GAM model")
    p.add_argument("--folds", metavar="INT", type=int, default=5,
                   help="Grouped CV folds (default 5; set equal to sample count for full leave-one-sample-out)")
    p.add_argument("-t", metavar="INT", type=int, default=4,
                   help="Threads per mosdepth run")
    p.add_argument("--training_table_cache", metavar="FILE", default=None,
                   help="Optional: save/reuse the assembled long-format training table (skips re-running mosdepth on reruns)")
    args = p.parse_args()

    if not os.path.exists(args.manifest):
        print(f"[-] Error: manifest not found: {args.manifest}")
        sys.exit(1)

    manifest_df = pd.read_csv(args.manifest, sep='\t')
    required_cols = {'sample_id', 'wes_bam', 'wgs_bam'}
    if not required_cols.issubset(manifest_df.columns):
        print(f"[-] Error: manifest must have columns {required_cols}")
        sys.exit(1)

    print(f"[+] Loaded manifest: {len(manifest_df)} matched WGS/WES pairs.")

    covariates_df = pd.read_csv(args.covariates, sep='\t')

    if args.training_table_cache and os.path.exists(args.training_table_cache):
        print(f"[+] Loading cached training table: {args.training_table_cache}")
        training_df = pd.read_csv(args.training_table_cache, sep='\t')
    else:
        training_df = build_training_table(
            manifest_df, args.windows_bed, covariates_df, args.temp_dir, args.t)
        if args.training_table_cache:
            training_df.to_csv(args.training_table_cache, sep='\t', index=False)
            print(f"[+] Cached training table to: {args.training_table_cache}")

    print(f"\n[+] Final pooled training table: {len(training_df)} rows across "
          f"{training_df['sample_id'].nunique()} samples.")

    # Cross-validate BEFORE fitting the final model on everything, so the
    # reported metrics reflect genuine held-out generalization.
    cv_metrics_df = run_grouped_cv(training_df, args.folds)
    cv_metrics_path = args.model_output + ".cv_metrics.tsv"
    cv_metrics_df.to_csv(cv_metrics_path, sep='\t', index=False)

    print(f"\n[+] Fitting final pooled GAM on all {training_df['sample_id'].nunique()} samples...")
    final_gam = fit_gam(training_df)
    r2_full, mad_full = evaluate(final_gam, training_df)
    print(f"    -> In-sample R^2={r2_full:.3f}, in-sample MAD={mad_full:.4f} "
          f"(compare against the held-out CV numbers above -- a big gap means overfitting)")

    with open(args.model_output, 'wb') as f:
        pickle.dump({
            'model': final_gam,
            'feature_order': ['gc_pct', 'mappability', 'dist_to_target'],
            'n_training_samples': training_df['sample_id'].nunique(),
            'n_training_windows': len(training_df),
        }, f)

    print("=" * 60)
    print(f"[+] POOLED BIAS MODEL TRAINING COMPLETE!")
    print(f"    -> Model saved to:      {args.model_output}")
    print(f"    -> CV metrics saved to: {cv_metrics_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()