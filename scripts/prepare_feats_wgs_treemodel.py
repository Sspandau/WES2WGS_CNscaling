#!/usr/bin/env python3
"""
prepare_wgs_depth_features.py

Discovers samples, loads and filters their *_off_target_copy_ratios.tsv
files, pools them, and builds the feature table (X) and target (y) that
train_wgs_depth_model.py trains on. Split out from the training step so
that experimenting with model hyperparameters (objective, CV grouping,
model choice) doesn't require re-reading 34+ TSVs each time.

OUTPUT (written to --outdir)
-------------------------------
  training_data.parquet   one row per (sample, bin): feature columns +
                           meta_sample / meta_chrom (kept even if not used
                           as a model feature, so the training script can
                           group CV folds by either) + y_wgs_tumor_depth
  metadata.json            location_encoding, feature/categorical column
                            lists, filtering params, sample count -- read
                            by train_wgs_depth_model.py to reconstruct
                            dtypes and know what it's working with

GENOMIC LOCATION ENCODING (--location-encoding)
--------------------------------------------------
coords (default): chrom (categorical) + start (numeric) as features.
bin-id:           adds a chrom:start categorical bin identity (high
                   cardinality -- LightGBM only, see train script).
none:             no location feature at all (ablation baseline).

Regardless of choice, meta_sample and meta_chrom are always retained as
non-feature columns, so the training script can run GroupKFold grouped by
either sample or chromosome even if chrom isn't itself a model feature.
"""

import os
import sys
import glob
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Discovery / loading
# --------------------------------------------------------------------------

def discover_samples_from_root(wes_root):
    """
    One subdirectory per sample under wes_root, each containing a single
    *_off_target_copy_ratios.tsv (mirrors the layout used by
    find_recurrent_amplified_bins.py's --wes-root discovery).
    """
    entries = []
    for d in sorted(os.listdir(wes_root)):
        sample_dir = os.path.join(wes_root, d)
        if not os.path.isdir(sample_dir):
            continue
        matches = sorted(glob.glob(os.path.join(sample_dir, "*_off_target_copy_ratios.tsv")))
        if not matches:
            print(f"[skip] {d}: no copy ratio TSV found")
            continue
        if len(matches) > 1:
            print(f"[warn] {d}: {len(matches)} TSVs found, using {matches[0]}")
        entries.append({"sample": d, "copy_ratio_tsv": matches[0]})
    return pd.DataFrame(entries)


def load_manifest(path):
    sep = "\t" if str(path).lower().endswith((".tsv", ".txt")) else None
    df = pd.read_csv(path, sep=sep, engine="python")
    required = {"sample", "copy_ratio_tsv"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"Error: manifest is missing required columns: {missing}")
    return df


def load_window_covariates(path):
    """
    Loads the per-window covariates TSV produced by compute_window_covariates.py
    (gc_pct, mappability, dist_to_target, gap_size, gap_fraction, keyed by
    chrom/start/end/window_id). This is sample-independent -- purely a
    function of the window design -- so it's merged in here at prep time rather
    than requiring scale_WES_tracks.py to be rerun (which would mean reprocessing
    every WES/WGS BAM just to pick up a few extra per-window columns).

    log_gap_size is intentionally NOT loaded here -- tree models are invariant
    to monotonic transforms of a single feature, so log1p(gap_size) and raw
    gap_size would produce identical splits. Raw gap_size is loaded instead,
    used downstream to build gap_size_per_bin (see aggregate_to_window_size /
    load_sample_tsv).
    """
    df = pd.read_csv(path, sep='\t')
    required = {'chrom', 'start'}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"Error: --window-covariates-tsv is missing required columns: {missing}")

    covariate_cols = [c for c in
                       ['gc_pct', 'mappability', 'dist_to_target', 'gap_size', 'gap_fraction']
                       if c in df.columns]
    if not covariate_cols:
        sys.exit("Error: --window-covariates-tsv has none of the expected covariate "
                  "columns (gc_pct, mappability, dist_to_target, gap_size, gap_fraction).")

    df = df[['chrom', 'start'] + covariate_cols].copy()
    for col in covariate_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # If a window appears more than once (shouldn't happen, but be defensive
    # rather than silently duplicating rows on merge)
    dupes = df.duplicated(subset=['chrom', 'start']).sum()
    if dupes > 0:
        print(f"    [-] Warning: {dupes} duplicate (chrom, start) rows in window "
              f"covariates file; keeping the first occurrence of each.")
        df = df.drop_duplicates(subset=['chrom', 'start'], keep='first')

    print(f"    -> Loaded window covariates: {len(df):,} windows, "
          f"columns={covariate_cols}")
    return df, covariate_cols


def aggregate_to_window_size(df, window_size):
    """
    Mean-aggregates rows into window_size bp genomic blocks, matching the
    windowing behavior of plot_WES_scaling.py's --window-size. Runs on one
    sample's already-filtered frame (filtering happens at native resolution
    first, same order as the plotting script, so a masked/low-depth bin
    doesn't quietly drag down a window's mean before it would have been
    dropped anyway).
    """
    df = df.copy()
    df['window_idx'] = df['start'] // window_size

    agg_dict = {'start': 'min', 'end': 'max'}
    for col in ('raw_wes_depth', 'wgs_tumor_depth', 'gc_pct', 'dist_to_target',
                'gap_size', 'gap_fraction', 'mappability'):
        if col in df.columns:
            agg_dict[col] = 'mean'

    agg_df = df.groupby(['chrom', 'window_idx'], as_index=False).agg(agg_dict)
    return agg_df.drop(columns=['window_idx']).reset_index(drop=True)


def load_sample_tsv(sample_name, tsv_path, min_wes_depth, min_wgs_depth, mask_col,
                     window_covariates_df=None, covariate_cols=None, window_size=0,
                     apply_min_wes_depth_filter=True):
    df = pd.read_csv(tsv_path, sep='\t')

    required_cols = {'chrom', 'start', 'end', 'gc_pct', 'raw_wes_depth', 'wgs_tumor_depth'}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"    [skip] {sample_name}: missing required columns {missing}")
        return None

    if mask_col.lower() != 'none' and mask_col in df.columns:
        df = df[df[mask_col].astype(float).fillna(0) == 0].copy()

    dist_col = None
    for cand in ('dist_to_target', 'distance_to_target', 'dist'):
        if cand in df.columns:
            dist_col = cand
            break

    keep_cols = ['chrom', 'start', 'end', 'gc_pct', 'raw_wes_depth', 'wgs_tumor_depth']
    if dist_col is not None:
        keep_cols.append(dist_col)

    df = df[keep_cols].copy()
    if dist_col is not None and dist_col != 'dist_to_target':
        df = df.rename(columns={dist_col: 'dist_to_target'})

    if window_covariates_df is not None:
        # The covariates TSV is the canonical, sample-independent source for
        # these columns -- overwrite whatever this sample's own TSV had for
        # any overlapping column names (gc_pct, dist_to_target), and bring in
        # the new ones (gap_size, gap_fraction, mappability)
        # that scale_WES_tracks.py's output doesn't have yet.
        overlap = [c for c in covariate_cols if c in df.columns]
        if overlap:
            df = df.drop(columns=overlap)
        n_before_merge = len(df)
        df = df.merge(window_covariates_df, on=['chrom', 'start'], how='left')
        if len(df) != n_before_merge:
            print(f"    [-] Warning: row count changed after covariate merge for "
                  f"{sample_name} ({n_before_merge} -> {len(df)}); check for duplicate "
                  f"(chrom, start) pairs in the copy-ratio TSV.")
        n_unmatched = df[covariate_cols[0]].isna().sum() if covariate_cols else 0
        if n_unmatched > 0:
            print(f"    [-] Warning: {n_unmatched:,} bins in {sample_name} had no matching "
                  f"window in the covariates file (chrom/start mismatch) -- those covariate "
                  f"values will be NaN for those rows.")

    numeric_check_cols = ['gc_pct', 'raw_wes_depth', 'wgs_tumor_depth', 'dist_to_target']
    if covariate_cols:
        numeric_check_cols += [c for c in covariate_cols if c not in numeric_check_cols]
    for col in numeric_check_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if not apply_min_wes_depth_filter:
        # Recovery pipeline only: a bin with literally no WES coverage may be
        # encoded as a blank/NaN cell in the source TSV rather than a literal
        # 0.0. Treat that as 0 here so these bins survive into
        # full_wes_range_data.parquet instead of being silently dropped by
        # dropna below. The TRAINING pipeline (apply_min_wes_depth_filter=
        # True) is untouched -- a bin with unknown/missing WES coverage still
        # has no business being trained on.
        df['raw_wes_depth'] = df['raw_wes_depth'].fillna(0.0)

    df = df.dropna(subset=['chrom', 'start', 'gc_pct', 'raw_wes_depth', 'wgs_tumor_depth'])

    if apply_min_wes_depth_filter:
        df = df[df['raw_wes_depth'] > min_wes_depth]

    n_before_wgs_filter = len(df)
    df = df[df['wgs_tumor_depth'] > min_wgs_depth]
    n_dropped_zero_wgs = n_before_wgs_filter - len(df)
    if n_dropped_zero_wgs > 0:
        print(f"    -> {sample_name}: dropped {n_dropped_zero_wgs:,} bins with "
              f"wgs_tumor_depth <= {min_wgs_depth} (likely WGS coverage gaps/mappability "
              f"artifacts rather than real signal)")

    if df.empty:
        print(f"    [skip] {sample_name}: no usable rows after filtering "
              f"(check that wgs_tumor_depth is populated -- was matched WGS given "
              f"when this TSV was generated?)")
        return None

    if window_size and window_size > 0:
        n_before_agg = len(df)
        df = aggregate_to_window_size(df, window_size)
        print(f"    -> {sample_name}: aggregated {n_before_agg:,} native bins into "
              f"{len(df):,} {window_size // 1000}kb windows")

    # Normalize distance features by each row's own actual bin span
    # (end - start) rather than absolute bp. This is NOT a no-op rescale:
    # an aggregated block near a chromosome edge, or one where most of its
    # native bins got masked/filtered out, can end up with a smaller actual
    # span than the nominal --window-size -- so raw bp distances aren't
    # directly comparable row to row the way "distance in units of this
    # bin's own width" is.
    bin_span = (df['end'] - df['start']).replace(0, np.nan)
    if 'dist_to_target' in df.columns:
        df['dist_to_target_per_bin'] = df['dist_to_target'] / bin_span
    if 'gap_size' in df.columns:
        df['gap_size_per_bin'] = df['gap_size'] / bin_span

    df['sample'] = sample_name
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# Feature construction
# --------------------------------------------------------------------------

def build_feature_table(df, location_encoding, has_dist, extra_numeric_cols=None):
    """
    Returns (X, categorical_cols, numeric_cols). chrom (and bin_id, if
    requested) are left as pandas 'category' dtype -- LightGBM's sklearn
    API auto-detects and splits on category-dtype columns natively.

    extra_numeric_cols: additional already-validated numeric columns to
    include as-is (e.g. gap_fraction, gap_size_per_bin, mappability from a
    merged window-covariates file).
    """
    df = df.copy()
    numeric_cols = ['raw_wes_depth', 'gc_pct']
    if has_dist:
        numeric_cols.append('dist_to_target_per_bin')
    if extra_numeric_cols:
        numeric_cols += [c for c in extra_numeric_cols if c not in numeric_cols]
    categorical_cols = []

    if location_encoding in ('coords', 'bin-id'):
        df['chrom'] = df['chrom'].astype('category')
        categorical_cols.append('chrom')
        numeric_cols.append('start')

    if location_encoding == 'bin-id':
        df['bin_id'] = (df['chrom'].astype(str) + ':' + df['start'].astype(str)).astype('category')
        categorical_cols.append('bin_id')

    feature_cols = numeric_cols + categorical_cols
    X = df[feature_cols].copy()
    return X, categorical_cols, numeric_cols


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--wes-root', default=None,
                   help='Root directory with one subdirectory per sample, each containing '
                        'a *_off_target_copy_ratios.tsv (auto-discovery mode)')
    p.add_argument('--manifest', default=None,
                   help='TSV/CSV with columns: sample, copy_ratio_tsv. Overrides --wes-root.')
    p.add_argument('--location-encoding', choices=['coords', 'bin-id', 'none'], default='coords',
                   help="How genomic location is encoded as a feature. 'coords' = chrom "
                        "(categorical) + start (numeric); 'bin-id' = full per-bin categorical "
                        "identity (lgbm only at training time); 'none' = no location feature. "
                        "meta_sample/meta_chrom are always retained regardless, for CV grouping. "
                        "Default: coords")
    p.add_argument('--min-wes-depth', type=float, default=1.0,
                   help='Minimum raw_wes_depth required to retain a bin. Default: 1.0')
    p.add_argument('--min-wgs-depth', type=float, default=0.0,
                   help='Minimum wgs_tumor_depth required to retain a bin (excludes rows with '
                        'the value <= this threshold, e.g. wgs_tumor_depth == 0). A depth of '
                        'exactly 0 at an off-target bin is more often a WGS coverage gap or '
                        'mappability artifact than a real deletion signal, and can inject noise '
                        'into training if left in. Default: 0.0 (drops exact zeros only; raise '
                        'this if you suspect low-coverage WGS bins below a few reads are also '
                        'unreliable)')
    p.add_argument('--mask-col', default='mask_rejected',
                   help="Column flagging rows to exclude (nonzero = excluded). "
                        "Set to 'none' to disable. Default: mask_rejected")
    p.add_argument('--window-size', type=int, default=0,
                   help='If > 0, mean-aggregates bins into this many bp before pooling '
                        '(e.g. 25000 for 25kb windows), matching plot_WES_scaling.py\'s '
                        '--window-size. Filtering (--min-wes-depth/--min-wgs-depth/--mask-col) '
                        'is applied at native resolution FIRST, then remaining bins are '
                        'aggregated -- so a masked/low-depth bin never drags down a window\'s '
                        'mean. Default: 0 (no aggregation -- trains at whatever resolution the '
                        'input TSVs are already at)')
    p.add_argument('--window-covariates-tsv', default=None,
                   help='Optional output TSV from compute_window_covariates.py (gc_pct, '
                        'mappability, dist_to_target, gap_size, gap_fraction, keyed by '
                        'chrom/start). Merged in here directly, on (chrom, start) -- '
                        'this is sample-independent, so it does NOT require rerunning '
                        'scale_WES_tracks.py. Values here take priority over any matching '
                        'column already present in a sample TSV (e.g. gc_pct, dist_to_target). '
                        'dist_to_target and gap_size are further normalized by each bin\'s own '
                        'actual span into dist_to_target_per_bin / gap_size_per_bin before '
                        'becoming model features.')
    p.add_argument('--outdir', required=True, help='Output directory')
    p.add_argument('--save-full-wes-range', action='store_true',
                   help="Also build a second parquet (full_wes_range_data.parquet) containing "
                        "every bin that would normally be dropped ONLY by --min-wes-depth -- "
                        "same mask/--min-wgs-depth/dropna/--window-size pipeline, just without "
                        "that one filter. Adds a 'below_min_wes_depth' boolean column. Lets "
                        "train_wgs_depth_model.py score bins the trained model never saw during "
                        "training, for coverage recovery (e.g. so a downstream recurrence "
                        "analysis isn't missing bins purely because they had thin WES coverage). "
                        "Doubles the per-sample TSV read/filter work, so it's opt-in. See that "
                        "script's --score-full-wes-range flag for the scoring side, and its "
                        "module docstring for the extrapolation caveat.")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not args.manifest and not args.wes_root:
        sys.exit("Error: provide either --manifest or --wes-root.")

    print("[+] Discovering samples...")
    sample_table = load_manifest(args.manifest) if args.manifest else discover_samples_from_root(args.wes_root)
    if sample_table.empty:
        sys.exit("Error: no samples found.")
    print(f"    -> Found {len(sample_table)} candidate samples.")

    window_covariates_df, covariate_cols = None, []
    if args.window_covariates_tsv:
        print(f"[+] Loading window covariates: {args.window_covariates_tsv}")
        window_covariates_df, covariate_cols = load_window_covariates(args.window_covariates_tsv)

    print("[+] Loading and filtering per-sample TSVs...")
    frames = []
    frames_full_wes = []
    for row in sample_table.itertuples(index=False):
        df = load_sample_tsv(row.sample, row.copy_ratio_tsv, args.min_wes_depth,
                              args.min_wgs_depth, args.mask_col,
                              window_covariates_df=window_covariates_df,
                              covariate_cols=covariate_cols,
                              window_size=args.window_size)
        if df is not None:
            frames.append(df)
            print(f"    -> {row.sample}: {len(df):,} usable bins")

        if args.save_full_wes_range:
            # Same pipeline, minus the --min-wes-depth filter -- kept as a
            # fully separate call (not derived by post-hoc filtering the
            # frame above) because --window-size aggregation happens BEFORE
            # this point when window_size > 0: a window that mixes some
            # passing and some low-WES-depth native bins would get a
            # DIFFERENT mean depending on whether the low-depth bins were
            # excluded pre-aggregation (as they are for the standard
            # training frame) or included (as intended here). Re-deriving
            # this from `df` instead of re-reading the TSV would silently
            # inherit the standard frame's aggregation and miss exactly the
            # recovered bins this flag exists to include.
            df_full = load_sample_tsv(row.sample, row.copy_ratio_tsv, args.min_wes_depth,
                                       args.min_wgs_depth, args.mask_col,
                                       window_covariates_df=window_covariates_df,
                                       covariate_cols=covariate_cols,
                                       window_size=args.window_size,
                                       apply_min_wes_depth_filter=False)
            if df_full is not None:
                # For window_size > 0 this is a per-window mean, so it flags
                # "this window's average native raw_wes_depth was at/under
                # threshold" rather than a single native bin's value -- the
                # closest available meaning once bins are pre-aggregated.
                df_full['below_min_wes_depth'] = df_full['raw_wes_depth'] <= args.min_wes_depth
                frames_full_wes.append(df_full)

    if not frames:
        sys.exit("Error: no samples produced usable rows. Check that wgs_tumor_depth is "
                  "populated in these TSVs.")

    pooled = pd.concat(frames, ignore_index=True)
    n_samples_used = pooled['sample'].nunique()
    print(f"[+] Pooled table: {len(pooled):,} rows across {n_samples_used} samples.")

    has_dist = bool('dist_to_target_per_bin' in pooled.columns
                     and pooled['dist_to_target_per_bin'].notna().any())
    if not has_dist:
        print("    -> No usable dist_to_target column found across samples; excluding it.")

    extra_numeric_cols = []
    for col in ('gap_fraction', 'gap_size_per_bin'):
        if col in pooled.columns and pooled[col].notna().any():
            extra_numeric_cols.append(col)
    if covariate_cols and not extra_numeric_cols:
        print("    -> Window covariates file was provided but produced no usable "
              "gap_fraction/gap_size_per_bin values (all NaN after merge/filtering).")
    elif extra_numeric_cols:
        print(f"    -> Including extra covariate features: {extra_numeric_cols}")

    X, categorical_cols, numeric_cols = build_feature_table(
        pooled, args.location_encoding, has_dist, extra_numeric_cols=extra_numeric_cols)

    out_df = X.copy()
    out_df['meta_sample'] = pooled['sample'].astype('category').values
    out_df['meta_chrom'] = pooled['chrom'].astype('category').values
    out_df['meta_start'] = pooled['start'].values
    out_df['y_wgs_tumor_depth'] = pooled['wgs_tumor_depth'].values.astype(float)

    parquet_path = outdir / 'training_data.parquet'
    out_df.to_parquet(parquet_path, index=False)

    full_wes_range_parquet = None
    n_recovered_bins = 0
    if args.save_full_wes_range:
        if not frames_full_wes:
            print("    -> --save-full-wes-range was set but produced no rows -- skipping "
                  "full_wes_range_data.parquet")
        else:
            pooled_full = pd.concat(frames_full_wes, ignore_index=True)
            X_full, _, _ = build_feature_table(
                pooled_full, args.location_encoding, has_dist, extra_numeric_cols=extra_numeric_cols)
            out_df_full = X_full.copy()
            out_df_full['meta_sample'] = pooled_full['sample'].astype('category').values
            out_df_full['meta_chrom'] = pooled_full['chrom'].astype('category').values
            out_df_full['meta_start'] = pooled_full['start'].values
            out_df_full['y_wgs_tumor_depth'] = pooled_full['wgs_tumor_depth'].values.astype(float)
            out_df_full['below_min_wes_depth'] = pooled_full['below_min_wes_depth'].values

            n_recovered_bins = int(out_df_full['below_min_wes_depth'].sum())
            full_wes_range_parquet = outdir / 'full_wes_range_data.parquet'
            out_df_full.to_parquet(full_wes_range_parquet, index=False)
            print(f"[+] Saved: {full_wes_range_parquet}  ({len(out_df_full):,} rows, of which "
                  f"{n_recovered_bins:,} are below --min-wes-depth={args.min_wes_depth} and "
                  f"weren't in training_data.parquet)")

    metadata = {
        'location_encoding': args.location_encoding,
        'feature_cols': list(X.columns),
        'categorical_cols': categorical_cols,
        'numeric_cols': numeric_cols,
        'has_dist': has_dist,
        'n_samples': int(n_samples_used),
        'n_rows': int(len(out_df)),
        'min_wes_depth': args.min_wes_depth,
        'min_wgs_depth': args.min_wgs_depth,
        'mask_col': args.mask_col,
        'window_size': args.window_size,
        'window_covariates_tsv': args.window_covariates_tsv,
        'extra_numeric_cols': extra_numeric_cols,
        'samples': sorted(pooled['sample'].unique().tolist()),
        'has_full_wes_range_data': full_wes_range_parquet is not None,
        'full_wes_range_parquet': full_wes_range_parquet.name if full_wes_range_parquet else None,
        'n_recovered_bins': n_recovered_bins,
    }
    metadata_path = outdir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"[+] Saved: {parquet_path}  ({len(out_df):,} rows, {len(X.columns)} feature cols)")
    print(f"[+] Saved: {metadata_path}")
    print("\n[+] prepare_wgs_depth_features.py COMPLETE")


if __name__ == '__main__':
    main()
