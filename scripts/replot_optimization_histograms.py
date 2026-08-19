#!/usr/bin/env python3
"""
replot_histogram_no_recurrent.py

Regenerates best_combo_histogram.png (and, if --highlight-sample is given,
best_combo_histogram_<sample>.png) for ONE already-known (quantile,
min_samples) combo, WITHOUT the purple "recurrent amplified region bins"
layer -- just blue background vs orange ecDNA/BFB/CNC.

Why a separate script instead of a flag on the main one: the merged
per-bin data that feeds these histograms is never written to disk by
optimize_recurrent_amplification_threshold_wgstreemodel.py (only the
aggregate counts/medians end up in the PNG itself), so there's no cached
data to replot from. This script rebuilds the same per-sample cache that
script builds (reusing its own loading/pooling code directly, so it can
never drift out of sync) and re-derives the merged dataframe for a SINGLE
combo -- skipping the full quantile x min_samples_ratio grid search and
all its KDE fitting, which is the expensive part. Cache-building itself
(reading every sample's depth data) is unavoidable and is the main cost
here, same as it was in the original run.

plot_best_histogram() already omits the purple layer entirely when handed
an empty recurrent_values array, so no plotting code needed to change --
this script just doesn't compute/pass that argument.

USAGE
-----
Pass the SAME --wes-root/--aa-root/--cv-csv/--classification-*/--sample-map/
etc. flags you used for the original optimize_recurrent_amplification_
threshold_wgstreemodel.py run, plus the winning combo's exact --quantile
and --min-samples (both printed in that run's console output / best_combo_
summary.txt -- --min-samples is the absolute count, e.g. 6, not a ratio).
"""

import sys
import argparse
from pathlib import Path

import numpy as np


def load_module(path, name):
    import importlib.util
    path = Path(path)
    if not path.exists():
        sys.exit(f"Error: {name} not found at {path}.")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--optimize-script", required=True,
                   help="Path to optimize_recurrent_amplification_threshold_wgstreemodel.py")
    p.add_argument("--base-script", required=True,
                   help="Path to find_recurrent_novel_amplifications_binlevel.py")

    # Same data-source / classification args as the main script -- pass
    # whichever ones match how you originally ran it.
    p.add_argument("--wes-root", default=None)
    p.add_argument("--aa-root", default=None)
    p.add_argument("--manifest", default=None)
    p.add_argument("--cv-csv", default=None)
    p.add_argument("--cv-sample-col", default="meta_sample")
    p.add_argument("--cv-chrom-col", default="meta_chrom")
    p.add_argument("--cv-start-col", default="meta_start")
    p.add_argument("--cv-value-col", default="oof_prediction")
    p.add_argument("--cv-raw-depth-col", default="y_wgs_tumor_depth")
    p.add_argument("--seed-bed-manifest", default=None)
    p.add_argument("--seed-bed-filename", default="*_AA_CNV_SEEDS.bed")
    p.add_argument("--column", default="predicted_loess_upscale_depth")
    p.add_argument("--mask-col", default="mask_rejected")
    p.add_argument("--raw-depth-col", default="raw_wes_depth")

    p.add_argument("--classification-tsv", required=True)
    p.add_argument("--classification-bed-dir", required=True)
    p.add_argument("--orange-categories", default="ecDNA,BFB,CNC")
    p.add_argument("--include-ecdna-bfb-samples-in-recurrent-calling", action="store_true")
    p.add_argument("--include-cnc-samples-in-recurrent-calling", action="store_true")
    p.add_argument("--sample-map", default=None)

    p.add_argument("--rebin-to", type=int, default=25000)
    p.add_argument("--smooth-window", type=int, default=1)
    p.add_argument("--min-overlap-bp", type=int, default=1)

    # The one known combo to replot -- from the original run's console
    # output / best_combo_summary.txt.
    p.add_argument("--quantile", type=float, required=True)
    p.add_argument("--min-samples", type=int, required=True,
                   help="Absolute count (e.g. 6), not a ratio -- this is what the original "
                        "run printed as 'min_samples=N' in its Best combo line.")
    p.add_argument("--min-samples-ratio-display", type=float, default=None,
                   help="Optional, cosmetic only: the min_samples_ratio value to show in the "
                        "plot title, matching the original run's title text.")

    p.add_argument("--exclude-recurrent-from-orange", action="store_true",
                   help="By default, orange includes every is_orange=True instance regardless "
                        "of that location's is_recurrent status (a location's recurrent status "
                        "is voted on by OTHER samples' non-orange instances, so a bin can be "
                        "both orange in one sample and recurrent as a location at the same "
                        "time). Pass this to instead drop any is_recurrent=True instance from "
                        "BOTH orange and blue, so recurrent bins are fully separated out of the "
                        "distributions rather than just omitted as a third plotted layer.")
    p.add_argument("--highlight-sample", default=None)
    p.add_argument("--min-depth", type=float, default=0.1)
    p.add_argument("--outdir", default="replotted_histograms")
    args = p.parse_args()

    if not args.cv_csv and not args.manifest and not (args.wes_root and args.aa_root):
        sys.exit("Error: provide either --cv-csv, --manifest, or both --wes-root and --aa-root.")
    if args.cv_csv and not (args.seed_bed_manifest or args.aa_root):
        sys.exit("Error: --cv-csv mode needs --seed-bed-manifest or --aa-root to locate seed BEDs.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    opt = load_module(args.optimize_script, "optimize_mod")
    base = opt.load_base_module(args.base_script)

    # --- reproduce main()'s setup, up through building `cache` -----------
    if args.cv_csv:
        cv_data = opt.load_cv_predictions_csv(
            base, args.cv_csv, args.cv_sample_col, args.cv_chrom_col, args.cv_start_col,
            args.cv_value_col, args.cv_raw_depth_col)
        if not cv_data:
            sys.exit(f"Error: no rows loaded from --cv-csv {args.cv_csv}.")
        seed_table = opt.load_seed_bed_manifest(args.seed_bed_manifest) if args.seed_bed_manifest \
            else opt.discover_seed_beds(args.aa_root, args.seed_bed_filename)
        if seed_table.empty:
            sys.exit("Error: no seed BEDs found.")
    else:
        sample_table = base.load_manifest(args.manifest) if args.manifest \
            else base.discover_samples(args.wes_root, args.aa_root)[0]
        if sample_table.empty:
            sys.exit("Error: no samples with both a copy ratio TSV and a seed BED were found.")

    qualifying_by_sample = opt.load_qualifying_amplicons_by_sample(
        args.classification_tsv, tuple(args.orange_categories.split(",")))

    excluded_samples = set()
    if not args.include_ecdna_bfb_samples_in_recurrent_calling:
        excluded_samples |= opt.load_samples_with_ecdna_or_bfb(args.classification_tsv)
    if not args.include_cnc_samples_in_recurrent_calling:
        excluded_samples |= opt.load_samples_with_cnc(args.classification_tsv)

    bed_index = opt.index_classification_bed_dir(args.classification_bed_dir)
    sample_name_map = opt.load_sample_name_map(args.sample_map) if args.sample_map else {}

    if args.cv_csv:
        cache, _ = opt.build_per_sample_cache_cv(
            base, cv_data, seed_table, args.rebin_to, args.smooth_window, args.min_overlap_bp,
            qualifying_by_sample, bed_index, sample_name_map, excluded_samples)
    else:
        cache, _ = opt.build_per_sample_cache(
            base, sample_table, args.column, args.mask_col, args.rebin_to,
            args.smooth_window, args.min_overlap_bp, qualifying_by_sample, bed_index, sample_name_map,
            args.raw_depth_col, excluded_samples)
    if not cache:
        sys.exit("Error: no samples loaded successfully.")
    print(f"[+] Cache built: {len(cache)} samples")

    # --- single-combo pooling (no grid search) ----------------------------
    pooled_novel, pooled_all = opt.pool_for_quantile(cache, args.quantile)
    merged, _ = opt.label_recurrent(pooled_novel, pooled_all, args.min_samples)
    merged = opt._apply_min_depth_filter(merged, args.min_depth, context_label="[pooled] ")

    orange_mask = merged["is_orange"] & (~merged["is_recurrent"] if args.exclude_recurrent_from_orange else True)
    orange_v = merged.loc[orange_mask, "value_raw"]
    blue_v = merged.loc[~merged["is_orange"] & ~merged["is_recurrent"], "value_raw"]
    empty = np.array([])  # deliberately empty -> plot_best_histogram omits the purple layer

    suffix = "no_recurrent_excluded" if args.exclude_recurrent_from_orange else "no_recurrent"
    n_excluded_orange = int((merged["is_orange"] & merged["is_recurrent"]).sum()) if args.exclude_recurrent_from_orange else 0
    if args.exclude_recurrent_from_orange:
        print(f"[+] Excluded {n_excluded_orange:,} orange bin-instances that were also is_recurrent "
              f"(orange n went from {int(merged['is_orange'].sum()):,} to {len(orange_v):,})")

    pooled_out = outdir / f"best_combo_histogram_{suffix}.png"
    opt.plot_best_histogram(orange_v, blue_v, empty, args.quantile, args.min_samples,
                             args.min_samples_ratio_display, pooled_out)
    print(f"[+] Saved: {pooled_out}")

    if args.highlight_sample:
        sample_rows = merged[merged["sample"] == args.highlight_sample]
        if sample_rows.empty:
            available = sorted(merged["sample"].unique())
            print(f"[warn] --highlight-sample '{args.highlight_sample}' not found. "
                  f"Available: {available[:15]}{' ...' if len(available) > 15 else ''}")
        else:
            orange_s_mask = sample_rows["is_orange"] & (~sample_rows["is_recurrent"] if args.exclude_recurrent_from_orange else True)
            orange_s = sample_rows.loc[orange_s_mask, "value_raw"]
            blue_s = sample_rows.loc[~sample_rows["is_orange"] & ~sample_rows["is_recurrent"], "value_raw"]
            sample_out = outdir / f"best_combo_histogram_{args.highlight_sample}_{suffix}.png"
            opt.plot_best_histogram(orange_s, blue_s, empty, args.quantile, args.min_samples,
                                     args.min_samples_ratio_display, sample_out,
                                     subtitle=f"sample: {args.highlight_sample}")
            print(f"[+] Saved: {sample_out}")


if __name__ == "__main__":
    main()
