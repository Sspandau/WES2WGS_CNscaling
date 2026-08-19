#!/usr/bin/env python3
"""
diagnose_missing_orange_bins.py

Runs the SAME per-sample orange-flagging logic twice -- once against legacy
per-sample off_target_copy_ratios.tsv files, once against a --cv-csv
consolidated file -- and diffs the two orange (ecDNA+/BFB+/CNC) location
sets to show EXACTLY which orange bin locations are present in one source
but not the other, and why.

Deliberately does NOT go through optimize_ecdna_bfb_cnc_thresholds.py's
build_per_sample_cache()/build_per_sample_cache_cv(): those require a
successful seed-BED lookup for a sample to be included at all, which is
irrelevant to orange coverage and would conflate two different possible
causes of a missing bin. This script only needs copy-ratio data + the
classification TSV/BED dir, so every sample with EITHER data source is
compared regardless of seed-BED availability.

For every orange location present in the legacy source but missing from
--cv-csv (the direction you're asking about), it also checks whether
--cv-csv had ANY native-resolution row at all inside that rebinned window,
for the specific sample(s) that carry the orange call there -- which
distinguishes "no cv-csv coverage there" (the sparse-coverage explanation)
from "cv-csv has rows there but they didn't get flagged" (which would be a
bug worth chasing separately).

OUTPUT (--outdir)
------------------
  orange_locations_summary.txt      counts: legacy-only, cv-only, shared
  missing_from_cv_csv.csv           every orange location the legacy source
                                     has that --cv-csv doesn't, with the
                                     contributing sample(s) and whether
                                     --cv-csv had ANY native row there
  missing_from_legacy.csv           the reverse direction, for completeness
                                     / sanity-checking
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_optimize_module(path):
    import importlib.util
    path = Path(path)
    if not path.exists():
        sys.exit(f"Error: --optimize-script not found at {path}. Pass the path to "
                  f"optimize_ecdna_bfb_cnc_thresholds.py (or your renamed copy of it).")
    spec = importlib.util.spec_from_file_location("optimize_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def discover_legacy_samples(wes_root):
    """One subdirectory per sample under wes_root, each containing a single
    *_off_target_copy_ratios.tsv -- same convention as
    prepare_feats_wgs_treemodel.py's discover_samples_from_root(). Written
    locally rather than reused from the base analysis script, which
    couples sample discovery to seed-BED availability (requires BOTH a
    copy-ratio TSV and a seed BED per sample) -- irrelevant here and would
    silently drop legacy samples for a reason unrelated to orange coverage."""
    import os
    import glob
    entries = []
    for d in sorted(os.listdir(wes_root)):
        sample_dir = os.path.join(wes_root, d)
        if not os.path.isdir(sample_dir):
            continue
        matches = sorted(glob.glob(os.path.join(sample_dir, "*_off_target_copy_ratios.tsv")))
        if matches:
            entries.append({"sample": d, "copy_ratio_tsv": matches[0]})
    return pd.DataFrame(entries)


def load_legacy_manifest(path):
    sep = "\t" if str(path).lower().endswith((".tsv", ".txt")) else None
    df = pd.read_csv(path, sep=sep, engine="python")
    required = {"sample", "copy_ratio_tsv"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"Error: --manifest is missing required columns: {missing}")
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--optimize-script", required=True,
                   help="Path to optimize_ecdna_bfb_cnc_thresholds.py -- reused for its "
                        "classification-loading, cv-csv-loading, and rebinning helpers so this "
                        "diagnostic can never drift out of sync with the main analysis.")
    p.add_argument("--base-script", required=True,
                   help="Path to find_recurrent_novel_amplifications_binlevel.py, same as the "
                        "main script's --base-script.")
    p.add_argument("--classification-tsv", required=True)
    p.add_argument("--classification-bed-dir", required=True)
    p.add_argument("--orange-categories", default="ecDNA,BFB,CNC")
    p.add_argument("--sample-map", default=None,
                   help="Same meaning as the main script's --sample-map: maps a sample id (used "
                        "in both --wes-root dir names and --cv-csv meta_sample values -- this "
                        "diagnostic assumes those two already match each other directly) to the "
                        "classification TSV's sample_name, if they differ.")

    # legacy source
    p.add_argument("--wes-root", required=True,
                    help="Root with one subdirectory per sample containing "
                         "*_off_target_copy_ratios.tsv (or pass --manifest instead)")
    p.add_argument("--manifest", default=None,
                    help="TSV/CSV with columns sample, copy_ratio_tsv -- overrides --wes-root")
    p.add_argument("--column", default="predicted_loess_upscale_depth")
    p.add_argument("--mask-col", default="mask_rejected")
    p.add_argument("--raw-depth-col", default="raw_wes_depth")

    # cv-csv source
    p.add_argument("--cv-csv", required=True)
    p.add_argument("--cv-sample-col", default="meta_sample")
    p.add_argument("--cv-chrom-col", default="meta_chrom")
    p.add_argument("--cv-start-col", default="meta_start")
    p.add_argument("--cv-value-col", default="oof_prediction")
    p.add_argument("--cv-raw-depth-col", default="y_wgs_tumor_depth")

    p.add_argument("--rebin-to", type=int, default=25000)
    p.add_argument("--min-overlap-bp", type=int, default=1)
    p.add_argument("--outdir", default="orange_coverage_diagnostic")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    opt = load_optimize_module(args.optimize_script)
    base = opt.load_base_module(args.base_script)

    # --- shared: classification data -----------------------------------
    qualifying_by_sample = opt.load_qualifying_amplicons_by_sample(
        args.classification_tsv, tuple(args.orange_categories.split(",")))
    bed_index = opt.index_classification_bed_dir(args.classification_bed_dir)
    sample_name_map = opt.load_sample_name_map(args.sample_map) if args.sample_map else {}

    # --- legacy source: sample -> copy_ratio_tsv path -------------------
    if args.manifest:
        sample_table = load_legacy_manifest(args.manifest)
    else:
        sample_table = discover_legacy_samples(args.wes_root)
    if sample_table.empty:
        sys.exit("Error: no legacy samples found via --wes-root/--manifest (looking for "
                  "<wes-root>/<sample>/*_off_target_copy_ratios.tsv).")
    legacy_tsv_by_sample = dict(zip(sample_table["sample"].astype(str), sample_table["copy_ratio_tsv"]))
    print(f"[legacy] {len(legacy_tsv_by_sample)} samples with a copy-ratio TSV")

    # --- cv-csv source: sample -> native-resolution DataFrame -----------
    cv_data = opt.load_cv_predictions_csv(
        base, args.cv_csv, args.cv_sample_col, args.cv_chrom_col, args.cv_start_col,
        args.cv_value_col, args.cv_raw_depth_col)
    print(f"[cv-csv] {len(cv_data)} samples")

    all_samples = sorted(set(legacy_tsv_by_sample) | set(cv_data))
    print(f"[+] {len(all_samples)} samples total (union of both sources)")

    # --- per-sample: compute orange-flagged locations from each source --
    legacy_rows, cv_rows = [], []          # each: DataFrame[chrom,start,end] for one sample's orange bins
    cv_native_by_sample = {}               # sample -> native (unbinned) cv DataFrame, kept for the "why missing" check

    for sample in all_samples:
        classification_sample = sample_name_map.get(sample, sample) if sample_name_map else sample
        qualifying = qualifying_by_sample.get(classification_sample, set())
        if not qualifying:
            continue  # this sample has no orange amplicons at all -- nothing to compare
        orange_df = opt.load_orange_intervals_for_sample(classification_sample, qualifying, bed_index, base)
        if orange_df.empty:
            continue

        if sample in legacy_tsv_by_sample:
            try:
                raw_df, _, _, _ = opt.load_copy_ratio_with_raw_depth(
                    base, legacy_tsv_by_sample[sample], args.column, args.mask_col, args.raw_depth_col)
                binned = opt.rebin_mean_with_raw_depth(raw_df, args.rebin_to)
                flag = base.flag_seed_overlap(binned, orange_df, args.min_overlap_bp)
                hits = binned.loc[flag, ["chrom", "start", "end"]].copy()
                hits["sample"] = sample
                legacy_rows.append(hits)
            except Exception as e:
                print(f"    [-] {sample}: legacy load failed ({e}) -- excluded from legacy side only")

        if sample in cv_data:
            native_df = cv_data[sample]
            cv_native_by_sample[sample] = native_df
            binned = opt.rebin_mean_with_raw_depth(native_df, args.rebin_to)
            flag = base.flag_seed_overlap(binned, orange_df, args.min_overlap_bp)
            hits = binned.loc[flag, ["chrom", "start", "end"]].copy()
            hits["sample"] = sample
            cv_rows.append(hits)

    legacy_df = pd.concat(legacy_rows, ignore_index=True) if legacy_rows else \
        pd.DataFrame(columns=["chrom", "start", "end", "sample"])
    cv_df = pd.concat(cv_rows, ignore_index=True) if cv_rows else \
        pd.DataFrame(columns=["chrom", "start", "end", "sample"])

    legacy_set = set(legacy_df[["chrom", "start", "end"]].itertuples(index=False, name=None))
    cv_set = set(cv_df[["chrom", "start", "end"]].itertuples(index=False, name=None))

    missing_from_cv = legacy_set - cv_set
    missing_from_legacy = cv_set - legacy_set
    shared = legacy_set & cv_set

    print(f"[+] Legacy orange locations: {len(legacy_set):,}")
    print(f"[+] cv-csv orange locations: {len(cv_set):,}")
    print(f"[+] Shared: {len(shared):,}")
    print(f"[+] Missing from cv-csv (present in legacy only): {len(missing_from_cv):,}")
    print(f"[+] Missing from legacy (present in cv-csv only): {len(missing_from_legacy):,}")

    # --- for each cv-csv-missing location: which sample(s) carry it in legacy,
    # and did cv-csv have ANY native row there at all for those samples? -----
    legacy_samples_by_loc = legacy_df.groupby(["chrom", "start", "end"])["sample"].apply(
        lambda s: ",".join(sorted(set(s)))).to_dict()

    def native_coverage_check(chrom, start, end, samples_csv):
        samples = samples_csv.split(",")
        total_rows = 0
        any_sample_covered = False
        for s in samples:
            native_df = cv_native_by_sample.get(s)
            if native_df is None:
                continue
            n = int(((native_df["chrom"] == chrom) & (native_df["start"] >= start) &
                     (native_df["start"] < end)).sum())
            total_rows += n
            if n > 0:
                any_sample_covered = True
        return total_rows, any_sample_covered

    missing_rows = []
    for chrom, start, end in sorted(missing_from_cv):
        samples_csv = legacy_samples_by_loc.get((chrom, start, end), "")
        n_native_rows, any_covered = native_coverage_check(chrom, start, end, samples_csv)
        missing_rows.append({
            "chrom": chrom, "start": start, "end": end,
            "samples_with_orange_here": samples_csv,
            "n_native_cv_rows_in_window": n_native_rows,
            "cv_csv_had_any_coverage_here": any_covered,
        })
    missing_df = pd.DataFrame(missing_rows)
    if not missing_df.empty:
        missing_df = missing_df.sort_values(
            ["cv_csv_had_any_coverage_here", "chrom", "start"])
    missing_df.to_csv(outdir / "missing_from_cv_csv.csv", index=False)

    n_no_coverage = int((~missing_df["cv_csv_had_any_coverage_here"]).sum()) if not missing_df.empty else 0
    n_had_coverage_but_missed = int(missing_df["cv_csv_had_any_coverage_here"].sum()) if not missing_df.empty else 0

    # --- reverse direction, for completeness -----------------------------
    cv_samples_by_loc = cv_df.groupby(["chrom", "start", "end"])["sample"].apply(
        lambda s: ",".join(sorted(set(s)))).to_dict()
    reverse_rows = [{"chrom": c, "start": s, "end": e, "samples_with_orange_here": cv_samples_by_loc.get((c, s, e), "")}
                     for c, s, e in sorted(missing_from_legacy)]
    pd.DataFrame(reverse_rows).to_csv(outdir / "missing_from_legacy.csv", index=False)

    summary_path = outdir / "orange_locations_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Orange bin coverage: legacy TSVs vs --cv-csv\n" + "=" * 46 + "\n\n")
        f.write(f"Legacy orange locations:                    {len(legacy_set):,}\n")
        f.write(f"cv-csv orange locations:                    {len(cv_set):,}\n")
        f.write(f"Shared:                                     {len(shared):,}\n")
        f.write(f"Missing from cv-csv (legacy only):          {len(missing_from_cv):,}\n")
        f.write(f"  -> of these, cv-csv had ZERO native rows "
                f"in that window:  {n_no_coverage:,}  (genuine sparse-coverage gap)\n")
        f.write(f"  -> of these, cv-csv HAD rows there but they "
                f"weren't flagged: {n_had_coverage_but_missed:,}  (worth investigating -- see below)\n")
        f.write(f"Missing from legacy (cv-csv only):          {len(missing_from_legacy):,}\n\n")
        if n_had_coverage_but_missed:
            f.write("NOTE: rows exist in missing_from_cv_csv.csv where "
                    "cv_csv_had_any_coverage_here=True -- these are NOT explained by sparse "
                    "coverage. Possible causes: the covering rows failed load_cv_predictions_csv's "
                    "dropna (NaN in --cv-value-col), a chrom-normalization mismatch, or a rebin "
                    "boundary effect. Inspect these specific (chrom,start,end) rows against the "
                    "raw --cv-csv directly.\n")

    print(f"[+] Saved: {summary_path}")
    print(f"[+] Saved: {outdir / 'missing_from_cv_csv.csv'}  "
          f"({n_no_coverage:,} genuine gaps, {n_had_coverage_but_missed:,} unexplained)")
    print(f"[+] Saved: {outdir / 'missing_from_legacy.csv'}")


if __name__ == "__main__":
    main()
