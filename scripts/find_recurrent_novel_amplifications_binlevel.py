#!/usr/bin/env python3
"""
find_recurrent_amplified_bins.py

Bin-level (not cluster-level) recurrence analysis across many samples,
using the *_off_target_copy_ratios.tsv files (the same inputs consumed by
genomic_cluster_analysis.py) directly, rather than pre-computed clusters.

WHY BIN-LEVEL INSTEAD OF CLUSTER-LEVEL
---------------------------------------
The cluster-level approach (find_recurrent_novel_amplifications.py) merges
overlapping cluster spans across samples, which can telescope into very
large "recurrent regions" if many samples each have a large, imprecisely-
bounded cluster nearby -- e.g. a chromosome-wide technical artifact (low
PoN mappability, near-zero pon_median inflating raw_wes_depth/pon_median)
can look identical to a chromosome-wide "recurrent amplification" once
clusters are merged. Working bin-by-bin avoids that merge-driven distortion
and makes the two cases visually and numerically distinguishable.

WHAT COUNTS AS "AMPLIFIED" HERE
---------------------------------
Same definition as genomic_cluster_analysis.py: per SAMPLE, the value
column (default predicted_loess_upscale_depth) is smoothed and any bin
above that sample's own --quantile (default 0.90) is called "high". This
is a RELATIVE, rank-based definition -- it is not an absolute copy-number
threshold (e.g. log2 ratio > 1). A bin can be "high" because of a genuine
focal amplification, or because of a systematic normalization artifact
that inflates values broadly for that sample. The pon_median diagnostic
columns in the output are meant to help you tell these apart: genuine
amplification hotspots should NOT correlate with low pon_median_wgs.

PIPELINE PER SAMPLE
--------------------
1. Load the sample's copy-ratio TSV.
2. Drop rows flagged by --mask-col (default mask_rejected), same as
   genomic_cluster_analysis.py.
3. Rebin to --rebin-to bp (mean aggregation) so all samples share a common
   bin grid (this matches the --rebin-to used when clusters.csv was made).
4. Smooth per chromosome, threshold at --quantile to get is_high per bin.
5. Exclude bins overlapping that sample's OWN AmpliconArchitect seed
   regions (>= --min-overlap-bp) -- these are seed-associated, not novel.

ACROSS SAMPLES
---------------
For every bin on the common grid, count how many samples had it flagged
as novel-high, and how many samples had usable (non-fully-masked) data
there at all. Bins with n_samples_high >= --min-samples are reported,
individually and merged into contiguous runs (adjacent recurrent bins
only -- no gap-bridging by default, use --merge-gap to allow bridging
small gaps of non-recurrent bins).

OUTPUT (written to --outdir)
-----------------------------
  recurrent_bins.csv       one row per individual recurrent bin
  recurrent_regions.csv    contiguous recurrent bins merged into intervals
  recurrent_regions.bed    same regions as BED4 (chrom, start, end, n_samples)
  recurrence_track.png     genome-wide recurrence-fraction plot, colored by
                            an artifact-risk proxy (mean PoN median) so
                            technical-artifact chromosomes stand out
  summary.txt              run summary, per-sample stats, skipped samples
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------

def normalize_chrom(c):
    c = str(c).strip()
    return c if c.lower().startswith("chr") else f"chr{c}"


def natural_chrom_key(chrom):
    s = str(chrom).lower().replace("chr", "")
    special = {"x": 1000, "y": 1001, "m": 1002, "mt": 1002}
    if s in special:
        return special[s]
    try:
        return int(s)
    except ValueError:
        return 999


def discover_samples(wes_root, aa_root):
    entries = []
    skip_log = []

    wes_dirs = {d for d in os.listdir(wes_root) if os.path.isdir(os.path.join(wes_root, d))}
    aa_dirs = {d for d in os.listdir(aa_root) if os.path.isdir(os.path.join(aa_root, d))}
    sample_dirs = sorted(wes_dirs & aa_dirs)

    ignored = sorted(wes_dirs - aa_dirs)
    if ignored:
        print(f"[info] Ignoring {len(ignored)} WES-root subdirectories with no matching "
              f"AA-root directory: {ignored[:5]}{' ...' if len(ignored) > 5 else ''}")

    for sample in sample_dirs:
        tsv_glob = os.path.join(wes_root, sample, "*_off_target_copy_ratios.tsv")
        matches = sorted(glob.glob(tsv_glob))
        if not matches:
            reason = f"no copy ratio TSV found matching {tsv_glob}"
            print(f"[skip] {sample}: {reason}")
            skip_log.append({"sample": sample, "stage": "discovery", "reason": reason})
            continue
        if len(matches) > 1:
            print(f"[warn] {sample}: found {len(matches)} copy ratio TSVs, "
                  f"using the first: {matches[0]}. Use --manifest to disambiguate.")
        copy_ratio_tsv = matches[0]

        seed_bed = os.path.join(aa_root, sample, f"{sample}_AA_CNV_SEEDS.bed")
        if not os.path.exists(seed_bed):
            reason = f"no seed BED found at {seed_bed}"
            print(f"[skip] {sample}: {reason}")
            skip_log.append({"sample": sample, "stage": "discovery", "reason": reason})
            continue

        entries.append({"sample": sample, "copy_ratio_tsv": copy_ratio_tsv, "seed_bed": seed_bed})
    return pd.DataFrame(entries), skip_log


def load_manifest(path):
    sep = "\t" if str(path).lower().endswith((".tsv", ".txt")) else None
    df = pd.read_csv(path, sep=sep, engine="python")
    required = {"sample", "copy_ratio_tsv", "seed_bed"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"Error: manifest is missing required columns: {missing}")
    return df


def load_seed_bed(path):
    if os.path.getsize(path) == 0:
        return pd.DataFrame(columns=["chrom", "start", "end"])
    try:
        df = pd.read_csv(path, sep="\t", header=None, comment="#", engine="python")
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=["chrom", "start", "end"])
    if df.empty:
        return pd.DataFrame(columns=["chrom", "start", "end"])
    if df.shape[1] < 3:
        raise ValueError(f"Seed BED {path} has fewer than 3 columns.")
    df = df.iloc[:, :3].copy()
    df.columns = ["chrom", "start", "end"]
    df["chrom"] = df["chrom"].map(normalize_chrom)
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    return df


# --------------------------------------------------------------------------
# Per-sample bin pipeline (mirrors genomic_cluster_analysis.py)
# --------------------------------------------------------------------------

def load_copy_ratio_tsv(path, value_col, mask_col, chrom_col="chrom", start_col="start", end_col="end"):
    sep = "\t" if str(path).lower().endswith((".tsv", ".txt")) else None
    df = pd.read_csv(path, sep=sep, engine="python")

    for col in (value_col, chrom_col, start_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {path}. Available: {list(df.columns)}")

    df = df.rename(columns={chrom_col: "chrom", start_col: "start"})
    df["chrom"] = df["chrom"].map(normalize_chrom)
    df["start"] = df["start"].astype(int)
    df["value_raw"] = df[value_col].astype(float)
    df = df.dropna(subset=["value_raw", "start"])

    pon_col = None
    for candidate in ("pon_median_wgs", "pon_median"):
        if candidate in df.columns:
            pon_col = candidate
            break
    df["pon_median"] = df[pon_col].astype(float) if pon_col else np.nan

    n_before = len(df)
    if mask_col.lower() != "none" and mask_col in df.columns:
        df = df[df[mask_col].astype(float).fillna(0) == 0]
    n_after = len(df)

    return df[["chrom", "start", "value_raw", "pon_median"]].sort_values(
        ["chrom", "start"]).reset_index(drop=True), n_before, n_after


def rebin_mean(df, new_bin_size):
    df = df.copy()
    df["bin_start"] = (df["start"] // new_bin_size) * new_bin_size
    agg = (
        df.groupby(["chrom", "bin_start"])
        .agg(value_raw=("value_raw", "mean"),
             pon_median=("pon_median", "mean"),
             n_fine_bins=("value_raw", "size"))
        .reset_index()
        .rename(columns={"bin_start": "start"})
    )
    agg["end"] = agg["start"] + new_bin_size
    return agg.sort_values(["chrom", "start"]).reset_index(drop=True)


def smooth_per_chrom(df, window):
    if window <= 1:
        df["smoothed"] = df["value_raw"]
        return df
    df["smoothed"] = (
        df.groupby("chrom", group_keys=False)["value_raw"]
        .apply(lambda s: s.rolling(window, center=True, min_periods=1).mean())
    )
    return df


def flag_seed_overlap(bins_df, seeds_df, min_overlap_bp):
    if seeds_df.empty:
        return np.zeros(len(bins_df), dtype=bool)

    flags = np.zeros(len(bins_df), dtype=bool)
    b_chrom = bins_df["chrom"].to_numpy()
    b_start = bins_df["start"].to_numpy()
    b_end = bins_df["end"].to_numpy()

    for _, seed in seeds_df.iterrows():
        same_chrom = b_chrom == seed["chrom"]
        overlap_bp = np.minimum(b_end, seed["end"]) - np.maximum(b_start, seed["start"])
        flags |= same_chrom & (overlap_bp >= min_overlap_bp)
    return flags


# --------------------------------------------------------------------------
# Merge adjacent recurrent bins into contiguous regions
# --------------------------------------------------------------------------

def merge_adjacent_bins(recurrent_df, bin_size, merge_gap=0):
    recurrent_df = recurrent_df.sort_values(["chrom", "start"]).reset_index(drop=True)
    records = []
    current = None

    for row in recurrent_df.itertuples(index=False):
        if current is None:
            current = {"chrom": row.chrom, "start": row.start, "end": row.end,
                       "n_samples_high_vals": [row.n_samples_high],
                       "n_samples_with_data_vals": [row.n_samples_with_data],
                       "mean_pon_median_vals": [row.mean_pon_median],
                       "n_bins": 1}
            continue

        same_chrom = row.chrom == current["chrom"]
        adjacent = row.start <= current["end"] + merge_gap * bin_size
        if same_chrom and adjacent:
            current["end"] = max(current["end"], row.end)
            current["n_samples_high_vals"].append(row.n_samples_high)
            current["n_samples_with_data_vals"].append(row.n_samples_with_data)
            current["mean_pon_median_vals"].append(row.mean_pon_median)
            current["n_bins"] += 1
        else:
            records.append(current)
            current = {"chrom": row.chrom, "start": row.start, "end": row.end,
                       "n_samples_high_vals": [row.n_samples_high],
                       "n_samples_with_data_vals": [row.n_samples_with_data],
                       "mean_pon_median_vals": [row.mean_pon_median],
                       "n_bins": 1}
    if current is not None:
        records.append(current)

    out_rows = []
    for r in records:
        out_rows.append({
            "chrom": r["chrom"], "start": r["start"], "end": r["end"],
            "width_bp": r["end"] - r["start"], "n_bins": r["n_bins"],
            "max_n_samples_high": int(np.max(r["n_samples_high_vals"])),
            "mean_n_samples_high": float(np.mean(r["n_samples_high_vals"])),
            "mean_n_samples_with_data": float(np.mean(r["n_samples_with_data_vals"])),
            "mean_pon_median": float(np.nanmean(r["mean_pon_median_vals"])),
        })
    return pd.DataFrame(out_rows)


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def plot_recurrence_track(all_bins_df, min_samples, out_path):
    if all_bins_df.empty:
        print("No bins to plot; skipping.")
        return

    chroms = sorted(all_bins_df["chrom"].unique(), key=natural_chrom_key)
    n = len(chroms)
    fig, axes = plt.subplots(n, 1, figsize=(14, 1.4 * n), sharex=False)
    if n == 1:
        axes = [axes]

    # Chromosome-level artifact-risk flag: compare each chromosome's typical
    # (median) PoN reference depth among its recurrent bins to the median
    # across chromosomes. A chromosome whose PoN median is far below the
    # rest (default: <10%) is flagged as likely low-mappability / unreliable
    # PoN reference rather than genuine recurrent amplification. This is
    # deliberately chromosome-level, not bin-level, since a single global
    # bin quantile can be skewed by how many bins a given artifact affects.
    chrom_pon_median = all_bins_df.groupby("chrom")["mean_pon_median"].median()
    overall_median_pon = chrom_pon_median.median()
    low_pon_chroms = set(chrom_pon_median[chrom_pon_median < overall_median_pon * 0.1].index)

    for ax, chrom in zip(axes, chroms):
        sub = all_bins_df[all_bins_df["chrom"] == chrom]
        is_artifact_risk = chrom in low_pon_chroms
        colors = np.where(
            sub["n_samples_high"] < min_samples, "#dddddd",
            "#ff9900" if is_artifact_risk else "#d62728"
        )
        ax.bar(sub["start"], sub["n_samples_high"], width=sub["end"] - sub["start"],
               color=colors, align="edge")
        ax.axhline(min_samples, color="black", ls="--", lw=0.6)
        label = f"{chrom} [low PoN]" if is_artifact_risk else chrom
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        if len(sub):
            ax.set_xlim(sub["start"].min(), sub["end"].max())

    axes[-1].set_xlabel("Genomic position")
    fig.suptitle(
        "Bin-level recurrence of novel (non-seed) amplification calls\n"
        "red = recurrent; orange = recurrent but on a chromosome with unusually low "
        "PoN median overall (possible normalization artifact, e.g. low-mappability "
        "regions); gray = below threshold",
        y=1.02, fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wes-root", default=None,
                   help="Root directory containing one subdirectory per sample, each with "
                        "a *_off_target_copy_ratios.tsv (auto-discovery mode)")
    p.add_argument("--aa-root", default=None,
                   help="Root directory containing one subdirectory per sample, each with "
                        "<sample>_AA_CNV_SEEDS.bed (auto-discovery mode)")
    p.add_argument("--manifest", default=None,
                   help="TSV/CSV with columns: sample, copy_ratio_tsv, seed_bed. "
                        "Overrides --wes-root/--aa-root auto-discovery.")
    p.add_argument("--column", default="predicted_loess_upscale_depth",
                   help="Value column to analyze (default: predicted_loess_upscale_depth)")
    p.add_argument("--mask-col", default="mask_rejected",
                   help="Column flagging rows to exclude before any analysis (nonzero = "
                        "excluded). Set to 'none' to disable. Default: mask_rejected")
    p.add_argument("--rebin-to", type=int, default=25000,
                   help="Common bin size in bp for cross-sample comparison (mean aggregation). "
                        "Should match what was used for the per-sample clustering. Default: 25000")
    p.add_argument("--smooth-window", type=int, default=1,
                   help="Rolling mean window in bins, applied per sample before thresholding. "
                        "Default: 1 (no extra smoothing, matching rebin-to default elsewhere).")
    p.add_argument("--quantile", type=float, default=0.90,
                   help="Per-sample quantile of the smoothed signal used as the 'high' "
                        "threshold. Default: 0.90")
    p.add_argument("--min-overlap-bp", type=int, default=1,
                   help="Minimum bp overlap with a seed region for a bin to be considered "
                        "seed-associated (and excluded). Default: 1 (any overlap)")
    p.add_argument("--min-samples", type=int, default=3,
                   help="Minimum number of samples with a bin flagged high (and not "
                        "seed-associated) for that bin to be reported as recurrent. Default: 3")
    p.add_argument("--merge-gap", type=int, default=0,
                   help="Max number of consecutive non-recurrent bins allowed when merging "
                        "recurrent bins into contiguous regions. Default: 0 (bins must be "
                        "directly adjacent).")
    p.add_argument("--outdir", default="recurrent_amplified_bins", help="Output directory")
    args = p.parse_args()

    if not args.manifest and not (args.wes_root and args.aa_root):
        sys.exit("Error: provide either --manifest, or both --wes-root and --aa-root.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.manifest:
        sample_table = load_manifest(args.manifest)
        skip_log = []
    else:
        sample_table, skip_log = discover_samples(args.wes_root, args.aa_root)

    if sample_table.empty:
        sys.exit("Error: no samples with both a copy ratio TSV and a seed BED were found.")

    print(f"Found {len(sample_table)} samples with both copy ratio TSV and seed BED.")

    per_sample_stats = []
    per_sample_bins = []  # list of (sample, DataFrame[chrom,start,end,is_novel_high,pon_median])

    for row in sample_table.itertuples(index=False):
        try:
            raw_df, n_before, n_after = load_copy_ratio_tsv(row.copy_ratio_tsv, args.column, args.mask_col)
        except Exception as e:
            reason = f"failed to load copy_ratio_tsv ({row.copy_ratio_tsv}): {e}"
            print(f"[skip] {row.sample}: {reason}")
            skip_log.append({"sample": row.sample, "stage": "load_copy_ratio_tsv", "reason": reason})
            continue

        try:
            seeds_df = load_seed_bed(row.seed_bed)
        except Exception as e:
            reason = f"failed to load seed_bed ({row.seed_bed}): {e}"
            print(f"[skip] {row.sample}: {reason}")
            skip_log.append({"sample": row.sample, "stage": "load_seed_bed", "reason": reason})
            continue

        binned = rebin_mean(raw_df, args.rebin_to)
        binned = smooth_per_chrom(binned, args.smooth_window)
        threshold = binned["smoothed"].quantile(args.quantile)
        binned["is_high"] = binned["smoothed"] > threshold

        seed_flag = flag_seed_overlap(binned, seeds_df, args.min_overlap_bp)
        binned["is_novel_high"] = binned["is_high"] & ~seed_flag

        per_sample_stats.append({
            "sample": row.sample,
            "n_fine_bins_before_mask": n_before,
            "n_fine_bins_after_mask": n_after,
            "n_coarse_bins": len(binned),
            "n_seeds": len(seeds_df),
            "n_high_bins": int(binned["is_high"].sum()),
            "n_seed_associated_high_bins": int((binned["is_high"] & seed_flag).sum()),
            "n_novel_high_bins": int(binned["is_novel_high"].sum()),
        })

        per_sample_bins.append(
            binned.loc[binned["is_novel_high"], ["chrom", "start", "end", "pon_median"]]
            .assign(sample=row.sample)
        )

    if not per_sample_bins or all(b.empty for b in per_sample_bins):
        sys.exit("Error: no novel-high bins were produced for any sample.")

    pooled = pd.concat(per_sample_bins, ignore_index=True)
    n_samples_processed = len(per_sample_stats)

    recurrence = (
        pooled.groupby(["chrom", "start", "end"])
        .agg(n_samples_high=("sample", "nunique"), mean_pon_median=("pon_median", "mean"))
        .reset_index()
    )
    # n_samples_with_data: for simplicity/robustness, use n_samples_processed as the
    # denominator baseline; a bin absent from a sample's rebinned table (fully masked
    # away) is treated as "no data" and does not count toward n_samples_high, but we
    # don't currently track a per-sample per-bin "present but not high" count, so this
    # is reported as an approximation.
    recurrence["n_samples_with_data"] = n_samples_processed

    all_bins_df = recurrence.sort_values(["chrom", "start"]).reset_index(drop=True)
    recurrent_bins_df = all_bins_df[all_bins_df["n_samples_high"] >= args.min_samples].copy()
    recurrent_bins_df = recurrent_bins_df.sort_values(
        ["n_samples_high", "chrom", "start"], ascending=[False, True, True]
    ).reset_index(drop=True)

    bins_csv_path = outdir / "recurrent_bins.csv"
    recurrent_bins_df.to_csv(bins_csv_path, index=False)

    recurrent_regions_df = merge_adjacent_bins(recurrent_bins_df, args.rebin_to, args.merge_gap)
    regions_csv_path = outdir / "recurrent_regions.csv"
    recurrent_regions_df = recurrent_regions_df.sort_values(
        ["max_n_samples_high", "chrom", "start"], ascending=[False, True, True]
    ).reset_index(drop=True)
    recurrent_regions_df.to_csv(regions_csv_path, index=False)

    bed_path = outdir / "recurrent_regions.bed"
    recurrent_regions_df[["chrom", "start", "end", "max_n_samples_high"]].to_csv(
        bed_path, sep="\t", header=False, index=False
    )

    plot_path = outdir / "recurrence_track.png"
    plot_recurrence_track(all_bins_df, args.min_samples, plot_path)

    stats_df = pd.DataFrame(per_sample_stats).sort_values("sample")
    summary_path = outdir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("Recurrent amplified bins (bin-level) analysis summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Samples processed:                  {n_samples_processed}\n")
        f.write(f"Samples skipped:                    {len(skip_log)}\n")
        f.write(f"Common bin size:                    {args.rebin_to} bp\n")
        f.write(f"Total pooled novel-high bins:       {len(pooled)}\n")
        f.write(f"Distinct bins with any recurrence:  {len(all_bins_df)}\n")
        f.write(f"Recurrent bins (>= {args.min_samples} samples):       {len(recurrent_bins_df)}\n")
        f.write(f"Recurrent regions (merged, gap={args.merge_gap}): {len(recurrent_regions_df)}\n\n")
        f.write("Parameters:\n")
        f.write(f"  column:            {args.column}\n")
        f.write(f"  mask_col:          {args.mask_col}\n")
        f.write(f"  rebin_to:          {args.rebin_to}\n")
        f.write(f"  smooth_window:     {args.smooth_window}\n")
        f.write(f"  quantile:          {args.quantile}\n")
        f.write(f"  min_overlap_bp:    {args.min_overlap_bp}\n")
        f.write(f"  min_samples:       {args.min_samples}\n")
        f.write(f"  merge_gap:         {args.merge_gap}\n\n")
        if skip_log:
            f.write("Skipped samples:\n")
            f.write(pd.DataFrame(skip_log).to_string(index=False))
            f.write("\n\n")
        f.write("Per-sample breakdown:\n")
        f.write(stats_df.to_string(index=False))
        f.write("\n\nTop recurrent regions (by max_n_samples_high):\n")
        if len(recurrent_regions_df):
            f.write(recurrent_regions_df.head(30).to_string(index=False))
        else:
            f.write("(none passed the --min-samples threshold)\n")
        f.write(
            "\n\nNote on mean_pon_median: low values here are a common signature of "
            "low-mappability / unreliable PoN reference regions (e.g. acrocentric "
            "chromosome arms, centromeres, satellite repeats). A recurrent region with "
            "a conspicuously low mean_pon_median relative to the genome-wide distribution "
            "is more likely to be a normalization artifact than a genuine recurrent "
            "amplification, and is worth checking manually before treating it as a hit.\n"
        )

    print(f"Recurrent bins (>= {args.min_samples} samples): {len(recurrent_bins_df)}")
    print(f"Recurrent regions (merged): {len(recurrent_regions_df)}")
    print(f"Saved: {bins_csv_path}")
    print(f"Saved: {regions_csv_path}")
    print(f"Saved: {bed_path}")
    print(f"Saved: {plot_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
