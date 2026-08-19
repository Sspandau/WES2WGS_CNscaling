#!/usr/bin/env python3
"""
find_recurrent_novel_amplifications.py

Across many samples, find genomic regions that are RECURRENTLY amplified
(i.e. flagged as clusters by genomic_cluster_analysis.py) but are NOT
associated with that sample's own AmpliconArchitect seed regions.

This answers: "which amplified bins show up consistently across my cohort,
independent of the known/seeded amplicon driver regions?"

EXPECTED FILE LAYOUT (auto-discovered by default)
--------------------------------------------------
  --wes-root/<SAMPLE>/*_cluster_results/clusters.csv
      e.g. CCLE_WXS/WES2WGS_CCLE/SW579_THYROID/SRR8619028_cluster_results/clusters.csv

  --aa-root/<SAMPLE>/<SAMPLE>_AA_CNV_SEEDS.bed
      e.g. /nucleus/projects/cancer_cell_lines/AA/CCLE/AA/SNU1_STOMACH/SNU1_STOMACH_AA_CNV_SEEDS.bed

Sample names are taken from the subdirectories of --wes-root. If your paths
don't follow this exact pattern, use --manifest instead (see below).

MANIFEST OVERRIDE (optional)
-----------------------------
--manifest a TSV/CSV with columns: sample, clusters_csv, seed_bed
Use this if auto-discovery doesn't fit your directory structure, or to
restrict/rename the sample set.

ALGORITHM
---------
1. For each sample, load its clusters.csv (output of genomic_cluster_analysis.py)
   and its AA seed BED.
2. Drop any cluster that overlaps (by >= --min-overlap-bp) a seed region in
   that SAME sample -- these are "seed-associated", not novel.
3. Pool the remaining ("candidate novel") clusters across all samples and
   merge overlapping/nearby intervals (allowing --merge-gap bp) into
   consensus regions, tracking which samples contributed to each.
4. Keep consensus regions supported by >= --min-samples distinct samples.

OUTPUT (written to --outdir)
-----------------------------
  recurrent_novel_amplifications.csv   full table, sorted by n_samples desc
  recurrent_novel_amplifications.bed   bed4 (chrom, start, end, n_samples)
  recurrence_track.png                 genome-wide plot of recurrence counts
  summary.txt                          run summary + per-sample stats
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
    """Auto-discover (sample, clusters_csv, seed_bed) triples from the
    directory layout described in the module docstring."""
    entries = []
    sample_dirs = sorted(
        d for d in os.listdir(wes_root) if os.path.isdir(os.path.join(wes_root, d))
    )
    for sample in sample_dirs:
        cluster_glob = os.path.join(wes_root, sample, "*_cluster_results", "clusters.csv")
        matches = sorted(glob.glob(cluster_glob))
        if not matches:
            print(f"[skip] {sample}: no clusters.csv found matching {cluster_glob}")
            continue
        if len(matches) > 1:
            print(f"[warn] {sample}: found {len(matches)} clusters.csv matches, "
                  f"using the first: {matches[0]}. Use --manifest to disambiguate "
                  f"if this is wrong.")
        clusters_csv = matches[0]

        seed_bed = os.path.join(aa_root, sample, f"{sample}_AA_CNV_SEEDS.bed")
        if not os.path.exists(seed_bed):
            print(f"[skip] {sample}: no seed BED found at {seed_bed}")
            continue

        entries.append({"sample": sample, "clusters_csv": clusters_csv, "seed_bed": seed_bed})
    return pd.DataFrame(entries)


def load_manifest(path):
    sep = "\t" if str(path).lower().endswith((".tsv", ".txt")) else None
    df = pd.read_csv(path, sep=sep, engine="python")
    required = {"sample", "clusters_csv", "seed_bed"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"Error: manifest is missing required columns: {missing}")
    return df


# --------------------------------------------------------------------------
# Per-sample: load clusters + seeds, exclude seed-overlapping clusters
# --------------------------------------------------------------------------

def load_seed_bed(path):
    df = pd.read_csv(path, sep="\t", header=None, comment="#", engine="python")
    if df.shape[1] < 3:
        raise ValueError(f"Seed BED {path} has fewer than 3 columns.")
    df = df.iloc[:, :3].copy()
    df.columns = ["chrom", "start", "end"]
    df["chrom"] = df["chrom"].map(normalize_chrom)
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    return df


def load_clusters_csv(path):
    df = pd.read_csv(path)
    required = {"chrom", "start", "end"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"clusters.csv {path} is missing required columns: {missing}")
    df["chrom"] = df["chrom"].map(normalize_chrom)
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    return df


def exclude_seed_overlaps(clusters_df, seeds_df, min_overlap_bp=1):
    """Returns clusters_df filtered to rows that do NOT overlap any seed
    region by >= min_overlap_bp, plus the count of rows dropped."""
    if seeds_df.empty:
        clusters_df = clusters_df.copy()
        clusters_df["seed_associated"] = False
        return clusters_df, 0

    is_seed_associated = np.zeros(len(clusters_df), dtype=bool)
    c_chrom = clusters_df["chrom"].to_numpy()
    c_start = clusters_df["start"].to_numpy()
    c_end = clusters_df["end"].to_numpy()

    for _, seed in seeds_df.iterrows():
        same_chrom = c_chrom == seed["chrom"]
        overlap_bp = np.minimum(c_end, seed["end"]) - np.maximum(c_start, seed["start"])
        overlaps = same_chrom & (overlap_bp >= min_overlap_bp)
        is_seed_associated |= overlaps

    out = clusters_df.copy()
    out["seed_associated"] = is_seed_associated
    n_dropped = int(is_seed_associated.sum())
    return out[~is_seed_associated].copy(), n_dropped


# --------------------------------------------------------------------------
# Cross-sample merging into consensus recurrent regions
# --------------------------------------------------------------------------

def merge_across_samples(pooled_df, merge_gap=0):
    """
    pooled_df: concatenated candidate-novel clusters from all samples, with
    a 'sample' column. Merges overlapping/nearby (within merge_gap bp)
    intervals across ALL samples into consensus regions, regardless of
    which sample they came from, then reports which/how many distinct
    samples contributed to each consensus region.
    """
    pooled_df = pooled_df.sort_values(["chrom", "start", "end"]).reset_index(drop=True)

    records = []
    current = None

    for row in pooled_df.itertuples(index=False):
        if current is None:
            current = {
                "chrom": row.chrom, "start": row.start, "end": row.end,
                "samples": {row.sample}, "n_clusters": 1,
                "mean_values": [getattr(row, "mean_value", np.nan)],
                "max_values": [getattr(row, "max_value", np.nan)],
            }
            continue

        same_chrom = row.chrom == current["chrom"]
        overlaps_or_close = row.start <= current["end"] + merge_gap
        if same_chrom and overlaps_or_close:
            current["end"] = max(current["end"], row.end)
            current["samples"].add(row.sample)
            current["n_clusters"] += 1
            current["mean_values"].append(getattr(row, "mean_value", np.nan))
            current["max_values"].append(getattr(row, "max_value", np.nan))
        else:
            records.append(current)
            current = {
                "chrom": row.chrom, "start": row.start, "end": row.end,
                "samples": {row.sample}, "n_clusters": 1,
                "mean_values": [getattr(row, "mean_value", np.nan)],
                "max_values": [getattr(row, "max_value", np.nan)],
            }

    if current is not None:
        records.append(current)

    out_rows = []
    for r in records:
        out_rows.append({
            "chrom": r["chrom"],
            "start": r["start"],
            "end": r["end"],
            "width_bp": r["end"] - r["start"],
            "n_samples": len(r["samples"]),
            "samples": ";".join(sorted(r["samples"])),
            "n_clusters": r["n_clusters"],
            "mean_value_avg": float(np.nanmean(r["mean_values"])) if r["mean_values"] else np.nan,
            "max_value_max": float(np.nanmax(r["max_values"])) if r["max_values"] else np.nan,
        })
    return pd.DataFrame(out_rows)


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def plot_recurrence_track(recurrent_df, all_regions_df, out_path, min_samples):
    if recurrent_df.empty:
        print("No regions passed the recurrence threshold; skipping plot.")
        return

    chroms = sorted(all_regions_df["chrom"].unique(), key=natural_chrom_key)
    n = len(chroms)
    fig, axes = plt.subplots(n, 1, figsize=(14, 1.6 * n), sharex=False)
    if n == 1:
        axes = [axes]

    max_n_samples = max(all_regions_df["n_samples"].max(), 1)

    for ax, chrom in zip(axes, chroms):
        sub = all_regions_df[all_regions_df["chrom"] == chrom]
        for _, row in sub.iterrows():
            color = "#d62728" if row["n_samples"] >= min_samples else "#cccccc"
            ax.bar(
                (row["start"] + row["end"]) / 2,
                row["n_samples"],
                width=max(row["width_bp"], 1),
                color=color,
                align="center",
            )
        ax.axhline(min_samples, color="black", ls="--", lw=0.6)
        ax.set_ylabel(chrom, rotation=0, ha="right", va="center", fontsize=9)
        ax.set_ylim(0, max_n_samples * 1.1)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        if len(sub):
            ax.set_xlim(sub["start"].min(), sub["end"].max())

    axes[-1].set_xlabel("Genomic position")
    fig.suptitle(
        f"Recurrence of candidate novel amplified regions across samples "
        f"(red = passes min-samples={min_samples} threshold)",
        y=1.0,
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
                        "a *_cluster_results/clusters.csv (auto-discovery mode)")
    p.add_argument("--aa-root", default=None,
                   help="Root directory containing one subdirectory per sample, each with "
                        "<sample>_AA_CNV_SEEDS.bed (auto-discovery mode)")
    p.add_argument("--manifest", default=None,
                   help="TSV/CSV with columns: sample, clusters_csv, seed_bed. "
                        "Overrides --wes-root/--aa-root auto-discovery.")
    p.add_argument("--min-overlap-bp", type=int, default=1,
                   help="Minimum bp overlap with a seed region for a cluster to be "
                        "considered seed-associated (and excluded). Default: 1 (any overlap).")
    p.add_argument("--merge-gap", type=int, default=0,
                   help="Max bp gap allowed when merging candidate regions across samples "
                        "into one consensus region. Default: 0 (must actually overlap).")
    p.add_argument("--min-samples", type=int, default=3,
                   help="Minimum number of distinct samples required for a consensus region "
                        "to be reported as 'recurrent'. Default: 3")
    p.add_argument("--outdir", default="recurrent_novel_amplifications", help="Output directory")
    args = p.parse_args()

    if not args.manifest and not (args.wes_root and args.aa_root):
        sys.exit("Error: provide either --manifest, or both --wes-root and --aa-root.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.manifest:
        sample_table = load_manifest(args.manifest)
    else:
        sample_table = discover_samples(args.wes_root, args.aa_root)

    if sample_table.empty:
        sys.exit("Error: no samples with both a clusters.csv and a seed BED were found.")

    print(f"Found {len(sample_table)} samples with both cluster and seed files.")

    per_sample_stats = []
    pooled_candidates = []

    for row in sample_table.itertuples(index=False):
        try:
            clusters_df = load_clusters_csv(row.clusters_csv)
            seeds_df = load_seed_bed(row.seed_bed)
        except Exception as e:
            print(f"[skip] {row.sample}: failed to load inputs ({e})")
            continue

        candidates, n_dropped = exclude_seed_overlaps(clusters_df, seeds_df, args.min_overlap_bp)
        candidates = candidates.copy()
        candidates["sample"] = row.sample

        per_sample_stats.append({
            "sample": row.sample,
            "n_seeds": len(seeds_df),
            "n_clusters_total": len(clusters_df),
            "n_seed_associated": n_dropped,
            "n_candidate_novel": len(candidates),
        })

        keep_cols = ["chrom", "start", "end", "sample"]
        for optional_col in ("mean_value", "max_value", "n_bins"):
            if optional_col in candidates.columns:
                keep_cols.append(optional_col)
        pooled_candidates.append(candidates[keep_cols])

    if not pooled_candidates:
        sys.exit("Error: no candidate novel regions were produced for any sample.")

    pooled_df = pd.concat(pooled_candidates, ignore_index=True)
    print(f"Pooled {len(pooled_df)} candidate novel cluster regions across "
          f"{pooled_df['sample'].nunique()} samples.")

    all_regions_df = merge_across_samples(pooled_df, merge_gap=args.merge_gap)
    all_regions_df = all_regions_df.sort_values(
        ["n_samples", "chrom", "start"], ascending=[False, True, True]
    ).reset_index(drop=True)

    recurrent_df = all_regions_df[all_regions_df["n_samples"] >= args.min_samples].copy()

    csv_path = outdir / "recurrent_novel_amplifications.csv"
    recurrent_df.to_csv(csv_path, index=False)

    bed_path = outdir / "recurrent_novel_amplifications.bed"
    bed_df = recurrent_df[["chrom", "start", "end", "n_samples"]].copy()
    bed_df.to_csv(bed_path, sep="\t", header=False, index=False)

    plot_path = outdir / "recurrence_track.png"
    plot_recurrence_track(recurrent_df, all_regions_df, plot_path, args.min_samples)

    stats_df = pd.DataFrame(per_sample_stats).sort_values("sample")

    summary_path = outdir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("Recurrent novel amplification analysis summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Samples processed:                  {len(per_sample_stats)}\n")
        f.write(f"Total candidate novel clusters:     {len(pooled_df)}\n")
        f.write(f"Consensus regions (any recurrence): {len(all_regions_df)}\n")
        f.write(f"Recurrent regions (>= {args.min_samples} samples):    {len(recurrent_df)}\n\n")
        f.write(f"Parameters:\n")
        f.write(f"  min_overlap_bp (seed exclusion): {args.min_overlap_bp}\n")
        f.write(f"  merge_gap (cross-sample merge):  {args.merge_gap}\n")
        f.write(f"  min_samples (recurrence cutoff): {args.min_samples}\n\n")
        f.write("Per-sample breakdown:\n")
        f.write(stats_df.to_string(index=False))
        f.write("\n\nTop recurrent regions:\n")
        if len(recurrent_df):
            f.write(recurrent_df.head(20).to_string(index=False))
        else:
            f.write("(none passed the --min-samples threshold)\n")

    print(f"Consensus regions (any recurrence): {len(all_regions_df)}")
    print(f"Recurrent regions (>= {args.min_samples} samples): {len(recurrent_df)}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {bed_path}")
    print(f"Saved: {plot_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
