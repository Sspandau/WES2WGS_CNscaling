#!/usr/bin/env python3
"""
genomic_cluster_analysis.py

Find spatial clusters of genomic bins with elevated predicted values,
allowing small gaps, with smoothing, plus a global/per-chromosome
spatial-autocorrelation check (Moran's I), and a genomic-track plot.

USAGE
-----
python genomic_cluster_analysis.py \
    --input predictions.csv \
    --column pred_score \
    --bin-col bin \
    --outdir results/

If your file already has separate chromosome/position columns instead of
one combined "bin" column, use --chrom-col and --start-col instead of
--bin-col.

OUTPUT (written to --outdir)
-----------------------------
  clusters.csv        one row per cluster (coords, size, mean/max value)
  morans_i.txt        global + per-chromosome Moran's I, z-score, p-value
  summary.txt         plain-language run summary
  genomic_track.png   one horizontal track per chromosome, clusters shaded
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import binary_closing


# --------------------------------------------------------------------------
# Parsing / loading
# --------------------------------------------------------------------------

BIN_COL_PATTERN = re.compile(
    r"^\s*(?P<chrom>chr[\w]+|[0-9XYMxym]+)\s*[:_\-]\s*"
    r"(?P<start>\d+)\s*[:_\-]\s*(?P<end>\d+)?\s*$"
)


def load_data(path):
    sep = "\t" if str(path).lower().endswith((".tsv", ".txt")) else None
    df = pd.read_csv(path, sep=sep, engine="python")
    return df


def split_bin_column(df, bin_col):
    """Parse a combined bin identifier like 'chr1:100000-200000' or
    'chr1_100000_200000' into chrom / start / end columns."""
    parsed = df[bin_col].astype(str).str.extract(BIN_COL_PATTERN)
    if parsed["chrom"].isna().any():
        bad = df.loc[parsed["chrom"].isna(), bin_col].head(3).tolist()
        raise ValueError(
            f"Could not parse bin column '{bin_col}'. Examples that failed: "
            f"{bad}. Expected formats like 'chr1:1000-2000', 'chr1_1000_2000', "
            f"or 'chr1-1000-2000'. If your file already has separate "
            f"chromosome/position columns, use --chrom-col/--start-col instead."
        )
    df["chrom"] = parsed["chrom"]
    df["start"] = parsed["start"].astype(int)
    df["end"] = parsed["end"].astype(float)  # may contain NaN -> filled later
    return df


def natural_chrom_key(chrom):
    """Sort chr1, chr2, ..., chr10, chrX, chrY, chrM in a sensible order."""
    s = str(chrom).lower().replace("chr", "")
    special = {"x": 1000, "y": 1001, "m": 1002, "mt": 1002}
    if s in special:
        return special[s]
    try:
        return int(s)
    except ValueError:
        return 999


# --------------------------------------------------------------------------
# Core analysis
# --------------------------------------------------------------------------

def smooth_per_chrom(df, value_col, window):
    """Centered rolling mean, computed independently within each chromosome
    so smoothing never bleeds across chromosome boundaries."""
    if window <= 1:
        df["smoothed"] = df[value_col]
        return df
    df["smoothed"] = (
        df.groupby("chrom", group_keys=False)[value_col]
        .apply(lambda s: s.rolling(window, center=True, min_periods=1).mean())
    )
    return df


def call_high_bins(df, quantile):
    threshold = df["smoothed"].quantile(quantile)
    df["is_high"] = df["smoothed"] > threshold
    return df, threshold


def cluster_bins(df, max_gap, min_cluster_size):
    """
    Group adjacent 'high' bins into clusters, allowing up to `max_gap`
    consecutive non-high bins in between (morphological closing), but never
    bridging a genuine large physical gap between bins (e.g. missing
    assembly region, centromere, chromosome end).
    """
    df = df.sort_values(["chrom", "start"]).reset_index(drop=True)
    df["cluster_id"] = -1

    cluster_counter = 0
    pieces = []

    for chrom, sub in df.groupby("chrom", sort=False):
        sub = sub.sort_values("start").reset_index()  # keep original idx in 'index'
        if len(sub) == 0:
            continue

        step = sub["start"].diff().dropna()
        expected_step = step.mode().iloc[0] if not step.mode().empty else 1
        # a "hard break" = physical gap far larger than what max_gap should bridge
        hard_break = sub["start"].diff().fillna(0) > expected_step * (max_gap + 1) * 1.5

        is_high = sub["is_high"].to_numpy()

        # fill small gaps between high bins (binary closing)
        if max_gap > 0 and len(is_high) > 2:
            closed = binary_closing(
                is_high, structure=np.ones(3, dtype=bool), iterations=max_gap
            )
        else:
            closed = is_high.copy()

        # never bridge a hard physical break, regardless of closing result
        hard_break_arr = hard_break.to_numpy()
        segment_id = np.cumsum(hard_break_arr)  # increments at each large physical gap

        # within each segment, split into runs of closed==True
        run_break = (~closed) | (np.diff(np.concatenate(([segment_id[0]], segment_id))) != 0)
        local_cluster = np.cumsum(run_break)

        sub["closed"] = closed
        sub["local_cluster"] = local_cluster
        sub["chrom_for_cluster"] = chrom

        for local_id, grp in sub[sub["closed"]].groupby("local_cluster"):
            n_high = int(grp["is_high"].sum())
            if n_high == 0:
                continue
            cluster_counter += 1
            df.loc[grp["index"], "cluster_id"] = cluster_counter
        pieces.append(sub)

    clustered = df[df["cluster_id"] > 0]
    if clustered.empty:
        return pd.DataFrame(
            columns=[
                "chrom", "start", "end", "n_bins", "n_high_bins",
                "mean_value", "max_value", "mean_smoothed",
            ]
        ), df

    agg = (
        clustered.groupby("cluster_id")
        .agg(
            chrom=("chrom", "first"),
            start=("start", "min"),
            end=("start", "max"),
            n_bins=("start", "size"),
            n_high_bins=("is_high", "sum"),
            mean_value=("value_raw", "mean"),
            max_value=("value_raw", "max"),
            mean_smoothed=("smoothed", "mean"),
        )
        .reset_index(drop=True)
    )
    agg = agg[agg["n_bins"] >= min_cluster_size]
    agg = agg.sort_values(["chrom", "start"]).reset_index(drop=True)
    return agg, df


# --------------------------------------------------------------------------
# Spatial autocorrelation: Moran's I on first-order (adjacent-bin) neighbors
# --------------------------------------------------------------------------

def morans_i_for_chrom(sub, value_col, max_neighbor_gap):
    """Moran's I restricted to pairs of physically adjacent bins
    (gap <= max_neighbor_gap * modal bin step)."""
    x = sub[value_col].to_numpy(dtype=float)
    pos = sub["start"].to_numpy()
    n = len(x)
    if n < 3:
        return None

    step = np.diff(pos)
    modal_step = pd.Series(step).mode().iloc[0] if len(step) else 1
    neighbor_mask = step <= modal_step * max_neighbor_gap

    xbar = x.mean()
    dev = x - xbar
    denom = np.sum(dev ** 2)
    if denom == 0:
        return None

    pair_products = dev[:-1][neighbor_mask] * dev[1:][neighbor_mask]
    w_sum = 2 * neighbor_mask.sum()  # symmetric weights, each pair counted twice
    if w_sum == 0:
        return None

    numerator = 2 * pair_products.sum()
    I = (n / w_sum) * (numerator / denom)
    return {
        "I": I,
        "n_bins": n,
        "n_neighbor_pairs": int(neighbor_mask.sum()),
    }


def permutation_pvalue(sub, value_col, max_neighbor_gap, observed_I, n_perm, rng):
    x = sub[value_col].to_numpy(dtype=float).copy()
    pos = sub["start"].to_numpy()
    step = np.diff(pos)
    modal_step = pd.Series(step).mode().iloc[0] if len(step) else 1
    neighbor_mask = step <= modal_step * max_neighbor_gap
    w_sum = 2 * neighbor_mask.sum()
    n = len(x)
    if w_sum == 0:
        return np.nan

    xbar = x.mean()
    denom = np.sum((x - xbar) ** 2)
    if denom == 0:
        return np.nan

    perm_Is = np.empty(n_perm)
    for i in range(n_perm):
        xp = rng.permutation(x)
        dev = xp - xp.mean()
        pair_products = dev[:-1][neighbor_mask] * dev[1:][neighbor_mask]
        numerator = 2 * pair_products.sum()
        perm_Is[i] = (n / w_sum) * (numerator / np.sum(dev ** 2))

    p = (np.sum(np.abs(perm_Is) >= abs(observed_I)) + 1) / (n_perm + 1)
    return p


def run_morans_i(df, value_col, max_neighbor_gap, n_perm, seed):
    rng = np.random.default_rng(seed)
    results = {}

    # global: pool neighbor pairs across all chromosomes (cross-chrom pairs excluded
    # naturally since we compute within each chromosome and concatenate)
    all_dev_products = []
    all_devs_sq = []
    total_n = 0
    total_w = 0

    per_chrom_rows = []
    for chrom, sub in df.groupby("chrom", sort=False):
        sub = sub.sort_values("start")
        res = morans_i_for_chrom(sub, value_col, max_neighbor_gap)
        if res is None:
            continue
        p = permutation_pvalue(sub, value_col, max_neighbor_gap, res["I"], n_perm, rng)
        per_chrom_rows.append(
            {"chrom": chrom, "morans_I": res["I"], "n_bins": res["n_bins"],
             "n_neighbor_pairs": res["n_neighbor_pairs"], "p_value": p}
        )

        x = sub[value_col].to_numpy(dtype=float)
        pos = sub["start"].to_numpy()
        step = np.diff(pos)
        modal_step = pd.Series(step).mode().iloc[0] if len(step) else 1
        neighbor_mask = step <= modal_step * max_neighbor_gap
        xbar = x.mean()
        dev = x - xbar
        all_dev_products.append((dev[:-1][neighbor_mask] * dev[1:][neighbor_mask]).sum())
        all_devs_sq.append(np.sum(dev ** 2) * (len(x) / len(x)))  # placeholder, recombine below
        total_n += len(x)
        total_w += 2 * neighbor_mask.sum()

    per_chrom_df = pd.DataFrame(per_chrom_rows).sort_values(
        "chrom", key=lambda s: s.map(natural_chrom_key)
    ).reset_index(drop=True)

    # global Moran's I computed properly (not from the placeholder above):
    # use grand mean across the whole genome, per-chromosome adjacency only
    x_all = df[value_col].to_numpy(dtype=float)
    global_mean = x_all.mean()
    global_denom = np.sum((x_all - global_mean) ** 2)
    numer = 0.0
    w_total = 0
    n_total = len(df)
    for chrom, sub in df.groupby("chrom", sort=False):
        sub = sub.sort_values("start")
        x = sub[value_col].to_numpy(dtype=float)
        pos = sub["start"].to_numpy()
        step = np.diff(pos)
        if len(step) == 0:
            continue
        modal_step = pd.Series(step).mode().iloc[0]
        neighbor_mask = step <= modal_step * max_neighbor_gap
        dev = x - global_mean
        numer += 2 * (dev[:-1][neighbor_mask] * dev[1:][neighbor_mask]).sum()
        w_total += 2 * neighbor_mask.sum()

    global_I = (n_total / w_total) * (numer / global_denom) if w_total > 0 and global_denom > 0 else np.nan

    # permutation p-value for the global statistic (shuffle within whole genome,
    # keep per-chromosome adjacency structure)
    rng2 = np.random.default_rng(seed + 1)
    perm_Is = np.empty(n_perm)
    for i in range(n_perm):
        xp = rng2.permutation(x_all)
        dfp = df.copy()
        dfp["_perm"] = xp
        pm = global_mean  # keep same mean for permuted data structure (values unchanged, just reordered)
        num_p = 0.0
        for chrom, sub in dfp.groupby("chrom", sort=False):
            sub = sub.sort_values("start")
            x = sub["_perm"].to_numpy(dtype=float)
            pos = sub["start"].to_numpy()
            step = np.diff(pos)
            if len(step) == 0:
                continue
            modal_step = pd.Series(step).mode().iloc[0]
            neighbor_mask = step <= modal_step * max_neighbor_gap
            dev = x - pm
            num_p += 2 * (dev[:-1][neighbor_mask] * dev[1:][neighbor_mask]).sum()
        perm_Is[i] = (n_total / w_total) * (num_p / global_denom)

    global_p = (np.sum(np.abs(perm_Is) >= abs(global_I)) + 1) / (n_perm + 1)

    results["global_I"] = global_I
    results["global_p_value"] = global_p
    results["n_bins"] = n_total
    results["per_chrom"] = per_chrom_df
    return results


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def plot_genomic_track(df, clusters, value_col, threshold, out_path):
    chroms = sorted(df["chrom"].unique(), key=natural_chrom_key)
    n = len(chroms)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.2 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, chrom in zip(axes, chroms):
        sub = df[df["chrom"] == chrom].sort_values("start")
        ax.plot(sub["start"], sub[value_col], color="#888888", lw=0.6, alpha=0.6, label="raw")
        ax.plot(sub["start"], sub["smoothed"], color="#1f77b4", lw=1.3, label="smoothed")
        ax.axhline(threshold, color="red", ls="--", lw=0.8, label="threshold")

        chrom_clusters = clusters[clusters["chrom"] == chrom]
        for _, row in chrom_clusters.iterrows():
            ax.axvspan(row["start"], row["end"], color="orange", alpha=0.35)

        ax.set_ylabel(chrom, rotation=0, ha="right", va="center", fontsize=9)
        ax.set_yticks([])
        ax.set_xlim(sub["start"].min(), sub["start"].max())
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)

    axes[0].legend(loc="upper right", fontsize=8, ncol=3, frameon=False)
    axes[-1].set_xlabel("Genomic position")
    fig.suptitle("Predicted value by genomic bin, with detected clusters shaded", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Path to input CSV/TSV file")
    p.add_argument("--column", required=True, help="Name of the column with predicted values")
    p.add_argument("--bin-col", default=None,
                   help="Name of a single combined bin column, e.g. 'chr1:1000-2000'. "
                        "If your file has separate columns instead, use --chrom-col/--start-col.")
    p.add_argument("--chrom-col", default="chrom", help="Chromosome column name (default: chrom)")
    p.add_argument("--start-col", default="start", help="Start position column name (default: start)")
    p.add_argument("--quantile", type=float, default=0.90,
                   help="Quantile of the smoothed signal used as the 'high' threshold (default: 0.90)")
    p.add_argument("--smooth-window", type=int, default=5,
                   help="Rolling mean window size in bins (default: 5)")
    p.add_argument("--max-gap", type=int, default=1,
                   help="Max number of consecutive non-high bins allowed inside a cluster (default: 1)")
    p.add_argument("--min-cluster-size", type=int, default=2,
                   help="Minimum number of bins for a cluster to be reported (default: 2)")
    p.add_argument("--neighbor-gap-factor", type=float, default=1.5,
                   help="Bins are treated as spatial neighbors for Moran's I if their position "
                        "gap is <= this factor times the modal bin step (default: 1.5)")
    p.add_argument("--n-permutations", type=int, default=999,
                   help="Number of permutations for Moran's I p-value (default: 999)")
    p.add_argument("--seed", type=int, default=0, help="Random seed for permutation test")
    p.add_argument("--outdir", default="peak_cluster_output", help="Output directory")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.input)

    if args.column not in df.columns:
        sys.exit(f"Error: column '{args.column}' not found in input. Available columns: {list(df.columns)}")

    if args.bin_col:
        if args.bin_col not in df.columns:
            sys.exit(f"Error: bin column '{args.bin_col}' not found. Available columns: {list(df.columns)}")
        df = split_bin_column(df, args.bin_col)
    else:
        if args.chrom_col not in df.columns or args.start_col not in df.columns:
            sys.exit(
                f"Error: expected columns '{args.chrom_col}' and '{args.start_col}' not found. "
                f"Available columns: {list(df.columns)}. Use --bin-col if you have a single "
                f"combined bin identifier column instead."
            )
        df = df.rename(columns={args.chrom_col: "chrom", args.start_col: "start"})

    df["chrom"] = df["chrom"].astype(str)
    df["value_raw"] = df[args.column].astype(float)
    df = df.dropna(subset=["value_raw", "start"]).sort_values(["chrom", "start"]).reset_index(drop=True)

    print(f"Loaded {len(df)} bins across {df['chrom'].nunique()} chromosomes/contigs.")

    df = smooth_per_chrom(df, "value_raw", args.smooth_window)
    df, threshold = call_high_bins(df, args.quantile)
    print(f"'High' threshold (quantile {args.quantile}) on smoothed values: {threshold:.4f}")
    print(f"Bins above threshold: {df['is_high'].sum()} / {len(df)}")

    clusters, df_annotated = cluster_bins(df, args.max_gap, args.min_cluster_size)
    clusters_path = outdir / "clusters.csv"
    clusters.to_csv(clusters_path, index=False)
    print(f"Found {len(clusters)} clusters (min size {args.min_cluster_size} bins). Saved to {clusters_path}")

    morans = run_morans_i(df, "value_raw", args.neighbor_gap_factor, args.n_permutations, args.seed)
    morans_path = outdir / "morans_i.txt"
    with open(morans_path, "w") as f:
        f.write("Spatial autocorrelation (Moran's I) on adjacent-bin neighbors\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Genome-wide (pooled across chromosomes):\n")
        f.write(f"  Moran's I  = {morans['global_I']:.4f}\n")
        f.write(f"  p-value    = {morans['global_p_value']:.4g}  ({args.n_permutations} permutations)\n")
        f.write(f"  n bins     = {morans['n_bins']}\n\n")
        f.write("Per chromosome:\n")
        f.write(morans["per_chrom"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        f.write("\n\nNote: Moran's I > 0 indicates neighboring bins tend to have similar "
                "(positively correlated) values -- i.e. real spatial clustering rather than "
                "noise. I ~ 0 indicates no spatial structure; I < 0 indicates neighboring bins "
                "tend to differ (checkerboard-like pattern).\n")
    print(f"Moran's I: global = {morans['global_I']:.4f} (p = {morans['global_p_value']:.4g}). "
          f"Details saved to {morans_path}")

    plot_path = outdir / "genomic_track.png"
    plot_genomic_track(df_annotated, clusters, "value_raw", threshold, plot_path)
    print(f"Genomic track plot saved to {plot_path}")

    summary_path = outdir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("Genomic bin clustering analysis summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Input file:            {args.input}\n")
        f.write(f"Value column:          {args.column}\n")
        f.write(f"Bins analyzed:         {len(df)}\n")
        f.write(f"Chromosomes/contigs:   {df['chrom'].nunique()}\n")
        f.write(f"Smoothing window:      {args.smooth_window} bins\n")
        f.write(f"'High' threshold:      {threshold:.4f} (quantile {args.quantile} of smoothed values)\n")
        f.write(f"Max gap allowed:       {args.max_gap} bin(s)\n")
        f.write(f"Min cluster size:      {args.min_cluster_size} bins\n")
        f.write(f"Clusters found:        {len(clusters)}\n")
        if len(clusters):
            f.write(f"Largest cluster:       {clusters.loc[clusters['n_bins'].idxmax(), 'chrom']}: "
                     f"{int(clusters.loc[clusters['n_bins'].idxmax(),'start'])}-"
                     f"{int(clusters.loc[clusters['n_bins'].idxmax(),'end'])} "
                     f"({int(clusters['n_bins'].max())} bins)\n")
        f.write(f"\nGenome-wide Moran's I: {morans['global_I']:.4f} (p = {morans['global_p_value']:.4g})\n")
        f.write("\nOutputs:\n")
        f.write(f"  - {clusters_path.name}: table of detected clusters\n")
        f.write(f"  - {morans_path.name}: spatial autocorrelation results\n")
        f.write(f"  - {plot_path.name}: genomic track visualization\n")
        f.write("\nNext steps you mentioned doing later: cluster-level significance testing "
                "(e.g. permutation testing of cluster size/strength) and changepoint/"
                "segmentation methods (e.g. PELT via the `ruptures` package, or circular "
                "binary segmentation) as an alternative to the threshold-based approach here.\n")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
