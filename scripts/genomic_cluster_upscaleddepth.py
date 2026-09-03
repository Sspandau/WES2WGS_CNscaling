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

OPTIONAL: rebinning to a coarser resolution
--------------------------------------------
If your input is at a fine resolution (e.g. 5kb loess-predicted upscale
depth) and you want to analyze it at a coarser resolution (e.g. 25kb),
pass --rebin-to 25000. Bins are merged by taking the MEAN of the
constituent fine bins (appropriate for a continuous predicted/fitted
signal like loess depth -- use raw counts/sums instead if your value is
a raw read count, which this script does not currently handle specially).
When --rebin-to is used, --smooth-window defaults to 1 (no extra
smoothing) since averaging fine bins into coarse bins is itself a form
of smoothing and double-smoothing would over-flatten the signal. You can
still override this with an explicit --smooth-window.

OPTIONAL: automatic gap size
-----------------------------
--max-gap auto computes, per chromosome, the largest lag at which the
signal's autocorrelation is still statistically distinguishable from
zero (using a 95% confidence band), and uses that as the number of
non-high bins allowed inside a cluster. This ties the gap-merging
tolerance to how far the signal itself stays spatially correlated,
rather than to an arbitrary fixed integer.

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


def load_mask_regions(path):
    """Load a CSV/TSV of regions to mask. Expected columns: chrom,start,end.
    If exact column names are not found, the first three columns are used.
    Returns a DataFrame with columns `chrom` (str), `start` (int), `end` (int).
    """
    m = load_data(path)
    cols = {c.lower(): c for c in m.columns}
    if "chrom" in cols and "start" in cols and "end" in cols:
        m = m.rename(columns={cols["chrom"]: "chrom", cols["start"]: "start", cols["end"]: "end"})
    else:
        # fallback: take first three columns
        if m.shape[1] < 3:
            raise ValueError("Mask regions file must have at least three columns: chrom,start,end")
        first_three = m.columns[:3]
        m = m.rename(columns={first_three[0]: "chrom", first_three[1]: "start", first_three[2]: "end"})

    m = m[["chrom", "start", "end"]].copy()
    m["chrom"] = m["chrom"].astype(str)
    m["start"] = m["start"].astype(int)
    m["end"] = m["end"].astype(int)
    return m


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
# Rebinning to a coarser resolution
# --------------------------------------------------------------------------

def rebin_to_resolution(df, value_col, new_bin_size):
    """Merge fine bins into coarser bins of size `new_bin_size` bp by
    taking the mean of the value column within each new bin. Appropriate
    for continuous/fitted signals (e.g. loess-predicted depth); use a
    sum instead if your value is a raw count."""
    step = df.groupby("chrom")["start"].diff().dropna()
    modal_step = step.mode().iloc[0] if not step.empty else new_bin_size
    if new_bin_size < modal_step:
        raise ValueError(
            f"--rebin-to ({new_bin_size}) must be >= the current bin size "
            f"({int(modal_step)}); rebinning only merges to a coarser resolution."
        )

    df = df.copy()
    df["_new_start"] = (df["start"] // new_bin_size) * new_bin_size

    rebinned = (
        df.groupby(["chrom", "_new_start"])
        .agg(**{
            value_col: (value_col, "mean"),
            "n_fine_bins": (value_col, "size"),
        })
        .reset_index()
        .rename(columns={"_new_start": "start"})
    )
    rebinned = rebinned.sort_values(["chrom", "start"]).reset_index(drop=True)

    expected_n = new_bin_size / modal_step
    partial = rebinned["n_fine_bins"] < expected_n
    if partial.any():
        print(
            f"Note: {partial.sum()} of {len(rebinned)} rebinned {new_bin_size}bp bins "
            f"are made up of fewer than the expected {expected_n:.0f} fine bins "
            f"(likely chromosome ends or gaps in the input). Their mean is still "
            f"computed over whatever fine bins were present."
        )
    return rebinned


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


def call_high_bins(df, quantile, cn_floor=None):
    threshold = df["smoothed"].quantile(quantile)
    if cn_floor is not None:
        threshold = max(float(threshold), float(cn_floor))
    df["is_high"] = df["smoothed"] > threshold
    return df, threshold


def compute_binarized_overlap_metrics(wes_df, wgs_df, wes_value_col="value_raw", wgs_value_col="value_raw",
                                     wes_quantile=0.90, wgs_quantile=0.90,
                                     wes_smooth_window=None, wgs_smooth_window=None,
                                     rebin_to=None, cn_floor=None):
    """Compare two rescaled bin tracks using a binary threshold-on-high approach.

    Each track is smoothed within chromosome, thresholded at its own quantile,
    and then reduced to a set of genomic bins with values above threshold. We
    compare the resulting high-bin sets using Jaccard similarity and the
    overlap coefficient.

    Returns a dict with:
      - wes_threshold, wgs_threshold
      - wes_high_bins, wgs_high_bins
      - jaccard, overlap_coefficient
      - intersection_bins, union_bins
    """
    wes_proc = wes_df.copy()
    wgs_proc = wgs_df.copy()

    if rebin_to is not None:
        wes_proc = rebin_to_resolution(wes_proc[["chrom", "start", wes_value_col]].copy(), wes_value_col, rebin_to)
        wgs_proc = rebin_to_resolution(wgs_proc[["chrom", "start", wgs_value_col]].copy(), wgs_value_col, rebin_to)

    wes_proc = smooth_per_chrom(wes_proc, wes_value_col, wes_smooth_window if wes_smooth_window is not None else 5)
    wgs_proc = smooth_per_chrom(wgs_proc, wgs_value_col, wgs_smooth_window if wgs_smooth_window is not None else 5)

    wes_q = float(wes_proc["smoothed"].quantile(wes_quantile))
    wgs_q = float(wgs_proc["smoothed"].quantile(wgs_quantile))

    if cn_floor is not None and wes_q < cn_floor and wgs_q < cn_floor:
        wes_threshold = float(cn_floor)
        wgs_threshold = float(cn_floor)
        wes_proc["is_high"] = wes_proc["smoothed"] > wes_threshold
        wgs_proc["is_high"] = wgs_proc["smoothed"] > wgs_threshold
    else:
        wes_proc, wes_threshold = call_high_bins(wes_proc, wes_quantile)
        wgs_proc, wgs_threshold = call_high_bins(wgs_proc, wgs_quantile)

    wes_high = set(
        (str(row["chrom"]), int(row["start"]))
        for _, row in wes_proc[(wes_proc["is_high"])][["chrom", "start"]].iterrows()
    )
    wgs_high = set(
        (str(row["chrom"]), int(row["start"]))
        for _, row in wgs_proc[(wgs_proc["is_high"])][["chrom", "start"]].iterrows()
    )

    intersection = wes_high & wgs_high
    union = wes_high | wgs_high
    n_inter = len(intersection)
    n_union = len(union)
    n_wes = len(wes_high)
    n_wgs = len(wgs_high)

    if n_union == 0:
        jaccard = 1.0 if n_inter == 0 else 0.0
    else:
        jaccard = n_inter / n_union

    if n_wes == 0 and n_wgs == 0:
        overlap = 1.0
    elif min(n_wes, n_wgs) == 0:
        overlap = 0.0
    else:
        overlap = n_inter / min(n_wes, n_wgs)

    return {
        "wes_threshold": float(wes_threshold),
        "wgs_threshold": float(wgs_threshold),
        "n_wes_high_bins": n_wes,
        "n_wgs_high_bins": n_wgs,
        "n_intersection_bins": n_inter,
        "n_union_bins": n_union,
        "jaccard": float(jaccard),
        "overlap_coefficient": float(overlap),
        "wes_high_bins": wes_high,
        "wgs_high_bins": wgs_high,
    }


# --------------------------------------------------------------------------
# Automatic gap size from autocorrelation decay
# --------------------------------------------------------------------------

def compute_acf(x, nlags):
    """Simple, dependency-free autocorrelation function."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    var = np.dot(x, x) / n
    acf_vals = np.empty(nlags + 1)
    acf_vals[0] = 1.0
    for lag in range(1, nlags + 1):
        if var == 0 or n - lag <= 0:
            acf_vals[lag] = 0.0
        else:
            acf_vals[lag] = np.dot(x[:-lag], x[lag:]) / ((n - lag) * var)
    return acf_vals


def suggest_max_gap_from_acf(sub, value_col, max_lag=20, alpha=0.05):
    """Largest lag at which the signal's autocorrelation is still
    outside the ~95% confidence band for white noise. Used as a
    data-driven number of non-high bins to allow inside a cluster."""
    x = sub[value_col].to_numpy(dtype=float)
    n = len(x)
    if n < 10:
        return 1
    max_lag = min(max_lag, n // 3) if n // 3 > 0 else 1
    acf_vals = compute_acf(x, max_lag)
    conf = 1.96 / np.sqrt(n)
    suggested = 0
    for lag in range(1, len(acf_vals)):
        if abs(acf_vals[lag]) >= conf:
            suggested = lag
        else:
            break  # stop at first lag that drops into the noise band
    return max(suggested, 0)


def resolve_max_gap(df, value_col, max_gap_arg, max_lag=20):
    """Returns a dict {chrom: max_gap} regardless of whether the user
    passed a fixed integer or 'auto'."""
    chroms = df["chrom"].unique()
    if max_gap_arg == "auto":
        gaps = {}
        for chrom in chroms:
            sub = df[df["chrom"] == chrom].sort_values("start")
            gaps[chrom] = suggest_max_gap_from_acf(sub, value_col, max_lag)
        return gaps
    else:
        fixed = int(max_gap_arg)
        return {chrom: fixed for chrom in chroms}


def cluster_bins(df, max_gap_by_chrom, min_cluster_size):
    """
    Group adjacent 'high' bins into clusters, allowing up to
    max_gap_by_chrom[chrom] consecutive non-high bins in between
    (morphological closing), but never bridging a genuine large physical
    gap between bins (e.g. missing assembly region, centromere, chromosome
    end).
    """
    df = df.sort_values(["chrom", "start"]).reset_index(drop=True)
    df["cluster_id"] = -1

    cluster_counter = 0

    for chrom, sub in df.groupby("chrom", sort=False):
        sub = sub.sort_values("start").reset_index()  # keep original idx in 'index'
        if len(sub) == 0:
            continue

        max_gap = max(int(max_gap_by_chrom.get(chrom, 1)), 0)

        step = sub["start"].diff().dropna()
        expected_step = step.mode().iloc[0] if not step.mode().empty else 1
        hard_break = sub["start"].diff().fillna(0) > expected_step * (max_gap + 1) * 1.5

        is_high = sub["is_high"].to_numpy()

        if max_gap > 0 and len(is_high) > 2:
            closed = binary_closing(
                is_high, structure=np.ones(3, dtype=bool), iterations=max_gap
            )
        else:
            closed = is_high.copy()

        hard_break_arr = hard_break.to_numpy()
        segment_id = np.cumsum(hard_break_arr)

        run_break = (~closed) | (np.diff(np.concatenate(([segment_id[0]], segment_id))) != 0)
        local_cluster = np.cumsum(run_break)

        sub["closed"] = closed
        sub["local_cluster"] = local_cluster

        for local_id, grp in sub[sub["closed"]].groupby("local_cluster"):
            n_high = int(grp["is_high"].sum())
            if n_high == 0:
                continue
            cluster_counter += 1
            df.loc[grp["index"], "cluster_id"] = cluster_counter

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
    w_sum = 2 * neighbor_mask.sum()
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

    per_chrom_df = pd.DataFrame(per_chrom_rows).sort_values(
        "chrom", key=lambda s: s.map(natural_chrom_key)
    ).reset_index(drop=True)

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

    rng2 = np.random.default_rng(seed + 1)
    perm_Is = np.empty(n_perm)
    for i in range(n_perm):
        xp = rng2.permutation(x_all)
        dfp = df.copy()
        dfp["_perm"] = xp
        pm = global_mean
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

    return {
        "global_I": global_I,
        "global_p_value": global_p,
        "n_bins": n_total,
        "per_chrom": per_chrom_df,
    }


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

def max_gap_type(value):
    if value == "auto":
        return value
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("--max-gap must be an integer or 'auto'")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Path to input CSV/TSV file")
    p.add_argument("--column", required=True, help="Name of the column with predicted values")
    p.add_argument("--bin-col", default=None,
                   help="Name of a single combined bin column, e.g. 'chr1:1000-2000'. "
                        "If your file has separate columns instead, use --chrom-col/--start-col.")
    p.add_argument("--chrom-col", default="chrom", help="Chromosome column name (default: chrom)")
    p.add_argument("--start-col", default="start", help="Start position column name (default: start)")
    p.add_argument("--rebin-to", type=int, default=None,
                   help="Merge fine bins into coarser bins of this size in bp (e.g. 25000), "
                        "by taking the mean of the value column. Intended for continuous/fitted "
                        "signals such as loess-predicted depth.")
    p.add_argument("--quantile", type=float, default=0.90,
                   help="Quantile of the smoothed signal used as the 'high' threshold (default: 0.90)")
    p.add_argument("--smooth-window", type=int, default=None,
                   help="Rolling mean window size in bins. Default: 5, or 1 (no extra smoothing) "
                        "if --rebin-to is used, since the rebinning itself smooths the signal.")
    p.add_argument("--max-gap", type=max_gap_type, default=1,
                   help="Max number of consecutive non-high bins allowed inside a cluster. "
                        "Pass an integer, or 'auto' to derive it per chromosome from how far "
                        "the signal's autocorrelation stays significant (default: 1)")
    p.add_argument("--max-gap-search-lags", type=int, default=20,
                   help="Max lag (in bins) to search when --max-gap auto is used (default: 20)")
    p.add_argument("--min-cluster-size", type=int, default=2,
                   help="Minimum number of bins for a cluster to be reported (default: 2)")
    p.add_argument("--neighbor-gap-factor", type=float, default=1.5,
                   help="Bins are treated as spatial neighbors for Moran's I if their position "
                        "gap is <= this factor times the modal bin step (default: 1.5)")
    p.add_argument("--n-permutations", type=int, default=999,
                   help="Number of permutations for Moran's I p-value (default: 999)")
    p.add_argument("--seed", type=int, default=0, help="Random seed for permutation test")
    p.add_argument("--mask-regions", default=None,
                   help="Optional CSV/TSV of regions to mask as non-amplified (columns: chrom,start,end)."
                        "Masked bins are treated as non-high for clustering but do not create hard breaks.")
    p.add_argument("--outdir", default="peak_cluster_output", help="Output directory")
    p.add_argument("--compare-wgs-input", default=None,
                   help="Optional second input file for a WGS-vs-WES binary-overlap comparison. "
                        "If supplied, this is compared against the primary --input and --column, "
                        "using same genomic bins after rescaling/smoothing.")
    p.add_argument("--compare-wgs-column", default=None,
                   help="Value column in --compare-wgs-input to compare against the primary input. "
                        "Defaults to the same value name as --column.")
    p.add_argument("--compare-wes-quantile", type=float, default=0.90,
                   help="Threshold quantile for the rescaled WES values in the binary overlap comparison")
    p.add_argument("--compare-wgs-quantile", type=float, default=0.90,
                   help="Threshold quantile for the WGS values in the binary overlap comparison")
    p.add_argument("--compare-rebin-to", type=int, default=None,
                   help="Optional resolution in bp to harmonize both tracks before computing the metric")
    p.add_argument("--cn-floor", type=float, default=3.0,
                   help="Absolute copy-number floor for 'high' bins: if both WES and WGS 0.9 quantiles are below this value, use this threshold instead (default: 3.0)")
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

    if args.rebin_to is not None:
        df_for_rebin = df[["chrom", "start", "value_raw"]].copy()
        df = rebin_to_resolution(df_for_rebin, "value_raw", args.rebin_to)
        print(f"Rebinned to {args.rebin_to}bp resolution: {len(df)} bins "
              f"(mean of constituent fine bins).")

    smooth_window = args.smooth_window
    if smooth_window is None:
        smooth_window = 1 if args.rebin_to is not None else 5
        print(f"Using smooth-window={smooth_window} "
              f"({'no extra smoothing after rebinning' if args.rebin_to is not None else 'default'})")

    df = smooth_per_chrom(df, "value_raw", smooth_window)
    df["masked"] = False

    # Apply mask regions before thresholding/fallback decisions so recurrent bins
    # do not contribute to the CN-floor check or to the effective quantile.
    if args.mask_regions is not None:
        mask_df = load_mask_regions(args.mask_regions)
        masked_count = 0
        for _, mrow in mask_df.iterrows():
            mch = str(mrow["chrom"])
            mstart = int(mrow["start"])
            mend = int(mrow["end"])
            sel = (df["chrom"] == mch) & (df["start"] >= mstart) & (df["start"] < mend)
            if sel.any():
                df.loc[sel, "masked"] = True
                masked_count += sel.sum()
        print(f"Applied mask regions from {args.mask_regions}: marked {int(masked_count)} bins as masked (non-amplified)")

    compare_wgs_threshold = None
    cn_floor_threshold = None

    if args.compare_wgs_input is not None:
        compare_path = Path(args.compare_wgs_input)
        if compare_path.exists():
            compare_df = load_data(compare_path)
            compare_col = args.compare_wgs_column or args.column
            if compare_col not in compare_df.columns:
                raise ValueError(
                    f"Comparison WGS column '{compare_col}' not found in {compare_path}. "
                    f"Available columns: {list(compare_df.columns)}"
                )
            if args.bin_col:
                wgs_df = split_bin_column(compare_df.copy(), args.bin_col)
            else:
                if args.chrom_col not in compare_df.columns or args.start_col not in compare_df.columns:
                    raise ValueError(
                        f"Comparison WGS input must have '{args.chrom_col}' and '{args.start_col}' columns "
                        f"or a combined '{args.bin_col}' column when --compare-wgs-input is used."
                    )
                wgs_df = compare_df.rename(columns={args.chrom_col: "chrom", args.start_col: "start"})
            wgs_df["chrom"] = wgs_df["chrom"].astype(str)
            wgs_df["value_raw"] = wgs_df[compare_col].astype(float)
            wgs_df = wgs_df.dropna(subset=["value_raw", "start"]).sort_values(["chrom", "start"]).reset_index(drop=True)
            if args.compare_rebin_to is not None:
                wgs_df = rebin_to_resolution(wgs_df[["chrom", "start", "value_raw"]].copy(), "value_raw", args.compare_rebin_to)
            wgs_q = float(smooth_per_chrom(wgs_df[["chrom", "start", "value_raw"]].copy(), "value_raw", smooth_window)["smoothed"].quantile(args.compare_wgs_quantile))
            wes_q = float(df.loc[~df["masked"], "smoothed"].quantile(args.compare_wes_quantile)) if df["masked"].any() else float(df["smoothed"].quantile(args.compare_wes_quantile))
            if args.cn_floor is not None and wes_q < args.cn_floor and wgs_q < args.cn_floor:
                cn_floor_threshold = float(args.cn_floor)
                compare_wgs_threshold = float(args.cn_floor)
                print(f"CN floor applied: both WES and WGS 0.9 quantiles are below {args.cn_floor}; setting high threshold to {compare_wgs_threshold:.2f} CN")

    if cn_floor_threshold is not None:
        df, threshold = call_high_bins(df, args.quantile, cn_floor=cn_floor_threshold)
    else:
        df, threshold = call_high_bins(df, args.quantile)

    # Ensure masked bins are not considered 'high' after thresholding.
    if args.mask_regions is not None:
        df.loc[df["masked"], "is_high"] = False
    print(f"'High' threshold (quantile {args.quantile}) on smoothed values: {threshold:.4f}")
    print(f"Bins above threshold: {df['is_high'].sum()} / {len(df)}")

    max_gap_by_chrom = resolve_max_gap(df, "value_raw", args.max_gap, args.max_gap_search_lags)
    if args.max_gap == "auto":
        gap_report = ", ".join(f"{c}={g}" for c, g in sorted(max_gap_by_chrom.items()))
        print(f"Auto-computed max-gap per chromosome (from autocorrelation decay): {gap_report}")

    clusters, df_annotated = cluster_bins(df, max_gap_by_chrom, args.min_cluster_size)
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
        if args.rebin_to is not None:
            f.write(f"Rebinned to:           {args.rebin_to} bp (mean aggregation)\n")
        f.write(f"Bins analyzed:         {len(df)}\n")
        f.write(f"Chromosomes/contigs:   {df['chrom'].nunique()}\n")
        f.write(f"Smoothing window:      {smooth_window} bins\n")
        f.write(f"'High' threshold:      {threshold:.4f} (quantile {args.quantile} of smoothed values)\n")
        if args.max_gap == "auto":
            f.write(f"Max gap allowed:       auto (per chromosome, from ACF decay): "
                     f"{dict(sorted(max_gap_by_chrom.items()))}\n")
        else:
            f.write(f"Max gap allowed:       {args.max_gap} bin(s)\n")
        f.write(f"Min cluster size:      {args.min_cluster_size} bins\n")
        f.write(f"Clusters found:        {len(clusters)}\n")
        if df["masked"].any():
            f.write(f"Masked bins:           {int(df['masked'].sum())} (from {args.mask_regions})\n")
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
    if args.compare_wgs_input is not None:
        compare_path = Path(args.compare_wgs_input)
        if not compare_path.exists():
            raise FileNotFoundError(f"Comparison WGS input not found: {compare_path}")
        compare_df = load_data(compare_path)
        compare_col = args.compare_wgs_column or args.column
        if compare_col not in compare_df.columns:
            raise ValueError(
                f"Comparison WGS column '{compare_col}' not found in {compare_path}. "
                f"Available columns: {list(compare_df.columns)}"
            )

        if args.bin_col:
            wgs_df = split_bin_column(compare_df.copy(), args.bin_col)
        else:
            if args.chrom_col not in compare_df.columns or args.start_col not in compare_df.columns:
                raise ValueError(
                    f"Comparison WGS input must have '{args.chrom_col}' and '{args.start_col}' columns "
                    f"or a combined '{args.bin_col}' column when --compare-wgs-input is used."
                )
            wgs_df = compare_df.rename(columns={args.chrom_col: "chrom", args.start_col: "start"})
        wgs_df["chrom"] = wgs_df["chrom"].astype(str)
        wgs_df["value_raw"] = wgs_df[compare_col].astype(float)
        wgs_df = wgs_df.dropna(subset=["value_raw", "start"]).sort_values(["chrom", "start"]).reset_index(drop=True)

        if args.compare_rebin_to is not None:
            wgs_df = rebin_to_resolution(wgs_df[["chrom", "start", "value_raw"]].copy(), "value_raw", args.compare_rebin_to)
            df = rebin_to_resolution(df[["chrom", "start", "value_raw"]].copy(), "value_raw", args.compare_rebin_to)

        metrics = compute_binarized_overlap_metrics(
            df, wgs_df,
            wes_value_col="value_raw",
            wgs_value_col="value_raw",
            wes_quantile=args.compare_wes_quantile,
            wgs_quantile=args.compare_wgs_quantile,
            wes_smooth_window=smooth_window,
            wgs_smooth_window=smooth_window,
            rebin_to=args.compare_rebin_to,
            cn_floor=args.cn_floor,
        )
        metrics_path = outdir / "wgs_wes_overlap_metrics.tsv"
        metric_rows = [{
            "wes_threshold": metrics["wes_threshold"],
            "wgs_threshold": metrics["wgs_threshold"],
            "n_wes_high_bins": metrics["n_wes_high_bins"],
            "n_wgs_high_bins": metrics["n_wgs_high_bins"],
            "n_intersection_bins": metrics["n_intersection_bins"],
            "n_union_bins": metrics["n_union_bins"],
            "jaccard": metrics["jaccard"],
            "overlap_coefficient": metrics["overlap_coefficient"],
        }]
        pd.DataFrame(metric_rows).to_csv(metrics_path, sep='\t', index=False)
        print(f"Binarized WES/WGS overlap metrics saved to {metrics_path}")
        print(
            "Jaccard = {:.4f}, overlap coefficient = {:.4f} "
            "(WES high bins = {}, WGS high bins = {}, shared = {})".format(
                metrics["jaccard"],
                metrics["overlap_coefficient"],
                metrics["n_wes_high_bins"],
                metrics["n_wgs_high_bins"],
                metrics["n_intersection_bins"],
            )
        )

    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
