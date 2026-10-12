#!/usr/bin/env python3
"""
plot_chrom_arm_track_own_ecdna.py

Genomic track of one sample's chromosome arm, at true bin-level precision,
colored by a 4-way overlap category computed directly (not via cluster
bounding-box approximation):

  purple  -- bin overlaps BOTH a recurrent bin AND one of THIS SAMPLE'S OWN
             ecDNA amplicon intervals
  blue    -- bin overlaps a recurrent bin only
  orange  -- bin overlaps this sample's own ecDNA only
  yellow  -- bin is in a cluster, but overlaps neither
  grey    -- background, not part of any cluster

WHY "OWN" ECDNA, NOT THE COHORT-POOLED SET
---------------------------------------------
Earlier analysis in this pipeline used a cohort-pooled ecDNA footprint
(any sample's classified amplicon, from AmpliconClassifier BEDs) checked
against a cluster's whole bounding-box span. That over-counts: a large
cluster can get flagged "ecDNA-overlapping" because some OTHER sample in
the cohort has an amplicon somewhere inside that cluster's span, even
though THIS sample's own signal there has nothing to do with ecDNA. This
script instead takes --own-ecdna-csv -- a small chrom/start/end interval
list specific to the one sample being plotted -- and checks bin-level
overlap directly against just those intervals. This is what settled an
earlier apparent discrepancy: a naive cohort-pooled bin-vs-cluster check
said 149 bins were both ecDNA and recurrent; checking bin-level overlap
against the sample's own real intervals instead gave 6, matching what the
recurrent CSV's own overlaps_orange column said all along.

RECURRENT BIN OVERLAP
------------------------
--recurrent-csv rows (e.g. best_combo_recurrent_orange_overlap.csv from
optimize_recurrent_amplification_threshold_wgstreemodel.py) are already
exact --bin-size-wide bins, so this is exact bin-index (start // bin_size)
matching, not an interval overlap check.

CLUSTER MEMBERSHIP (for the yellow/grey distinction only)
--------------------------------------------------------------
--clusters-csv is this sample's own clusters.csv from
genomic_cluster_upscaleddepth.py. Its 'end' column is the last member
bin's OWN start (not that bin's end coordinate) -- see that script's own
cluster_bins() aggregation -- so member bin starts are
range(start, end + bin_size, bin_size).

OUTPUT (--outdir)
------------------
  <chrom>_<arm>_<sample>_own_ecdna_track.png    full-arm track
  <chrom>_<arm>_<sample>_own_ecdna_track.csv    per-bin data with all flags
  <chrom>_<sample>_own_ecdna_zoom.png           zoomed view around
                                                 --zoom-start/--zoom-end,
                                                 only produced if both are given
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HG38_CENTROMERE_START = {
    "chr1": 123_400_000, "chr2": 93_900_000, "chr3": 90_900_000,
    "chr4": 50_000_000, "chr5": 48_800_000, "chr6": 59_800_000,
    "chr7": 60_100_000, "chr8": 45_200_000, "chr9": 43_000_000,
    "chr10": 39_800_000, "chr11": 53_400_000, "chr12": 35_500_000,
    "chr13": 17_700_000, "chr14": 17_200_000, "chr15": 19_000_000,
    "chr16": 36_800_000, "chr17": 25_100_000, "chr18": 18_500_000,
    "chr19": 26_200_000, "chr20": 28_100_000, "chr21": 12_000_000,
    "chr22": 15_000_000, "chrX": 61_000_000, "chrY": 10_400_000,
}

CATEGORY_COLORS = {
    "recurrent+ecdna": "#7F77DD",
    "recurrent only": "#378ADD",
    "ecdna only": "#EF9F27",
    "neither": "#FAC775",
}
BACKGROUND_COLOR = "#B4B2A9"


def load_copy_ratio_tsv(path, value_col):
    with open(path) as f:
        first_line = f.readline()
    sep = "\t" if "\t" in first_line else r"\s+"
    df = pd.read_csv(path, sep=sep, engine="python")
    required = {"chrom", "start", value_col}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"Error: {path} missing columns {missing}. Found: {list(df.columns)}")
    df["chrom"] = df["chrom"].astype(str)
    df["start"] = df["start"].astype(int)
    df["value_raw"] = df[value_col].astype(float)
    return df.dropna(subset=["value_raw", "start"])


def rebin_mean(df, chrom, bin_size):
    sub = df.loc[df["chrom"] == chrom, ["start", "value_raw"]].copy()
    if sub.empty:
        sys.exit(f"Error: no rows found for chrom={chrom} in the copy-ratio TSV.")
    sub["bin_start"] = (sub["start"] // bin_size) * bin_size
    agg = sub.groupby("bin_start", as_index=False)["value_raw"].mean()
    agg = agg.rename(columns={"bin_start": "start"})
    agg["chrom"] = chrom
    agg["end"] = agg["start"] + bin_size
    return agg.sort_values("start").reset_index(drop=True)


def load_cluster_member_bins(clusters_csv, chrom, bin_size):
    clusters = pd.read_csv(clusters_csv)
    required = {"chrom", "start", "end"}
    missing = required - set(clusters.columns)
    if missing:
        sys.exit(f"Error: {clusters_csv} missing columns {missing}")
    sub = clusters[clusters["chrom"] == chrom]
    member_bins = set()
    for row in sub.itertuples(index=False):
        member_bins.update(range(int(row.start), int(row.end) + bin_size, bin_size))
    print(f"[+] {len(sub):,} clusters on {chrom}, covering {len(member_bins):,} member 25kb bins")
    return member_bins


def load_recurrent_bin_starts(recurrent_csv, chrom):
    recurrent = pd.read_csv(recurrent_csv)
    required = {"chrom", "start"}
    missing = required - set(recurrent.columns)
    if missing:
        sys.exit(f"Error: {recurrent_csv} missing columns {missing}")
    return set(recurrent.loc[recurrent["chrom"] == chrom, "start"])


def load_own_ecdna_intervals(own_ecdna_csv, chrom):
    df = pd.read_csv(own_ecdna_csv)
    required = {"chrom", "start", "end"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"Error: {own_ecdna_csv} missing columns {missing}")
    sub = df[df["chrom"] == chrom]
    print(f"[+] {len(sub):,} own-ecDNA intervals on {chrom}")
    return sub[["start", "end"]].reset_index(drop=True)


def overlaps_intervals_vectorized(bin_starts, bin_ends, interval_starts, interval_ends):
    if len(interval_starts) == 0:
        return np.zeros(len(bin_starts), dtype=bool)
    result = np.zeros(len(bin_starts), dtype=bool)
    for i_start, i_end in zip(interval_starts, interval_ends):
        result |= (bin_starts < i_end) & (bin_ends > i_start)
    return result


def categorize(row):
    if not row["in_cluster"]:
        return "background"
    if row["overlaps_recurrent"] and row["overlaps_own_ecdna"]:
        return "recurrent+ecdna"
    if row["overlaps_recurrent"]:
        return "recurrent only"
    if row["overlaps_own_ecdna"]:
        return "ecdna only"
    return "neither"


def make_legend_handles():
    return [
        Patch(color=CATEGORY_COLORS["recurrent+ecdna"], label="recurrent + ecDNA (own)"),
        Patch(color=CATEGORY_COLORS["recurrent only"], label="recurrent only"),
        Patch(color=CATEGORY_COLORS["ecdna only"], label="ecDNA (own) only"),
        Patch(color=CATEGORY_COLORS["neither"], label="cluster, neither"),
        Patch(color=BACKGROUND_COLOR, label="background"),
    ]


def plot_track(track, bin_size, title, out_path, y_max=None, y_percentile_cap=None, log_y=False,
                xlim_mb=None):
    fig, ax = plt.subplots(figsize=(14, 4))
    bin_width_mb = bin_size / 1e6 * 0.95
    colors = track["category"].map(lambda c: CATEGORY_COLORS.get(c, BACKGROUND_COLOR))

    title_suffix = ""
    if log_y:
        floor = track.loc[track["value_raw"] > 0, "value_raw"].min() / 2 if (track["value_raw"] > 0).any() else 0.1
        plot_vals = track["value_raw"].clip(lower=floor)
        ax.bar(track["start"] / 1e6, plot_vals, width=bin_width_mb, color=colors, align="edge", linewidth=0)
        ax.set_yscale("log")
        title_suffix = " (log y-axis)"
    else:
        ax.bar(track["start"] / 1e6, track["value_raw"], width=bin_width_mb, color=colors,
               align="edge", linewidth=0)
        y_cap = y_max
        if y_percentile_cap is not None:
            y_cap = float(np.percentile(track["value_raw"], y_percentile_cap))
        if y_cap is not None:
            n_clipped = int((track["value_raw"] > y_cap).sum())
            ax.set_ylim(0, y_cap)
            title_suffix = f" (y capped at {y_cap:.0f}, {n_clipped} bins clipped)"

    ax.legend(handles=make_legend_handles(), loc="upper right", fontsize=8, frameon=False, ncol=2)
    ax.set_title(title + title_suffix)
    ax.set_xlabel(f"position (Mb)")
    ax.set_ylabel("value" + (" (log scale)" if log_y else ""))
    if xlim_mb:
        ax.set_xlim(*xlim_mb)
    else:
        ax.set_xlim(track["start"].min() / 1e6, (track["start"].max() + bin_size) / 1e6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--copy-ratio-tsv", required=True)
    p.add_argument("--clusters-csv", required=True,
                   help="This sample's own clusters.csv from genomic_cluster_upscaleddepth.py")
    p.add_argument("--recurrent-csv", required=True,
                   help="best_combo_recurrent_orange_overlap.csv (or any chrom/start/end recurrent-bin list)")
    p.add_argument("--own-ecdna-csv", required=True,
                   help="CSV with columns chrom,start,end -- THIS SAMPLE's own ecDNA amplicon intervals")
    p.add_argument("--sample", required=True, help="Sample label, used in titles/filenames")
    p.add_argument("--chrom", default="chr1")
    p.add_argument("--arm", choices=["p", "q"], default="p")
    p.add_argument("--arm-end", type=int, default=None)
    p.add_argument("--value-col", default="predicted_loess_upscale_depth")
    p.add_argument("--rebin-to", type=int, default=25000)
    p.add_argument("--y-max", type=float, default=None)
    p.add_argument("--y-percentile-cap", type=float, default=None)
    p.add_argument("--log-y", action="store_true")
    p.add_argument("--zoom-start", type=int, default=None, help="bp start of an optional zoomed-in plot")
    p.add_argument("--zoom-end", type=int, default=None, help="bp end of an optional zoomed-in plot")
    p.add_argument("--outdir", default="chrom_arm_track_own_ecdna")
    args = p.parse_args()

    if args.log_y and (args.y_max is not None or args.y_percentile_cap is not None):
        sys.exit("Error: --log-y is mutually exclusive with --y-max/--y-percentile-cap.")
    if args.y_max is not None and args.y_percentile_cap is not None:
        sys.exit("Error: pass only one of --y-max or --y-percentile-cap.")
    if (args.zoom_start is None) != (args.zoom_end is None):
        sys.exit("Error: pass both --zoom-start and --zoom-end, or neither.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    arm_end = args.arm_end
    if arm_end is None:
        arm_end = HG38_CENTROMERE_START.get(args.chrom)
        if arm_end is None:
            sys.exit(f"Error: no default --arm-end known for {args.chrom}; pass --arm-end explicitly.")
        print(f"[warn] Using approximate hg38 --arm-end={arm_end:,} for {args.chrom} -- verify against "
              f"your reference build. Pass --arm-end to override.")

    df = load_copy_ratio_tsv(args.copy_ratio_tsv, args.value_col)
    rebinned = rebin_mean(df, args.chrom, args.rebin_to)
    if args.arm == "p":
        track = rebinned[rebinned["start"] < arm_end].copy()
    else:
        track = rebinned[rebinned["start"] >= arm_end].copy()
    if track.empty:
        sys.exit(f"Error: no bins found for {args.chrom} {args.arm} arm with --arm-end={arm_end:,}.")

    member_bins = load_cluster_member_bins(args.clusters_csv, args.chrom, args.rebin_to)
    recurrent_starts = load_recurrent_bin_starts(args.recurrent_csv, args.chrom)
    own_ecdna = load_own_ecdna_intervals(args.own_ecdna_csv, args.chrom)

    track["in_cluster"] = track["start"].isin(member_bins)
    track["overlaps_recurrent"] = track["start"].isin(recurrent_starts)
    track["overlaps_own_ecdna"] = overlaps_intervals_vectorized(
        track["start"].values, track["end"].values, own_ecdna["start"].values, own_ecdna["end"].values)
    track["category"] = track.apply(categorize, axis=1)

    n_ecdna = int(track["overlaps_own_ecdna"].sum())
    n_ecdna_and_recurrent = int((track["overlaps_own_ecdna"] & track["overlaps_recurrent"]).sum())
    print(f"[+] {args.chrom} {args.arm} arm: {len(track):,} bins in view")
    print(f"[+] bins overlapping own ecDNA: {n_ecdna} ({n_ecdna_and_recurrent} also recurrent, "
          f"{n_ecdna - n_ecdna_and_recurrent} ecDNA-only)")
    print(track["category"].value_counts().to_string())

    csv_path = outdir / f"{args.chrom}_{args.arm}_{args.sample}_own_ecdna_track.csv"
    track.to_csv(csv_path, index=False)
    print(f"[+] Saved: {csv_path}")

    png_path = outdir / f"{args.chrom}_{args.arm}_{args.sample}_own_ecdna_track.png"
    plot_track(track, args.rebin_to, f"{args.sample}: {args.chrom} {args.arm} arm -- {args.value_col}",
               png_path, args.y_max, args.y_percentile_cap, args.log_y)
    print(f"[+] Saved: {png_path}")

    if args.zoom_start is not None:
        zoom = track[(track["start"] >= args.zoom_start) & (track["start"] <= args.zoom_end)]
        if zoom.empty:
            print(f"[warn] no bins in zoom range {args.zoom_start:,}-{args.zoom_end:,} -- skipping zoom plot")
        else:
            zoom_png_path = outdir / f"{args.chrom}_{args.sample}_own_ecdna_zoom.png"
            plot_track(zoom, args.rebin_to,
                       f"{args.sample}: {args.chrom} {args.zoom_start/1e6:.2f}-{args.zoom_end/1e6:.2f} Mb "
                       f"(zoom) -- {args.value_col}",
                       zoom_png_path, args.y_max, args.y_percentile_cap, args.log_y,
                       xlim_mb=(args.zoom_start / 1e6, args.zoom_end / 1e6))
            print(f"[+] Saved: {zoom_png_path}")


if __name__ == "__main__":
    main()
