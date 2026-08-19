#!/usr/bin/env python3
"""
plot_chrom_arm_track.py

Genomic track (bar plot) of one chromosome arm's 25kb-rebinned predicted
depth, with bars colored orange if that 25kb bin belongs to a cluster
(per clusters.csv from genomic_cluster_upscaleddepth.py), grey otherwise.

Rebinning matches genomic_cluster_upscaleddepth.py's own --rebin-to
exactly: 5kb bins -> 25kb bins by taking the MEAN of the value column
within each 25kb bin (floor(start / bin_size) * bin_size). No masking is
applied here (mask_rejected or similar), matching that script's own
input handling -- it doesn't filter on any mask column either, only
dropna on the value column and start.

CLUSTER MEMBERSHIP
--------------------
clusters.csv's 'end' column is NOT the last bin's end coordinate -- per
genomic_cluster_upscaleddepth.py's own aggregation (end=("start","max")),
it's the last member bin's OWN start. So a cluster's full set of member
25kb-bin starts is range(cluster.start, cluster.end + bin_size, bin_size)
-- this correctly includes bridged/filler bins the clustering step
merged in via morphological closing, not just the bins that were
individually above threshold.

CHROMOSOME ARM BOUNDARY
--------------------------
--arm-end is the bp position marking the end of the p arm / start of the
centromeric region. THIS DEFAULT IS A ROUGH hg38 APPROXIMATION and you
should confirm it matches your reference build before trusting the plot
boundary -- pass --arm-end explicitly if you know the exact value used
elsewhere in your pipeline (e.g. from your reference FASTA's centromere
annotation or AA's data repo).

OUTPUT (--outdir)
------------------
  <chrom>_<arm>_track.png    the genomic track plot
  <chrom>_<arm>_track.csv    the exact per-bin data that was plotted
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Rough hg38 centromere start positions (bp), for --arm-end defaults.
# APPROXIMATE -- verify against your own reference build before trusting
# the arm boundary in the plot.
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
    df = df.dropna(subset=["value_raw", "start"])
    return df


def rebin_mean(df, chrom, bin_size):
    sub = df.loc[df["chrom"] == chrom, ["start", "value_raw"]].copy()
    if sub.empty:
        sys.exit(f"Error: no rows found for chrom={chrom} in the copy-ratio TSV.")
    sub["bin_start"] = (sub["start"] // bin_size) * bin_size
    agg = sub.groupby("bin_start", as_index=False)["value_raw"].mean()
    agg = agg.rename(columns={"bin_start": "start"})
    agg["chrom"] = chrom
    return agg.sort_values("start").reset_index(drop=True)


def load_clustered_bin_starts(clusters_csv, chrom, bin_size):
    clusters = pd.read_csv(clusters_csv)
    required = {"chrom", "start", "end"}
    missing = required - set(clusters.columns)
    if missing:
        sys.exit(f"Error: {clusters_csv} missing columns {missing}")
    sub = clusters[clusters["chrom"] == chrom]
    clustered = set()
    for row in sub.itertuples(index=False):
        clustered.update(range(int(row.start), int(row.end) + bin_size, bin_size))
    print(f"[+] {len(sub):,} clusters on {chrom}, covering {len(clustered):,} member 25kb bins "
          f"(including bridged/filler bins)")
    return clustered


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--copy-ratio-tsv", required=True,
                   help="Sample's *_off_target_copy_ratios.tsv (5kb native resolution)")
    p.add_argument("--clusters-csv", required=True,
                   help="clusters.csv from genomic_cluster_upscaleddepth.py for this same sample")
    p.add_argument("--chrom", default="chr1")
    p.add_argument("--arm", choices=["p", "q"], default="p")
    p.add_argument("--arm-end", type=int, default=None,
                   help="bp position of the p/q arm boundary (centromere start). Default: a rough "
                        "hg38 approximation for --chrom -- VERIFY against your reference build.")
    p.add_argument("--value-col", default="predicted_loess_upscale_depth")
    p.add_argument("--rebin-to", type=int, default=25000)
    p.add_argument("--sample-label", default=None, help="Label for the plot title, e.g. NCIH889_LUNG")
    p.add_argument("--outdir", default="chrom_arm_track")
    args = p.parse_args()

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
        sys.exit(f"Error: no bins found for {args.chrom} {args.arm} arm with --arm-end={arm_end:,} -- "
                  f"check --arm-end is correct and on the same side as intended.")

    clustered_starts = load_clustered_bin_starts(args.clusters_csv, args.chrom, args.rebin_to)
    track["in_cluster"] = track["start"].isin(clustered_starts)

    n_clustered = int(track["in_cluster"].sum())
    print(f"[+] {args.chrom} {args.arm} arm: {len(track):,} bins in view, "
          f"{n_clustered:,} ({n_clustered/len(track):.1%}) belong to a cluster")

    csv_path = outdir / f"{args.chrom}_{args.arm}_track.csv"
    track.to_csv(csv_path, index=False)
    print(f"[+] Saved: {csv_path}")

    fig, ax = plt.subplots(figsize=(14, 4))
    bin_width_mb = args.rebin_to / 1e6 * 0.95
    colors = np.where(track["in_cluster"], "#D85A30", "#B4B2A9")
    ax.bar(track["start"] / 1e6, track["value_raw"], width=bin_width_mb, color=colors,
           align="edge", linewidth=0)

    from matplotlib.patches import Patch
    legend_handles = [Patch(color="#D85A30", label="in cluster"), Patch(color="#B4B2A9", label="background")]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9, frameon=False)

    title = f"{args.chrom} {args.arm} arm -- {args.value_col}"
    if args.sample_label:
        title = f"{args.sample_label}: {title}"
    ax.set_title(title)
    ax.set_xlabel(f"{args.chrom} position (Mb)")
    ax.set_ylabel(args.value_col)
    ax.set_xlim(track["start"].min() / 1e6, (track["start"].max() + args.rebin_to) / 1e6)
    fig.tight_layout()

    png_path = outdir / f"{args.chrom}_{args.arm}_track.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"[+] Saved: {png_path}")


if __name__ == "__main__":
    main()
