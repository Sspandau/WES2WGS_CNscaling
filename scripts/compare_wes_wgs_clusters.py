#!/usr/bin/env python3
"""
compare_wes_wgs_clusters.py

Compare WES-derived clusters (<PREFIX>_cluster_results/clusters.csv) with
WGS-derived clusters (<PREFIX>_wgs_cluster_results/clusters.csv) for every
sample under --root, treating WGS as the reference.

Both runs used the same 25 kb grid, so each cluster is expanded to the set of
grid bins it spans (clusters.csv 'end' is the START of the last bin, so a
cluster covers start .. end + bin_size). All overlap is computed on bins.

Optionally pass --mask-regions (the same recurrent-region file used for WES).
Masked bins can never be high in the WES run, so WGS clusters there would
always count as WES misses. Masking removes those bins from BOTH tracks so the
comparison only covers regions WES was allowed to call.

OUTPUT (--outdir)
-----------------
  per_sample_metrics.tsv    one row per sample
  per_cluster_matches.tsv   one row per cluster (both tracks), with overlap
  pooled_summary.txt        totals and medians across samples
  comparison_overview.png   cluster counts and per-sample Jaccard

METRICS (per sample)
--------------------
  bin level:      shared / union (Jaccard), shared / WGS bins (recall),
                  shared / WES bins (precision)
  cluster level:  fraction of WGS clusters hit by WES (recall), fraction of
                  WES clusters hit by WGS (precision), and their F1.
                  A cluster is "hit" if more than --min-overlap-frac of its bins
                  are covered by the other track (default 0 = any overlap).

USAGE
-----
python compare_wes_wgs_clusters.py \
    --root /metropolis/projects/sspandau/depmap/wes/scaled_output \
    --mask-regions /metropolis/projects/sspandau/depmap/recurrent_bins_search/best_combo_recurrent_orange_overlap.csv \
    --outdir /metropolis/projects/sspandau/depmap/wes_vs_wgs_clusters
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WGS_SUFFIX = "_wgs_cluster_results"
WES_SUFFIX = "_cluster_results"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def find_pairs(root):
    """Yield (sample, prefix, wes_csv, wgs_csv) for every WGS results dir."""
    for wgs_dir in sorted(root.rglob(f"*{WGS_SUFFIX}")):
        if not wgs_dir.is_dir():
            continue
        prefix = wgs_dir.name[: -len(WGS_SUFFIX)]
        wes_dir = wgs_dir.parent / f"{prefix}{WES_SUFFIX}"
        sample = wgs_dir.relative_to(root).parts[0]
        yield sample, prefix, wes_dir / "clusters.csv", wgs_dir / "clusters.csv"


def load_clusters(path):
    cols = ["chrom", "start", "end", "n_bins"]
    if not path.is_file():
        return None
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=cols)
    if df.empty:
        return pd.DataFrame(columns=cols)
    df["chrom"] = df["chrom"].astype(str)
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    return df.reset_index(drop=True)


def load_mask_bins(path, bin_size):
    """Grid bins whose start falls in [start, end) of any mask region
    (same rule the clustering script uses to mask bins)."""
    m = pd.read_csv(path, sep=None, engine="python")
    cols = {c.lower(): c for c in m.columns}
    if {"chrom", "start", "end"} <= set(cols):
        m = m.rename(columns={cols["chrom"]: "chrom", cols["start"]: "start", cols["end"]: "end"})
    else:
        m = m.rename(columns=dict(zip(m.columns[:3], ["chrom", "start", "end"])))
    bins = set()
    for chrom, s, e in m[["chrom", "start", "end"]].itertuples(index=False):
        first = -(-int(s) // bin_size) * bin_size   # first grid start >= s
        for b in range(first, int(e), bin_size):
            bins.add((str(chrom), b))
    return bins


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------

def expand(clusters, bin_size, mask):
    """List of bin sets, one per cluster, with masked bins removed.
    Clusters that are entirely masked are dropped."""
    out = []
    for i, row in clusters.iterrows():
        bins = {(row["chrom"], b) for b in range(row["start"], row["end"] + 1, bin_size)}
        bins -= mask
        if bins:
            out.append((i, row, bins))
    return out


def match_clusters(track, own, other_all, min_frac, sample):
    rows, n_hit = [], 0
    for i, row, bins in own:
        frac = len(bins & other_all) / len(bins)
        hit = frac > min_frac
        n_hit += hit
        rows.append({
            "sample": sample, "track": track, "chrom": row["chrom"],
            "start": int(row["start"]), "end": int(row["end"]),
            "n_bins_compared": len(bins),
            "frac_bins_in_other": round(frac, 4), "matched": bool(hit),
        })
    return rows, n_hit


def safe_div(a, b):
    return a / b if b else np.nan


def compare_sample(sample, prefix, wes, wgs, bin_size, mask, min_frac):
    wes_x = expand(wes, bin_size, mask)
    wgs_x = expand(wgs, bin_size, mask)
    wes_all = set().union(*(b for _, _, b in wes_x)) if wes_x else set()
    wgs_all = set().union(*(b for _, _, b in wgs_x)) if wgs_x else set()

    wes_rows, wes_hit = match_clusters("WES", wes_x, wgs_all, min_frac, sample)
    wgs_rows, wgs_hit = match_clusters("WGS", wgs_x, wes_all, min_frac, sample)

    shared = len(wes_all & wgs_all)
    union = len(wes_all | wgs_all)
    recall = safe_div(wgs_hit, len(wgs_x))
    precision = safe_div(wes_hit, len(wes_x))
    f1 = (2 * precision * recall / (precision + recall)
          if np.isfinite(recall) and np.isfinite(precision) and (precision + recall) > 0
          else np.nan)

    metrics = {
        "sample": sample, "prefix": prefix, "status": "ok",
        "n_wes_clusters": len(wes_x), "n_wgs_clusters": len(wgs_x),
        "n_wes_clusters_masked_out": len(wes) - len(wes_x),
        "n_wgs_clusters_masked_out": len(wgs) - len(wgs_x),
        "wes_bins": len(wes_all), "wgs_bins": len(wgs_all),
        "shared_bins": shared, "union_bins": union,
        "bin_jaccard": safe_div(shared, union),   # NaN if neither track has clusters
        "bin_recall": safe_div(shared, len(wgs_all)),
        "bin_precision": safe_div(shared, len(wes_all)),
        "wgs_clusters_hit_by_wes": wgs_hit, "wes_clusters_hit_by_wgs": wes_hit,
        "cluster_recall": recall, "cluster_precision": precision, "cluster_f1": f1,
    }
    return metrics, wes_rows + wgs_rows


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def write_pooled(ok, path, args, n_total):
    tot = ok[["wes_bins", "wgs_bins", "shared_bins", "union_bins",
              "n_wes_clusters", "n_wgs_clusters",
              "wgs_clusters_hit_by_wes", "wes_clusters_hit_by_wgs"]].sum()
    with open(path, "w") as f:
        f.write("WES vs WGS cluster comparison (WGS = reference)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Root:                 {args.root}\n")
        f.write(f"Bin size:             {args.bin_size} bp\n")
        f.write(f"Mask regions:         {args.mask_regions or 'none'}\n")
        f.write(f"Cluster hit rule:     > {args.min_overlap_frac} of bins covered by other track\n")
        f.write(f"Samples compared:     {len(ok)} of {n_total}\n\n")
        f.write("Pooled over all samples:\n")
        f.write(f"  clusters            WES {int(tot.n_wes_clusters)}, WGS {int(tot.n_wgs_clusters)}\n")
        f.write(f"  bin Jaccard         {safe_div(tot.shared_bins, tot.union_bins):.4f}\n")
        f.write(f"  bin recall          {safe_div(tot.shared_bins, tot.wgs_bins):.4f}  (WGS bins also in WES)\n")
        f.write(f"  bin precision       {safe_div(tot.shared_bins, tot.wes_bins):.4f}  (WES bins also in WGS)\n")
        f.write(f"  cluster recall      {safe_div(tot.wgs_clusters_hit_by_wes, tot.n_wgs_clusters):.4f}\n")
        f.write(f"  cluster precision   {safe_div(tot.wes_clusters_hit_by_wgs, tot.n_wes_clusters):.4f}\n\n")
        f.write("Per-sample medians (samples with clusters in the relevant track):\n")
        for col in ["bin_jaccard", "bin_recall", "bin_precision",
                    "cluster_recall", "cluster_precision", "cluster_f1"]:
            f.write(f"  {col:<19} {ok[col].median(skipna=True):.4f}  (n={ok[col].notna().sum()})\n")
        both_empty = ((ok.n_wes_clusters == 0) & (ok.n_wgs_clusters == 0)).sum()
        f.write(f"\nSamples with no clusters in either track: {both_empty}\n")


def plot_overview(ok, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    mx = max(ok["n_wes_clusters"].max(), ok["n_wgs_clusters"].max(), 1)
    a1.scatter(ok["n_wgs_clusters"], ok["n_wes_clusters"], s=14, alpha=0.6)
    a1.plot([0, mx], [0, mx], ls="--", lw=0.8, color="grey")
    a1.set_xlabel("WGS clusters per sample")
    a1.set_ylabel("WES clusters per sample")
    a1.set_title("Cluster counts")

    j = ok["bin_jaccard"].dropna()
    a2.hist(j, bins=np.linspace(0, 1, 21), edgecolor="white")
    a2.axvline(j.median(), color="red", ls="--", lw=1, label=f"median {j.median():.2f}")
    a2.set_xlabel("Bin-level Jaccard (WES vs WGS)")
    a2.set_ylabel("Samples")
    a2.set_title("Per-sample agreement")
    a2.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--bin-size", type=int, default=25000,
                   help="Grid size used in both runs (the --rebin-to value; default 25000)")
    p.add_argument("--mask-regions", default=None,
                   help="Optional recurrent-region file; masked bins are removed from both tracks")
    p.add_argument("--min-overlap-frac", type=float, default=0.0,
                   help="A cluster counts as hit if MORE than this fraction of its bins "
                        "are covered by the other track (default 0 = any overlap)")
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    mask = load_mask_bins(args.mask_regions, args.bin_size) if args.mask_regions else set()
    if mask:
        print(f"Mask: {len(mask)} grid bins removed from both tracks")

    sample_rows, cluster_rows = [], []
    for sample, prefix, wes_csv, wgs_csv in find_pairs(args.root):
        wes, wgs = load_clusters(wes_csv), load_clusters(wgs_csv)
        if wes is None or wgs is None:
            status = "missing_wes" if wes is None else "missing_wgs"
            sample_rows.append({"sample": sample, "prefix": prefix, "status": status})
            continue
        m, c = compare_sample(sample, prefix, wes, wgs, args.bin_size, mask, args.min_overlap_frac)
        sample_rows.append(m)
        cluster_rows.extend(c)

    if not sample_rows:
        raise SystemExit(f"No *{WGS_SUFFIX} directories found under {args.root}")

    per_sample = pd.DataFrame(sample_rows)
    per_sample.to_csv(args.outdir / "per_sample_metrics.tsv", sep="\t", index=False, float_format="%.4f")
    pd.DataFrame(cluster_rows).to_csv(args.outdir / "per_cluster_matches.tsv", sep="\t", index=False)

    ok = per_sample[per_sample["status"] == "ok"]
    skipped = per_sample["status"].value_counts().drop("ok", errors="ignore")
    print(f"Compared {len(ok)} samples" +
          (f"; skipped: {', '.join(f'{k}={v}' for k, v in skipped.items())}" if len(skipped) else ""))
    if ok.empty:
        raise SystemExit("No sample had both WES and WGS clusters.csv")

    write_pooled(ok, args.outdir / "pooled_summary.txt", args, len(per_sample))
    plot_overview(ok, args.outdir / "comparison_overview.png")
    print((args.outdir / "pooled_summary.txt").read_text())
    print(f"Outputs written to {args.outdir}")


if __name__ == "__main__":
    main()
