#!/usr/bin/env python3
"""
analyze_cluster_tightness_vs_recurrent_orange.py

For every sample's clusters.csv (legacy per-sample clustering pipeline:
<clusters-root>/<sample>/<srr>_cluster_results/clusters.csv), computes a
cluster "tightness" ratio and checks overlap against BOTH:
  - RECURRENT locations, from a best_combo_recurrent_orange_overlap.csv
    (an optimize_recurrent_amplification_threshold_wgstreemodel.py run's
    winning-combo output -- only the chrom/start/end/n_samples_high
    columns are used; its own overlaps_orange column is ignored, see
    below)
  - ORANGE (ecDNA+/BFB+/CNC) locations, computed FRESH and DIRECTLY from
    the AmpliconClassifier outputs (classification TSV + BED dir), using
    the SAME functions optimize_recurrent_amplification_threshold_
    wgstreemodel.py itself uses (load_qualifying_amplicons_by_sample /
    index_classification_bed_dir / load_orange_intervals_for_sample),
    imported directly from that script so this can never drift out of
    sync with its own orange definition.

WHY ORANGE IS COMPUTED FRESH HERE, NOT READ FROM best_combo_recurrent_
orange_overlap.csv
-----------------------------------------------------------------------
That CSV only lists RECURRENT locations, each flagged with whether it's
ALSO orange -- by construction it can never contain an orange location
that ISN'T recurrent. That made "orange but not recurrent" structurally
invisible. Reading the classification BEDs directly here instead gives
the TRUE genome-wide orange footprint, independent of whether any given
orange region ever got voted recurrent -- so "orange but not recurrent"
is now a real, populated category. This also means orange here is NOT
gated by whatever depth-data coverage the optimize script's cache had
(--cv-csv sparsity, --wes-root filtering, etc.) -- it's the raw AC BED
footprint, full stop.

TIGHTNESS RATIO (cluster_size / max_gap_size)
------------------------------------------------
clusters.csv has n_bins (bins actually included in the cluster) and the
cluster's genomic span (end - start), but NOT individual bin positions or
a literal "max single gap". From n_bins and (end - start) at a fixed
--bin-size (default 25000 = 25kb), the number of "missing" bin slots
across the whole cluster span is

    n_gap_bins_estimate = round((end - start) / bin_size) - n_bins

and tightness_ratio = n_bins / n_gap_bins_estimate. This is a genome-span
estimate of TOTAL missing coverage, not the true single largest gap (not
recoverable from clusters.csv alone) -- a cluster with one huge gap and a
cluster with the same total gap spread over many small gaps get the same
estimate. n_gap_bins_estimate <= 0 (fully contiguous, or bin-boundary
rounding) gets tightness_ratio = inf.

OVERLAP DETECTION
-------------------
- Recurrent: best_combo_recurrent_orange_overlap.csv rows are already
  exact --bin-size-wide bins (start/end differ by exactly --bin-size), so
  this is done in fast bin-index (start // bin_size) set-membership space.
- Orange: raw AC BED intervals have ARBITRARY bp boundaries, not aligned
  to any bin grid, so this is done via genuine interval overlap
  (orange_start < cluster_end AND orange_end > cluster_start) per
  chromosome, vectorized with numpy.

CATEGORIES (now a true 4-way split)
--------------------------------------
  "orange+recurrent"  -- overlaps both
  "orange only"        -- overlaps orange, never recurrent  (NEW -- see above)
  "recurrent only"      -- overlaps recurrent, never orange
  "neither"

OUTPUT (--outdir)
------------------
  all_clusters_annotated.csv         every cluster, every sample, with
                                      tightness_ratio + overlap flags/category
  overlap_summary.txt                counts/percentages, top/bottom
                                      clusters by tightness_ratio, stats
  tightness_hist_recurrent_overlap.png
  tightness_hist_no_recurrent_overlap.png
  tightness_hist_orange_overlap.png
  tightness_hist_no_orange_overlap.png
                                      separate tightness_ratio histograms
  tightness_by_category_boxplot.png  tightness_ratio grouped by all 4 categories
  size_by_category_boxplot.png       n_bins (cluster size) grouped by all 4 categories
  size_vs_gap_recurrent_overlap.png
  size_vs_gap_no_recurrent_overlap.png
                                      separate n_bins-vs-gap_bins scatter plots
"""

import sys
import importlib.util
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


def load_module(path, name):
    path = Path(path)
    if not path.exists():
        sys.exit(f"Error: {name} not found at {path}.")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def discover_cluster_csvs(clusters_root, glob_pattern):
    paths = sorted(Path(clusters_root).glob(glob_pattern))
    print(f"[+] Found {len(paths)} clusters.csv files under {clusters_root} "
          f"(pattern: {glob_pattern})")
    return paths


def load_all_clusters(cluster_paths):
    frames = []
    for path in cluster_paths:
        srr_dir = path.parent.name
        srr = srr_dir[:-len("_cluster_results")] if srr_dir.endswith("_cluster_results") else srr_dir
        sample = path.parent.parent.name
        df = pd.read_csv(path)
        required = {"chrom", "start", "end", "n_bins", "n_high_bins", "mean_value", "max_value"}
        missing = required - set(df.columns)
        if missing:
            print(f"    [skip] {path}: missing columns {missing}")
            continue
        df = df.copy()
        df["sample"] = sample
        df["srr"] = srr
        frames.append(df)
    if not frames:
        sys.exit("Error: no usable clusters.csv files found.")
    all_df = pd.concat(frames, ignore_index=True)
    print(f"[+] Loaded {len(all_df):,} clusters across {all_df['sample'].nunique()} samples")
    return all_df


def load_recurrent_csv(path):
    df = pd.read_csv(path)
    required = {"chrom", "start", "end"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"Error: {path} missing required columns: {missing}")
    return df


def build_recurrent_bin_sets(recurrent_df, bin_size):
    out = {}
    for chrom, grp in recurrent_df.groupby("chrom"):
        out[chrom] = set((grp["start"].values // bin_size).astype(np.int64))
    return out


def load_full_orange_intervals(opt, base, classification_tsv, classification_bed_dir, categories):
    """Full genome-wide orange footprint, pooled across ALL samples in the
    classification TSV, computed directly from the AC BED files -- NOT
    gated by any depth-data source. Returns {chrom: (starts_array, ends_array)}."""
    qualifying_by_sample = opt.load_qualifying_amplicons_by_sample(classification_tsv, categories)
    bed_index = opt.index_classification_bed_dir(classification_bed_dir)

    frames = []
    for sample_name, qualifying in qualifying_by_sample.items():
        if not qualifying:
            continue
        intervals = opt.load_orange_intervals_for_sample(sample_name, qualifying, bed_index, base)
        if not intervals.empty:
            frames.append(intervals)
    if not frames:
        print(f"[warn] 0 orange intervals found for categories={categories} -- check "
              f"--classification-tsv/--classification-bed-dir and --orange-categories.")
        return {}

    pooled = pd.concat(frames, ignore_index=True).drop_duplicates()
    print(f"[+] Loaded {len(pooled):,} raw orange BED intervals (categories={categories}, "
          f"before merging across samples/features)")
    out = {}
    for chrom, grp in pooled.groupby("chrom"):
        order = np.argsort(grp["start"].values)
        out[chrom] = (grp["start"].values[order], grp["end"].values[order])
    return out


def cluster_bin_indices(start, end, bin_size):
    lo = int(start) // bin_size
    hi = int(np.ceil(end / bin_size))
    return range(lo, hi)


def overlaps_any_interval(cluster_start, cluster_end, starts, ends):
    if starts is None or len(starts) == 0:
        return False
    return bool(np.any((starts < cluster_end) & (ends > cluster_start)))


def annotate_clusters(all_df, recurrent_bins, orange_all_intervals, orange_ecdna_intervals, bin_size):
    n_gap_bins_estimate = all_df["n_bins"]-all_df["n_high_bins"]
    n_gap_bins_estimate = n_gap_bins_estimate.clip(lower=0)
    tightness_ratio = np.where(
        n_gap_bins_estimate > 0, all_df["n_bins"] / n_gap_bins_estimate.replace(0, np.nan), np.inf
    )

    overlaps_recurrent, overlaps_orange_all, overlaps_orange_ecdna = [], [], []
    for row in all_df.itertuples(index=False):
        idx_set = set(cluster_bin_indices(row.start, row.end, bin_size))
        rec = bool(idx_set & recurrent_bins.get(row.chrom, set()))

        starts_all, ends_all = orange_all_intervals.get(row.chrom, (None, None))
        org_all = overlaps_any_interval(row.start, row.end, starts_all, ends_all)

        starts_ecd, ends_ecd = orange_ecdna_intervals.get(row.chrom, (None, None))
        org_ecd = overlaps_any_interval(row.start, row.end, starts_ecd, ends_ecd)

        overlaps_recurrent.append(rec)
        overlaps_orange_all.append(org_all)
        overlaps_orange_ecdna.append(org_ecd)

    out = all_df.copy()
    out["n_gap_bins_estimate"] = n_gap_bins_estimate.values
    out["tightness_ratio"] = tightness_ratio
    out["overlaps_recurrent"] = overlaps_recurrent
    out["overlaps_orange_all"] = overlaps_orange_all
    out["overlaps_orange_ecdna"] = overlaps_orange_ecdna

    def categorize(r):
        if r.overlaps_orange_all and r.overlaps_recurrent:
            return "orange+recurrent"
        if r.overlaps_orange_all:
            return "orange only"
        if r.overlaps_recurrent:
            return "recurrent only"
        return "neither"
    out["category"] = out.apply(categorize, axis=1)
    return out


def plot_tightness_hist(df, title, out_path):
    vals = df["tightness_ratio"].replace(np.inf, np.nan).dropna()
    n_inf = int(np.isinf(df["tightness_ratio"]).sum())
    fig, ax = plt.subplots(figsize=(7, 5))
    if len(vals):
        bins = np.logspace(np.log10(max(vals.min(), 1e-3)), np.log10(vals.max()), 40)
        ax.hist(vals, bins=bins, color="#4C72B0", alpha=0.8)
        ax.set_xscale("log")
    ax.set_xlabel("tightness_ratio (n_bins / n_gap_bins_estimate)")
    ax.set_ylabel("count")
    ax.set_title(f"{title}\n(n={len(df):,}, {n_inf:,} fully-contiguous/inf excluded from plot)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_size_vs_gap(df, title, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    # Consistent across both size_vs_gap plots regardless of which
    # categories happen to be present: any category involving orange gets
    # the same orange, anything not involving orange gets the same blue --
    # so "orange only" and "orange+recurrent" always read as one color
    # family, and "recurrent only"/"neither" always read as the other.
    colors = {"orange+recurrent": "#cc7a00", "orange only": "#cc7a00",
              "recurrent only": "#4C72B0", "neither": "#4C72B0"}
    for cat, sub in df.groupby("category"):
        ax.scatter(sub["n_gap_bins_estimate"] + 0.5, sub["n_bins"], s=10, alpha=0.5,
                   label=f"{cat} (n={len(sub):,})", color=colors.get(cat, "black"))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n_gap_bins_estimate (+0.5, log scale)")
    ax.set_ylabel("n_bins (cluster size, log scale)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metric_by_category(df, metric_col, ylabel, title, out_path, log_scale=True, annotate_inf=False):
    """General-purpose boxplot of any numeric column across the 4 overlap
    categories, with jittered points. Used for both tightness_ratio and
    n_bins (cluster size).

    annotate_inf: when the metric can be inf (tightness_ratio, for a
    fully-contiguous cluster with n_gap_bins_estimate == 0), those rows
    get dropped before boxplot() sees them (a box can't render an infinite
    value) -- if a category is ALL inf, its box silently disappears even
    though the category has members. Rather than clip inf to some
    arbitrary finite value, this reports the inf count/fraction directly
    in the tick label, since "100% fully contiguous" is itself the
    informative result for that category, not a plotting inconvenience to
    paper over."""
    cats = ["orange+recurrent", "orange only", "recurrent only", "neither"]
    is_inf = np.isinf(df[metric_col]) if log_scale else pd.Series(False, index=df.index)
    series = df[metric_col].replace(np.inf, np.nan) if log_scale else df[metric_col]
    data = [series[df["category"] == c].dropna().values for c in cats]

    labels = []
    for c in cats:
        n_cat = int((df["category"] == c).sum())
        label = f"{c}\n(n={n_cat:,})"
        if annotate_inf and n_cat:
            n_inf = int(is_inf[df["category"] == c].sum())
            if n_inf:
                label += f"\n({n_inf:,} inf, {n_inf/n_cat:.0%})"
        labels.append(label)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    for i, d in enumerate(data, start=1):
        if len(d):
            n_show = min(len(d), 2000)
            sample_d = np.random.default_rng(0).choice(d, size=n_show, replace=False)
            jitter = np.random.default_rng(0).normal(i, 0.05, size=n_show)
            ax.scatter(jitter, sample_d, s=4, alpha=0.15, color="black")
    if log_scale:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel + (" (log scale)" if log_scale else ""))
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clusters-root", required=True)
    p.add_argument("--cluster-glob", default="*/*_cluster_results/clusters.csv")
    p.add_argument("--recurrent-csv", required=True,
                   help="best_combo_recurrent_orange_overlap.csv -- only chrom/start/end used "
                        "(its own overlaps_orange column is ignored; orange is computed fresh, see "
                        "module docstring)")
    p.add_argument("--optimize-script", required=True,
                   help="Path to optimize_recurrent_amplification_threshold_wgstreemodel.py")
    p.add_argument("--base-script", required=True,
                   help="Path to find_recurrent_novel_amplifications_binlevel.py")
    p.add_argument("--classification-tsv", required=True)
    p.add_argument("--classification-bed-dir", required=True)
    p.add_argument("--orange-categories", default="ecDNA,BFB,CNC")
    p.add_argument("--ecdna-categories", default="ecDNA",
                   help="Category set for the ecDNA-specific subset flag. Default: ecDNA only.")
    p.add_argument("--bin-size", type=int, default=25000)
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--outdir", default="cluster_tightness_analysis")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    opt = load_module(args.optimize_script, "optimize_mod")
    base = opt.load_base_module(args.base_script)

    cluster_paths = discover_cluster_csvs(args.clusters_root, args.cluster_glob)
    all_df = load_all_clusters(cluster_paths)

    recurrent_df = load_recurrent_csv(args.recurrent_csv)
    recurrent_bins = build_recurrent_bin_sets(recurrent_df, args.bin_size)
    print(f"[+] {len(recurrent_df):,} recurrent bin locations loaded from {args.recurrent_csv}")

    orange_all_intervals = load_full_orange_intervals(
        opt, base, args.classification_tsv, args.classification_bed_dir,
        tuple(args.orange_categories.split(",")))
    orange_ecdna_intervals = load_full_orange_intervals(
        opt, base, args.classification_tsv, args.classification_bed_dir,
        tuple(args.ecdna_categories.split(",")))

    annotated = annotate_clusters(all_df, recurrent_bins, orange_all_intervals, orange_ecdna_intervals, args.bin_size)
    out_csv = outdir / "all_clusters_annotated.csv"
    annotated.to_csv(out_csv, index=False)
    print(f"[+] Saved: {out_csv}")

    # --- summary ------------------------------------------------------------
    n_total = len(annotated)
    cat_counts = annotated["category"].value_counts()
    n_ecdna = int(annotated["overlaps_orange_ecdna"].sum())

    rec_ratios = annotated.loc[annotated["overlaps_recurrent"], "tightness_ratio"].replace(np.inf, np.nan).dropna()
    non_rec_ratios = annotated.loc[~annotated["overlaps_recurrent"], "tightness_ratio"].replace(np.inf, np.nan).dropna()
    mw_rec_stat, mw_rec_p = (stats.mannwhitneyu(rec_ratios, non_rec_ratios, alternative="two-sided")
                              if len(rec_ratios) >= 5 and len(non_rec_ratios) >= 5 else (np.nan, np.nan))

    org_ratios = annotated.loc[annotated["overlaps_orange_all"], "tightness_ratio"].replace(np.inf, np.nan).dropna()
    non_org_ratios = annotated.loc[~annotated["overlaps_orange_all"], "tightness_ratio"].replace(np.inf, np.nan).dropna()
    mw_org_stat, mw_org_p = (stats.mannwhitneyu(org_ratios, non_org_ratios, alternative="two-sided")
                              if len(org_ratios) >= 5 and len(non_org_ratios) >= 5 else (np.nan, np.nan))

    cat_groups = [annotated.loc[annotated["category"] == c, "tightness_ratio"].replace(np.inf, np.nan).dropna()
                  for c in ["orange+recurrent", "orange only", "recurrent only", "neither"]]
    cat_groups = [g for g in cat_groups if len(g) >= 5]
    kw_stat, kw_p = stats.kruskal(*cat_groups) if len(cat_groups) >= 2 else (np.nan, np.nan)

    size_cat_groups = [annotated.loc[annotated["category"] == c, "n_bins"]
                        for c in ["orange+recurrent", "orange only", "recurrent only", "neither"]]
    size_cat_groups = [g for g in size_cat_groups if len(g) >= 5]
    kw_size_stat, kw_size_p = stats.kruskal(*size_cat_groups) if len(size_cat_groups) >= 2 else (np.nan, np.nan)

    loosest = annotated.replace({np.inf: np.nan}).dropna(subset=["tightness_ratio"]) \
        .sort_values("tightness_ratio").head(args.top_n)

    summary_path = outdir / "overlap_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Cluster tightness vs recurrent/orange overlap\n" + "=" * 47 + "\n\n")
        f.write(f"Total clusters: {n_total:,} across {annotated['sample'].nunique()} samples\n\n")
        f.write("Category counts (4-way: orange is the FULL AC footprint, not gated by recurrence):\n")
        for cat in ["orange+recurrent", "orange only", "recurrent only", "neither"]:
            n = int(cat_counts.get(cat, 0))
            f.write(f"  {cat:20s} {n:6,}  ({n/n_total:.1%})\n")
        f.write(f"\nOf which overlap the ecDNA-specific subset (overlaps_orange_ecdna): {n_ecdna:,} "
                f"({n_ecdna/n_total:.1%})\n\n")
        f.write("Fully-contiguous clusters per category (tightness_ratio == inf, i.e. "
                 "n_gap_bins_estimate == 0 -- excluded from the boxplot/Mann-Whitney/Kruskal-Wallis "
                 "below since a box can't render an infinite value; reported here instead since "
                 "'100% fully contiguous' is itself the informative result for a category, not just "
                 "a plotting gap):\n")
        for cat in ["orange+recurrent", "orange only", "recurrent only", "neither"]:
            n_cat = int(cat_counts.get(cat, 0))
            n_inf = int(np.isinf(annotated.loc[annotated["category"] == cat, "tightness_ratio"]).sum())
            frac = f"{n_inf/n_cat:.1%}" if n_cat else "n/a"
            f.write(f"  {cat:20s} {n_inf:6,} / {n_cat:6,}  ({frac})\n")
        f.write("\n")
        f.write("Mann-Whitney U, tightness_ratio: overlaps_recurrent vs not:\n")
        f.write(f"  statistic={mw_rec_stat}, p={mw_rec_p}\n\n")
        f.write("Mann-Whitney U, tightness_ratio: overlaps_orange vs not:\n")
        f.write(f"  statistic={mw_org_stat}, p={mw_org_p}\n\n")
        f.write("Kruskal-Wallis, tightness_ratio across 4 categories:\n")
        f.write(f"  statistic={kw_stat}, p={kw_p}\n\n")
        f.write("Kruskal-Wallis, n_bins (cluster size) across 4 categories:\n")
        f.write(f"  statistic={kw_size_stat}, p={kw_size_p}\n\n")
        f.write(f"Top {args.top_n} loosest clusters (smallest tightness_ratio):\n")
        f.write(loosest[["sample", "chrom", "start", "end", "n_bins", "n_gap_bins_estimate",
                          "tightness_ratio", "category"]].to_string(index=False))
        f.write("\n")
    print(f"[+] Saved: {summary_path}")
    print(cat_counts.to_string())
    print(f"Mann-Whitney (recurrent vs not): p={mw_rec_p}")
    print(f"Mann-Whitney (orange vs not):    p={mw_org_p}")
    print(f"Kruskal-Wallis (4 categories):   p={kw_p}")
    print(f"Kruskal-Wallis, n_bins (4 categories): p={kw_size_p}")

    # --- plots ---------------------------------------------------------------
    rec_df = annotated[annotated["overlaps_recurrent"]]
    non_rec_df = annotated[~annotated["overlaps_recurrent"]]
    org_df = annotated[annotated["overlaps_orange_all"]]
    non_org_df = annotated[~annotated["overlaps_orange_all"]]

    plot_tightness_hist(rec_df, "Tightness ratio: clusters overlapping recurrent regions",
                         outdir / "tightness_hist_recurrent_overlap.png")
    plot_tightness_hist(non_rec_df, "Tightness ratio: clusters NOT overlapping recurrent regions",
                         outdir / "tightness_hist_no_recurrent_overlap.png")
    plot_tightness_hist(org_df, "Tightness ratio: clusters overlapping orange (ecDNA/BFB/CNC) regions",
                         outdir / "tightness_hist_orange_overlap.png")
    plot_tightness_hist(non_org_df, "Tightness ratio: clusters NOT overlapping orange regions",
                         outdir / "tightness_hist_no_orange_overlap.png")
    plot_size_vs_gap(rec_df, "Cluster size vs estimated gap: overlapping recurrent regions",
                      outdir / "size_vs_gap_recurrent_overlap.png")
    plot_size_vs_gap(non_rec_df, "Cluster size vs estimated gap: NOT overlapping recurrent regions",
                      outdir / "size_vs_gap_no_recurrent_overlap.png")
    plot_metric_by_category(annotated, "tightness_ratio", "tightness_ratio",
                             "Cluster tightness by recurrent/orange overlap category",
                             outdir / "tightness_by_category_boxplot.png", log_scale=True, annotate_inf=True)
    plot_metric_by_category(annotated, "n_bins", "n_bins (cluster size)",
                             "Cluster size by recurrent/orange overlap category",
                             outdir / "size_by_category_boxplot.png", log_scale=True)
    print(f"[+] Saved plots to {outdir}")


if __name__ == "__main__":
    main()
