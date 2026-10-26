#!/usr/bin/env python3
"""
wes_wgs_signal_concordance.py

Per-sample agreement between upscaled WES depth and WGS tumor depth, taken
from the same *_off_target_copy_ratios.tsv, after rebinning both to 25 kb
(mean of fine bins, as in the clustering runs).

Per sample it reports:
  spearman, pearson, pearson_log       overall agreement (log1p for the last)
  wes_q99_over_med, wgs_q99_over_med   tail heaviness of each track
  top1pct_wes_wgs_ratio                median WGS / WGS-median within the WES top 1%
                                       (1.0 = the WES extremes are ordinary in WGS)
  top1pct_wes_in_wgs_top10             fraction of WES top-1% bins that are in the WGS top 10%
  top_decile_jaccard                   top-10% vs top-10% Jaccard (as the built-in metric)

Masked bins (--mask-regions) are dropped before anything is computed.

USAGE
  python wes_wgs_signal_concordance.py \
      --root /metropolis/projects/sspandau/depmap/wes/scaled_output \
      --mask-regions /metropolis/projects/sspandau/depmap/best_combo_recurrent_orange_overlap_093026.csv \
      --out /metropolis/projects/sspandau/depmap/wes_wgs_signal_concordance.tsv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

WES_COL = "predicted_loess_upscale_depth"
WGS_COL = "wgs_tumor_depth"


def load_mask(path, bin_size):
    m = pd.read_csv(path, sep=None, engine="python")
    cols = {c.lower(): c for c in m.columns}
    if {"chrom", "start", "end"} <= set(cols):
        m = m.rename(columns={cols["chrom"]: "chrom", cols["start"]: "start", cols["end"]: "end"})
    else:
        m = m.rename(columns=dict(zip(m.columns[:3], ["chrom", "start", "end"])))
    keys = set()
    for chrom, s, e in m[["chrom", "start", "end"]].itertuples(index=False):
        first = -(-int(s) // bin_size) * bin_size
        for b in range(first, int(e), bin_size):
            keys.add((str(chrom), b))
    return keys


def sample_metrics(tsv, wes_col, wgs_col, bin_size, mask):
    d = pd.read_csv(tsv, sep="\t", usecols=["chrom", "start", wes_col, wgs_col])
    d = d.dropna()
    d["chrom"] = d["chrom"].astype(str)
    d["bstart"] = (d["start"] // bin_size) * bin_size
    g = d.groupby(["chrom", "bstart"])[[wes_col, wgs_col]].mean().reset_index()
    if mask:
        keep = [(c, b) not in mask for c, b in zip(g["chrom"], g["bstart"])]
        g = g[keep]
    wes, wgs = g[wes_col].to_numpy(float), g[wgs_col].to_numpy(float)
    n = len(g)
    if n < 100:
        return {"n_bins": n}

    wes_med, wgs_med = np.median(wes), np.median(wgs)
    top1 = wes >= np.quantile(wes, 0.99)
    wes_top10 = wes > np.quantile(wes, 0.90)
    wgs_top10 = wgs > np.quantile(wgs, 0.90)
    inter, union = (wes_top10 & wgs_top10).sum(), (wes_top10 | wgs_top10).sum()

    return {
        "n_bins": n,
        "spearman": pd.Series(wes).corr(pd.Series(wgs), method="spearman"),
        "pearson": np.corrcoef(wes, wgs)[0, 1],
        "pearson_log": np.corrcoef(np.log1p(np.clip(wes, 0, None)),
                                   np.log1p(np.clip(wgs, 0, None)))[0, 1],
        "wes_median": wes_med,
        "wgs_median": wgs_med,
        "wes_q99_over_med": np.quantile(wes, 0.99) / wes_med if wes_med > 0 else np.nan,
        "wgs_q99_over_med": np.quantile(wgs, 0.99) / wgs_med if wgs_med > 0 else np.nan,
        "top1pct_wes_wgs_ratio": np.median(wgs[top1]) / wgs_med if wgs_med > 0 else np.nan,
        "top1pct_wes_in_wgs_top10": wgs_top10[top1].mean(),
        "top_decile_jaccard": inter / union if union else np.nan,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--pattern", default="*_off_target_copy_ratios.tsv")
    p.add_argument("--wes-column", default=WES_COL)
    p.add_argument("--wgs-column", default=WGS_COL)
    p.add_argument("--bin-size", type=int, default=25000)
    p.add_argument("--mask-regions", default=None)
    args = p.parse_args()

    mask = load_mask(args.mask_regions, args.bin_size) if args.mask_regions else set()

    rows = []
    sample_dirs = sorted(d for d in args.root.iterdir() if d.is_dir() and not d.name.startswith("."))
    for i, d in enumerate(sample_dirs, 1):
        hits = sorted(d.rglob(args.pattern), key=lambda x: (len(x.relative_to(d).parts), str(x)))
        if not hits:
            continue
        tsv = hits[0]
        try:
            m = sample_metrics(tsv, args.wes_column, args.wgs_column, args.bin_size, mask)
            m["status"] = "ok" if "spearman" in m else "too_few_bins"
        except ValueError as e:          # missing column
            m = {"status": f"error: {e}"}
        rows.append({"sample": d.name, "tsv": str(tsv), **m})
        print(f"[{i}/{len(sample_dirs)}] {d.name}: "
              + (f"spearman={m['spearman']:.3f} pearson_log={m['pearson_log']:.3f} "
                 f"WES q99/med={m['wes_q99_over_med']:.1f}" if m["status"] == "ok" else m["status"]),
              flush=True)

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, sep="\t", index=False, float_format="%.4f")

    ok = df[df["status"] == "ok"]
    print(f"\n{len(ok)} samples. Medians across samples:")
    cols = ["spearman", "pearson", "pearson_log", "wes_q99_over_med", "wgs_q99_over_med",
            "top1pct_wes_wgs_ratio", "top1pct_wes_in_wgs_top10", "top_decile_jaccard"]
    print(ok[cols].median().round(3).to_string())
    print(f"\nWritten: {args.out}")


if __name__ == "__main__":
    main()
