#!/usr/bin/env python3
"""Compute a binarized WES-vs-WGS overlap score. The script:

1. loads WES and WGS bin-level tracks,
2. smooths both within chromosome,
3. applies an optional recurrent-bin mask,
4. thresholds high bins at either the requested quantile or a CN floor,
5. computes Jaccard similarity and overlap coefficient,
6. writes a TSV summary.

The fallback logic is:
- if the filtered WES q90 and the WGS q90 are both below --cn-floor,
  then the effective threshold becomes --cn-floor for both tracks.
- bins in mask regions are excluded from this decision and are forced to
  non-high before clustering/overlap is computed.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


BIN_COL_PATTERN = re.compile(
    r"^\s*(?P<chrom>chr[\w]+|[0-9XYMxym]+)\s*[:_\-]\s*"
    r"(?P<start>\d+)\s*[:_\-]\s*(?P<end>\d+)?\s*$"
)


def load_data(path):
    sep = "\t" if str(path).lower().endswith((".tsv", ".txt")) else None
    return pd.read_csv(path, sep=sep, engine="python")


def load_mask_regions(path):
    m = load_data(path)
    cols = {c.lower(): c for c in m.columns}
    if "chrom" in cols and "start" in cols and "end" in cols:
        m = m.rename(columns={cols["chrom"]: "chrom", cols["start"]: "start", cols["end"]: "end"})
    else:
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
    parsed = df[bin_col].astype(str).str.extract(BIN_COL_PATTERN)
    if parsed["chrom"].isna().any():
        bad = df.loc[parsed["chrom"].isna(), bin_col].head(3).tolist()
        raise ValueError(
            f"Could not parse bin column '{bin_col}'. Examples that failed: {bad}. "
            "Expected formats like 'chr1:1000-2000', 'chr1_1000_2000', or 'chr1-1000-2000'."
        )
    df = df.copy()
    df["chrom"] = parsed["chrom"]
    df["start"] = parsed["start"].astype(int)
    df["end"] = parsed["end"].astype(float)
    return df


def smooth_per_chrom(df, value_col, window):
    df = df.copy()
    if window <= 1:
        df["smoothed"] = df[value_col]
        return df
    df["smoothed"] = (
        df.groupby("chrom", group_keys=False)[value_col]
        .apply(lambda s: s.rolling(window, center=True, min_periods=1).mean())
    )
    return df


def apply_mask(df, mask_df):
    df = df.copy()
    df["masked"] = False
    if mask_df is None or mask_df.empty:
        return df

    for _, row in mask_df.iterrows():
        chrom = str(row["chrom"])
        start = int(row["start"])
        end = int(row["end"])
        sel = (df["chrom"] == chrom) & (df["start"] >= start) & (df["start"] < end)
        if sel.any():
            df.loc[sel, "masked"] = True
    return df


def normalize_to_cn_like(df, value_col, mask_col):
    df = df.copy()
    valid = df.loc[(~df[mask_col]) & df[value_col].notna(), value_col]
    denom = float(valid.median()) if not valid.empty else np.nan
    if not np.isfinite(denom) or denom <= 0:
        df["cn_like"] = df[value_col]
    else:
        df["cn_like"] = df[value_col] / denom
    return df


def compute_high_thresholds(df, wes_quantile, wgs_quantile, cn_floor):
    wes_values = df["wes_cn_like"]
    wgs_values = df["wgs_cn_like"]

    wes_clean = wes_values[~df["wes_masked"]].dropna()
    wgs_clean = wgs_values[~df["wgs_masked"]].dropna()

    wes_q = float(wes_clean.quantile(wes_quantile)) if not wes_clean.empty else float("nan")
    wgs_q = float(wgs_clean.quantile(wgs_quantile)) if not wgs_clean.empty else float("nan")

    if np.isfinite(wes_q) and np.isfinite(wgs_q) and cn_floor is not None and wes_q < cn_floor and wgs_q < cn_floor:
        wes_threshold = float(cn_floor)
        wgs_threshold = float(cn_floor)
        floor_applied = True
    else:
        wes_threshold = float(wes_clean.quantile(wes_quantile)) if not wes_clean.empty else float("nan")
        wgs_threshold = float(wgs_clean.quantile(wgs_quantile)) if not wgs_clean.empty else float("nan")
        floor_applied = False

    return wes_threshold, wgs_threshold, floor_applied


def high_bin_set(df, value_col, threshold_col, mask_col):
    df = df.copy()
    df["is_high"] = False
    sel = (~df[mask_col]) & (df[value_col] > df[threshold_col])
    df.loc[sel, "is_high"] = True
    return df


def compute_overlap_metrics(wes_df, wgs_df, wes_value_col, wgs_value_col,
                           wes_quantile=0.90, wgs_quantile=0.90,
                           smooth_window=5, rebin_to=None, cn_floor=3.0,
                           mask_df=None):
    wes = wes_df.copy()
    wgs = wgs_df.copy()

    if rebin_to is not None:
        wes = wes[["chrom", "start", wes_value_col]].copy()
        wgs = wgs[["chrom", "start", wgs_value_col]].copy()
        wes = wes.groupby("chrom", group_keys=False).apply(lambda x: x.assign(start=(x["start"] // rebin_to) * rebin_to))
        wgs = wgs.groupby("chrom", group_keys=False).apply(lambda x: x.assign(start=(x["start"] // rebin_to) * rebin_to))
        wes = wes.groupby(["chrom", "start"], as_index=False)[wes_value_col].mean()
        wgs = wgs.groupby(["chrom", "start"], as_index=False)[wgs_value_col].mean()

    wes = apply_mask(wes, mask_df)
    wgs = apply_mask(wgs, mask_df)

    wes = smooth_per_chrom(wes, wes_value_col, smooth_window)
    wgs = smooth_per_chrom(wgs, wgs_value_col, smooth_window)

    wes["wes_smoothed"] = wes["smoothed"]
    wgs["wgs_smoothed"] = wgs["smoothed"]
    wes["wes_masked"] = wes["masked"]
    wgs["wgs_masked"] = wgs["masked"]

    wes = normalize_to_cn_like(wes, "wes_smoothed", "wes_masked")
    wgs = normalize_to_cn_like(wgs, "wgs_smoothed", "wgs_masked")

    wes["wes_cn_like"] = wes["cn_like"]
    wgs["wgs_cn_like"] = wgs["cn_like"]

    wes_threshold, wgs_threshold, floor_applied = compute_high_thresholds(
        pd.DataFrame({
            "wes_cn_like": wes["wes_cn_like"],
            "wgs_cn_like": wgs["wgs_cn_like"],
            "wes_masked": wes["wes_masked"],
            "wgs_masked": wgs["wgs_masked"],
        }),
        wes_quantile=wes_quantile,
        wgs_quantile=wgs_quantile,
        cn_floor=cn_floor,
    )

    if np.isfinite(wes_threshold):
        wes["wes_threshold"] = wes_threshold
        wes = high_bin_set(wes, "wes_cn_like", "wes_threshold", "wes_masked")
    else:
        wes["is_high"] = False
    if np.isfinite(wgs_threshold):
        wgs["wgs_threshold"] = wgs_threshold
        wgs = high_bin_set(wgs, "wgs_cn_like", "wgs_threshold", "wgs_masked")
    else:
        wgs["is_high"] = False

    wes_high = set((str(r["chrom"]), int(r["start"])) for _, r in wes[wes["is_high"]][["chrom", "start"]].iterrows())
    wgs_high = set((str(r["chrom"]), int(r["start"])) for _, r in wgs[wgs["is_high"]][["chrom", "start"]].iterrows())
    shared_high = wes_high & wgs_high
    wes_only = wes_high - wgs_high
    wgs_only = wgs_high - wes_high

    intersection = shared_high
    union = wes_high | wgs_high
    n_intersection = len(intersection)
    n_union = len(union)
    n_wes = len(wes_high)
    n_wgs = len(wgs_high)

    if n_union == 0:
        jaccard = 1.0 if n_intersection == 0 else 0.0
    else:
        jaccard = n_intersection / n_union

    if n_wes == 0 and n_wgs == 0:
        overlap = 1.0
    elif min(n_wes, n_wgs) == 0:
        overlap = 0.0
    else:
        overlap = n_intersection / min(n_wes, n_wgs)

    return {
        "wes_threshold": float(wes_threshold),
        "wgs_threshold": float(wgs_threshold),
        "cn_floor_applied": bool(floor_applied),
        "n_wes_high_bins": n_wes,
        "n_wgs_high_bins": n_wgs,
        "n_intersection_bins": n_intersection,
        "n_union_bins": n_union,
        "jaccard": float(jaccard),
        "overlap_coefficient": float(overlap),
        "wes_high_bins": wes_high,
        "wgs_high_bins": wgs_high,
        "shared_high_bins": shared_high,
        "wes_only_bins": wes_only,
        "wgs_only_bins": wgs_only,
    }


def prepare_track(df, value_col, bin_col=None, chrom_col="chrom", start_col="start"):
    df = df.copy()
    if bin_col:
        if bin_col not in df.columns:
            raise ValueError(f"Bin column '{bin_col}' not found in input.")
        df = split_bin_column(df, bin_col)
    else:
        if chrom_col not in df.columns or start_col not in df.columns:
            raise ValueError(
                f"Expected '{chrom_col}' and '{start_col}' columns, or a combined '{bin_col}' column."
            )
        df = df.rename(columns={chrom_col: "chrom", start_col: "start"})
    df["chrom"] = df["chrom"].astype(str)
    df["start"] = df["start"].astype(int)
    df[value_col] = df[value_col].astype(float)
    return df.dropna(subset=["chrom", "start", value_col]).sort_values(["chrom", "start"]).reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wes-input", required=True, help="Path to WES bin-level TSV/CSV")
    p.add_argument("--wes-column", required=True, help="Column with WES values")
    p.add_argument("--wgs-input", required=True, help="Path to WGS bin-level TSV/CSV")
    p.add_argument("--wgs-column", default=None, help="Column with WGS values (defaults to the WES column name)")
    p.add_argument("--bin-col", default=None, help="Combined bin column like 'chr1:1000-2000'")
    p.add_argument("--chrom-col", default="chrom", help="Chromosome column name")
    p.add_argument("--start-col", default="start", help="Start position column name")
    p.add_argument("--rebin-to", type=int, default=None, help="Optional rebinning size in bp")
    p.add_argument("--smooth-window", type=int, default=5, help="Rolling smoothing window in bins")
    p.add_argument("--wes-quantile", type=float, default=0.90, help="WES threshold quantile")
    p.add_argument("--wgs-quantile", type=float, default=0.90, help="WGS threshold quantile")
    p.add_argument("--cn-floor", type=float, default=3.0, help="Copy-number floor applied when both q90 values are below this value")
    p.add_argument("--mask-regions", default=None, help="Optional CSV/TSV of recurrent bins to mask/gap regions")
    p.add_argument("--outdir", default="wes_wgs_overlap_output", help="Output directory")
    args = p.parse_args()

    default_mask = Path("/home/sspandau/CCLE_WXS/WES2WGS_CCLE/recurrent_amplification_v3_optimization/best_combo_recurrent_orange_overlap.csv")
    if args.mask_regions is None and default_mask.exists():
        args.mask_regions = str(default_mask)

    wgs_col = args.wgs_column or args.wes_column
    wes_df = prepare_track(load_data(args.wes_input), args.wes_column, args.bin_col, args.chrom_col, args.start_col)
    wgs_df = prepare_track(load_data(args.wgs_input), wgs_col, args.bin_col, args.chrom_col, args.start_col)

    mask_df = load_mask_regions(args.mask_regions) if args.mask_regions else None
    metrics = compute_overlap_metrics(
        wes_df,
        wgs_df,
        wes_value_col=args.wes_column,
        wgs_value_col=wgs_col,
        wes_quantile=args.wes_quantile,
        wgs_quantile=args.wgs_quantile,
        smooth_window=args.smooth_window,
        rebin_to=args.rebin_to,
        cn_floor=args.cn_floor,
        mask_df=mask_df,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary = {k: v for k, v in metrics.items() if k not in {"wes_high_bins", "wgs_high_bins", "shared_high_bins", "wes_only_bins", "wgs_only_bins"}}
    outpath = outdir / "wgs_wes_overlap_metrics.tsv"
    pd.DataFrame([summary]).to_csv(outpath, sep="\t", index=False)

    for name, key in [("wes_high_bins", "wes_high_bins"), ("wgs_high_bins", "wgs_high_bins"), ("shared_high_bins", "shared_high_bins")]:
        rows = [{"chrom": chrom, "start": start} for chrom, start in sorted(metrics[key], key=lambda x: (x[0], x[1]))]
        pd.DataFrame(rows).to_csv(outdir / f"{name}.tsv", sep="\t", index=False)

    print(f"WES/WGS overlap metrics written to {outpath}")
    print(f"Shared high bins written to {outdir / 'shared_high_bins.tsv'}")
    print(
        "Jaccard = {jaccard:.4f}, overlap coefficient = {overlap:.4f}, "
        "WES high bins = {wes}, WGS high bins = {wgs}, shared = {shared}".format(
            jaccard=metrics["jaccard"],
            overlap=metrics["overlap_coefficient"],
            wes=metrics["n_wes_high_bins"],
            wgs=metrics["n_wgs_high_bins"],
            shared=metrics["n_intersection_bins"],
        )
    )


if __name__ == "__main__":
    main()
