#!/usr/bin/env python3
"""Minimal WES/WGS amplification-threshold search.

This script is intentionally lightweight:

1. load WES and WGS bin-level tracks
2. average smaller bins into 25kb bins
3. align WES and WGS on chrom/start
4. evaluate a grid of amplification thresholds for Jaccard and overlap
5. repeat the search in a CN-like space defined as value / mean(value on positive bins)
6. write TSVs and plots highlighting the best threshold pair

No smoothing is used anywhere.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data(path):
    sep = "\t" if str(path).lower().endswith((".tsv", ".txt")) else None
    return pd.read_csv(path, sep=sep, engine="python")


def split_bin_column(df, bin_col):
    parsed = df[bin_col].astype(str).str.extract(
        r"^\s*(?P<chrom>chr[\w]+|[0-9XYMxym]+)\s*[:_\-]\s*(?P<start>\d+)\s*[:_\-]\s*(?P<end>\d+)?\s*$"
    )
    if parsed["chrom"].isna().any():
        bad = df.loc[parsed["chrom"].isna(), bin_col].head(3).tolist()
        raise ValueError(
            f"Could not parse bin column '{bin_col}'. Examples that failed: {bad}. "
            "Expected formats like 'chr1:1000-2000', 'chr1_1000_2000', or 'chr1-1000-2000'."
        )
    df = df.copy()
    df["chrom"] = parsed["chrom"]
    df["start"] = parsed["start"].astype(int)
    return df


def prepare_track(df, value_col, bin_col=None, chrom_col="chrom", start_col="start"):
    df = df.copy()
    if bin_col is not None:
        if bin_col not in df.columns:
            raise ValueError(f"Bin column '{bin_col}' not found in input.")
        df = split_bin_column(df, bin_col)
    else:
        chrom_candidates = [chrom_col, "chrom", "meta_chrom", "chr", "chromosome"]
        start_candidates = [start_col, "start", "meta_start", "pos", "position"]
        resolved_chrom = next((c for c in chrom_candidates if c in df.columns), None)
        resolved_start = next((c for c in start_candidates if c in df.columns), None)
        if resolved_chrom is None or resolved_start is None:
            raise ValueError(
                "Could not resolve chromosome/start columns. "
                f"Available columns: {list(df.columns[:20])}."
            )
        df = df.rename(columns={resolved_chrom: "chrom", resolved_start: "start"})

    df["chrom"] = df["chrom"].astype(str)
    df["start"] = df["start"].astype(int)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    return df.dropna(subset=["chrom", "start", value_col]).sort_values(["chrom", "start"]).reset_index(drop=True)


def rebin_track(df, value_col, target_bin_size=25000):
    df = df.copy()
    if target_bin_size is None or target_bin_size <= 0:
        return df
    df["bin_start"] = (df["start"] // target_bin_size) * target_bin_size
    rebinned = (
        df.groupby(["chrom", "bin_start"], as_index=False)[value_col]
        .mean()
        .rename(columns={"bin_start": "start"})
        .sort_values(["chrom", "start"])
        .reset_index(drop=True)
    )
    return rebinned


def align_tracks(wes_df, wgs_df, wes_col, wgs_col):
    merged = wes_df[["chrom", "start", wes_col]].rename(columns={wes_col: "wes_value"}).merge(
        wgs_df[["chrom", "start", wgs_col]].rename(columns={wgs_col: "wgs_value"}),
        on=["chrom", "start"],
        how="outer",
    )
    merged["wes_value"] = pd.to_numeric(merged["wes_value"], errors="coerce")
    merged["wgs_value"] = pd.to_numeric(merged["wgs_value"], errors="coerce")
    return merged.sort_values(["chrom", "start"]).reset_index(drop=True)


def cn_like_from_track(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    s_pos = s[s > 0]
    denom = float(s_pos.mean()) if not s_pos.empty else np.nan
    if np.isfinite(denom) and denom > 0:
        return series / denom
    return series


def make_cn_like_version(df, wes_col="wes_value", wgs_col="wgs_value"):
    out = df.copy()
    out["wes_cn_like"] = cn_like_from_track(out[wes_col])
    out["wgs_cn_like"] = cn_like_from_track(out[wgs_col])
    return out


def threshold_grid(values, n_points=60):
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return np.array([0.0])
    lo = float(values.min())
    hi = float(values.max())
    if np.isclose(lo, hi):
        return np.array([lo])
    qs = np.linspace(0.0, 1.0, n_points)
    grid = np.quantile(values, qs)
    grid = np.unique(np.round(grid, 12))
    return grid.astype(float)


def compute_overlap_for_thresholds(wes_vals, wgs_vals, wes_thresh, wgs_thresh):
    wes_high = set(np.where(wes_vals > wes_thresh)[0].tolist())
    wgs_high = set(np.where(wgs_vals > wgs_thresh)[0].tolist())
    shared = wes_high & wgs_high
    union = wes_high | wgs_high

    n_inter = len(shared)
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
        "wes_threshold": float(wes_thresh),
        "wgs_threshold": float(wgs_thresh),
        "jaccard": float(jaccard),
        "overlap_coefficient": float(overlap),
        "n_wes_high_bins": int(n_wes),
        "n_wgs_high_bins": int(n_wgs),
        "n_intersection_bins": int(n_inter),
        "n_union_bins": int(n_union),
    }


def search_threshold_grid(df, wes_col, wgs_col, n_points=60):
    wes_vals = pd.to_numeric(df[wes_col], errors="coerce").fillna(-np.inf).to_numpy(dtype=float)
    wgs_vals = pd.to_numeric(df[wgs_col], errors="coerce").fillna(-np.inf).to_numpy(dtype=float)

    wes_thresholds = threshold_grid(df[wes_col], n_points=n_points)
    wgs_thresholds = threshold_grid(df[wgs_col], n_points=n_points)

    rows = []
    for wes_thresh in wes_thresholds:
        for wgs_thresh in wgs_thresholds:
            rows.append(compute_overlap_for_thresholds(wes_vals, wgs_vals, wes_thresh, wgs_thresh))

    out = pd.DataFrame(rows)
    return out.sort_values(["wes_threshold", "wgs_threshold"]).reset_index(drop=True)


def best_row(df, metric):
    if df.empty:
        raise ValueError("No threshold grid results found.")
    i = df[metric].idxmax()
    return df.loc[i].to_dict()


def plot_threshold_grid(df, outpath, metric, title):
    table = df.pivot(index="wes_threshold", columns="wgs_threshold", values=metric)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(table.to_numpy(), origin="lower", aspect="auto", cmap="viridis")

    best = df.loc[df[metric].idxmax()]
    y_idx = np.argmin(np.abs(table.index.to_numpy() - best["wes_threshold"]))
    x_idx = np.argmin(np.abs(table.columns.to_numpy() - best["wgs_threshold"]))
    ax.plot(x_idx, y_idx, "o", color="red", markersize=7, label="best")
    ax.set_title(title)
    ax.set_xlabel("WGS threshold")
    ax.set_ylabel("WES threshold")
    fig.colorbar(image, ax=ax, label=metric)
    ax.legend(frameon=False)

    if table.shape[1] > 1:
        xticks = np.linspace(0, table.shape[1] - 1, min(5, table.shape[1]), dtype=int)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{table.columns[i]:.3f}" for i in xticks], rotation=45, ha="right")
    if table.shape[0] > 1:
        yticks = np.linspace(0, table.shape[0] - 1, min(5, table.shape[0]), dtype=int)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{table.index[i]:.3f}" for i in yticks])

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wes-input", required=True, help="WES copy-ratio or loess-depth TSV/CSV")
    parser.add_argument("--wes-column", required=True, help="WES value column")
    parser.add_argument("--wgs-input", required=True, help="WGS tumor-depth TSV/CSV")
    parser.add_argument("--wgs-column", default=None, help="WGS value column; defaults to the WES column name")
    parser.add_argument("--bin-col", default=None, help="Combined bin column like 'chr1:1000-2000'")
    parser.add_argument("--chrom-col", default="chrom", help="Chromosome column name")
    parser.add_argument("--start-col", default="start", help="Start coordinate column")
    parser.add_argument("--target-bin-size", type=int, default=25000, help="Final bin size in bp for comparison")
    parser.add_argument("--search-grid", type=int, default=60, help="Number of thresholds to evaluate per track")
    parser.add_argument("--outdir", default="wes_wgs_overlap_output", help="Output directory")
    args = parser.parse_args()

    wgs_col = args.wgs_column or args.wes_column

    wes_df = prepare_track(load_data(args.wes_input), args.wes_column, args.bin_col, args.chrom_col, args.start_col)
    wgs_df = prepare_track(load_data(args.wgs_input), wgs_col, args.bin_col, args.chrom_col, args.start_col)

    wes_df = rebin_track(wes_df, args.wes_column, target_bin_size=args.target_bin_size)
    wgs_df = rebin_track(wgs_df, wgs_col, target_bin_size=args.target_bin_size)

    aligned = align_tracks(wes_df, wgs_df, args.wes_column, wgs_col)

    raw_results = search_threshold_grid(aligned, "wes_value", "wgs_value", n_points=args.search_grid)
    cn_like_df = make_cn_like_version(aligned, "wes_value", "wgs_value")
    cn_results = search_threshold_grid(cn_like_df, "wes_cn_like", "wgs_cn_like", n_points=args.search_grid)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw_results.to_csv(outdir / "threshold_grid_raw.tsv", sep="\t", index=False)
    cn_results.to_csv(outdir / "threshold_grid_cn_like.tsv", sep="\t", index=False)

    plot_threshold_grid(raw_results, outdir / "jaccard_raw.png", "jaccard", "Raw amplification search: Jaccard")
    plot_threshold_grid(raw_results, outdir / "overlap_raw.png", "overlap_coefficient", "Raw amplification search: overlap coefficient")
    plot_threshold_grid(cn_results, outdir / "jaccard_cn_like.png", "jaccard", "CN-like amplification search: Jaccard")
    plot_threshold_grid(cn_results, outdir / "overlap_cn_like.png", "overlap_coefficient", "CN-like amplification search: overlap coefficient")

    raw_best_jaccard = best_row(raw_results, "jaccard")
    raw_best_overlap = best_row(raw_results, "overlap_coefficient")
    cn_best_jaccard = best_row(cn_results, "jaccard")
    cn_best_overlap = best_row(cn_results, "overlap_coefficient")

    for name, row in {
        "raw_best_jaccard": raw_best_jaccard,
        "raw_best_overlap": raw_best_overlap,
        "cn_best_jaccard": cn_best_jaccard,
        "cn_best_overlap": cn_best_overlap,
    }.items():
        pd.DataFrame([row]).to_csv(outdir / f"{name}.tsv", sep="\t", index=False)

    print(f"Raw threshold grid saved to {outdir / 'threshold_grid_raw.tsv'}")
    print(f"CN-like threshold grid saved to {outdir / 'threshold_grid_cn_like.tsv'}")
    print(f"Best raw Jaccard: {raw_best_jaccard}")
    print(f"Best raw overlap: {raw_best_overlap}")
    print(f"Best CN-like Jaccard: {cn_best_jaccard}")
    print(f"Best CN-like overlap: {cn_best_overlap}")


if __name__ == "__main__":
    main()
