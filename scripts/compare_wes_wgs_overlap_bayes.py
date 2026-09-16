#!/usr/bin/env python3
"""Bayesian-optimization version of the WES/WGS amplification-threshold search.

This keeps the same 25kb, no-smoothing logic as the grid-search script, but uses a
black-box Bayesian optimizer to search over the WES and WGS amplification thresholds
for the best Jaccard or overlap score.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from skopt import gp_minimize
    from skopt.space import Real
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This script requires scikit-optimize. Install it with: pip install scikit-optimize"
    ) from exc

from compare_wes_wgs_overlap import (
    align_tracks,
    apply_mask_regions_to_df,
    load_data,
    load_mask_regions,
    make_cn_like_version,
    prepare_track,
    rebin_track,
)


def objective_from_values(wes_vals, wgs_vals, metric="jaccard"):
    def objective(x):
        wes_thresh = float(x[0])
        wgs_thresh = float(x[1])
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

        score = jaccard if metric == "jaccard" else overlap
        return -float(score)

    return objective


def run_bayes_search(df, wes_col, wgs_col, metric="jaccard", n_init=12, n_iter=40, seed=0):
    wes_vals = pd.to_numeric(df[wes_col], errors="coerce").fillna(-np.inf).to_numpy(dtype=float)
    wgs_vals = pd.to_numeric(df[wgs_col], errors="coerce").fillna(-np.inf).to_numpy(dtype=float)

    wes_min = float(np.nanmin(np.asarray(pd.to_numeric(df[wes_col], errors="coerce").dropna())))
    wes_max = float(np.nanmax(np.asarray(pd.to_numeric(df[wes_col], errors="coerce").dropna())))
    wgs_min = float(np.nanmin(np.asarray(pd.to_numeric(df[wgs_col], errors="coerce").dropna())))
    wgs_max = float(np.nanmax(np.asarray(pd.to_numeric(df[wgs_col], errors="coerce").dropna())))

    if not np.isfinite(wes_min) or not np.isfinite(wes_max):
        raise ValueError(f"No valid WES values found in column '{wes_col}'")
    if not np.isfinite(wgs_min) or not np.isfinite(wgs_max):
        raise ValueError(f"No valid WGS values found in column '{wgs_col}'")

    if wes_min == wes_max:
        wes_min = wes_min - 1.0
        wes_max = wes_max + 1.0
    if wgs_min == wgs_max:
        wgs_min = wgs_min - 1.0
        wgs_max = wgs_max + 1.0

    bounds = [
        Real(wes_min, wes_max, prior="uniform"),
        Real(wgs_min, wgs_max, prior="uniform"),
    ]

    rng = np.random.default_rng(seed)
    initial_points = np.vstack([
        np.array([
            rng.uniform(wes_min, wes_max),
            rng.uniform(wgs_min, wgs_max),
        ])
        for _ in range(max(1, n_init))
    ])

    objective = objective_from_values(wes_vals, wgs_vals, metric=metric)
    result = gp_minimize(
        objective,
        dimensions=bounds,
        n_calls=n_init + n_iter,
        x0=initial_points,
        n_random_starts=min(n_init, 10),
        random_state=seed,
        acq_func="EI",
    )

    best = np.asarray(result.x)
    wes_thresh = float(best[0])
    wgs_thresh = float(best[1])
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

    history = pd.DataFrame(result.x_iters, columns=["wes_threshold", "wgs_threshold"])
    history["objective"] = np.asarray(result.func_vals)
    history["metric"] = metric
    history["jaccard"] = [
        objective_from_values(wes_vals, wgs_vals, metric="jaccard")(row) * -1.0 for row in result.x_iters
    ]
    history["overlap_coefficient"] = [
        objective_from_values(wes_vals, wgs_vals, metric="overlap")(row) * -1.0 for row in result.x_iters
    ]

    best_row = {
        "wes_threshold": wes_thresh,
        "wgs_threshold": wgs_thresh,
        "metric": metric,
        "jaccard": float(jaccard),
        "overlap_coefficient": float(overlap),
        "n_wes_high_bins": int(n_wes),
        "n_wgs_high_bins": int(n_wgs),
        "n_intersection_bins": int(n_inter),
        "n_union_bins": int(n_union),
    }

    return best_row, history


def plot_bayes_history(history, outpath, metric="jaccard"):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(history["wes_threshold"], history["wgs_threshold"], ".", alpha=0.5)
    best = history.loc[history[metric].idxmax()]
    ax.scatter([best["wes_threshold"]], [best["wgs_threshold"]], color="red", s=40, label="best")
    ax.set_xlabel("WES threshold")
    ax.set_ylabel("WGS threshold")
    ax.set_title(f"Bayesian search: best {metric}")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wes-input", required=True, help="WES file")
    parser.add_argument("--wes-column", required=True, help="WES value column")
    parser.add_argument("--wgs-input", required=True, help="WGS file")
    parser.add_argument("--wgs-column", default=None, help="WGS value column")
    parser.add_argument("--bin-col", default=None, help="Combined bin column like chr1:1000-2000")
    parser.add_argument("--chrom-col", default="chrom")
    parser.add_argument("--start-col", default="start")
    parser.add_argument("--target-bin-size", type=int, default=25000)
    parser.add_argument("--metric", choices=["jaccard", "overlap"], default="jaccard")
    parser.add_argument("--n-init", type=int, default=12)
    parser.add_argument("--n-iter", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mask-regions", default=None, help="Optional CSV/TSV of regions to mask, with columns chrom,start,end")
    parser.add_argument("--outdir", default="wes_wgs_overlap_output")
    args = parser.parse_args()

    wgs_col = args.wgs_column or args.wes_column
    wes_df = prepare_track(load_data(args.wes_input), args.wes_column, args.bin_col, args.chrom_col, args.start_col)
    wgs_df = prepare_track(load_data(args.wgs_input), wgs_col, args.bin_col, args.chrom_col, args.start_col)
    wes_df = rebin_track(wes_df, args.wes_column, target_bin_size=args.target_bin_size)
    wgs_df = rebin_track(wgs_df, wgs_col, target_bin_size=args.target_bin_size)
    aligned = align_tracks(wes_df, wgs_df, args.wes_column, wgs_col)
    if args.mask_regions is not None:
        aligned = apply_mask_regions_to_df(aligned, load_mask_regions(args.mask_regions))
        masked_count = int(aligned["masked"].sum())
        print(f"Applied mask regions from {args.mask_regions}: masked {masked_count} aligned bins")

    raw = aligned.copy()
    raw_best, raw_hist = run_bayes_search(raw, "wes_value", "wgs_value", metric=args.metric, n_init=args.n_init, n_iter=args.n_iter, seed=args.seed)

    cn_df = make_cn_like_version(raw, "wes_value", "wgs_value")
    cn_best, cn_hist = run_bayes_search(cn_df, "wes_cn_like", "wgs_cn_like", metric=args.metric, n_init=args.n_init, n_iter=args.n_iter, seed=args.seed + 1)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw_hist.to_csv(outdir / "bayes_history_raw.tsv", sep="\t", index=False)
    cn_hist.to_csv(outdir / "bayes_history_cn_like.tsv", sep="\t", index=False)
    pd.DataFrame([raw_best]).to_csv(outdir / "bayes_best_raw.tsv", sep="\t", index=False)
    pd.DataFrame([cn_best]).to_csv(outdir / "bayes_best_cn_like.tsv", sep="\t", index=False)

    plot_bayes_history(raw_hist, outdir / "bayes_history_raw.png", metric=args.metric)
    plot_bayes_history(cn_hist, outdir / "bayes_history_cn_like.png", metric=args.metric)

    print(f"Best raw {args.metric}: {raw_best}")
    print(f"Best CN-like {args.metric}: {cn_best}")


if __name__ == "__main__":
    main()
