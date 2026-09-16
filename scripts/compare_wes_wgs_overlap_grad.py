#!/usr/bin/env python3
"""Gradient-descent surrogate version of the WES/WGS amplification-threshold search.

This keeps the same 25kb, no-smoothing input pipeline as the grid-search script, but
optimizes thresholds using a smooth surrogate objective rather than a hard thresholded
set-based objective.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compare_wes_wgs_overlap import (
    align_tracks,
    apply_mask_regions_to_df,
    load_data,
    load_mask_regions,
    make_cn_like_version,
    prepare_track,
    rebin_track,
)


def sigmoid(x, temp):
    return 1.0 / (1.0 + np.exp(-(x / temp)))


def soft_metric(wes_vals, wgs_vals, wes_thresh, wgs_thresh, metric="jaccard", temp=0.05):
    s_wes = sigmoid(wes_vals - wes_thresh, temp)
    s_wgs = sigmoid(wgs_vals - wgs_thresh, temp)

    overlap = np.minimum(s_wes, s_wgs)
    union = np.maximum(s_wes, s_wgs)

    inter = float(overlap.sum())
    union_sum = float(union.sum())
    wes_sum = float(s_wes.sum())
    wgs_sum = float(s_wgs.sum())

    if union_sum <= 0:
        jaccard = 1.0 if inter == 0 else 0.0
    else:
        jaccard = inter / union_sum

    if wes_sum == 0 and wgs_sum == 0:
        overlap_score = 1.0
    elif min(wes_sum, wgs_sum) == 0:
        overlap_score = 0.0
    else:
        overlap_score = inter / min(wes_sum, wgs_sum)

    return jaccard if metric == "jaccard" else overlap_score


def objective_for_metric(wes_vals, wgs_vals, metric="jaccard", temp=0.05, cn_floor=8.0, empty_penalty=1.0):
    def objective(params):
        wes_thresh, wgs_thresh = params
        score = soft_metric(wes_vals, wgs_vals, wes_thresh, wgs_thresh, metric=metric, temp=temp)
        real_amp = ((wes_vals > cn_floor).any() or (wgs_vals > cn_floor).any())
        if real_amp:
            if (wes_vals > wes_thresh).sum() == 0 and (wgs_vals > wgs_thresh).sum() == 0:
                score -= empty_penalty
            if wes_thresh >= float(np.nanmax(wes_vals)) or wgs_thresh >= float(np.nanmax(wgs_vals)):
                score -= empty_penalty
        return -score

    return objective


def gradient_descent_search(df, wes_col, wgs_col, metric="jaccard", n_steps=200, lr=0.05, seed=0, temp=0.05, cn_floor=8.0, empty_penalty=1.0):
    wes_vals = pd.to_numeric(df[wes_col], errors="coerce").dropna().to_numpy(dtype=float)
    wgs_vals = pd.to_numeric(df[wgs_col], errors="coerce").dropna().to_numpy(dtype=float)
    if len(wes_vals) == 0 or len(wgs_vals) == 0:
        raise ValueError("No valid values found for optimization")

    rng = np.random.default_rng(seed)
    wes_start = float(np.median(wes_vals))
    wgs_start = float(np.median(wgs_vals))
    params = np.array([wes_start, wgs_start], dtype=float)

    wes_min = float(np.min(np.concatenate([wes_vals, wgs_vals])) * 0.5)
    wes_max = float(np.max(np.concatenate([wes_vals, wgs_vals])) * 1.5)
    wgs_min = float(np.min(np.concatenate([wes_vals, wgs_vals])) * 0.5)
    wgs_max = float(np.max(np.concatenate([wes_vals, wgs_vals])) * 1.5)

    obj = objective_for_metric(wes_vals, wgs_vals, metric=metric, temp=temp, cn_floor=cn_floor, empty_penalty=empty_penalty)
    history = []
    for _ in range(n_steps):
        grad = np.zeros(2, dtype=float)
        for i in range(2):
            eps = 1e-3 * (abs(params[i]) + 1e-6)
            p_plus = params.copy()
            p_minus = params.copy()
            p_plus[i] += eps
            p_minus[i] -= eps
            loss_plus = obj(p_plus)
            loss_minus = obj(p_minus)
            grad[i] = (loss_plus - loss_minus) / (2 * eps)
        params -= lr * grad
        params[0] = np.clip(params[0], wes_min, wes_max)
        params[1] = np.clip(params[1], wgs_min, wgs_max)
        score = -obj(params)
        history.append({
            "step": len(history),
            "wes_threshold": float(params[0]),
            "wgs_threshold": float(params[1]),
            "metric_value": float(score),
            "metric": metric,
        })

    hist = pd.DataFrame(history)
    best = hist.loc[hist["metric_value"].idxmax()]
    best_dict = {
        "wes_threshold": float(best["wes_threshold"]),
        "wgs_threshold": float(best["wgs_threshold"]),
        "metric": metric,
        "metric_value": float(best["metric_value"]),
    }
    return best_dict, hist


def plot_trajectory(hist, outpath, metric="jaccard"):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(hist["wes_threshold"], hist["wgs_threshold"], "-o", alpha=0.8)
    best = hist.loc[hist["metric_value"].idxmax()]
    ax.scatter([best["wes_threshold"]], [best["wgs_threshold"]], color="red", s=40, label="best")
    ax.set_xlabel("WES threshold")
    ax.set_ylabel("WGS threshold")
    ax.set_title(f"Gradient-descent search: {metric}")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wes-input", required=True)
    parser.add_argument("--wes-column", required=True)
    parser.add_argument("--wgs-input", required=True)
    parser.add_argument("--wgs-column", default=None)
    parser.add_argument("--bin-col", default=None)
    parser.add_argument("--chrom-col", default="chrom")
    parser.add_argument("--start-col", default="start")
    parser.add_argument("--target-bin-size", type=int, default=25000)
    parser.add_argument("--metric", choices=["jaccard", "overlap"], default="jaccard")
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--temp", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cn-floor", type=float, default=8.0, help="Absolute amplification floor (CN-like units) used to penalize empty-set extremes when true amplification is present")
    parser.add_argument("--empty-penalty", type=float, default=1.0, help="Penalty applied when an empty high-bin set is observed despite real amplification beyond --cn-floor")
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

    raw_best, raw_hist = gradient_descent_search(aligned, "wes_value", "wgs_value", metric=args.metric, n_steps=args.n_steps, lr=args.lr, seed=args.seed, temp=args.temp, cn_floor=args.cn_floor, empty_penalty=args.empty_penalty)
    cn_df = make_cn_like_version(aligned, "wes_value", "wgs_value")
    cn_best, cn_hist = gradient_descent_search(cn_df, "wes_cn_like", "wgs_cn_like", metric=args.metric, n_steps=args.n_steps, lr=args.lr, seed=args.seed + 1, temp=args.temp, cn_floor=args.cn_floor, empty_penalty=args.empty_penalty)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw_hist.to_csv(outdir / "grad_history_raw.tsv", sep="\t", index=False)
    cn_hist.to_csv(outdir / "grad_history_cn_like.tsv", sep="\t", index=False)
    pd.DataFrame([raw_best]).to_csv(outdir / "grad_best_raw.tsv", sep="\t", index=False)
    pd.DataFrame([cn_best]).to_csv(outdir / "grad_best_cn_like.tsv", sep="\t", index=False)

    plot_trajectory(raw_hist, outdir / "grad_history_raw.png", metric=args.metric)
    plot_trajectory(cn_hist, outdir / "grad_history_cn_like.png", metric=args.metric)

    print(f"Best raw gradient-descent {args.metric}: {raw_best}")
    print(f"Best CN-like gradient-descent {args.metric}: {cn_best}")


if __name__ == "__main__":
    main()
