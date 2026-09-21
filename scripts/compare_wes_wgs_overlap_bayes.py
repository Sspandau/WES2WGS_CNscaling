#!/usr/bin/env python3
"""Bayesian-optimization version of the WES/WGS amplification-threshold search.

This keeps the same 25kb, no-smoothing logic as the grid-search script, but uses a
black-box Bayesian optimizer to search for the WES and WGS amplification thresholds
that give the best Jaccard or overlap score.

To avoid degenerate solutions (threshold ~0 calling every bin amplified, or a
threshold above the data maximum calling none), the optimizer searches over the
FRACTION of bins called amplified in each track. Thresholds are derived from
quantiles of the data, so both extremes are impossible by construction. A soft
penalty on the actual bin counts handles ties and absolute threshold floors.
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


def compute_stats(wes_vals, wgs_vals, wes_thresh, wgs_thresh):
    """Unpenalized overlap statistics for a pair of thresholds."""
    wes_high = wes_vals > wes_thresh
    wgs_high = wgs_vals > wgs_thresh
    n_wes = int(wes_high.sum())
    n_wgs = int(wgs_high.sum())
    n_inter = int((wes_high & wgs_high).sum())
    n_union = int((wes_high | wgs_high).sum())
    jaccard = n_inter / n_union if n_union else 0.0
    overlap = n_inter / min(n_wes, n_wgs) if min(n_wes, n_wgs) else 0.0
    return dict(
        jaccard=jaccard,
        overlap_coefficient=overlap,
        n_wes_high_bins=n_wes,
        n_wgs_high_bins=n_wgs,
        n_intersection_bins=n_inter,
        n_union_bins=n_union,
    )


def raw_floor_from_cn(raw_vals, cn_vals, cn_min):
    """Raw-scale value equivalent to `cn_min` on the CN-like scale.

    Takes the smallest raw value among bins whose CN-like value is >= cn_min. This
    assumes the CN-like conversion is monotone increasing in the raw value (true for a
    per-track linear rescale). Returns +inf if no bin reaches cn_min.
    """
    raw = pd.to_numeric(pd.Series(np.asarray(raw_vals)), errors="coerce").to_numpy(dtype=float)
    cn = pd.to_numeric(pd.Series(np.asarray(cn_vals)), errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(raw) & np.isfinite(cn) & (cn >= cn_min)
    return float(raw[ok].min()) if ok.any() else float("inf")


def amplification_status(stats):
    e_wes = stats["n_wes_high_bins"] == 0
    e_wgs = stats["n_wgs_high_bins"] == 0
    if e_wes and e_wgs:
        return "no_amplification_both"
    if e_wes:
        return "no_amplification_wes"
    if e_wgs:
        return "no_amplification_wgs"
    return "ok"


def finalize_best(best, wes_vals, wgs_vals, wes_finite, wgs_finite, min_wes_thresh, min_wgs_thresh):
    """Add status and floor diagnostics; Jaccard is NaN (not 0) when nothing is amplified."""
    best["status"] = amplification_status(best)
    best["n_wes_bins_above_floor"] = int((wes_vals > min_wes_thresh).sum()) if min_wes_thresh is not None else int(wes_finite.size)
    best["n_wgs_bins_above_floor"] = int((wgs_vals > min_wgs_thresh).sum()) if min_wgs_thresh is not None else int(wgs_finite.size)
    if best["status"] == "no_amplification_both":
        best["jaccard"] = float("nan")
        best["overlap_coefficient"] = float("nan")
    return best


def thresholds_from_fracs(wes_finite, wgs_finite, f_wes, f_wgs, min_wes_thresh, min_wgs_thresh):
    """Threshold = value such that the top `frac` of bins are called amplified."""
    t_wes = np.quantile(wes_finite, 1.0 - f_wes)
    t_wgs = np.quantile(wgs_finite, 1.0 - f_wgs)
    if min_wes_thresh is not None:
        t_wes = max(t_wes, min_wes_thresh)
    if min_wgs_thresh is not None:
        t_wgs = max(t_wgs, min_wgs_thresh)
    return float(t_wes), float(t_wgs)


def run_bayes_search(
    df,
    wes_col,
    wgs_col,
    metric="jaccard",
    n_init=12,
    n_iter=40,
    seed=0,
    min_frac=0.005,
    max_frac=0.10,
    min_bins=100,
    min_wes_thresh=None,
    min_wgs_thresh=None,
    count_penalty=1.0,
):
    wes_vals = pd.to_numeric(df[wes_col], errors="coerce").to_numpy(dtype=float)
    wgs_vals = pd.to_numeric(df[wgs_col], errors="coerce").to_numpy(dtype=float)
    wes_vals = np.where(np.isfinite(wes_vals), wes_vals, -np.inf)
    wgs_vals = np.where(np.isfinite(wgs_vals), wgs_vals, -np.inf)
    wes_finite = wes_vals[np.isfinite(wes_vals)]
    wgs_finite = wgs_vals[np.isfinite(wgs_vals)]
    if wes_finite.size == 0 or wgs_finite.size == 0:
        raise ValueError("No valid WES/WGS values found")

    # Use the number of usable (non-masked, finite) bins as the denominator.
    n_bins = min(wes_finite.size, wgs_finite.size)
    max_bins = max_frac * n_bins
    lo = max(min_frac, min_bins / n_bins)
    if lo >= max_frac:
        raise ValueError("min_frac/min_bins is >= max_frac; widen the allowed range")

    # If the floor leaves fewer than min_bins eligible bins, relax the minimum for that track
    # so the count penalty is not a permanent constant.
    n_avail_wes = int((wes_vals > min_wes_thresh).sum()) if min_wes_thresh is not None else int(wes_finite.size)
    n_avail_wgs = int((wgs_vals > min_wgs_thresh).sum()) if min_wgs_thresh is not None else int(wgs_finite.size)
    min_bins_wes = min(min_bins, n_avail_wes)
    min_bins_wgs = min(min_bins, n_avail_wgs)

    def to_thresholds(x):
        return thresholds_from_fracs(
            wes_finite, wgs_finite, float(x[0]), float(x[1]), min_wes_thresh, min_wgs_thresh
        )

    def objective(x):
        t_wes, t_wgs = to_thresholds(x)
        s = compute_stats(wes_vals, wgs_vals, t_wes, t_wgs)
        score = s["jaccard"] if metric == "jaccard" else s["overlap_coefficient"]
        # Soft penalty if actual counts fall outside the allowed range (ties, floors).
        for n, mb in ((s["n_wes_high_bins"], min_bins_wes), (s["n_wgs_high_bins"], min_bins_wgs)):
            if n < mb:
                score -= count_penalty * (mb - n) / mb
            elif n > max_bins:
                score -= count_penalty * (n - max_bins) / max_bins
        return -float(score)

    bounds = [
        Real(lo, max_frac, prior="log-uniform"),
        Real(lo, max_frac, prior="log-uniform"),
    ]
    result = gp_minimize(
        objective,
        dimensions=bounds,
        n_calls=n_init + n_iter,
        n_initial_points=max(1, min(n_init, 10)),
        random_state=seed,
        acq_func="EI",
    )

    rows = []
    for x in result.x_iters:
        t_wes, t_wgs = to_thresholds(x)
        rows.append(
            dict(
                wes_frac=x[0],
                wgs_frac=x[1],
                wes_threshold=t_wes,
                wgs_threshold=t_wgs,
                **compute_stats(wes_vals, wgs_vals, t_wes, t_wgs),
            )
        )
    history = pd.DataFrame(rows)
    history["objective"] = np.asarray(result.func_vals)
    history["metric"] = metric

    t_wes, t_wgs = to_thresholds(result.x)
    best_row = dict(
        wes_threshold=t_wes,
        wgs_threshold=t_wgs,
        wes_frac=float(result.x[0]),
        wgs_frac=float(result.x[1]),
        metric=metric,
        **compute_stats(wes_vals, wgs_vals, t_wes, t_wgs),
    )
    best_row = finalize_best(best_row, wes_vals, wgs_vals, wes_finite, wgs_finite, min_wes_thresh, min_wgs_thresh)
    return best_row, history


def plot_bayes_history(history, outpath, metric="jaccard"):
    history = history[np.isfinite(history["wes_threshold"]) & np.isfinite(history["wgs_threshold"])]
    if history.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(history["wes_threshold"], history["wgs_threshold"], ".", alpha=0.5)
    # Best by the penalized objective so the plot marks the reported best point.
    best = history.loc[history["objective"].idxmin()]
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
    parser.add_argument("--min-frac", type=float, default=0.005, help="Min fraction of bins called amplified per track")
    parser.add_argument("--max-frac", type=float, default=0.10, help="Max fraction of bins called amplified per track")
    parser.add_argument("--min-bins", type=int, default=100, help="Min number of amplified bins per track")
    parser.add_argument("--min-cn-like", type=float, default=4.0, help="Minimum CN-like value for a bin to count as amplified. Applied directly in the CN-like run and converted to an equivalent raw-scale floor for the raw run. 0 effectively disables it.")
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

    search_kwargs = dict(
        metric=args.metric,
        n_init=args.n_init,
        n_iter=args.n_iter,
        min_frac=args.min_frac,
        max_frac=args.max_frac,
        min_bins=args.min_bins,
    )

    raw = aligned.copy()
    cn_df = make_cn_like_version(raw, "wes_value", "wgs_value")
    cn_min = args.min_cn_like
    if len(cn_df) != len(raw):
        raise SystemExit("make_cn_like_version changed the number of rows; cannot translate the CN-like floor to the raw scale")
    raw_floor_wes = raw_floor_from_cn(raw["wes_value"], cn_df["wes_cn_like"], cn_min)
    raw_floor_wgs = raw_floor_from_cn(raw["wgs_value"], cn_df["wgs_cn_like"], cn_min)
    print(f"Amplification floor: CN-like > {cn_min} (raw-scale equivalents: WES > {raw_floor_wes:.4g}, WGS > {raw_floor_wgs:.4g})")

    raw_best, raw_hist = run_bayes_search(
        raw, "wes_value", "wgs_value", seed=args.seed,
        min_wes_thresh=raw_floor_wes, min_wgs_thresh=raw_floor_wgs, **search_kwargs
    )
    cn_best, cn_hist = run_bayes_search(
        cn_df, "wes_cn_like", "wgs_cn_like", seed=args.seed + 1,
        min_wes_thresh=cn_min, min_wgs_thresh=cn_min, **search_kwargs
    )

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