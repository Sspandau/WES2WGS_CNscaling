"""Gradient-descent surrogate version of the WES/WGS amplification-threshold search.

This keeps the same 25kb, no-smoothing input pipeline as the grid-search script, but
optimizes thresholds using a smooth surrogate objective rather than a hard thresholded
set-based objective.

To avoid degenerate solutions (threshold ~0 calling every bin amplified, or a threshold
above the data maximum calling none), the optimizer works on the FRACTION of bins called
amplified in each track, restricted to [min_frac, max_frac]. Thresholds are derived from
quantiles of the data, so both extremes are impossible by construction.

Parameterization:
    z (unconstrained)  ->  log f = log(lo) + (log(hi) - log(lo)) * sigmoid(z)
    f                  ->  threshold = quantile(values, 1 - f)

The bounds on f are therefore enforced smoothly, with no clipping. Because the
objective is non-convex, several starting fractions are tried and the best result
(judged by the hard, non-smoothed Jaccard/overlap) is reported.
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


def expit(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))


def compute_stats(wes_vals, wgs_vals, wes_thresh, wgs_thresh):
    """Unpenalized hard (set-based) overlap statistics for a pair of thresholds."""
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


def soft_stats(wes_vals, wgs_vals, wes_thresh, wgs_thresh, wes_temp, wgs_temp):
    """Smooth surrogate: sigmoid membership per bin, fuzzy intersection/union."""
    s_wes = expit((wes_vals - wes_thresh) / wes_temp)
    s_wgs = expit((wgs_vals - wgs_thresh) / wgs_temp)
    inter = float(np.minimum(s_wes, s_wgs).sum())
    union = float(np.maximum(s_wes, s_wgs).sum())
    wes_sum = float(s_wes.sum())
    wgs_sum = float(s_wgs.sum())
    jaccard = inter / union if union > 0 else 0.0
    overlap = inter / min(wes_sum, wgs_sum) if min(wes_sum, wgs_sum) > 0 else 0.0
    return jaccard, overlap, wes_sum, wgs_sum


def count_penalty_term(n, min_bins, max_bins, count_penalty):
    if n < min_bins:
        return count_penalty * (min_bins - n) / min_bins
    if n > max_bins:
        return count_penalty * (n - max_bins) / max_bins
    return 0.0


def gradient_descent_search(
    df,
    wes_col,
    wgs_col,
    metric="jaccard",
    n_steps=100,
    lr=0.2,
    n_starts=5,
    seed=0,
    temp=0.25,
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

    n_bins = min(wes_finite.size, wgs_finite.size)
    max_bins = max_frac * n_bins
    lo = max(min_frac, min_bins / n_bins)
    if lo >= max_frac:
        raise ValueError("min_frac/min_bins is >= max_frac; widen the allowed range")
    log_lo, log_hi = np.log(lo), np.log(max_frac)

    # Sigmoid temperature is set per track, relative to how far apart the thresholds
    # for the allowed fraction range are, so `temp` is scale-free (raw vs CN-like).
    def track_temp(finite):
        span = np.quantile(finite, 1.0 - lo) - np.quantile(finite, 1.0 - max_frac)
        if not np.isfinite(span) or span <= 0:
            span = float(np.std(finite))
        if not np.isfinite(span) or span <= 0:
            span = 1.0
        return temp * span

    wes_temp = track_temp(wes_finite)
    wgs_temp = track_temp(wgs_finite)

    # If the floor leaves fewer than min_bins eligible bins, relax the minimum for that track
    # so the count penalty is not a permanent constant.
    n_avail_wes = int((wes_vals > min_wes_thresh).sum()) if min_wes_thresh is not None else int(wes_finite.size)
    n_avail_wgs = int((wgs_vals > min_wgs_thresh).sum()) if min_wgs_thresh is not None else int(wgs_finite.size)
    min_bins_wes = min(min_bins, n_avail_wes)
    min_bins_wgs = min(min_bins, n_avail_wgs)

    def z_to_frac(z):
        return float(np.exp(log_lo + (log_hi - log_lo) * expit(z)))

    def to_thresholds(z):
        return thresholds_from_fracs(
            wes_finite, wgs_finite, z_to_frac(z[0]), z_to_frac(z[1]), min_wes_thresh, min_wgs_thresh
        )

    def soft_loss(z):
        t_wes, t_wgs = to_thresholds(z)
        jac, ovl, wes_sum, wgs_sum = soft_stats(wes_vals, wgs_vals, t_wes, t_wgs, wes_temp, wgs_temp)
        score = jac if metric == "jaccard" else ovl
        score -= count_penalty_term(wes_sum, min_bins_wes, max_bins, count_penalty)
        score -= count_penalty_term(wgs_sum, min_bins_wgs, max_bins, count_penalty)
        return -score

    def hard_score(stats):
        score = stats["jaccard"] if metric == "jaccard" else stats["overlap_coefficient"]
        score -= count_penalty_term(stats["n_wes_high_bins"], min_bins_wes, max_bins, count_penalty)
        score -= count_penalty_term(stats["n_wgs_high_bins"], min_bins_wgs, max_bins, count_penalty)
        return score

    rng = np.random.default_rng(seed)
    start_u = np.linspace(0.1, 0.9, max(1, n_starts))  # positions within log-fraction range
    beta1, beta2, adam_eps = 0.9, 0.999, 1e-8
    fd_eps = 1e-3

    rows = []
    for start_idx, u0 in enumerate(start_u):
        z = np.array([logit(u0), logit(u0)], dtype=float) + rng.normal(0, 0.1, size=2)
        m = np.zeros(2)
        v = np.zeros(2)
        for step in range(n_steps + 1):
            # Record the current point (step 0 = the starting point).
            t_wes, t_wgs = to_thresholds(z)
            stats = compute_stats(wes_vals, wgs_vals, t_wes, t_wgs)
            rows.append(dict(
                start=start_idx,
                step=step,
                wes_frac=z_to_frac(z[0]),
                wgs_frac=z_to_frac(z[1]),
                wes_threshold=t_wes,
                wgs_threshold=t_wgs,
                soft_objective=-soft_loss(z),
                hard_score=hard_score(stats),
                metric=metric,
                **stats,
            ))
            if step == n_steps:
                break

            # Central finite-difference gradient in z-space (2 parameters).
            grad = np.zeros(2)
            for i in range(2):
                zp, zm = z.copy(), z.copy()
                zp[i] += fd_eps
                zm[i] -= fd_eps
                grad[i] = (soft_loss(zp) - soft_loss(zm)) / (2 * fd_eps)

            # Adam update.
            t = step + 1
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad ** 2
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            z = z - lr * m_hat / (np.sqrt(v_hat) + adam_eps)

    hist = pd.DataFrame(rows)
    best = hist.loc[hist["hard_score"].idxmax()]
    best_dict = {
        "wes_threshold": float(best["wes_threshold"]),
        "wgs_threshold": float(best["wgs_threshold"]),
        "wes_frac": float(best["wes_frac"]),
        "wgs_frac": float(best["wgs_frac"]),
        "metric": metric,
        "jaccard": float(best["jaccard"]),
        "overlap_coefficient": float(best["overlap_coefficient"]),
        "n_wes_high_bins": int(best["n_wes_high_bins"]),
        "n_wgs_high_bins": int(best["n_wgs_high_bins"]),
        "n_intersection_bins": int(best["n_intersection_bins"]),
        "n_union_bins": int(best["n_union_bins"]),
        "start": int(best["start"]),
    }
    best_dict = finalize_best(best_dict, wes_vals, wgs_vals, wes_finite, wgs_finite, min_wes_thresh, min_wgs_thresh)
    return best_dict, hist


def plot_trajectory(hist, outpath, metric="jaccard"):
    hist = hist[np.isfinite(hist["wes_threshold"]) & np.isfinite(hist["wgs_threshold"])]
    if hist.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    for start_idx, sub in hist.groupby("start"):
        ax.plot(sub["wes_threshold"], sub["wgs_threshold"], "-o", ms=2, alpha=0.6, label=f"start {start_idx}")
    best = hist.loc[hist["hard_score"].idxmax()]
    ax.scatter([best["wes_threshold"]], [best["wgs_threshold"]], color="red", s=50, zorder=5, label="best")
    ax.set_xlabel("WES threshold")
    ax.set_ylabel("WGS threshold")
    ax.set_title(f"Gradient-descent search: {metric}")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wes-input", required=True)
    parser.add_argument("--wes-column", required=True)
    parser.add_argument("--wgs-input", required=True)
    parser.add_argument("--wgs-column", default=None)
    parser.add_argument("--bin-col", default=None)
    parser.add_argument("--chrom-col", default="chrom")
    parser.add_argument("--start-col", default="start")
    parser.add_argument("--target-bin-size", type=int, default=25000)
    parser.add_argument("--metric", choices=["jaccard", "overlap"], default="jaccard")
    parser.add_argument("--n-steps", type=int, default=100, help="Adam steps per starting point")
    parser.add_argument("--n-starts", type=int, default=5, help="Number of starting fractions (log-spaced across the allowed range)")
    parser.add_argument("--lr", type=float, default=0.2, help="Adam learning rate in the unconstrained (logit) space")
    parser.add_argument("--temp", type=float, default=0.25, help="Sigmoid temperature, relative to the threshold spread across the allowed fraction range")
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
        n_steps=args.n_steps,
        lr=args.lr,
        n_starts=args.n_starts,
        temp=args.temp,
        min_frac=args.min_frac,
        max_frac=args.max_frac,
        min_bins=args.min_bins,
    )

    cn_df = make_cn_like_version(aligned, "wes_value", "wgs_value")
    cn_min = args.min_cn_like
    if len(cn_df) != len(aligned):
        raise SystemExit("make_cn_like_version changed the number of rows; cannot translate the CN-like floor to the raw scale")
    raw_floor_wes = raw_floor_from_cn(aligned["wes_value"], cn_df["wes_cn_like"], cn_min)
    raw_floor_wgs = raw_floor_from_cn(aligned["wgs_value"], cn_df["wgs_cn_like"], cn_min)
    print(f"Amplification floor: CN-like > {cn_min} (raw-scale equivalents: WES > {raw_floor_wes:.4g}, WGS > {raw_floor_wgs:.4g})")

    raw_best, raw_hist = gradient_descent_search(
        aligned, "wes_value", "wgs_value", seed=args.seed,
        min_wes_thresh=raw_floor_wes, min_wgs_thresh=raw_floor_wgs, **search_kwargs
    )
    cn_best, cn_hist = gradient_descent_search(
        cn_df, "wes_cn_like", "wgs_cn_like", seed=args.seed + 1,
        min_wes_thresh=cn_min, min_wgs_thresh=cn_min, **search_kwargs
    )

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