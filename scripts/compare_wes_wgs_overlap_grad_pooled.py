#!/usr/bin/env python3
"""Pooled (cohort-wide) WES/WGS amplification-threshold search, gradient-descent version.

Same goal as compare_wes_wgs_overlap_pooled.py (find ONE WES threshold and ONE WGS
threshold shared by all samples), but optimizes a smooth surrogate objective with
gradient descent instead of Bayesian optimization. No scikit-optimize dependency.

Design
------
* Every sample is preprocessed exactly as in the per-sample scripts (rebin to 25 kb,
  align, mask, CN-like = value / mean of positive bins), then only bins that are finite
  in BOTH tracks are kept and pooled into one big array per track. Preprocessing can run
  in parallel (--n-jobs) and is cached.
* Thresholds live on the CN-like scale, the only scale comparable across samples.
* To avoid degenerate solutions, the optimizer works on the POOLED FRACTION of bins
  called amplified in each track, reparameterized through a sigmoid so it stays inside
  [min_frac, max_frac] with no clipping (same trick as the per-sample gradient script).
  A CN-like floor (--min-cn-like) is applied on top: threshold = max(quantile, floor).
* Two ways to aggregate across samples (--aggregate):
    micro : pool all bins into one big set (sum of intersections / sum of unions).
    macro : mean of per-sample Jaccard/overlap, over samples with an amplified bin,
            with a penalty if fewer than --min-samples samples contribute.
  Both are always reported; --aggregate picks which one is optimized (and which one the
  smooth surrogate approximates, for macro, via per-sample reduceat sums).
* The optimizer only ever sees the smooth surrogate; the reported best point is chosen by
  the true, non-smoothed hard score across all evaluated points (same as the per-sample
  gradient script).
"""

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
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


# --------------------------------------------------------------------------------------
# Per-sample preprocessing (parallel + cached) -- identical to the Bayes pooled script
# --------------------------------------------------------------------------------------

def _mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def preprocess_sample(job):
    """Return dict with jointly-finite CN-like WES/WGS arrays for one sample."""
    try:
        cache_file = None
        if job["cache_dir"] is not None:
            key = json.dumps(
                [job["wes_input"], _mtime(job["wes_input"]), job["wgs_input"], _mtime(job["wgs_input"]),
                 job["wes_col"], job["wgs_col"], job["bin_size"], job["bin_col"], job["chrom_col"],
                 job["start_col"], job["mask_path"], _mtime(job["mask_path"]) if job["mask_path"] else None],
                sort_keys=True,
            )
            digest = hashlib.md5(key.encode()).hexdigest()[:16]
            safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in job["sample"])
            cache_file = Path(job["cache_dir"]) / f"{safe}.{digest}.npz"
            if cache_file.exists():
                z = np.load(cache_file)
                return dict(sample=job["sample"], wes=z["wes"], wgs=z["wgs"],
                            n_total=int(z["n_total"]), n_masked=int(z["n_masked"]), error=None)

        mask_df = load_mask_regions(job["mask_path"]) if job["mask_path"] else None
        wes_raw = load_data(job["wes_input"])
        wgs_raw = wes_raw if job["wgs_input"] == job["wes_input"] else load_data(job["wgs_input"])
        wes_df = prepare_track(wes_raw, job["wes_col"], job["bin_col"], job["chrom_col"], job["start_col"])
        wgs_df = prepare_track(wgs_raw, job["wgs_col"], job["bin_col"], job["chrom_col"], job["start_col"])
        wes_df = rebin_track(wes_df, job["wes_col"], target_bin_size=job["bin_size"])
        wgs_df = rebin_track(wgs_df, job["wgs_col"], target_bin_size=job["bin_size"])
        aligned = align_tracks(wes_df, wgs_df, job["wes_col"], job["wgs_col"])
        n_masked = 0
        if mask_df is not None:
            aligned = apply_mask_regions_to_df(aligned, mask_df)
            n_masked = int(aligned["masked"].sum()) if "masked" in aligned.columns else 0
        cn = make_cn_like_version(aligned, "wes_value", "wgs_value")
        wes = pd.to_numeric(cn["wes_cn_like"], errors="coerce").to_numpy(dtype=float)
        wgs = pd.to_numeric(cn["wgs_cn_like"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(wes) & np.isfinite(wgs)
        out = dict(sample=job["sample"], wes=wes[ok].astype(np.float32), wgs=wgs[ok].astype(np.float32),
                   n_total=int(len(aligned)), n_masked=n_masked, error=None)
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_file, wes=out["wes"], wgs=out["wgs"],
                     n_total=out["n_total"], n_masked=out["n_masked"])
        return out
    except Exception as exc:  # keep going; the sample is reported as skipped
        return dict(sample=job["sample"], wes=None, wgs=None, n_total=0, n_masked=0, error=repr(exc))


class PooledData:
    def __init__(self, results):
        self.samples = [r["sample"] for r in results]
        lengths = np.array([len(r["wes"]) for r in results], dtype=np.int64)
        self.n_per_sample = lengths
        self.starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
        self.wes = np.concatenate([r["wes"] for r in results]).astype(np.float64, copy=False)
        self.wgs = np.concatenate([r["wgs"] for r in results]).astype(np.float64, copy=False)
        self.n_total = int(self.wes.size)
        self.wes_sorted = np.sort(self.wes)
        self.wgs_sorted = np.sort(self.wgs)


# --------------------------------------------------------------------------------------
# Hard (set-based) statistics -- identical to the Bayes pooled script
# --------------------------------------------------------------------------------------

def quantile_sorted(sorted_arr, q):
    n = sorted_arr.size
    pos = q * (n - 1)
    lo = int(np.floor(pos))
    hi = min(lo + 1, n - 1)
    w = pos - lo
    return float(sorted_arr[lo]) * (1.0 - w) + float(sorted_arr[hi]) * w


def thresholds_from_fracs(data, f_wes, f_wgs, floor_wes, floor_wgs):
    t_wes = quantile_sorted(data.wes_sorted, 1.0 - f_wes)
    t_wgs = quantile_sorted(data.wgs_sorted, 1.0 - f_wgs)
    if floor_wes is not None:
        t_wes = max(t_wes, floor_wes)
    if floor_wgs is not None:
        t_wgs = max(t_wgs, floor_wgs)
    return float(t_wes), float(t_wgs)


def compute_counts(data, t_wes, t_wgs):
    """Pooled and per-sample counts of amplified bins at a threshold pair."""
    wes_high = data.wes > t_wes
    wgs_high = data.wgs > t_wgs
    inter = wes_high & wgs_high
    per = dict(
        wes=np.add.reduceat(wes_high.view(np.uint8), data.starts, dtype=np.int64),
        wgs=np.add.reduceat(wgs_high.view(np.uint8), data.starts, dtype=np.int64),
        inter=np.add.reduceat(inter.view(np.uint8), data.starts, dtype=np.int64),
    )
    per["union"] = per["wes"] + per["wgs"] - per["inter"]
    return per


def per_sample_scores(per):
    union = per["union"].astype(float)
    mn = np.minimum(per["wes"], per["wgs"]).astype(float)
    inter = per["inter"].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        jac = np.where(union > 0, inter / union, np.nan)
        ovl = np.where(union > 0, np.where(mn > 0, inter / mn, 0.0), np.nan)
    return jac, ovl


def summarize(per):
    n_wes = int(per["wes"].sum())
    n_wgs = int(per["wgs"].sum())
    n_inter = int(per["inter"].sum())
    n_union = n_wes + n_wgs - n_inter
    micro_j = n_inter / n_union if n_union else 0.0
    micro_o = n_inter / min(n_wes, n_wgs) if min(n_wes, n_wgs) else 0.0
    jac, ovl = per_sample_scores(per)
    n_contrib = int(np.isfinite(jac).sum())
    macro_j = float(np.nanmean(jac)) if n_contrib else 0.0
    macro_o = float(np.nanmean(ovl)) if n_contrib else 0.0
    return dict(
        micro_jaccard=micro_j, micro_overlap=micro_o,
        macro_jaccard=macro_j, macro_overlap=macro_o,
        n_samples_with_amplification=n_contrib,
        n_wes_high_bins=n_wes, n_wgs_high_bins=n_wgs,
        n_intersection_bins=n_inter, n_union_bins=n_union,
    )


def count_penalty_term(n, mn, mx, weight):
    if n < mn:
        return weight * (mn - n) / mn
    if n > mx:
        return weight * (n - mx) / mx
    return 0.0


def status_from_counts(n_wes, n_wgs):
    if n_wes == 0 and n_wgs == 0:
        return "no_amplification_both"
    if n_wes == 0:
        return "no_amplification_wes"
    if n_wgs == 0:
        return "no_amplification_wgs"
    return "ok"


def per_sample_table(data, t_wes, t_wgs):
    per = compute_counts(data, t_wes, t_wgs)
    jac, ovl = per_sample_scores(per)
    return pd.DataFrame(dict(
        sample=data.samples,
        n_bins=data.n_per_sample,
        n_wes_high_bins=per["wes"],
        n_wgs_high_bins=per["wgs"],
        n_intersection_bins=per["inter"],
        n_union_bins=per["union"],
        jaccard=jac,
        overlap_coefficient=ovl,
        status=[status_from_counts(a, b) for a, b in zip(per["wes"], per["wgs"])],
    ))


# --------------------------------------------------------------------------------------
# Soft (smooth surrogate) statistics, pooled across samples
# --------------------------------------------------------------------------------------

def soft_pooled_stats(data, t_wes, t_wgs, wes_temp, wgs_temp, aggregate):
    """Sigmoid membership per bin; micro sums globally, macro sums per-sample via reduceat."""
    s_wes = expit((data.wes - t_wes) / wes_temp)
    s_wgs = expit((data.wgs - t_wgs) / wgs_temp)
    s_min = np.minimum(s_wes, s_wgs)
    s_max = np.maximum(s_wes, s_wgs)

    wes_sum = float(s_wes.sum())
    wgs_sum = float(s_wgs.sum())
    inter_sum = float(s_min.sum())
    union_sum = float(s_max.sum())
    micro_j = inter_sum / union_sum if union_sum > 0 else 0.0
    micro_o = inter_sum / min(wes_sum, wgs_sum) if min(wes_sum, wgs_sum) > 0 else 0.0

    if aggregate == "macro":
        per_wes = np.add.reduceat(s_wes, data.starts)
        per_wgs = np.add.reduceat(s_wgs, data.starts)
        per_inter = np.add.reduceat(s_min, data.starts)
        per_union = np.add.reduceat(s_max, data.starts)
        per_mn = np.minimum(per_wes, per_wgs)
        with np.errstate(divide="ignore", invalid="ignore"):
            per_jac = np.where(per_union > 1e-9, per_inter / per_union, 0.0)
            per_ovl = np.where(per_mn > 1e-9, per_inter / per_mn, 0.0)
        # Soft "contributes" weight: how much amplified mass this sample has (0..1-ish),
        # used so near-empty samples don't get the same vote as clearly-amplified ones.
        weight = np.tanh(per_mn)
        wsum = float(weight.sum())
        macro_j = float((per_jac * weight).sum() / wsum) if wsum > 1e-9 else 0.0
        macro_o = float((per_ovl * weight).sum() / wsum) if wsum > 1e-9 else 0.0
    else:
        macro_j = micro_j
        macro_o = micro_o

    return dict(micro_jaccard=micro_j, micro_overlap=micro_o, macro_jaccard=macro_j,
                macro_overlap=macro_o, wes_sum=wes_sum, wgs_sum=wgs_sum)


# --------------------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------------------

def run_pooled_search(data, metric, aggregate, n_steps, n_starts, lr, seed, temp,
                      min_frac, max_frac, min_bins, min_samples, floor, count_penalty=1.0):
    n_total = data.n_total
    max_bins = max_frac * n_total
    lo = max(min_frac, min_bins / n_total)
    if lo >= max_frac:
        raise ValueError("min_frac/min_bins is >= max_frac; widen the allowed range")
    log_lo, log_hi = np.log(lo), np.log(max_frac)

    floor_wes = floor_wgs = floor
    if floor is not None:
        n_avail_wes = int((data.wes > floor).sum())
        n_avail_wgs = int((data.wgs > floor).sum())
    else:
        n_avail_wes = n_avail_wgs = n_total
    min_bins_wes = min(min_bins, n_avail_wes)
    min_bins_wgs = min(min_bins, n_avail_wgs)

    # Sigmoid temperature, relative to the threshold spread across the allowed fraction
    # range, so `temp` is scale-free.
    def track_temp(sorted_arr):
        span = quantile_sorted(sorted_arr, 1.0 - lo) - quantile_sorted(sorted_arr, 1.0 - max_frac)
        if not np.isfinite(span) or span <= 0:
            span = float(np.std(sorted_arr))
        if not np.isfinite(span) or span <= 0:
            span = 1.0
        return temp * span

    wes_temp = track_temp(data.wes_sorted)
    wgs_temp = track_temp(data.wgs_sorted)

    def z_to_frac(z):
        return float(np.exp(log_lo + (log_hi - log_lo) * expit(z)))

    def to_thresholds(z):
        return thresholds_from_fracs(data, z_to_frac(z[0]), z_to_frac(z[1]), floor_wes, floor_wgs)

    key = "jaccard" if metric == "jaccard" else "overlap"

    def soft_loss(z):
        t_wes, t_wgs = to_thresholds(z)
        s = soft_pooled_stats(data, t_wes, t_wgs, wes_temp, wgs_temp, aggregate)
        score = s[f"{aggregate}_{key}"]
        score -= count_penalty_term(s["wes_sum"], min_bins_wes, max_bins, count_penalty)
        score -= count_penalty_term(s["wgs_sum"], min_bins_wgs, max_bins, count_penalty)
        return -score

    def hard_score_and_stats(z):
        t_wes, t_wgs = to_thresholds(z)
        per = compute_counts(data, t_wes, t_wgs)
        s = summarize(per)
        score = s[f"{aggregate}_{key}"]
        if aggregate == "macro" and s["n_samples_with_amplification"] < min_samples:
            score -= count_penalty * (min_samples - s["n_samples_with_amplification"]) / min_samples
        score -= count_penalty_term(s["n_wes_high_bins"], min_bins_wes, max_bins, count_penalty)
        score -= count_penalty_term(s["n_wgs_high_bins"], min_bins_wgs, max_bins, count_penalty)
        return score, t_wes, t_wgs, s

    rng = np.random.default_rng(seed)
    start_u = np.linspace(0.1, 0.9, max(1, n_starts))
    beta1, beta2, adam_eps = 0.9, 0.999, 1e-8
    fd_eps = 1e-3

    rows = []
    for start_idx, u0 in enumerate(start_u):
        z = np.array([logit(u0), logit(u0)], dtype=float) + rng.normal(0, 0.1, size=2)
        m = np.zeros(2)
        v = np.zeros(2)
        for step in range(n_steps + 1):
            hard_score, t_wes, t_wgs, s = hard_score_and_stats(z)
            rows.append(dict(
                start=start_idx, step=step,
                wes_frac=z_to_frac(z[0]), wgs_frac=z_to_frac(z[1]),
                wes_threshold=t_wes, wgs_threshold=t_wgs,
                soft_objective=-soft_loss(z), hard_score=hard_score,
                metric=metric, aggregate=aggregate, **s,
            ))
            if step == n_steps:
                break
            grad = np.zeros(2)
            for i in range(2):
                zp, zm = z.copy(), z.copy()
                zp[i] += fd_eps
                zm[i] -= fd_eps
                grad[i] = (soft_loss(zp) - soft_loss(zm)) / (2 * fd_eps)
            t = step + 1
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad ** 2
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            z = z - lr * m_hat / (np.sqrt(v_hat) + adam_eps)

    history = pd.DataFrame(rows)
    best = history.loc[history["hard_score"].idxmax()].to_dict()
    status = status_from_counts(best["n_wes_high_bins"], best["n_wgs_high_bins"])
    if status == "no_amplification_both":
        for k in ("micro_jaccard", "micro_overlap", "macro_jaccard", "macro_overlap"):
            best[k] = float("nan")
    best.update(status=status, n_samples=len(data.samples), n_total_bins=n_total,
                cn_like_floor=floor, n_wes_bins_above_floor=n_avail_wes,
                n_wgs_bins_above_floor=n_avail_wgs)
    return best, history


def plot_search(history, best, outpath, metric, aggregate):
    h = history[np.isfinite(history["wes_threshold"]) & np.isfinite(history["wgs_threshold"])]
    if h.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    for start_idx, sub in h.groupby("start"):
        ax.plot(sub["wes_threshold"], sub["wgs_threshold"], "-o", ms=2, alpha=0.6, label=f"start {start_idx}")
    ax.scatter([best["wes_threshold"]], [best["wgs_threshold"]], color="red", s=60, marker="*", label="best", zorder=5)
    ax.set_xlabel("WES CN-like threshold")
    ax.set_ylabel("WGS CN-like threshold")
    ax.set_title(f"Pooled gradient-descent search: {aggregate} {metric}")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="TSV with columns: sample, wes_input, wgs_input")
    parser.add_argument("--wes-column", default="predicted_loess_upscale_depth")
    parser.add_argument("--wgs-column", default="wgs_tumor_depth")
    parser.add_argument("--bin-col", default=None)
    parser.add_argument("--chrom-col", default="chrom")
    parser.add_argument("--start-col", default="start")
    parser.add_argument("--target-bin-size", type=int, default=25000)
    parser.add_argument("--mask-regions", default=None)
    parser.add_argument("--metric", choices=["jaccard", "overlap", "both"], default="both")
    parser.add_argument("--aggregate", choices=["micro", "macro"], default="micro",
                        help="micro: pool all bins; macro: (weighted) mean of per-sample scores")
    parser.add_argument("--n-steps", type=int, default=100, help="Adam steps per starting point")
    parser.add_argument("--n-starts", type=int, default=5, help="Number of starting fractions (log-spaced across the allowed range)")
    parser.add_argument("--lr", type=float, default=0.2, help="Adam learning rate in the unconstrained (logit) space")
    parser.add_argument("--temp", type=float, default=0.25, help="Sigmoid temperature, relative to the threshold spread across the allowed fraction range")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-frac", type=float, default=0.0005, help="Min POOLED fraction of bins called amplified per track")
    parser.add_argument("--max-frac", type=float, default=0.01, help="Max POOLED fraction of bins called amplified per track")
    parser.add_argument("--min-bins", type=int, default=1000, help="Min pooled amplified bins per track")
    parser.add_argument("--min-samples", type=int, default=10, help="Macro only: min samples with an amplified bin")
    parser.add_argument("--min-cn-like", type=float, default=4.0, help="Minimum CN-like value for a bin to count as amplified (0 effectively disables)")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel workers for per-sample preprocessing")
    parser.add_argument("--cache-dir", default=None, help="Preprocessed-array cache (default: <outdir>/cache)")
    parser.add_argument("--outdir", default="wes_wgs_overlap_pooled_grad_output")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or str(outdir / "cache")

    manifest = pd.read_csv(args.manifest, sep="\t")
    missing = {"sample", "wes_input", "wgs_input"} - set(manifest.columns)
    if missing:
        raise SystemExit(f"Manifest is missing columns: {sorted(missing)}")

    jobs = [dict(sample=str(r.sample), wes_input=str(r.wes_input), wgs_input=str(r.wgs_input),
                 wes_col=args.wes_column, wgs_col=args.wgs_column, bin_size=args.target_bin_size,
                 bin_col=args.bin_col, chrom_col=args.chrom_col, start_col=args.start_col,
                 mask_path=args.mask_regions, cache_dir=cache_dir)
            for r in manifest.itertuples(index=False)]
    print(f"Preprocessing {len(jobs)} samples with {args.n_jobs} worker(s); cache: {cache_dir}", flush=True)

    results = []
    if args.n_jobs > 1:
        with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
            for i, r in enumerate(ex.map(preprocess_sample, jobs, chunksize=1), 1):
                results.append(r)
                if i % 25 == 0:
                    print(f"  preprocessed {i}/{len(jobs)}", flush=True)
    else:
        for i, job in enumerate(jobs, 1):
            results.append(preprocess_sample(job))
            if i % 25 == 0:
                print(f"  preprocessed {i}/{len(jobs)}", flush=True)

    used = [r for r in results if r["error"] is None and r["wes"] is not None and len(r["wes"]) > 0]
    skipped = [dict(sample=r["sample"], reason=r["error"] or "no jointly valid bins") for r in results
               if not (r["error"] is None and r["wes"] is not None and len(r["wes"]) > 0)]
    pd.DataFrame(skipped, columns=["sample", "reason"]).to_csv(outdir / "pooled_skipped_samples.tsv", sep="\t", index=False)
    if not used:
        raise SystemExit("No usable samples after preprocessing.")
    pd.DataFrame(dict(sample=[r["sample"] for r in used],
                      n_bins_total=[r["n_total"] for r in used],
                      n_bins_masked=[r["n_masked"] for r in used],
                      n_bins_used=[len(r["wes"]) for r in used])).to_csv(
        outdir / "pooled_samples_used.tsv", sep="\t", index=False)
    total_masked = sum(r["n_masked"] for r in used)
    print(f"Using {len(used)} samples ({len(skipped)} skipped); masked bins summed over samples: {total_masked}", flush=True)
    if args.mask_regions and total_masked == 0:
        print("WARNING: mask file given but no bins were masked (check chromosome naming and path).", flush=True)

    data = PooledData(used)
    del results, used
    print(f"Pooled bins: {data.n_total}", flush=True)

    floor = args.min_cn_like
    metrics = ["jaccard", "overlap"] if args.metric == "both" else [args.metric]
    for i, metric in enumerate(metrics):
        best, history = run_pooled_search(
            data, metric, args.aggregate, args.n_steps, args.n_starts, args.lr, args.seed + i,
            args.temp, args.min_frac, args.max_frac, args.min_bins, args.min_samples, floor,
        )
        pd.DataFrame([best]).to_csv(outdir / f"pooled_grad_best_{metric}.tsv", sep="\t", index=False)
        history.to_csv(outdir / f"pooled_grad_history_{metric}.tsv", sep="\t", index=False)
        per_sample_table(data, best["wes_threshold"], best["wgs_threshold"]).to_csv(
            outdir / f"pooled_grad_per_sample_{metric}.tsv", sep="\t", index=False)
        plot_search(history, best, outdir / f"pooled_grad_search_{metric}.png", metric, args.aggregate)
        print(f"Best pooled {args.aggregate} {metric}: {best}", flush=True)


if __name__ == "__main__":
    main()
