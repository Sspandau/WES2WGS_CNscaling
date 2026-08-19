#!/usr/bin/env python3
"""
bayesian_optimize_ecdna_bfb_cnc_thresholds.py

Replaces the exhaustive grid search in optimize_ecdna_bfb_cnc_thresholds.py
with a sequential search over the same two parameters (--quantile,
--min-samples-ratio), using the same objective (minimize the KDE overlap
coefficient between ORANGE = AC-classified ecDNA+/BFB+/Complex-non-cyclic
bins, and BLUE = background bins not caught by the statistical recurrence
caller).

WHY NOT TRUE GRADIENT DESCENT
------------------------------
The objective is not differentiable: quantile thresholding and the integer
min-samples cutoff both make hard, discontinuous decisions about which bins
land in which bucket, so d(overlap)/d(quantile) doesn't exist in the usual
sense (it's a step function almost everywhere, zero-or-undefined). Two
practical options given that:

  --method bayesian     (default) Sequential model-based optimization via
                         a Gaussian Process surrogate (scikit-optimize's
                         gp_minimize). This is the standard approach for
                         optimizing a cheap-ish, noisy, non-smooth, LOW-
                         DIMENSIONAL (here: 2-parameter) black-box function
                         -- it models the response surface, is not
                         confused by the non-smoothness, and typically
                         needs far fewer evaluations than a fine grid to
                         find the optimum.

  --method nelder-mead   Derivative-free local simplex search
                         (scipy.optimize.minimize). This is the closest
                         practical analogue to "gradient descent" for a
                         function with no real gradient -- it estimates a
                         local descent direction from function VALUES
                         alone, no finite-difference gradient needed. It's
                         a local method (can get stuck in a local
                         optimum), so it's run from several random
                         restarts and the best result is kept.

Both share the exact same objective function and per-sample caching as the
grid-search script (imported directly, so all three scripts can never
silently disagree on what "overlap" or "orange"/"blue" mean).

IMPORTANT: MINIMIZING OVERLAP ALONE HAS A DEGENERATE OPTIMUM
---------------------------------------------------------------
Pulling MORE bins out of "blue" always mechanically shrinks the overlap
with orange, whether or not those bins represent genuine cross-sample
recurrence -- so an unconstrained search on overlap_coefficient alone will
push min_samples toward its loosest possible value (min_samples=1, i.e.
"high in at least one sample", which isn't recurrence at all). To prevent
this, every evaluated combo must also produce at least
--min-recurrent-regions contiguous recurrent regions (default 2) and
min_samples >= 2 to be considered eligible; combos that don't meet this
are penalized rather than reported as the optimum.

OUTPUT (--outdir)
------------------
  search_trace.csv          every evaluated (quantile, min_samples_ratio)
                             point in the order it was tried, with its score
  convergence.png           best-score-so-far vs. evaluation number
  search_space.png          all evaluated points over the quantile x
                             min_samples_ratio plane, colored by score
  best_combo_summary.txt    recommended combo
  best_combo_histogram.png  same 3-way histogram as the grid-search script
"""

import argparse
import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# Import both prior scripts as modules (must sit in the same directory, or
# pass --base-script / --grid-script explicitly)
# --------------------------------------------------------------------------

def load_module(path, name):
    path = Path(path)
    if not path.exists():
        sys.exit(f"Error: {name} not found at {path}. Pass the matching --*-script flag.")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# Objective function: shared by both search methods
# --------------------------------------------------------------------------

class Objective:
    """Wraps the grid-search script's pooling/labeling/metric functions into
    a plain f(quantile, min_samples_ratio) -> overlap_coefficient callable,
    with per-quantile pooled results cached (min_samples_ratio is cheap to
    re-evaluate against an already-pooled quantile, exactly like the grid
    script's inner loop)."""

    def __init__(self, cache, gs, base, rebin_to, merge_gap, min_bin_instances, min_recurrent_regions=2,
                 max_kde_points=20000, penalty=1.0):
        self.cache = cache
        self.gs = gs                      # optimize_ecdna_bfb_cnc_thresholds module
        self.base = base                  # find_recurrent_novel_amplifications_binlevel module
        self.rebin_to = rebin_to
        self.merge_gap = merge_gap
        self.min_bin_instances = min_bin_instances
        self.min_recurrent_regions = min_recurrent_regions
        self.max_kde_points = max_kde_points
        self.penalty = penalty            # score returned for ineligible/degenerate combos
        self.n_samples = len(cache)
        self._pool_cache = {}             # quantile -> (pooled_novel, pooled_all, orange_values)
        self.trace = []                   # list of dicts, one per evaluation, in call order

    def _pooled(self, quantile):
        # round to keep the cache from growing unboundedly on continuous
        # Bayesian proposals that differ in the 10th decimal place
        key = round(float(quantile), 6)
        if key not in self._pool_cache:
            pooled_novel, pooled_all = self.gs.pool_for_quantile(self.cache, key)
            orange_values = pooled_all.loc[pooled_all["is_orange"], "value_raw"]
            self._pool_cache[key] = (pooled_novel, pooled_all, orange_values)
        return self._pool_cache[key]

    def evaluate(self, quantile, min_samples_ratio):
        quantile = float(np.clip(quantile, 0.5, 0.999))
        min_samples_ratio = float(np.clip(min_samples_ratio, 1.0 / self.n_samples, 1.0))

        pooled_novel, pooled_all, orange_values = self._pooled(quantile)
        min_samples = max(1, int(np.ceil(min_samples_ratio * self.n_samples)))
        merged, recurrence_counts = self.gs.label_recurrent(pooled_novel, pooled_all, min_samples)

        blue_mask = ~merged["is_orange"] & ~merged["is_recurrent"]
        blue_values = merged.loc[blue_mask, "value_raw"]
        recurrent_values = merged.loc[merged["is_recurrent"], "value_raw"]

        # Guardrail: minimizing overlap alone has a degenerate optimum at
        # the loosest possible min_samples cutoff (min_samples=1 just means
        # "high in >=1 sample", not genuine cross-sample recurrence, but it
        # still shrinks the overlap by pulling more bins out of blue). We
        # require actual evidence of cross-sample agreement -- at least
        # min_recurrent_regions contiguous recurrent regions -- before a
        # combo is allowed to be considered a real candidate.
        n_recurrent_bins, n_recurrent_regions = 0, 0
        if len(recurrence_counts):
            rec_bins_df = recurrence_counts[recurrence_counts["n_samples_high"] >= min_samples].copy()
            n_recurrent_bins = len(rec_bins_df)
            if n_recurrent_bins:
                rec_bins_df["mean_pon_median"] = np.nan
                rec_bins_df["n_samples_with_data"] = self.n_samples
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    regions_df = self.base.merge_adjacent_bins(rec_bins_df, self.rebin_to, self.merge_gap)
                n_recurrent_regions = len(regions_df)

        eligible = (
            len(orange_values) >= self.min_bin_instances
            and len(blue_values) >= self.min_bin_instances
            and n_recurrent_regions >= self.min_recurrent_regions
            and min_samples >= 2
        )
        metrics = self.gs.compute_separation_metrics(orange_values, blue_values, self.max_kde_points) if eligible else {
            "overlap_coefficient": np.nan, "ks_statistic": np.nan, "ks_pvalue": np.nan, "auc": np.nan
        }
        score = metrics["overlap_coefficient"]
        score_for_optimizer = self.penalty if (score is None or np.isnan(score)) else score

        self.trace.append({
            "eval_index": len(self.trace),
            "quantile": quantile,
            "min_samples_ratio": min_samples_ratio,
            "min_samples": min_samples,
            "n_orange_bin_instances": len(orange_values),
            "n_blue_bin_instances": len(blue_values),
            "n_recurrent_bins": n_recurrent_bins,
            "n_recurrent_regions": n_recurrent_regions,
            "eligible": eligible,
            "overlap_coefficient": score,
            "score_used": score_for_optimizer,
            **{k: v for k, v in metrics.items() if k != "overlap_coefficient"},
        })
        return score_for_optimizer, (orange_values, blue_values, recurrent_values)

    def __call__(self, params):
        """skopt/scipy-compatible: params = [quantile, min_samples_ratio]"""
        score, _ = self.evaluate(params[0], params[1])
        return score


# --------------------------------------------------------------------------
# Method 1: Bayesian optimization (GP surrogate)
# --------------------------------------------------------------------------

def run_bayesian(objective, quantile_bounds, ratio_bounds, n_calls, n_initial_points, random_state):
    try:
        from skopt import gp_minimize
        from skopt.space import Real
    except ImportError:
        sys.exit("Error: scikit-optimize is required for --method bayesian. "
                 "Install with: pip install scikit-optimize --break-system-packages")

    space = [Real(*quantile_bounds, name="quantile"),
             Real(*ratio_bounds, name="min_samples_ratio")]

    result = gp_minimize(
        objective, space,
        n_calls=n_calls, n_initial_points=min(n_initial_points, n_calls),
        random_state=random_state, acq_func="EI",
    )
    best_quantile, best_ratio = result.x
    best_score = result.fun
    return best_quantile, best_ratio, best_score


# --------------------------------------------------------------------------
# Method 2: Nelder-Mead, multi-start (closest derivative-free analogue to
# gradient descent for a non-differentiable objective)
# --------------------------------------------------------------------------

def _initial_simplex(x0, quantile_bounds, ratio_bounds):
    """A wider initial simplex than scipy's default (5% of x0) gives
    Nelder-Mead a real chance to sense a gradient across the large flat
    'ineligible' plateaus this objective has (see module docstring); too
    small a simplex can land entirely inside one flat region and 'converge'
    immediately with zero information."""
    q_span = (quantile_bounds[1] - quantile_bounds[0]) * 0.25
    r_span = (ratio_bounds[1] - ratio_bounds[0]) * 0.25
    p0 = np.array(x0, dtype=float)
    p1 = p0 + np.array([q_span, 0.0])
    p2 = p0 + np.array([0.0, r_span])
    simplex = np.vstack([p0, p1, p2])
    simplex[:, 0] = np.clip(simplex[:, 0], *quantile_bounds)
    simplex[:, 1] = np.clip(simplex[:, 1], *ratio_bounds)
    return simplex


def run_nelder_mead(objective, quantile_bounds, ratio_bounds, n_restarts, random_state):
    from scipy.optimize import minimize

    rng = np.random.default_rng(random_state)
    best = None

    for i in range(n_restarts):
        x0 = [
            rng.uniform(*quantile_bounds),
            rng.uniform(*ratio_bounds),
        ]

        def bounded_objective(x):
            # Nelder-Mead has no native bound support; penalize outside the box
            q, r = x
            if not (quantile_bounds[0] <= q <= quantile_bounds[1]) or \
               not (ratio_bounds[0] <= r <= ratio_bounds[1]):
                return 10.0
            return objective(x)

        res = minimize(bounded_objective, x0, method="Nelder-Mead",
                        options={"xatol": 1e-3, "fatol": 1e-4, "maxiter": 100, "adaptive": True,
                                 "initial_simplex": _initial_simplex(x0, quantile_bounds, ratio_bounds)})
        print(f"[nelder-mead restart {i+1}/{n_restarts}] x0={[round(v,3) for v in x0]} "
              f"-> x*={[round(v,3) for v in res.x]}, f*={res.fun:.4f}")
        if best is None or res.fun < best.fun:
            best = res

    return best.x[0], best.x[1], best.fun


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def plot_convergence(trace_df, out_path):
    trace_df = trace_df.sort_values("eval_index")
    best_so_far = trace_df["score_used"].cummin()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(trace_df["eval_index"], trace_df["score_used"], "o", alpha=0.35, ms=4, color="#7f9fc9",
            label="each evaluation")
    ax.plot(trace_df["eval_index"], best_so_far, "-", lw=2, color="#cc7a00", label="best so far")
    ax.set_xlabel("Evaluation #")
    ax.set_ylabel("overlap_coefficient (lower = better)")
    ax.set_title("Search convergence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_search_space(trace_df, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(trace_df["min_samples_ratio"], trace_df["quantile"],
                     c=trace_df["score_used"], cmap="viridis_r", s=40, edgecolors="k", linewidths=0.3)
    best_row = trace_df.loc[trace_df["score_used"].idxmin()]
    ax.scatter([best_row["min_samples_ratio"]], [best_row["quantile"]],
               marker="*", s=400, color="red", edgecolors="k", linewidths=0.5, label="best found", zorder=5)
    ax.set_xlabel("min_samples_ratio")
    ax.set_ylabel("quantile")
    ax.set_title("Evaluated points (color = overlap_coefficient, lower = better)")
    ax.legend()
    fig.colorbar(sc, ax=ax, shrink=0.8, label="overlap_coefficient")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wes-root", default=None)
    p.add_argument("--aa-root", default=None)
    p.add_argument("--manifest", default=None)
    p.add_argument("--classification-tsv", required=True)
    p.add_argument("--classification-bed-dir", required=True)
    p.add_argument("--sample-map", default=None)
    p.add_argument("--base-script",
                    default=str(Path(__file__).parent / "find_recurrent_novel_amplifications_binlevel.py"))
    p.add_argument("--grid-script",
                    default=str(Path(__file__).parent / "optimize_ecdna_bfb_cnc_thresholds.py"),
                    help="Path to optimize_ecdna_bfb_cnc_thresholds.py -- its pooling/labeling/metric "
                         "functions are reused directly so all scripts share one definition of the objective")
    p.add_argument("--column", default="predicted_loess_upscale_depth")
    p.add_argument("--mask-col", default="mask_rejected")
    p.add_argument("--rebin-to", type=int, default=25000)
    p.add_argument("--smooth-window", type=int, default=1)
    p.add_argument("--min-overlap-bp", type=int, default=1)
    p.add_argument("--merge-gap", type=int, default=0)

    p.add_argument("--method", choices=["bayesian", "nelder-mead"], default="bayesian")
    p.add_argument("--quantile-bounds", default="0.80,0.99")
    p.add_argument("--min-samples-ratio-bounds", default="0.05,1.0")
    p.add_argument("--n-calls", type=int, default=40,
                    help="[bayesian] total objective evaluations, including initial random points")
    p.add_argument("--n-initial-points", type=int, default=10,
                    help="[bayesian] random evaluations before the GP surrogate starts guiding the search")
    p.add_argument("--n-restarts", type=int, default=8,
                    help="[nelder-mead] number of random-start local searches (best of all kept)")
    p.add_argument("--random-state", type=int, default=0)

    p.add_argument("--min-bin-instances-for-selection", type=int, default=30)
    p.add_argument("--min-recurrent-regions", type=int, default=2,
                    help="Guardrail: a combo must produce at least this many recurrent regions "
                         "(real cross-sample agreement) to be eligible, otherwise minimizing overlap "
                         "alone degenerates toward min_samples=1 (see module docstring note on this)")
    p.add_argument("--max-kde-points", type=int, default=20000,
                    help="Cap each group's size before KDE fitting -- KDE cost scales linearly with n "
                         "and dominates runtime at genome-wide bin resolution (see grid script for "
                         "the same flag / rationale)")
    p.add_argument("--outdir", default="ecdna_bfb_cnc_bayesian_search")
    p.add_argument("--highlight-sample", default=None,
                    help="A wes-root/aa-root sample id (e.g. an SRR accession) -- if set, also produce "
                         "a single-sample version of best_combo_histogram.png filtered to just that "
                         "sample's own bins, matching the original per-sample plot style")
    p.add_argument("--min-depth", type=float, default=0.1,
                    help="Histogram plots only include bins with raw_wes_depth above this value. Does "
                         "not affect the search itself, only what's drawn.")
    p.add_argument("--raw-depth-col", default="raw_wes_depth",
                    help="Column name in the copy-ratio TSVs used for --min-depth filtering. Separate "
                         "from --column (the value actually optimized on).")
    args = p.parse_args()

    if not args.manifest and not (args.wes_root and args.aa_root):
        sys.exit("Error: provide either --manifest, or both --wes-root and --aa-root.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    base = load_module(args.base_script, "base_recurrence")
    gs = load_module(args.grid_script, "grid_search_ecdna")

    sample_table, skip_log = (base.load_manifest(args.manifest), []) if args.manifest \
        else base.discover_samples(args.wes_root, args.aa_root)
    if sample_table.empty:
        sys.exit("Error: no samples with both a copy ratio TSV and a seed BED were found.")
    print(f"Found {len(sample_table)} samples with depth + seed data.")

    qualifying_by_sample = gs.load_qualifying_amplicons_by_sample(args.classification_tsv)
    bed_index = gs.index_classification_bed_dir(args.classification_bed_dir)
    sample_name_map = gs.load_sample_name_map(args.sample_map) if args.sample_map else {}

    cache, more_skips = gs.build_per_sample_cache(
        base, sample_table, args.column, args.mask_col, args.rebin_to,
        args.smooth_window, args.min_overlap_bp, qualifying_by_sample, bed_index, sample_name_map,
        args.raw_depth_col
    )
    skip_log += more_skips
    if not cache:
        sys.exit("Error: no samples loaded successfully.")

    total_orange = sum(e["n_orange_bins"] for e in cache)
    if total_orange == 0:
        sys.exit("Error: 0 orange (ecDNA/BFB/CNC) bins found across all samples. See per-sample [warn] lines.")

    quantile_bounds = tuple(float(x) for x in args.quantile_bounds.split(","))
    ratio_bounds = tuple(float(x) for x in args.min_samples_ratio_bounds.split(","))

    objective = Objective(cache, gs, base, args.rebin_to, args.merge_gap,
                          args.min_bin_instances_for_selection, args.min_recurrent_regions,
                          args.max_kde_points)

    print(f"\nStarting {args.method} search over quantile in {quantile_bounds}, "
          f"min_samples_ratio in {ratio_bounds}\n")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        if args.method == "bayesian":
            best_q, best_r, best_score = run_bayesian(
                objective, quantile_bounds, ratio_bounds, args.n_calls, args.n_initial_points, args.random_state
            )
        else:
            best_q, best_r, best_score = run_nelder_mead(
                objective, quantile_bounds, ratio_bounds, args.n_restarts, args.random_state
            )

    trace_df = pd.DataFrame(objective.trace)
    trace_df.to_csv(outdir / "search_trace.csv", index=False)
    print(f"\nSaved: {outdir / 'search_trace.csv'} ({len(trace_df)} evaluations)")

    plot_convergence(trace_df, outdir / "convergence.png")
    plot_search_space(trace_df, outdir / "search_space.png")
    print(f"Saved: {outdir / 'convergence.png'}, {outdir / 'search_space.png'}")

    # best_min_samples is derived here (not from a re-evaluate() call) since
    # plot_pooled_histogram recomputes the merged data itself, fresh, with
    # raw_depth filtering applied -- the Objective's cached value-only
    # tuples don't carry raw_depth, so they're no longer used for plotting
    best_min_samples = max(1, int(np.ceil(best_r * len(cache))))

    summary_path = outdir / "best_combo_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"ecDNA/BFB/CNC vs background threshold search ({args.method})\n" + "=" * 55 + "\n\n")
        f.write(f"Samples processed: {len(cache)} (skipped: {len(skip_log)})\n")
        f.write(f"Total orange (ecDNA/BFB/CNC) bin-instances: {total_orange}\n")
        f.write(f"Search bounds: quantile in {quantile_bounds}, min_samples_ratio in {ratio_bounds}\n")
        f.write(f"Total evaluations: {len(trace_df)}\n\n")
        f.write("BEST COMBO FOUND:\n")
        f.write(f"  quantile           = {best_q:.4f}\n")
        f.write(f"  min_samples_ratio   = {best_r:.4f}\n")
        f.write(f"  min_samples         = {best_min_samples} / {len(cache)}\n")
        f.write(f"  overlap_coefficient = {best_score:.4f}\n\n")
        f.write("Top 15 evaluated points by overlap_coefficient:\n")
        f.write(trace_df.sort_values("score_used").head(15).to_string(index=False))
        f.write("\n")
        if skip_log:
            f.write("\nSkipped samples/stages:\n")
            f.write(pd.DataFrame(skip_log).to_string(index=False))
            f.write("\n")
    print(f"Saved: {summary_path}")

    gs.plot_pooled_histogram(
        cache, round(best_q, 4), best_min_samples, round(best_r, 4),
        outdir / "best_combo_histogram.png", min_depth=args.min_depth,
    )

    if args.highlight_sample:
        gs.plot_single_sample_histogram(
            cache, round(best_q, 4), best_min_samples, round(best_r, 4),
            args.highlight_sample, outdir / f"best_combo_histogram_{args.highlight_sample}.png",
            min_depth=args.min_depth,
        )

    print(f"\nBest combo: quantile={best_q:.4f}, min_samples_ratio={best_r:.4f} "
          f"(min_samples={best_min_samples}/{len(cache)}), overlap_coefficient={best_score:.4f}")


if __name__ == "__main__":
    main()
