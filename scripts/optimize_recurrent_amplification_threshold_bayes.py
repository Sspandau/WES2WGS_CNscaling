#!/usr/bin/env python3
"""
optimize_recurrent_amplification_threshold_bayes.py

Replaces the exhaustive grid search in optimize_recurrent_amplification_threshold.py
with a sequential search over the same two parameters (--quantile,
--min-samples-ratio), using the same objective (minimize the KDE overlap
coefficient between ORANGE = AC-classified ecDNA+/BFB+/FAN+/Complex-non-cyclic
bins, and BLUE = background bins not caught by the statistical recurrence
caller).

RECURRENT-BIN VOTING: WHO VOTES, AND WHAT THEY VOTE ON
--------------------------------------------------------
Identical to the grid script. By default, samples with >=1 ecDNA+, BFB+
(or FAN+), OR Complex-non-cyclic (CNC) amplicon are excluded from voting on
what counts as "recurrent" (their complex amplicon architecture can produce
broadly elevated/unstable depth beyond just their classified region, which
would bias the recurrence caller if allowed to contribute). Toggle each
category independently with
--include-ecdna-bfb-samples-in-recurrent-calling /
--include-cnc-samples-in-recurrent-calling. Excluded samples still appear
in the pooled background/orange data -- they just don't cast a vote.

min_samples is computed against the number of VOTING samples:
    min_samples = ceil(min_samples_ratio * n_voting_samples)
(using the full cohort size would silently make every ratio stricter than
intended once any samples are excluded -- same rule as the grid script).

For samples that DO vote, seed regions (AA_CNV_SEEDS.bed) are NOT excluded
from the recurrent-bin search -- a seed-overlapping bin can compete for
and become a recurrent call the same as any other bin. (Seed regions
still play no role in orange, which is AC ground truth and untouched by
any of this.)

The voting logic itself lives in the grid script's pool_for_quantile() and
build_per_sample_cache(), which this script imports -- so the two scripts
cannot drift apart on it.

WHY NOT TRUE GRADIENT DESCENT
------------------------------
The objective is not differentiable: quantile thresholding and the integer
min-samples cutoff both make hard, discontinuous decisions about which bins
land in which bucket. Two practical options given that:

  --method bayesian     (default) Gaussian-Process surrogate search
                         (scikit-optimize's gp_minimize) -- the standard
                         approach for a cheap-ish, noisy, non-smooth,
                         low-dimensional black-box function.

  --method nelder-mead   Derivative-free local simplex search from several
                         random restarts; best result kept.

IMPORTANT: MINIMIZING OVERLAP ALONE HAS A DEGENERATE OPTIMUM
---------------------------------------------------------------
Pulling MORE bins out of "blue" always mechanically shrinks the overlap
with orange, whether or not those bins represent genuine cross-sample
recurrence. Unconstrained, the search drifts to the loosest corner of the
space and labels a large fraction of the genome "recurrent". Three
guardrails make such combos ineligible (penalized, never reported as best):
  - min_samples >= 2
  - at least --min-recurrent-regions contiguous recurrent regions
  - recurrent bin-instances <= --max-recurrent-fraction of
    (recurrent + blue) bin-instances (default 0.15, same as the grid script)

OUTPUT (--outdir)
------------------
  search_trace.csv                      every evaluated point, in order, with its score
  convergence.png                       best-score-so-far vs. evaluation number
  search_space.png                      evaluated points over quantile x min_samples_ratio
  best_combo_summary.txt                recommended combo
  best_combo_histogram.png              3-way histogram (blue / orange / recurrent)
  best_combo_recurrent_orange_overlap.csv
                                        every recurrent bin for the best combo, with
                                        n_samples_high and overlaps_orange
  best_combo_recurrent_regions.tsv      those bins merged into contiguous regions
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
# Import both prior scripts as modules
# --------------------------------------------------------------------------

REQUIRED_GRID_FUNCS = (
    "load_qualifying_amplicons_by_sample", "load_samples_with_ecdna_or_bfb", "load_samples_with_cnc",
    "index_classification_bed_dir", "load_sample_name_map", "build_per_sample_cache",
    "pool_for_quantile", "label_recurrent", "compute_separation_metrics", "compute_orange_locations",
    "save_recurrent_orange_overlap_detail", "plot_pooled_histogram", "plot_single_sample_histogram",
)


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
    """Wraps the grid script's pooling/labeling/metric functions into a plain
    f(quantile, min_samples_ratio) -> overlap_coefficient callable, with
    per-quantile pooled results cached."""

    def __init__(self, cache, gs, base, rebin_to, merge_gap, min_bin_instances, min_recurrent_regions=2,
                 max_recurrent_fraction=0.15, max_kde_points=20000, penalty=1.0):
        self.cache = cache
        self.gs = gs
        self.base = base
        self.rebin_to = rebin_to
        self.merge_gap = merge_gap
        self.min_bin_instances = min_bin_instances
        self.min_recurrent_regions = min_recurrent_regions
        self.max_recurrent_fraction = max_recurrent_fraction
        self.max_kde_points = max_kde_points
        self.penalty = penalty
        self.n_samples = len(cache)
        # min_samples is relative to samples that can actually vote (see docstring)
        self.n_voting = sum(1 for e in cache if not e.get("excluded_from_recurrent", False))
        self._pool_cache = {}
        self.trace = []

    def min_samples_for(self, min_samples_ratio):
        ratio = float(np.clip(min_samples_ratio, 1.0 / self.n_voting, 1.0))
        return ratio, max(1, int(np.ceil(ratio * self.n_voting)))

    @staticmethod
    def quantile_key(quantile):
        # round so continuous proposals differing in the 10th decimal share a cache entry
        return round(float(np.clip(quantile, 0.5, 0.999)), 6)

    def _pooled(self, key):
        if key not in self._pool_cache:
            pooled_novel, pooled_all = self.gs.pool_for_quantile(self.cache, key)
            orange_values = pooled_all.loc[pooled_all["is_orange"], "value_raw"]
            self._pool_cache[key] = (pooled_novel, pooled_all, orange_values)
        return self._pool_cache[key]

    def evaluate(self, quantile, min_samples_ratio):
        quantile = self.quantile_key(quantile)
        min_samples_ratio, min_samples = self.min_samples_for(min_samples_ratio)

        pooled_novel, pooled_all, orange_values = self._pooled(quantile)
        merged, recurrence_counts = self.gs.label_recurrent(pooled_novel, pooled_all, min_samples)

        blue_mask = ~merged["is_orange"] & ~merged["is_recurrent"]
        blue_values = merged.loc[blue_mask, "value_raw"]
        n_recurrent_instances = int(merged["is_recurrent"].sum())
        denom = n_recurrent_instances + len(blue_values)
        recurrent_fraction = n_recurrent_instances / denom if denom else np.nan

        n_recurrent_bins, n_recurrent_regions = 0, 0
        if len(recurrence_counts):
            rec_bins_df = recurrence_counts[recurrence_counts["n_samples_high"] >= min_samples].copy()
            n_recurrent_bins = len(rec_bins_df)
            if n_recurrent_bins:
                rec_bins_df["mean_pon_median"] = np.nan
                rec_bins_df["n_samples_with_data"] = self.n_voting
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    regions_df = self.base.merge_adjacent_bins(rec_bins_df, self.rebin_to, self.merge_gap)
                n_recurrent_regions = len(regions_df)

        eligible = bool(
            len(orange_values) >= self.min_bin_instances
            and len(blue_values) >= self.min_bin_instances
            and n_recurrent_regions >= self.min_recurrent_regions
            and min_samples >= 2
            and np.isfinite(recurrent_fraction)
            and recurrent_fraction <= self.max_recurrent_fraction
        )
        metrics = self.gs.compute_separation_metrics(orange_values, blue_values, self.max_kde_points) \
            if eligible else {"overlap_coefficient": np.nan, "ks_statistic": np.nan,
                              "ks_pvalue": np.nan, "auc": np.nan}
        score = metrics["overlap_coefficient"]
        score_for_optimizer = self.penalty if (score is None or np.isnan(score)) else score

        self.trace.append({
            "eval_index": len(self.trace),
            "quantile": quantile,
            "min_samples_ratio": min_samples_ratio,
            "min_samples": min_samples,
            "n_voting_samples": self.n_voting,
            "n_orange_bin_instances": len(orange_values),
            "n_blue_bin_instances": len(blue_values),
            "n_recurrent_bin_instances": n_recurrent_instances,
            "recurrent_fraction_of_genome": recurrent_fraction,
            "n_recurrent_bins": n_recurrent_bins,
            "n_recurrent_regions": n_recurrent_regions,
            "eligible": eligible,
            "overlap_coefficient": score,
            "score_used": score_for_optimizer,
            **{k: v for k, v in metrics.items() if k != "overlap_coefficient"},
        })
        return score_for_optimizer

    def __call__(self, params):
        """skopt/scipy-compatible: params = [quantile, min_samples_ratio]"""
        return self.evaluate(params[0], params[1])


# --------------------------------------------------------------------------
# Method 1: Bayesian optimization (GP surrogate)
# --------------------------------------------------------------------------

def run_bayesian(objective, quantile_bounds, ratio_bounds, n_calls, n_initial_points, random_state):
    try:
        from skopt import gp_minimize
        from skopt.space import Real
    except ImportError:
        sys.exit("Error: scikit-optimize is required for --method bayesian. "
                 "Install with: pip install scikit-optimize")

    space = [Real(*quantile_bounds, name="quantile"),
             Real(*ratio_bounds, name="min_samples_ratio")]
    gp_minimize(
        objective, space,
        n_calls=n_calls, n_initial_points=min(n_initial_points, n_calls),
        random_state=random_state, acq_func="EI",
    )


# --------------------------------------------------------------------------
# Method 2: Nelder-Mead, multi-start
# --------------------------------------------------------------------------

def _initial_simplex(x0, quantile_bounds, ratio_bounds):
    """Wider-than-default initial simplex so Nelder-Mead can sense a direction
    across the large flat 'ineligible' plateaus this objective has."""
    q_span = (quantile_bounds[1] - quantile_bounds[0]) * 0.25
    r_span = (ratio_bounds[1] - ratio_bounds[0]) * 0.25
    p0 = np.array(x0, dtype=float)
    simplex = np.vstack([p0, p0 + [q_span, 0.0], p0 + [0.0, r_span]])
    simplex[:, 0] = np.clip(simplex[:, 0], *quantile_bounds)
    simplex[:, 1] = np.clip(simplex[:, 1], *ratio_bounds)
    return simplex


def run_nelder_mead(objective, quantile_bounds, ratio_bounds, n_restarts, random_state):
    from scipy.optimize import minimize

    rng = np.random.default_rng(random_state)

    def bounded_objective(x):
        q, r = x
        if not (quantile_bounds[0] <= q <= quantile_bounds[1]) or \
           not (ratio_bounds[0] <= r <= ratio_bounds[1]):
            return 10.0          # outside the box; not recorded in the trace
        return objective(x)

    for i in range(n_restarts):
        x0 = [rng.uniform(*quantile_bounds), rng.uniform(*ratio_bounds)]
        res = minimize(bounded_objective, x0, method="Nelder-Mead",
                       options={"xatol": 1e-3, "fatol": 1e-4, "maxiter": 100, "adaptive": True,
                                "initial_simplex": _initial_simplex(x0, quantile_bounds, ratio_bounds)})
        print(f"[nelder-mead restart {i+1}/{n_restarts}] x0={[round(v, 3) for v in x0]} "
              f"-> x*={[round(v, 3) for v in res.x]}, f*={res.fun:.4f}")


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
    ax.set_ylabel("overlap_coefficient (lower = better; ineligible = penalty)")
    ax.set_title("Search convergence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_search_space(trace_df, best_row, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    elig = trace_df["eligible"].astype(bool)
    if (~elig).any():
        ax.scatter(trace_df.loc[~elig, "min_samples_ratio"], trace_df.loc[~elig, "quantile"],
                   marker="x", s=30, color="grey", label="ineligible (guardrail)")
    if elig.any():
        sc = ax.scatter(trace_df.loc[elig, "min_samples_ratio"], trace_df.loc[elig, "quantile"],
                        c=trace_df.loc[elig, "overlap_coefficient"], cmap="viridis_r", s=40,
                        edgecolors="k", linewidths=0.3)
        fig.colorbar(sc, ax=ax, shrink=0.8, label="overlap_coefficient")
    if best_row is not None:
        ax.scatter([best_row["min_samples_ratio"]], [best_row["quantile"]], marker="*", s=400,
                   color="red", edgecolors="k", linewidths=0.5, label="best eligible", zorder=5)
    ax.set_xlabel("min_samples_ratio (of voting samples)")
    ax.set_ylabel("quantile")
    ax.set_title("Evaluated points (color = overlap_coefficient, lower = better)")
    ax.legend()
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
                   default=str(Path(__file__).parent / "optimize_recurrent_amplification_threshold.py"),
                   help="The grid-search script -- its loading/voting/pooling/metric functions are "
                        "reused directly so both scripts share one definition of the objective")
    p.add_argument("--orange-categories", default="ecDNA,BFB,CNC,FAN",
                   help="Which AC categories count as orange (any of ecDNA, BFB, CNC, FAN)")
    p.add_argument("--include-ecdna-bfb-samples-in-recurrent-calling", action="store_true",
                   help="Let samples with >=1 ecDNA+/BFB+/FAN+ amplicon vote on recurrence "
                        "(default: they're excluded from voting)")
    p.add_argument("--include-cnc-samples-in-recurrent-calling", action="store_true",
                   help="Let samples with >=1 Complex-non-cyclic amplicon vote on recurrence "
                        "(default: they're excluded from voting)")
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
                   help="[nelder-mead] number of random-start local searches")
    p.add_argument("--random-state", type=int, default=0)

    p.add_argument("--min-bin-instances-for-selection", type=int, default=30)
    p.add_argument("--min-recurrent-regions", type=int, default=2)
    p.add_argument("--max-recurrent-fraction", type=float, default=0.15,
                   help="Guardrail: combos whose recurrent bin-instances exceed this fraction of "
                        "(recurrent + blue) are ineligible (same rule as the grid script)")
    p.add_argument("--max-kde-points", type=int, default=20000)
    p.add_argument("--outdir", default="recurrent_threshold_bayesian_search")
    p.add_argument("--highlight-sample", default=None)
    p.add_argument("--min-depth", type=float, default=0.1,
                   help="Histogram plots only include bins with raw depth above this value")
    p.add_argument("--raw-depth-col", default="raw_wes_depth")
    args = p.parse_args()

    if not args.manifest and not (args.wes_root and args.aa_root):
        sys.exit("Error: provide either --manifest, or both --wes-root and --aa-root.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    base = load_module(args.base_script, "base_recurrence")
    gs = load_module(args.grid_script, "grid_search_ecdna")
    missing = [f for f in REQUIRED_GRID_FUNCS if not hasattr(gs, f)]
    if missing:
        sys.exit(f"Error: {args.grid_script} is missing {missing} -- point --grid-script at the "
                 f"current optimize_recurrent_amplification_threshold.py (with sample voting logic).")

    sample_table, skip_log = (base.load_manifest(args.manifest), []) if args.manifest \
        else base.discover_samples(args.wes_root, args.aa_root)
    if sample_table.empty:
        sys.exit("Error: no samples with both a copy ratio TSV and a seed BED were found.")
    print(f"Found {len(sample_table)} samples with depth + seed data.")

    qualifying_by_sample = gs.load_qualifying_amplicons_by_sample(
        args.classification_tsv, tuple(args.orange_categories.split(",")))

    # --- who votes on recurrence (same as the grid script's main()) ---
    excluded_samples = set()
    if not args.include_ecdna_bfb_samples_in_recurrent_calling:
        s = gs.load_samples_with_ecdna_or_bfb(args.classification_tsv)
        print(f"{len(s)} samples have >=1 ecDNA+/BFB+/FAN+ amplicon -> excluded from recurrent voting")
        excluded_samples |= s
    if not args.include_cnc_samples_in_recurrent_calling:
        s = gs.load_samples_with_cnc(args.classification_tsv)
        print(f"{len(s)} samples have >=1 Complex-non-cyclic amplicon -> excluded from recurrent voting")
        excluded_samples |= s

    bed_index = gs.index_classification_bed_dir(args.classification_bed_dir)
    sample_name_map = gs.load_sample_name_map(args.sample_map) if args.sample_map else {}

    cache, more_skips = gs.build_per_sample_cache(
        base, sample_table, args.column, args.mask_col, args.rebin_to,
        args.smooth_window, args.min_overlap_bp, qualifying_by_sample, bed_index, sample_name_map,
        args.raw_depth_col, excluded_samples,
    )
    skip_log += more_skips
    if not cache:
        sys.exit("Error: no samples loaded successfully.")

    total_orange = sum(e["n_orange_bins"] for e in cache)
    if total_orange == 0:
        sys.exit("Error: 0 orange bins found across all samples. See per-sample [warn] lines.")

    quantile_bounds = tuple(float(x) for x in args.quantile_bounds.split(","))
    ratio_bounds = tuple(float(x) for x in args.min_samples_ratio_bounds.split(","))

    objective = Objective(cache, gs, base, args.rebin_to, args.merge_gap,
                          args.min_bin_instances_for_selection, args.min_recurrent_regions,
                          args.max_recurrent_fraction, args.max_kde_points)
    if objective.n_voting < 2:
        sys.exit(f"Error: only {objective.n_voting} sample(s) are eligible to vote on recurrence "
                 f"(min_samples must be >= 2). Use the --include-*-in-recurrent-calling flags.")
    print(f"\n{len(cache)} samples loaded; {objective.n_voting} vote on recurrence, "
          f"{len(cache) - objective.n_voting} excluded (ecDNA/BFB/FAN/CNC). "
          f"min_samples_ratio is relative to the {objective.n_voting} voting samples.")

    print(f"\nStarting {args.method} search over quantile in {quantile_bounds}, "
          f"min_samples_ratio in {ratio_bounds}\n")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        if args.method == "bayesian":
            run_bayesian(objective, quantile_bounds, ratio_bounds,
                         args.n_calls, args.n_initial_points, args.random_state)
        else:
            run_nelder_mead(objective, quantile_bounds, ratio_bounds, args.n_restarts, args.random_state)

    trace_df = pd.DataFrame(objective.trace)
    trace_df.to_csv(outdir / "search_trace.csv", index=False)
    print(f"\nSaved: {outdir / 'search_trace.csv'} ({len(trace_df)} evaluations)")

    # best = lowest overlap among ELIGIBLE points only (tie-break: higher AUC)
    eligible_df = trace_df[trace_df["eligible"].astype(bool) & trace_df["overlap_coefficient"].notna()]
    best_row = None if eligible_df.empty else \
        eligible_df.sort_values(["overlap_coefficient", "auc"], ascending=[True, False]).iloc[0]

    plot_convergence(trace_df, outdir / "convergence.png")
    plot_search_space(trace_df, best_row, outdir / "search_space.png")
    print(f"Saved: {outdir / 'convergence.png'}, {outdir / 'search_space.png'}")

    summary_path = outdir / "best_combo_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Recurrent-amplification threshold search ({args.method})\n" + "=" * 55 + "\n\n")
        f.write(f"Samples processed: {len(cache)} (skipped: {len(skip_log)})\n")
        f.write(f"Voting on recurrence: {objective.n_voting} "
                f"(excluded: {len(cache) - objective.n_voting} ecDNA/BFB/FAN/CNC samples)\n")
        f.write(f"Total orange bin-instances: {total_orange}\n")
        f.write(f"Search bounds: quantile in {quantile_bounds}, min_samples_ratio in {ratio_bounds}\n")
        f.write(f"Guardrails: min_samples >= 2, recurrent regions >= {args.min_recurrent_regions}, "
                f"recurrent fraction <= {args.max_recurrent_fraction}\n")
        f.write(f"Total evaluations: {len(trace_df)} ({len(eligible_df)} eligible)\n\n")
        if best_row is None:
            f.write("No eligible combo found -- widen the bounds, raise --max-recurrent-fraction, "
                    "or increase --n-calls.\n")
        else:
            f.write("BEST ELIGIBLE COMBO:\n")
            f.write(f"  quantile                = {best_row['quantile']:.4f}\n")
            f.write(f"  min_samples_ratio       = {best_row['min_samples_ratio']:.4f}\n")
            f.write(f"  min_samples             = {int(best_row['min_samples'])} / {objective.n_voting} voting\n")
            f.write(f"  overlap_coefficient     = {best_row['overlap_coefficient']:.4f}\n")
            f.write(f"  auc                     = {best_row['auc']:.4f}\n")
            f.write(f"  recurrent bins/regions  = {int(best_row['n_recurrent_bins'])} / "
                    f"{int(best_row['n_recurrent_regions'])}\n")
            f.write(f"  recurrent fraction      = {best_row['recurrent_fraction_of_genome']:.3f}\n\n")
            f.write("Top 15 eligible points by overlap_coefficient:\n")
            f.write(eligible_df.sort_values(["overlap_coefficient", "auc"], ascending=[True, False])
                    .head(15).to_string(index=False))
            f.write("\n")
        if skip_log:
            f.write("\nSkipped samples/stages:\n")
            f.write(pd.DataFrame(skip_log).to_string(index=False))
            f.write("\n")
    print(f"Saved: {summary_path}")

    if best_row is None:
        print("No eligible combo found -- see best_combo_summary.txt.")
        return

    best_q = float(best_row["quantile"])            # already the exact key the objective used
    best_r = float(best_row["min_samples_ratio"])
    best_min_samples = int(best_row["min_samples"])

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        gs.plot_pooled_histogram(cache, best_q, best_min_samples, round(best_r, 4),
                                 outdir / "best_combo_histogram.png", min_depth=args.min_depth)
        if args.highlight_sample:
            gs.plot_single_sample_histogram(
                cache, best_q, best_min_samples, round(best_r, 4), args.highlight_sample,
                outdir / f"best_combo_histogram_{args.highlight_sample}.png", min_depth=args.min_depth)

        # --- recurrent bins + regions for the best combo ---
        orange_locations = gs.compute_orange_locations(cache)
        bins_path = outdir / "best_combo_recurrent_orange_overlap.csv"
        n_rec, n_overlap = gs.save_recurrent_orange_overlap_detail(
            cache, best_q, best_min_samples, orange_locations, bins_path)
        print(f"Saved: {bins_path} ({n_rec} recurrent bins, {n_overlap} overlap orange)")
        if n_rec:
            rec = pd.read_csv(bins_path).drop(columns="overlaps_orange")
            rec["mean_pon_median"] = np.nan
            rec["n_samples_with_data"] = objective.n_voting
            regions = base.merge_adjacent_bins(rec, args.rebin_to, args.merge_gap)
            regions_path = outdir / "best_combo_recurrent_regions.tsv"
            regions.to_csv(regions_path, sep="\t", index=False)
            print(f"Saved: {regions_path} ({len(regions)} regions)")

    print(f"\nBest eligible combo: quantile={best_q:.4f}, min_samples_ratio={best_r:.4f} "
          f"(min_samples={best_min_samples}/{objective.n_voting} voting), "
          f"overlap_coefficient={best_row['overlap_coefficient']:.4f}")


if __name__ == "__main__":
    main()
