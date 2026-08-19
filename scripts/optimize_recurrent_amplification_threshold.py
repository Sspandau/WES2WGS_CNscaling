#!/usr/bin/env python3
"""
optimize_ecdna_bfb_cnc_thresholds.py

Grid-search --quantile and --min-samples (the two parameters of
find_recurrent_novel_amplifications_binlevel.py) to MINIMIZE the overlap
between:

  ORANGE = bins belonging to amplicons AmpliconClassifier (AC) called
           ecDNA+, BFB+, or "Complex non-cyclic" (CNC) -- i.e. real
           AC-classified ecDNA/BFB/CNC amplicons. This is GROUND TRUTH and
           does not depend on quantile/min-samples at all.

  BLUE   = every bin that is BOTH (a) not part of an orange amplicon, AND
           (b) not caught by the statistical recurrent-bin caller at the
           current (quantile, min_samples) setting.

WHY THIS IS THE RIGHT OPTIMIZATION TARGET
-------------------------------------------
Orange is fixed -- it doesn't change as you sweep parameters. What DOES
change is which bins get pulled out of "everything else" and into
"recurrent" as quantile/min-samples move. If those two parameters are
well-tuned, the statistical caller successfully identifies and removes
truly-elevated bins (real amplification signal that isn't AC-classified,
e.g. because AA didn't call/reconstruct it, or because it's recurrent but
subthreshold in any single sample) from the background, leaving blue as
clean/low as possible -- which is exactly what minimizes overlap with
orange. A badly-tuned combo (quantile too high, min-samples too strict)
leaves elevated bins sitting in blue, inflating the overlap with orange.

DATA SOURCES
------------
1. Same off_target_copy_ratios.tsv + AA_CNV_SEEDS.bed inputs as
   find_recurrent_novel_amplifications_binlevel.py, for the per-sample
   depth values and the existing seed-exclusion logic (imported directly
   from that script so the two never drift apart).

2. NEW: per-sample AmpliconClassifier outputs, expected at
      <classification-root>/<sample>/<sample>_amplicon_classification_profiles.tsv
      <classification-root>/<sample>/<sample>_classification_bed_files/*.bed
   matching AC's actual documented output layout
   (github.com/AmpliconSuite/AmpliconClassifier):
     - amplicon_classification_profiles.tsv columns used:
         amplicon_number, amplicon_decomposition_class, ecDNA+, BFB+
       An amplicon QUALIFIES as orange if:
         ecDNA+ == "Positive" OR BFB+ == "Positive"
         OR amplicon_decomposition_class == "Complex non-cyclic"
       (this deliberately excludes Linear, No amp/Invalid, and Virus)
     - classification_bed_files/*.bed: AC writes one BED per classified
       feature per amplicon. Any file whose name contains "unknown" is
       skipped (AC's own convention for regions it couldn't confidently
       assign). A file is matched to its amplicon via an "amplicon<N>"
       substring in its filename, and only kept if that amplicon number
       qualified above.
   If your directory layout or filenames differ, adjust
   --classification-glob / edit discover_classification() -- the matching
   logic is intentionally isolated in one place.

OUTPUT (--outdir)
------------------
  grid_results.csv           quantile x min_samples_ratio x separation metrics
  overlap_heatmap.png        heatmap of overlap_coefficient (orange vs blue)
  auc_heatmap.png            heatmap of auc (orange vs blue)
  n_bg_removed_heatmap.png   how many background bins got reclassified as
                              recurrent at each combo (sanity check -- this
                              should generally track with less overlap)
  best_combo_summary.txt     recommended combo + ranked table
  best_combo_histogram.png   three-way histogram (blue / orange / recurrent)
                              styled like your example plot, for the winner
"""

import argparse
import glob
import importlib.util
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# Import the base script (per-sample pipeline reused verbatim)
# --------------------------------------------------------------------------

def load_base_module(path):
    path = Path(path)
    if not path.exists():
        sys.exit(
            f"Error: base script not found at {path}. Pass --base-script "
            f"/path/to/find_recurrent_novel_amplifications_binlevel.py"
        )
    spec = importlib.util.spec_from_file_location("base_recurrence", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_grid(spec):
    if ":" in spec:
        start, stop, step = (float(x) for x in spec.split(":"))
        n = int(round((stop - start) / step)) + 1
        vals = [round(start + i * step, 10) for i in range(n)]
        return [v for v in vals if v <= stop + 1e-9]
    return [float(x) for x in spec.split(",") if x.strip()]


# --------------------------------------------------------------------------
# AmpliconClassifier outputs: which amplicons qualify, and their intervals.
#
# Real-world layout observed: AC was run in BATCH mode over the whole
# cohort with one output prefix, so there is ONE consolidated
# <prefix>_amplicon_classification_profiles.tsv (sample_name is a column,
# not part of a per-sample directory), and correspondingly ONE
# <prefix>_classification_bed_files/ directory holding every sample's BED
# files together, named by AC's own convention:
#     <sample_name>_amplicon<N>_<feature>_<index>_intervals.bed
# (feature files marked "unknown" are AC's own low-confidence bucket and
# are skipped, same as before.)
#
# amplicon_decomposition_class values seen in the real file: "Cyclic",
# "Complex-non-cyclic" (hyphenated -- note, NOT "Complex non-cyclic"),
# "Linear". Matching below is case/hyphen/space-insensitive to be safe
# across AC versions.
# --------------------------------------------------------------------------

FILENAME_SAMPLE_AMPLICON_RE = re.compile(r"^(?P<sample>.+)_amplicon(?P<num>\d+)_", re.IGNORECASE)


def _normalize_class_label(s):
    return re.sub(r"[\s_-]+", "", str(s).strip().lower())


def load_qualifying_amplicons_by_sample(classification_tsv, categories=("ecDNA", "BFB", "CNC")):
    """Reads the ONE consolidated classification TSV and returns
    {sample_name: {qualifying amplicon_number strings}} for amplicons AC
    called ecDNA+, BFB+, and/or Complex-non-cyclic, restricted to whichever
    of those three `categories` are requested. Useful for isolating ecDNA
    alone, since ecDNA tends to sit at much higher/more sharply focal depth
    than BFB or Complex-non-cyclic amplicons -- pooling all three widens
    the orange distribution and can visually (and numerically) look less
    separated from background even with nothing wrong in the pipeline."""
    categories = {c.strip().lower() for c in categories}
    unknown = categories - {"ecdna", "bfb", "cnc"}
    if unknown:
        raise ValueError(f"Unknown --orange-categories value(s): {unknown}. Valid: ecDNA, BFB, CNC")

    df = pd.read_csv(classification_tsv, sep="\t", engine="python")

    required = {"sample_name", "amplicon_number", "amplicon_decomposition_class"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{classification_tsv} missing required columns: {missing}. "
                          f"Found: {list(df.columns)}")

    is_ecdna = (df["ecDNA+"].astype(str).str.strip() == "Positive") if "ecDNA+" in df.columns \
        else pd.Series(False, index=df.index)
    is_bfb = (df["BFB+"].astype(str).str.strip() == "Positive") if "BFB+" in df.columns \
        else pd.Series(False, index=df.index)
    normalized_class = df["amplicon_decomposition_class"].map(_normalize_class_label)
    is_cnc = normalized_class == "complexnoncyclic"

    qualifies = pd.Series(False, index=df.index)
    if "ecdna" in categories:
        qualifies |= is_ecdna
    if "bfb" in categories:
        qualifies |= is_bfb
    if "cnc" in categories:
        qualifies |= is_cnc

    sub = df.loc[qualifies, ["sample_name", "amplicon_number"]].copy()
    sub["sample_name"] = sub["sample_name"].astype(str).str.strip()
    sub["amplicon_number"] = sub["amplicon_number"].astype(str).str.strip()

    qualifying_by_sample = {}
    for sample_name, grp in sub.groupby("sample_name"):
        qualifying_by_sample[sample_name] = set(grp["amplicon_number"])
    return qualifying_by_sample


def load_samples_with_ecdna_or_bfb(classification_tsv):
    """Reads the classification TSV and returns the set of sample_names that
    have AT LEAST ONE amplicon called ecDNA+ or BFB+ (independent of
    --orange-categories -- this always checks the true ecDNA+/BFB+ status,
    since it's used to decide which samples are trusted to vote on what
    counts as 'recurrent', not what counts as 'orange'). CNC-only samples
    (no ecDNA+/BFB+) are NOT included here."""
    df = pd.read_csv(classification_tsv, sep="\t", engine="python")
    required = {"sample_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{classification_tsv} missing required columns: {missing}.")

    is_ecdna = (df["ecDNA+"].astype(str).str.strip() == "Positive") if "ecDNA+" in df.columns \
        else pd.Series(False, index=df.index)
    is_bfb = (df["BFB+"].astype(str).str.strip() == "Positive") if "BFB+" in df.columns \
        else pd.Series(False, index=df.index)

    flagged = df.loc[is_ecdna | is_bfb, "sample_name"].astype(str).str.strip()
    return set(flagged)


def index_classification_bed_dir(bed_dir):
    """Scans the ONE shared classification_bed_files/ directory once and
    indexes files by (sample_name, amplicon_number) -> [bed paths], parsed
    from AC's '<sample_name>_amplicon<N>_...' filename convention. Sample
    names may themselves contain underscores, so the split point is
    anchored on the literal '_amplicon<N>_' substring rather than on
    underscore-splitting."""
    bed_dir = Path(bed_dir)
    index = {}
    n_skipped_unknown, n_unmatched = 0, 0
    for bed_path in sorted(bed_dir.glob("*.bed")):
        name = bed_path.name
        if "unknown" in name.lower():
            n_skipped_unknown += 1
            continue
        m = FILENAME_SAMPLE_AMPLICON_RE.match(name)
        if not m:
            n_unmatched += 1
            continue
        sample_name = m.group("sample")
        amp_num = f"amplicon{m.group('num')}"
        index.setdefault((sample_name, amp_num), []).append(bed_path)
    print(f"[classification-bed-index] {len(index)} (sample, amplicon) keys indexed from {bed_dir} "
          f"({n_skipped_unknown} 'unknown' files skipped, {n_unmatched} files didn't match the naming pattern)")
    return index


def load_orange_intervals_for_sample(sample_name, qualifying_amplicons, bed_index, base):
    """Pools genomic intervals for one sample's qualifying amplicons using
    the pre-built bed_index."""
    frames = []
    for amp_num in qualifying_amplicons:
        for bed_path in bed_index.get((sample_name, amp_num), []):
            try:
                df = pd.read_csv(bed_path, sep="\t", header=None, comment="#", engine="python")
            except pd.errors.EmptyDataError:
                continue
            if df.empty or df.shape[1] < 3:
                continue
            df = df.iloc[:, :3].copy()
            df.columns = ["chrom", "start", "end"]
            df["chrom"] = df["chrom"].map(base.normalize_chrom)
            df["start"] = df["start"].astype(int)
            df["end"] = df["end"].astype(int)
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["chrom", "start", "end"])
    return pd.concat(frames, ignore_index=True)


def load_sample_name_map(path):
    """Optional TSV/CSV mapping this pipeline's sample ids (wes-root/aa-root
    directory names) to the sample_name values used in the classification
    TSV, for cohorts where the two don't match verbatim (e.g. SRR accession
    vs. cell-line name). Columns: sample, classification_sample_name."""
    sep = "\t" if str(path).lower().endswith((".tsv", ".txt")) else None
    df = pd.read_csv(path, sep=sep, engine="python")
    required = {"sample", "classification_sample_name"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"Error: --sample-map is missing required columns: {missing}")
    return dict(zip(df["sample"].astype(str).str.strip(), df["classification_sample_name"].astype(str).str.strip()))




# --------------------------------------------------------------------------
# Per-sample cache: depth bins + seed flag + orange flag, built ONCE
# --------------------------------------------------------------------------

def load_copy_ratio_with_raw_depth(base, path, value_col, mask_col, raw_depth_col,
                                    chrom_col="chrom", start_col="start"):
    """Mirrors base.load_copy_ratio_tsv but additionally carries through a
    raw depth column (e.g. raw_wes_depth) used ONLY for histogram plot
    filtering (--min-depth). Kept separate from base.load_copy_ratio_tsv so
    the original pipeline script's column contract never changes for its
    other callers. If raw_depth_col isn't present in a sample's TSV, that
    sample's raw_depth is set to NaN and a warning is printed once --
    NaN is treated as 'unknown, don't filter out' downstream, not as fail-closed."""
    sep = "\t" if str(path).lower().endswith((".tsv", ".txt")) else None
    df = pd.read_csv(path, sep=sep, engine="python")

    for col in (value_col, chrom_col, start_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {path}. Available: {list(df.columns)}")

    df = df.rename(columns={chrom_col: "chrom", start_col: "start"})
    df["chrom"] = df["chrom"].map(base.normalize_chrom)
    df["start"] = df["start"].astype(int)
    df["value_raw"] = df[value_col].astype(float)

    has_raw_depth = raw_depth_col in df.columns
    df["raw_depth"] = df[raw_depth_col].astype(float) if has_raw_depth else np.nan

    df = df.dropna(subset=["value_raw", "start"])

    n_before = len(df)
    if mask_col.lower() != "none" and mask_col in df.columns:
        df = df[df[mask_col].astype(float).fillna(0) == 0]
    n_after = len(df)

    out = df[["chrom", "start", "value_raw", "raw_depth"]].sort_values(["chrom", "start"]).reset_index(drop=True)
    return out, n_before, n_after, has_raw_depth


def rebin_mean_with_raw_depth(df, new_bin_size):
    """Same mean-aggregation as base.rebin_mean, plus raw_depth."""
    df = df.copy()
    df["bin_start"] = (df["start"] // new_bin_size) * new_bin_size
    agg = (
        df.groupby(["chrom", "bin_start"])
        .agg(value_raw=("value_raw", "mean"), raw_depth=("raw_depth", "mean"),
             n_fine_bins=("value_raw", "size"))
        .reset_index()
        .rename(columns={"bin_start": "start"})
    )
    agg["end"] = agg["start"] + new_bin_size
    return agg.sort_values(["chrom", "start"]).reset_index(drop=True)


def build_per_sample_cache(base, sample_table, column, mask_col, rebin_to,
                            smooth_window, min_overlap_bp, qualifying_by_sample,
                            bed_index, sample_name_map, raw_depth_col="raw_wes_depth",
                            ecdna_bfb_samples=None):
    cache, skip_log = [], []
    warned_missing_raw_depth = False
    ecdna_bfb_samples = ecdna_bfb_samples or set()
    for row in sample_table.itertuples(index=False):
        try:
            raw_df, n_before, n_after, has_raw_depth = load_copy_ratio_with_raw_depth(
                base, row.copy_ratio_tsv, column, mask_col, raw_depth_col)
        except Exception as e:
            skip_log.append({"sample": row.sample, "stage": "load_copy_ratio_tsv", "reason": str(e)})
            continue
        if not has_raw_depth and not warned_missing_raw_depth:
            print(f"[warn] '{raw_depth_col}' column not found in {row.copy_ratio_tsv} (checked on first "
                  f"occurrence only) -- --min-depth filtering will be skipped for samples missing it. "
                  f"Pass --raw-depth-col if your column is named differently.")
            warned_missing_raw_depth = True
        try:
            seeds_df = base.load_seed_bed(row.seed_bed)
        except Exception as e:
            skip_log.append({"sample": row.sample, "stage": "load_seed_bed", "reason": str(e)})
            continue

        classification_sample = sample_name_map.get(row.sample, row.sample) if sample_name_map else row.sample
        qualifying = qualifying_by_sample.get(classification_sample, set())
        if not qualifying:
            print(f"[warn] {row.sample}: 0 qualifying (ecDNA+/BFB+/Complex-non-cyclic) amplicons found "
                  f"for classification sample_name '{classification_sample}' -- check --sample-map if "
                  f"your wes-root/aa-root sample ids don't match the classification TSV's sample_name column")
        orange_df = load_orange_intervals_for_sample(classification_sample, qualifying, bed_index, base)

        binned = rebin_mean_with_raw_depth(raw_df, rebin_to)
        binned = base.smooth_per_chrom(binned, smooth_window)
        seed_flag = base.flag_seed_overlap(binned, seeds_df, min_overlap_bp)
        orange_flag = base.flag_seed_overlap(binned, orange_df, min_overlap_bp)

        n_orange_bins = int(orange_flag.sum())
        if n_orange_bins == 0 and qualifying:
            print(f"[warn] {row.sample}: {len(qualifying)} qualifying amplicons but 0 orange bins after "
                  f"BED lookup -- check that classification_bed_files filenames start with "
                  f"'{classification_sample}_amplicon<N>_'")

        has_ecdna_or_bfb = classification_sample in ecdna_bfb_samples

        cache.append({
            "sample": row.sample, "binned": binned,
            "seed_flag": seed_flag, "orange_flag": orange_flag,
            "n_orange_bins": n_orange_bins,
            "has_ecdna_or_bfb": has_ecdna_or_bfb,
        })
        print(f"[cache] {row.sample}: {len(binned)} coarse bins, {n_orange_bins} orange bins "
              f"({len(qualifying)} qualifying amplicons), "
              f"{'EXCLUDED from recurrent-calling (has ecDNA/BFB)' if has_ecdna_or_bfb else 'eligible for recurrent-calling'}")
    return cache, skip_log


# --------------------------------------------------------------------------
# Per-quantile pooling
# --------------------------------------------------------------------------

def pool_for_quantile(cache, quantile):
    novel_frames, all_frames = [], []
    for entry in cache:
        binned = entry["binned"]
        threshold = binned["smoothed"].quantile(quantile)
        is_high = binned["smoothed"] > threshold
        # novel-high excludes both seed-overlap AND orange bins -- orange is
        # already known/classified, so it should never compete to be called
        # "novel recurrent"
        is_novel_high = is_high & ~entry["seed_flag"] & ~entry["orange_flag"]

        all_frames.append(
            binned.loc[:, ["chrom", "start", "end", "value_raw", "raw_depth"]]
            .assign(sample=entry["sample"], is_orange=entry["orange_flag"])
        )
        # samples with a known ecDNA+/BFB+ amplicon don't get a vote in
        # what counts as "recurrent" -- their complex amplicon architecture
        # can produce broadly elevated/unstable depth beyond just their
        # classified region, which would bias the recurrence caller if
        # allowed to contribute. They still appear in pooled_all (their own
        # non-orange bins can still be labeled recurrent if that LOCATION
        # was called recurrent by the clean-sample vote, and their orange
        # bins are unaffected either way).
        if entry.get("has_ecdna_or_bfb", False):
            continue
        novel_frames.append(
            binned.loc[is_novel_high, ["chrom", "start", "end", "value_raw"]]
            .assign(sample=entry["sample"])
        )
    cols_novel = ["chrom", "start", "end", "value_raw", "sample"]
    cols_all = ["chrom", "start", "end", "value_raw", "raw_depth", "sample", "is_orange"]
    pooled_novel = pd.concat(novel_frames, ignore_index=True) if novel_frames else pd.DataFrame(columns=cols_novel)
    pooled_all = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame(columns=cols_all)
    return pooled_novel, pooled_all


def label_recurrent(pooled_novel, pooled_all, min_samples):
    if pooled_novel.empty:
        merged = pooled_all.copy()
        merged["is_recurrent"] = False
        return merged, pd.DataFrame(columns=["chrom", "start", "end", "n_samples_high"])

    recurrence_counts = (
        pooled_novel.groupby(["chrom", "start", "end"])["sample"].nunique()
        .rename("n_samples_high").reset_index()
    )
    recurrent_keys = recurrence_counts.loc[
        recurrence_counts["n_samples_high"] >= min_samples, ["chrom", "start", "end"]
    ].assign(is_recurrent=True)

    merged = pooled_all.merge(recurrent_keys, on=["chrom", "start", "end"], how="left")
    merged["is_recurrent"] = merged["is_recurrent"].fillna(False).astype(bool)
    return merged, recurrence_counts


# --------------------------------------------------------------------------
# Separation metrics (orange vs blue)
# --------------------------------------------------------------------------

def overlap_coefficient(x, y, grid_points=2000, max_points=20000, random_state=0):
    """
    Overlap coefficient via log-space Gaussian KDE. gaussian_kde's cost is
    O(n * grid_points) at evaluation time, which is fine for thousands of
    points but becomes the dominant cost of the whole search at genome-wide
    scale (e.g. 5kb bins x dozens of samples pools into tens of millions of
    background bin-instances -- verified ~23 microseconds/point, i.e.
    ~8 minutes per combo at ~20M points, unacceptable across a 100+ combo
    search). The overlap coefficient estimate itself does not need every
    point -- it's a density-shape statistic, and stabilizes well below
    max_points -- so each group is randomly subsampled (capped, not
    truncated by value) down to max_points before fitting, which is why
    this must be applied even for the fast/exact case: consistent, fixed
    cost per combo regardless of cohort size.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    x, y = x[x > 0], y[y > 0]
    if len(x) < 5 or len(y) < 5:
        return np.nan

    rng = np.random.default_rng(random_state)
    if len(x) > max_points:
        x = rng.choice(x, size=max_points, replace=False)
    if len(y) > max_points:
        y = rng.choice(y, size=max_points, replace=False)

    x_t, y_t = np.log10(x), np.log10(y)
    try:
        kde_x, kde_y = stats.gaussian_kde(x_t), stats.gaussian_kde(y_t)
    except np.linalg.LinAlgError:
        return np.nan
    lo, hi = min(x_t.min(), y_t.min()), max(x_t.max(), y_t.max())
    grid = np.linspace(lo, hi, grid_points)
    min_density = np.minimum(kde_x(grid), kde_y(grid))
    trapz_fn = getattr(np, "trapezoid", None) or np.trapz
    return float(np.clip(trapz_fn(min_density, grid), 0, 1))


def compute_separation_metrics(orange_values, blue_values, max_kde_points=20000):
    ov = np.asarray(orange_values, dtype=float)
    bv = np.asarray(blue_values, dtype=float)
    ov, bv = ov[np.isfinite(ov) & (ov > 0)], bv[np.isfinite(bv) & (bv > 0)]
    if len(ov) < 5 or len(bv) < 5:
        return {"overlap_coefficient": np.nan, "ks_statistic": np.nan, "ks_pvalue": np.nan, "auc": np.nan}

    ovl = overlap_coefficient(ov, bv, max_points=max_kde_points)
    ks_stat, ks_p = stats.ks_2samp(ov, bv)
    n_o, n_b = len(ov), len(bv)
    ranks = stats.rankdata(np.concatenate([ov, bv]))
    u = ranks[:n_o].sum() - n_o * (n_o + 1) / 2
    auc = u / (n_o * n_b)
    return {"overlap_coefficient": ovl, "ks_statistic": float(ks_stat), "ks_pvalue": float(ks_p), "auc": float(auc)}


# --------------------------------------------------------------------------
# Grid runner
# --------------------------------------------------------------------------

def run_grid(cache, quantiles, min_samples_ratios, base, rebin_to, merge_gap=0, max_kde_points=20000):
    n_samples = len(cache)
    # min_samples_ratio is meant relative to samples that can actually vote
    # on recurrence -- samples with ecDNA+/BFB+ never contribute to
    # pooled_novel (see pool_for_quantile), so using the full cohort size
    # here would silently make every ratio stricter than intended once any
    # samples are excluded.
    n_recurrent_eligible = sum(1 for e in cache if not e.get("has_ecdna_or_bfb", False))
    if n_recurrent_eligible < n_samples:
        print(f"[info] {n_samples - n_recurrent_eligible}/{n_samples} samples have ecDNA+/BFB+ and are "
              f"excluded from recurrent-calling; min_samples_ratio is computed against the remaining "
              f"{n_recurrent_eligible} eligible samples.")
    results = []
    instances_cache = {}

    for q in quantiles:
        pooled_novel, pooled_all = pool_for_quantile(cache, q)
        orange_values = pooled_all.loc[pooled_all["is_orange"], "value_raw"]

        for ratio in min_samples_ratios:
            min_samples = max(1, int(np.ceil(ratio * n_recurrent_eligible)))
            merged, recurrence_counts = label_recurrent(pooled_novel, pooled_all, min_samples)

            blue_mask = ~merged["is_orange"] & ~merged["is_recurrent"]
            blue_instances = merged.loc[blue_mask]
            recurrent_instances = merged.loc[merged["is_recurrent"]]

            metrics = compute_separation_metrics(orange_values, blue_instances["value_raw"], max_kde_points)

            n_recurrent_bins, n_recurrent_regions = 0, 0
            if len(recurrence_counts):
                rec_bins_df = recurrence_counts[recurrence_counts["n_samples_high"] >= min_samples].copy()
                n_recurrent_bins = len(rec_bins_df)
                if n_recurrent_bins:
                    rec_bins_df["mean_pon_median"] = np.nan
                    rec_bins_df["n_samples_with_data"] = n_recurrent_eligible
                    with warnings.catch_warnings():
                        # mean_pon_median isn't tracked in this script (it's
                        # only used for the base script's own artifact-risk
                        # coloring); nanmean-of-all-NaN is expected here.
                        warnings.filterwarnings("ignore", category=RuntimeWarning)
                        regions_df = base.merge_adjacent_bins(rec_bins_df, rebin_to, merge_gap)
                    n_recurrent_regions = len(regions_df)

            results.append({
                "quantile": q,
                "min_samples_ratio": ratio,
                "min_samples": min_samples,
                "n_samples_total": n_samples,
                "n_recurrent_eligible_samples": n_recurrent_eligible,
                "n_orange_bin_instances": len(orange_values),
                "n_blue_bin_instances": len(blue_instances),
                "n_recurrent_bin_instances": len(recurrent_instances),
                "n_recurrent_bins": n_recurrent_bins,
                "n_recurrent_regions": n_recurrent_regions,
                **metrics,
            })
            instances_cache[(q, ratio)] = (orange_values, blue_instances["value_raw"], recurrent_instances["value_raw"])

    return pd.DataFrame(results), instances_cache


# --------------------------------------------------------------------------
# Selection + plotting
# --------------------------------------------------------------------------

def select_best(grid_df, min_bin_instances, max_recurrent_fraction=0.15):
    """max_recurrent_fraction guards against the metric's degenerate optimum:
    overlap_coefficient mechanically shrinks whenever more of the genome is
    pulled out of blue into 'recurrent', regardless of whether that removal
    reflects real biology -- a loose quantile combined with a low-but-not-
    trivial min-samples bar can still sweep a large fraction of background
    into 'recurrent' via shared technical artifacts (mappability, GC bias)
    rather than genuine focal amplification. Real recurrent amplification
    should be a small minority of the genome, not comparable in size to
    background itself."""
    recurrent_fraction = grid_df["n_recurrent_bin_instances"] / (
        grid_df["n_recurrent_bin_instances"] + grid_df["n_blue_bin_instances"]
    ).replace(0, np.nan)

    eligible = grid_df[
        (grid_df["n_blue_bin_instances"] >= min_bin_instances)
        & (grid_df["n_orange_bin_instances"] >= min_bin_instances)
        & (recurrent_fraction <= max_recurrent_fraction)
        & grid_df["overlap_coefficient"].notna()
    ]
    if eligible.empty:
        return None, eligible
    best_row = eligible.sort_values(["overlap_coefficient", "auc"], ascending=[True, False]).iloc[0]
    return best_row, eligible


def plot_heatmap(grid_df, value_col, title, out_path, cmap="viridis_r"):
    pivot = grid_df.pivot(index="quantile", columns="min_samples_ratio", values=value_col)
    fig, ax = plt.subplots(figsize=(1.1 * len(pivot.columns) + 2, 0.5 * len(pivot.index) + 2))
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{r:.3f}" for r in pivot.index])
    ax.set_xlabel("min_samples_ratio")
    ax.set_ylabel("quantile")
    ax.set_title(title)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_best_histogram(orange_values, blue_values, recurrent_values, quantile, min_samples,
                         min_samples_ratio, out_path, subtitle="orange vs blue overlap minimized"):
    ov = np.asarray(orange_values, dtype=float)
    bv = np.asarray(blue_values, dtype=float)
    rv = np.asarray(recurrent_values, dtype=float)
    ov, bv, rv = ov[ov > 0], bv[bv > 0], rv[rv > 0]  # positivity only, needed for log-x axis

    if len(bv) < 2 or len(ov) + len(bv) < 2:
        print(f"[warn] not enough values to plot histogram at {out_path} "
              f"(orange={len(ov)}, blue={len(bv)}) -- skipping")
        return

    all_vals = np.concatenate([ov, bv, rv]) if len(rv) else np.concatenate([ov, bv])
    lo, hi = np.log10(all_vals.min()), np.log10(all_vals.max())
    bins = np.logspace(lo, hi, 60)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(bv, bins=bins, density=True, alpha=0.55, color="#7f9fc9",
            label=f"Blue: background (n={len(bv):,}, median={np.median(bv):.2f})")
    if len(ov):
        ax.hist(ov, bins=bins, density=True, alpha=0.6, color="#ffb84d",
                label=f"Orange: ecDNA/BFB/CNC (n={len(ov):,}, median={np.median(ov):.2f})")
    if len(rv):
        ax.hist(rv, bins=bins, density=True, alpha=0.45, color="#b07fc9",
                label=f"Recurrent amplified region bins (n={len(rv):,}, median={np.median(rv):.2f})")
    ax.axvline(np.median(bv), color="#3355aa", ls="--", lw=1.5)
    if len(ov):
        ax.axvline(np.median(ov), color="#cc7a00", ls="--", lw=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("Predicted upscale depth")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Best combo: quantile={quantile}, min_samples_ratio={min_samples_ratio} "
        f"(min_samples={min_samples})\n{subtitle}"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _apply_min_depth_filter(df, min_depth, context_label=""):
    """Filters a merged dataframe on raw_depth > min_depth. Rows with NaN
    raw_depth (sample's TSV lacked the raw-depth column) are KEPT rather
    than dropped -- we can't confirm they fail the filter, so silently
    excluding them would misrepresent the plot as more filtered than it
    is. Warns once if that happens so it's visible rather than silent."""
    if min_depth is None:
        return df
    missing = df["raw_depth"].isna()
    if missing.any():
        print(f"[warn] {context_label}{missing.sum():,} bin-instances have no raw_depth value "
              f"(column missing from source TSV) -- these are NOT filtered by --min-depth")
    keep = df["raw_depth"].isna() | (df["raw_depth"] > min_depth)
    return df[keep]


def plot_pooled_histogram(cache, quantile, min_samples, min_samples_ratio, out_path, min_depth=0.1):
    """Recomputes pool_for_quantile + label_recurrent for one combo (cheap)
    and applies --min-depth (on raw_wes_depth, not the plotted value) before
    producing the all-samples-pooled 3-way histogram."""
    pooled_novel, pooled_all = pool_for_quantile(cache, quantile)
    merged, _ = label_recurrent(pooled_novel, pooled_all, min_samples)
    merged = _apply_min_depth_filter(merged, min_depth, context_label="[pooled] ")

    orange_v = merged.loc[merged["is_orange"], "value_raw"]
    recurrent_v = merged.loc[merged["is_recurrent"], "value_raw"]
    blue_v = merged.loc[~merged["is_orange"] & ~merged["is_recurrent"], "value_raw"]

    plot_best_histogram(orange_v, blue_v, recurrent_v, quantile, min_samples, min_samples_ratio, out_path)
    print(f"Saved: {out_path}")


def plot_single_sample_histogram(cache, quantile, min_samples, min_samples_ratio,
                                  highlight_sample, out_path, min_depth=0.1):
    """Recomputes pool_for_quantile + label_recurrent for one combo (cheap --
    only re-thresholds already-cached per-sample bins) and filters the
    merged result down to one sample's own bins, reproducing the original
    per-sample plot style (orange/blue/purple for just that sample) rather
    than the pooled-across-all-samples view."""
    pooled_novel, pooled_all = pool_for_quantile(cache, quantile)
    merged, _ = label_recurrent(pooled_novel, pooled_all, min_samples)

    sample_rows = merged[merged["sample"] == highlight_sample]
    if sample_rows.empty:
        available = sorted(merged["sample"].unique())
        print(f"[warn] --highlight-sample '{highlight_sample}' not found among wes-root/aa-root sample ids. "
              f"Available: {available[:15]}{' ...' if len(available) > 15 else ''}")
        return
    sample_rows = _apply_min_depth_filter(sample_rows, min_depth, context_label=f"[{highlight_sample}] ")

    orange_s = sample_rows.loc[sample_rows["is_orange"], "value_raw"]
    recurrent_s = sample_rows.loc[sample_rows["is_recurrent"], "value_raw"]
    blue_s = sample_rows.loc[~sample_rows["is_orange"] & ~sample_rows["is_recurrent"], "value_raw"]

    plot_best_histogram(orange_s, blue_s, recurrent_s, quantile, min_samples, min_samples_ratio,
                         out_path, subtitle=f"sample: {highlight_sample}")
    print(f"Saved: {out_path}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wes-root", default=None)
    p.add_argument("--aa-root", default=None)
    p.add_argument("--manifest", default=None)
    p.add_argument("--classification-tsv", required=True,
                    help="Path to the consolidated <prefix>_amplicon_classification_profiles.tsv "
                         "(one file, sample_name is a column -- AC's batch-mode output)")
    p.add_argument("--classification-bed-dir", required=True,
                    help="Path to the consolidated <prefix>_classification_bed_files/ directory "
                         "(one directory shared across all samples)")
    p.add_argument("--orange-categories", default="ecDNA,BFB,CNC",
                    help="Comma list of which AC categories count as 'orange': any of ecDNA, BFB, CNC. "
                         "Default includes all three. Restrict to e.g. 'ecDNA' to isolate ecDNA-only "
                         "amplicons, which tend to sit at higher/more sharply focal depth than BFB or "
                         "Complex-non-cyclic amplicons -- pooling all three widens the orange "
                         "distribution and can look/measure less separated from background.")
    p.add_argument("--include-ecdna-bfb-samples-in-recurrent-calling", action="store_true",
                    help="By default, samples with >=1 ecDNA+ or BFB+ amplicon are excluded from voting "
                         "on what counts as 'recurrent' (their complex amplicon architecture can produce "
                         "broadly elevated/unstable depth beyond just their classified region, biasing "
                         "the recurrence caller). Orange bins and min_samples_ratio's denominator are "
                         "unaffected either way -- this only controls who votes. Pass this flag to "
                         "restore the old behavior and let all samples vote.")
    p.add_argument("--sample-map", default=None,
                    help="Optional TSV/CSV with columns 'sample','classification_sample_name' if your "
                         "wes-root/aa-root sample ids differ from the classification TSV's sample_name "
                         "(e.g. SRR accession vs. cell-line name). Default: assume they match verbatim.")
    p.add_argument("--base-script",
                    default=str(Path(__file__).parent / "find_recurrent_novel_amplifications_binlevel.py"))
    p.add_argument("--column", default="predicted_loess_upscale_depth")
    p.add_argument("--mask-col", default="mask_rejected")
    p.add_argument("--rebin-to", type=int, default=25000)
    p.add_argument("--smooth-window", type=int, default=1)
    p.add_argument("--min-overlap-bp", type=int, default=1)
    p.add_argument("--merge-gap", type=int, default=0)
    p.add_argument("--quantiles", default="0.80:0.98:0.02")
    p.add_argument("--min-samples-ratios", default="0.05:0.40:0.05")
    p.add_argument("--min-bin-instances-for-selection", type=int, default=30)
    p.add_argument("--max-recurrent-fraction", type=float, default=0.15,
                    help="A combo is only eligible if n_recurrent_bin_instances is at most this "
                         "fraction of (recurrent + blue) bin-instances. Guards against the metric's "
                         "degenerate optimum: overlap shrinks mechanically whenever more of the genome "
                         "is swept into 'recurrent', whether or not that reflects real biology. "
                         "Real recurrent amplification should be a small minority of the genome.")
    p.add_argument("--max-kde-points", type=int, default=20000,
                    help="Cap each group's size before KDE fitting (overlap_coefficient only -- "
                         "KS/AUC always use full data). KDE cost scales linearly with n and dominates "
                         "runtime at genome-wide bin resolution; 20000 is stable for density-shape "
                         "estimation. Raise if you have time to spare and want less subsampling noise.")
    p.add_argument("--outdir", default="ecdna_bfb_cnc_threshold_search")
    p.add_argument("--highlight-sample", default=None,
                    help="A wes-root/aa-root sample id (e.g. an SRR accession) -- if set, also produce "
                         "a single-sample version of best_combo_histogram.png filtered to just that "
                         "sample's own bins, matching the original per-sample plot style")
    p.add_argument("--min-depth", type=float, default=0.1,
                    help="Histogram plots (best_combo_histogram*.png) only include bins with "
                         "raw_wes_depth above this value. Does not affect the optimization/search "
                         "itself, only what's drawn.")
    p.add_argument("--raw-depth-col", default="raw_wes_depth",
                    help="Column name in the copy-ratio TSVs used for --min-depth filtering. This is "
                         "separate from --column (the value that's actually plotted/optimized on).")
    args = p.parse_args()

    if not args.manifest and not (args.wes_root and args.aa_root):
        sys.exit("Error: provide either --manifest, or both --wes-root and --aa-root.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    base = load_base_module(args.base_script)

    sample_table, skip_log = (base.load_manifest(args.manifest), []) if args.manifest \
        else base.discover_samples(args.wes_root, args.aa_root)
    if sample_table.empty:
        sys.exit("Error: no samples with both a copy ratio TSV and a seed BED were found.")
    print(f"Found {len(sample_table)} samples with depth + seed data.")

    qualifying_by_sample = load_qualifying_amplicons_by_sample(
        args.classification_tsv, tuple(args.orange_categories.split(",")))
    total_qualifying_amplicons = sum(len(v) for v in qualifying_by_sample.values())
    print(f"Classification TSV: {len(qualifying_by_sample)} samples have >=1 qualifying "
          f"(ecDNA+/BFB+/Complex-non-cyclic) amplicon, {total_qualifying_amplicons} qualifying amplicons total")

    ecdna_bfb_samples = set()
    if not args.include_ecdna_bfb_samples_in_recurrent_calling:
        ecdna_bfb_samples = load_samples_with_ecdna_or_bfb(args.classification_tsv)
        print(f"Classification TSV: {len(ecdna_bfb_samples)} samples have >=1 ecDNA+/BFB+ amplicon and "
              f"will be excluded from recurrent-calling (use "
              f"--include-ecdna-bfb-samples-in-recurrent-calling to disable this)")

    bed_index = index_classification_bed_dir(args.classification_bed_dir)

    sample_name_map = load_sample_name_map(args.sample_map) if args.sample_map else {}

    cache, more_skips = build_per_sample_cache(
        base, sample_table, args.column, args.mask_col, args.rebin_to,
        args.smooth_window, args.min_overlap_bp, qualifying_by_sample, bed_index, sample_name_map,
        args.raw_depth_col, ecdna_bfb_samples
    )
    skip_log += more_skips
    if not cache:
        sys.exit("Error: no samples loaded successfully.")

    total_orange = sum(e["n_orange_bins"] for e in cache)
    if total_orange == 0:
        sys.exit("Error: 0 orange (ecDNA/BFB/CNC) bins found across all samples. Check that sample "
                 "names in wes-root/aa-root match sample_name in --classification-tsv (use --sample-map "
                 "if not), and that --classification-bed-dir filenames follow the "
                 "'<sample_name>_amplicon<N>_..._intervals.bed' convention -- see the per-sample "
                 "[warn] lines above.")

    quantiles = parse_grid(args.quantiles)
    min_samples_ratios = parse_grid(args.min_samples_ratios)
    print(f"Grid: {len(quantiles)} quantiles x {len(min_samples_ratios)} min_samples_ratios "
          f"= {len(quantiles) * len(min_samples_ratios)} combos")

    grid_df, instances_cache = run_grid(cache, quantiles, min_samples_ratios, base, args.rebin_to,
                                        args.merge_gap, args.max_kde_points)
    recurrent_fraction = grid_df["n_recurrent_bin_instances"] / (
        grid_df["n_recurrent_bin_instances"] + grid_df["n_blue_bin_instances"]
    ).replace(0, np.nan)
    grid_df = grid_df.assign(recurrent_fraction_of_genome=recurrent_fraction)
    grid_df.to_csv(outdir / "grid_results.csv", index=False)
    print(f"Saved: {outdir / 'grid_results.csv'}")

    plot_heatmap(grid_df, "overlap_coefficient", "Overlap coefficient: orange vs blue (lower = better)",
                 outdir / "overlap_heatmap.png")
    plot_heatmap(grid_df, "auc", "AUC: orange vs blue (higher = better)",
                 outdir / "auc_heatmap.png", cmap="viridis")
    plot_heatmap(grid_df, "n_recurrent_bins", "n_recurrent_bins removed from background (sanity check)",
                 outdir / "n_bg_removed_heatmap.png", cmap="magma")
    print(f"Saved heatmaps to {outdir}")

    best_row, eligible = select_best(grid_df, args.min_bin_instances_for_selection, args.max_recurrent_fraction)

    summary_path = outdir / "best_combo_summary.txt"
    with open(summary_path, "w") as f:
        f.write("ecDNA/BFB/CNC vs background threshold search\n" + "=" * 45 + "\n\n")
        f.write(f"Samples processed: {len(cache)} (skipped: {len(skip_log)})\n")
        f.write(f"Total orange (ecDNA/BFB/CNC) bin-instances (fixed, all quantiles): {total_orange}\n")
        f.write(f"Grid: {len(quantiles)} quantiles x {len(min_samples_ratios)} min_samples_ratios\n")
        f.write(f"Eligibility filter: n_orange/n_blue bin instances >= {args.min_bin_instances_for_selection}, "
                f"recurrent fraction of genome <= {args.max_recurrent_fraction}\n")
        f.write(f"Eligible combos: {len(eligible)} / {len(grid_df)}\n\n")
        if best_row is None:
            f.write("No combo met the eligibility filter -- loosen --min-bin-instances-for-selection "
                    "or --max-recurrent-fraction, or widen the grid.\n")
        else:
            f.write("BEST COMBO (min overlap_coefficient, tie-broken by max AUC):\n")
            f.write(best_row.to_string() + "\n\n")
            f.write("Top 15 eligible combos by overlap_coefficient:\n")
            f.write(eligible.sort_values(["overlap_coefficient", "auc"], ascending=[True, False])
                    .head(15).to_string(index=False))
            f.write("\n")
        if skip_log:
            f.write("\nSkipped samples/stages:\n")
            f.write(pd.DataFrame(skip_log).to_string(index=False))
            f.write("\n")
    print(f"Saved: {summary_path}")

    if best_row is not None:
        plot_pooled_histogram(
            cache, best_row["quantile"], int(best_row["min_samples"]), best_row["min_samples_ratio"],
            outdir / "best_combo_histogram.png", min_depth=args.min_depth,
        )
        print(f"\nBest combo: quantile={best_row['quantile']}, "
              f"min_samples_ratio={best_row['min_samples_ratio']} "
              f"(min_samples={int(best_row['min_samples'])}/{len(cache)}), "
              f"overlap_coefficient={best_row['overlap_coefficient']:.3f}, "
              f"auc={best_row['auc']:.3f}")

        if args.highlight_sample:
            plot_single_sample_histogram(
                cache, best_row["quantile"], int(best_row["min_samples"]),
                best_row["min_samples_ratio"], args.highlight_sample,
                outdir / f"best_combo_histogram_{args.highlight_sample}.png", min_depth=args.min_depth,
            )
    else:
        print("No eligible combo found -- see best_combo_summary.txt.")


if __name__ == "__main__":
    main()
