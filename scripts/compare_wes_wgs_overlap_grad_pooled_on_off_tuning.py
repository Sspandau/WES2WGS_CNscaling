#!/usr/bin/env python3
"""
pooled_onoff_clustered_experiments.py

Pooled (cohort-wide) WES-vs-WGS overlap search over on-target + off-target bins,
with off-target bins restricted to each sample's genomic clusters.

ORDER OF OPERATIONS (per sample)
--------------------------------
  1. Load each region, drop QC-rejected rows (--offtarget-qc-column, default
     mask_rejected), then rebin and align on- and off-target tracks separately.
  2. Apply the recurrent-bin mask (compare_wes_wgs_overlap.apply_mask_regions_to_df).
  3. Compute CN-like values SEPARATELY per region and per track:
        CN-like = scale x depth / region_mean
     where scale is the constant used by your cn_like_from_track (probed at start-up,
     normally 2) and region_mean is taken over that region's rebinned bins BEFORE any
     masking or cluster filtering:
        on-target : mean of ALL on-target bins            (--ontarget-cn-mean all)
        off-target: mean of off-target bins with depth > 0 (--offtarget-cn-mean positive)
     The means used for every sample are written to samples_used_<split>.tsv.
  4. Drop masked bins (unless --keep-masked-in-clusters keeps those inside the sample's
     clusters), then drop off-target bins outside the sample's clusters.
     (Normalizing first keeps the off-target CN-like scale identical to an
     unfiltered run; clustered bins are often elevated, so normalizing by their
     own mean would pull them toward baseline.)
  5. Keep bins where all four values (raw WES, raw WGS, CN-like WES, CN-like WGS)
     are finite, so every experiment scores the exact same bins.

EXPERIMENTS (each run for --metrics jaccard and/or overlap)
----------------------------------------------------------
  exp1_raw     3 thresholds on RAW depth: WES on-target, WES off-target, one WGS
  exp1_cnlike  3 thresholds on CN-like:   WES on-target, WES off-target, one WGS
  exp2_cnlike  2 thresholds on CN-like:   ONE shared WES threshold for both
               regions, one WGS threshold

  A bin is WES-positive if its WES value exceeds the threshold for its region
  (exp1) or the shared threshold (exp2); WGS-positive if WGS exceeds the WGS
  threshold. CN-like floor: in every experiment a bin is only positive if its CN-like
  value (WES for the WES call, WGS for the WGS call) exceeds --min-cn-like. In the raw
  experiment the thresholds are optimised on raw depth and CN-like is used ONLY for this
  floor (--no-raw-cn-floor turns it off). Scores: jaccard = |W&G| / |W|G|, overlap = |W&G| / min(|W|,|G|).
  --aggregate micro pools bin counts across samples; macro averages per-sample
  scores. In macro, a sample with nothing called on EITHER side is left out of the
  mean; a sample where only one side calls bins scores 0 (for both metrics).

SEARCH (gradient descent -- same method as compare_wes_wgs_overlap_grad_pooled.py)
------
  One free parameter per threshold: the fraction of that axis' bins called positive
  (exp1: WES on-target, WES off-target, WGS; exp2: shared WES, WGS), kept in logit space
  and mapped log-uniformly into its allowed range; threshold = quantile at (1 - fraction),
  floored at --min-cn-like on the CN-like scale. Adam (--lr, --n-steps, --n-starts)
  optimizes a smooth surrogate (sigmoid membership in quantile-rank space, width --temp x
  the current positive fraction; micro = pooled sums,
  macro = tanh-weighted per-sample sums) minus count penalties; gradients by finite
  differences. Every step is also scored with the TRUE hard-threshold score, and the
  reported thresholds are the step with the highest hard score that satisfies all
  eligibility constraints:
    - WES-positive and WGS-positive pooled bins each within
      [max(--min-bins, --min-frac x N), --max-frac x N]
    - >= --min-samples samples with at least one WES-positive AND one WGS-positive bin
    - CN-like thresholds >= --min-cn-like (enforced by construction)
  The full step-by-step trace for the chosen combo is <exp>_<metric>_search_history.tsv.

TRAIN / EVAL / TEST
-------------------
  Thresholds are searched on --manifest (train) only. With --tune, every combination
  of the listed constraint values is searched on train and scored on --eval-manifest;
  the combo with the best eval score is chosen (tuning table: <exp>_<metric>_tuning.tsv).
  --test-manifest is scored ONCE, at the chosen combo, and never used for any choice --
  report the test score as the final, unbiased result.

OUTPUT (--outdir)
-----------------
  summary.tsv                                one row per experiment x metric:
                                             thresholds, train + holdout scores
  <exp>_<metric>_best.tsv                    full detail for the best combo
  <exp>_<metric>_top_eligible.tsv            top 50 distinct eligible steps
  <exp>_<metric>_search_history.tsv         every Adam step (soft + hard scores)
  <exp>_<metric>_per_sample_train.tsv        per-sample counts/scores at best
  <exp>_<metric>_per_sample_holdout.tsv      same, held-out samples
  samples_used_{train,holdout}.tsv, samples_skipped_{train,holdout}.tsv
"""

import argparse
import glob
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from compare_wes_wgs_overlap import (
    align_tracks,
    apply_mask_regions_to_df,
    load_data,
    load_mask_regions,
    prepare_track,
    rebin_track,
    cn_like_from_track,
)

EXPERIMENTS = {
    # name: (value scale, separate on/off WES thresholds?)
    "exp1_raw": ("raw", True),
    "exp1_cnlike": ("cn", True),
    "exp2_cnlike": ("cn", False),
}


def _mtime(path):
    try:
        return os.path.getmtime(path)
    except (OSError, TypeError):
        return None


# --------------------------------------------------------------------------
# Cluster intervals
# --------------------------------------------------------------------------

CHROM_NAMES = ("chrom", "chr", "chromosome", "seqnames", "contig", "seqname")
START_NAMES = ("start", "bin_start", "chromstart", "start_pos", "pos")
END_NAMES = ("end", "bin_end", "chromend", "end_pos", "stop")
LABEL_NAMES = ("cluster", "cluster_id", "cluster_label", "label")


def norm_chrom(values):
    s = pd.Series(values).astype(str).str.strip().str.upper()
    return s.str.replace(r"^CHR", "", regex=True).to_numpy()


def _pick(cols_lower, names):
    for n in names:
        if n in cols_lower:
            return cols_lower[n]
    return None


def load_cluster_intervals(path, default_width, label_col=None, exclude_labels=(), keep_labels=None):
    """Merged, sorted, non-overlapping intervals DataFrame[chrom, start, end]."""
    p = str(path)
    sep = "," if p.lower().endswith((".csv", ".csv.gz")) else "\t"
    df = pd.read_csv(p, sep=sep, low_memory=False)
    cols_lower = {str(c).strip().lower(): c for c in df.columns}
    c_col, s_col, e_col = _pick(cols_lower, CHROM_NAMES), _pick(cols_lower, START_NAMES), _pick(cols_lower, END_NAMES)
    if c_col is None or s_col is None:                      # headerless BED
        df = pd.read_csv(p, sep=sep, header=None, comment="#", low_memory=False)
        if df.shape[1] < 2:
            raise ValueError(f"cannot find chrom/start columns in cluster file {p}")
        c_col, s_col, e_col = 0, 1, (2 if df.shape[1] >= 3 else None)
        cols_lower = {}

    lab = None
    if label_col:
        lab = cols_lower.get(label_col.lower())
        if lab is None:
            raise ValueError(f"--cluster-label-column '{label_col}' not in {p}")
    elif cols_lower:
        lab = _pick(cols_lower, LABEL_NAMES)
    if lab is not None:
        labels = df[lab].astype(str).str.strip()
        mask = pd.Series(True, index=df.index)
        if keep_labels:
            mask &= labels.isin(keep_labels)
        if exclude_labels:
            mask &= ~labels.isin(exclude_labels)
        df = df[mask]
    if df.empty:
        return pd.DataFrame(columns=["chrom", "start", "end"])

    out = pd.DataFrame({"chrom": norm_chrom(df[c_col]), "start": pd.to_numeric(df[s_col], errors="coerce")})
    out["end"] = pd.to_numeric(df[e_col], errors="coerce") if e_col is not None else out["start"] + default_width
    out = out.dropna().astype({"start": np.int64, "end": np.int64})
    out = out[out["end"] > out["start"]].sort_values(["chrom", "start"])
    merged = []
    for ch, g in out.groupby("chrom", sort=False):
        cs = ce = None
        for s, e in zip(g["start"].to_numpy(), g["end"].to_numpy()):
            if cs is None:
                cs, ce = s, e
            elif s <= ce:
                ce = max(ce, e)
            else:
                merged.append((ch, cs, ce))
                cs, ce = s, e
        merged.append((ch, cs, ce))
    return pd.DataFrame(merged, columns=["chrom", "start", "end"])


def bins_in_intervals(chrom, start, end, intervals, min_overlap_bp=1):
    keep = np.zeros(len(chrom), dtype=bool)
    if intervals is None or intervals.empty:
        return keep
    for ch, g in intervals.groupby("chrom", sort=False):
        m = chrom == ch
        if not m.any():
            continue
        s, e = g["start"].to_numpy(), g["end"].to_numpy()
        bs, be = start[m], end[m]
        idx = np.searchsorted(s, be, side="left") - 1
        ok = idx >= 0
        ov = np.zeros(len(bs), dtype=np.int64)
        ov[ok] = np.minimum(be[ok], e[idx[ok]]) - np.maximum(bs[ok], s[idx[ok]])
        keep[np.flatnonzero(m)] = ov >= min_overlap_bp
    return keep


def resolve_cluster_file(sample, manifest_value, glob_template):
    if manifest_value is not None and str(manifest_value).strip() and str(manifest_value).lower() != "nan":
        return str(manifest_value).strip()
    if glob_template:
        hits = sorted(glob.glob(glob_template.replace("{sample}", sample), recursive=True))
        if hits:
            return hits[0]
    return None


# --------------------------------------------------------------------------
# Per-sample preprocessing
# --------------------------------------------------------------------------

def cn_scale_factor():
    """Constant your compare_wes_wgs_overlap.cn_like_from_track multiplies value/mean by
    (e.g. 2 for a diploid-baseline CN-like scale). Probed once on a constant track so this
    script uses YOUR convention for the scale, while the mean it divides by is defined
    explicitly below."""
    try:
        c = float(np.nanmedian(cn_like_from_track(pd.Series([3.0, 3.0, 3.0, 3.0])).to_numpy(dtype=float)))
        return c if np.isfinite(c) and c > 0 else 2.0
    except Exception:
        return 2.0


CN_SCALE = cn_scale_factor()


def region_mean(values, rule):
    """Reference mean for CN-like: 'all' = mean of all finite bins in the region,
    'positive' = mean of finite bins with value > 0 only."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if rule == "positive":
        v = v[v > 0]
    return float(v.mean()) if len(v) else np.nan


def load_region(path, wes_col, wgs_col, bin_size, mask_df, qc_col=None, mean_rule="all"):
    """Align one region and return a DataFrame with chrom/start/end, raw + CN-like WES/WGS,
    the masked flag, and the reference means used.

    CN-like = CN_SCALE * value / region_mean, with region_mean computed from THIS region's
    rebinned bins (on-target and off-target never mixed), BEFORE the recurrent-bin mask and
    the cluster filter are applied, using mean_rule ('all' or 'positive' = bins > 0 only),
    separately for the WES and the WGS track. Rows flagged in qc_col (non-zero) are dropped
    before rebinning (e.g. off-target windows with mask_rejected=1)."""
    df = load_data(path)
    if qc_col and qc_col.lower() != "none" and qc_col in df.columns:
        df = df[pd.to_numeric(df[qc_col], errors="coerce").fillna(0) == 0]
    wes_df = rebin_track(prepare_track(df, wes_col, None, "chrom", "start"), wes_col, target_bin_size=bin_size)
    wgs_df = rebin_track(prepare_track(df, wgs_col, None, "chrom", "start"), wgs_col, target_bin_size=bin_size)
    aligned = align_tracks(wes_df, wgs_df, wes_col, wgs_col)

    wes_raw = aligned["wes_value"].to_numpy(dtype=float)
    wgs_raw = aligned["wgs_value"].to_numpy(dtype=float)
    wes_mean, wgs_mean = region_mean(wes_raw, mean_rule), region_mean(wgs_raw, mean_rule)

    masked = np.zeros(len(aligned), dtype=bool)
    if mask_df is not None:
        m = apply_mask_regions_to_df(aligned, mask_df)
        if "masked" in m.columns:
            masked = m["masked"].to_numpy(dtype=bool)

    out = pd.DataFrame({
        "chrom": norm_chrom(aligned["chrom"]),
        "start": aligned["start"].to_numpy(dtype=np.int64),
        "wes_raw": wes_raw, "wgs_raw": wgs_raw,
        "wes_cn": CN_SCALE * wes_raw / wes_mean,
        "wgs_cn": CN_SCALE * wgs_raw / wgs_mean,
    })
    out["end"] = aligned["end"].to_numpy(dtype=np.int64) if "end" in aligned.columns else out["start"] + bin_size
    out["masked"] = masked
    return out, dict(wes_mean=wes_mean, wgs_mean=wgs_mean,
                     n_bins=len(aligned), n_wes_pos=int((wes_raw > 0).sum()))


def preprocess_sample(job):
    cols = ("wes_raw", "wgs_raw", "wes_cn", "wgs_cn")
    try:
        cfile = job["cluster_file"]
        if cfile is None and job["missing_clusters"] == "skip":
            raise FileNotFoundError("no off-target cluster file (manifest column / glob), "
                                    "or use --missing-clusters keep")
        cache_file = None
        if job["cache_dir"]:
            key = json.dumps([job["ontarget_input"], _mtime(job["ontarget_input"]),
                              job["offtarget_input"], _mtime(job["offtarget_input"]),
                              job["cols"], job["bin_size"], job["mask_path"], _mtime(job["mask_path"]),
                              cfile, _mtime(cfile), job["cluster_opts"], "v5"], sort_keys=True)
            digest = hashlib.md5(key.encode()).hexdigest()[:16]
            safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in job["sample"])
            cache_file = Path(job["cache_dir"]) / f"{safe}.{digest}.npz"
            if cache_file.exists():
                z = np.load(cache_file)
                return dict(sample=job["sample"], error=None, **{k: z[k] for k in z.files})

        mask_df = load_mask_regions(job["mask_path"]) if job["mask_path"] else None
        c = job["cols"]
        on, on_ref = load_region(job["ontarget_input"], c["on_wes"], c["on_wgs"], job["bin_size"], mask_df,
                                 c.get("on_qc"), c["on_mean"])
        off, off_ref = load_region(job["offtarget_input"], c["off_wes"], c["off_wgs"], job["bin_size"], mask_df,
                                   c.get("off_qc"), c["off_mean"])
        stats = dict(n_on_aligned=len(on), n_off_aligned=len(off),
                     n_on_masked=int(on["masked"].sum()), n_off_masked=int(off["masked"].sum()),
                     n_off_wes_positive=off_ref["n_wes_pos"])
        means = dict(on_wes_mean=on_ref["wes_mean"], on_wgs_mean=on_ref["wgs_mean"],
                     off_wes_mean=off_ref["wes_mean"], off_wgs_mean=off_ref["wgs_mean"])
        o = job["cluster_opts"]
        iv = None
        n_iv = 0
        if cfile is not None:
            iv = load_cluster_intervals(cfile, job["bin_size"], o["label_col"], o["exclude_labels"], o["keep_labels"])
            n_iv = len(iv)
        in_clu = lambda d: (bins_in_intervals(d["chrom"].to_numpy(), d["start"].to_numpy(), d["end"].to_numpy(),
                                              iv, o["min_overlap_bp"]) if iv is not None
                            else np.zeros(len(d), dtype=bool))
        on_clu, off_clu = in_clu(on), in_clu(off)

        # recurrent-bin mask, optionally NOT applied to bins inside the sample's clusters
        rescue = o.get("keep_masked_in_clusters", "off")
        on_rescue = on["masked"].to_numpy() & on_clu if rescue == "both" else np.zeros(len(on), dtype=bool)
        off_rescue = off["masked"].to_numpy() & off_clu if rescue in ("both", "offtarget") else np.zeros(len(off), dtype=bool)
        stats.update(n_on_masked_kept_in_cluster=int(on_rescue.sum()),
                     n_off_masked_kept_in_cluster=int(off_rescue.sum()))
        on_keep = ~on["masked"].to_numpy() | on_rescue
        off_keep = ~off["masked"].to_numpy() | off_rescue
        on, off, off_clu = on[on_keep], off[off_keep], off_clu[off_keep]

        if cfile is not None:                      # off-target: only bins inside clusters are scored
            off = off[off_clu]

        parts = []
        for region, d in ((0, on), (1, off)):
            arr = d[list(cols)].to_numpy(dtype=float)
            ok = np.isfinite(arr).all(axis=1)
            parts.append((region, arr[ok]))
        out = dict(
            region=np.concatenate([np.full(len(a), r, dtype=np.int8) for r, a in parts]),
            values=np.concatenate([a for _, a in parts]).astype(np.float32),
            n_on_used=np.int64(len(parts[0][1])), n_off_used=np.int64(len(parts[1][1])),
            n_cluster_intervals=np.int64(n_iv), **{k: np.int64(v) for k, v in stats.items()},
            **{k: np.float64(v) for k, v in means.items()},
        )
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_file, **out)
        return dict(sample=job["sample"], error=None, **out)
    except Exception as exc:
        return dict(sample=job["sample"], error=repr(exc))


# --------------------------------------------------------------------------
# Pooled data + exact grid engine
# --------------------------------------------------------------------------

class Pool:
    """All usable samples' bins: sample code, region (0 on / 1 off), and the
    four value columns wes_raw, wgs_raw, wes_cn, wgs_cn."""
    COL = {"wes_raw": 0, "wgs_raw": 1, "wes_cn": 2, "wgs_cn": 3}

    def __init__(self, results):
        self.samples = [r["sample"] for r in results]
        self.sample = np.concatenate([np.full(len(r["region"]), i, dtype=np.int32) for i, r in enumerate(results)])
        self.region = np.concatenate([r["region"] for r in results])
        self.values = np.concatenate([r["values"] for r in results])
        self.n = len(self.region)

    def wes(self, scale):
        return self.values[:, self.COL[f"wes_{scale}"]]

    def wgs(self, scale):
        return self.values[:, self.COL[f"wgs_{scale}"]]

    def groups(self, separate):
        """Boolean masks, one per WES-threshold group."""
        return [self.region == 0, self.region == 1] if separate else [np.ones(self.n, dtype=bool)]


def expit(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))


def quantile_sorted(sorted_arr, q):
    n = sorted_arr.size
    pos = q * (n - 1)
    lo = int(np.floor(pos))
    hi = min(lo + 1, n - 1)
    w = pos - lo
    return float(sorted_arr[lo]) * (1.0 - w) + float(sorted_arr[hi]) * w


def count_penalty_term(n, mn, mx, weight):
    if n < mn:
        return weight * (mn - n) / max(mn, 1e-9)
    if n > mx:
        return weight * (n - mx) / max(mx, 1e-9)
    return 0.0


def score_from_counts(I, A, B, metric):
    """Hard score + validity. Scored whenever anything is called on either side (union > 0).
    For overlap, one side calling bins and the other none scores 0 (nothing confirmed)."""
    union = A + B - I
    with np.errstate(divide="ignore", invalid="ignore"):
        if metric == "jaccard":
            s = np.where(union > 0, I / np.where(union > 0, union, 1), 0.0)
        else:
            mn = np.minimum(A, B)
            s = np.where(mn > 0, I / np.where(mn > 0, mn, 1), 0.0)
    return s, union > 0


TUNABLE = ("min_frac", "max_frac", "min_bins", "min_samples", "min_cn_like",   # constraints
           "temp", "lr", "n_steps", "n_starts")                               # optimizer settings


def with_overrides(a, combo):
    b = argparse.Namespace(**vars(a))
    for k, v in combo.items():
        setattr(b, k, v)
    return b


class GradSearch:
    """Gradient-descent threshold search, generalising compare_wes_wgs_overlap_grad_pooled.py
    from (1 WES + 1 WGS) to (G WES groups + 1 WGS):

      * one free parameter per threshold: the fraction of that axis' bins called positive,
        kept in logit space and mapped (log-uniformly) into an allowed range, so it can never
        leave the range; threshold = quantile of that axis at (1 - fraction), then
        max(threshold, --min-cn-like) on the CN-like scale (same as the original script)
          - exp1: WES on-target, WES off-target (fractions of each region's own bins), WGS
          - exp2: one shared WES fraction over all bins, WGS
      * smooth surrogate: sigmoid bin membership in quantile-rank space (width --temp x the
        current positive fraction, so it is scale-free on every axis), soft min/max for
        intersection/union,
        micro = pooled sums, macro = per-sample sums with tanh(soft min-count) weights,
        minus count penalties keeping total WES+ / WGS+ inside the allowed bin range
      * Adam (--lr, --n-steps) from --n-starts starting fractions, finite-difference gradients
      * every step is also scored with the TRUE hard-threshold score; the reported best point
        is the highest hard score among steps that satisfy all eligibility constraints."""

    def __init__(self, pool, scale, separate, metric, aggregate, a):
        self.metric, self.aggregate, self.a = metric, aggregate, a
        self.wes = pool.wes(scale).astype(np.float64)
        self.wgs = pool.wgs(scale).astype(np.float64)
        self.N = pool.n
        self.gidx = pool.region.astype(np.int64) if separate else np.zeros(self.N, dtype=np.int64)
        self.G = 2 if separate else 1
        self.names = group_names(separate)
        self.floor = a.min_cn_like if scale == "cn" else None       # threshold floor (CN-like scale only)
        # CN-like gate: a bin can only be called positive if its CN-like value exceeds --min-cn-like.
        # For CN-like experiments this is the same as flooring the threshold; for RAW experiments it is
        # the only floor -- raw thresholds are optimised, CN-like is used just to discount low-CN bins.
        self.gate_floor = a.min_cn_like if (scale == "cn" or getattr(a, "raw_cn_floor", True)) else None
        if self.gate_floor is not None:
            self.gate_w = pool.wes("cn").astype(np.float64) > self.gate_floor
            self.gate_g = pool.wgs("cn").astype(np.float64) > self.gate_floor
        else:
            self.gate_w = np.ones(self.N, dtype=bool)
            self.gate_g = np.ones(self.N, dtype=bool)
        # samples are stored contiguously in the pool
        self.starts = np.flatnonzero(np.r_[True, pool.sample[1:] != pool.sample[:-1]])

        # allowed total positives per side, as in the original (min_bins / min_frac / max_frac)
        self.lo_tot = max(a.min_frac, a.min_bins / self.N)
        self.hi_tot = a.max_frac
        if self.lo_tot >= self.hi_tot:
            raise ValueError(f"min_frac/min_bins ({self.lo_tot:.4g}) >= max_frac ({self.hi_tot:.4g}); "
                             f"widen the allowed range")
        self.max_bins = self.hi_tot * self.N
        self.min_bins = self.lo_tot * self.N

        # per-axis sorted values and fraction ranges
        self.wes_sorted, self.frange = [], []
        for g in range(self.G):
            v = np.sort(self.wes[self.gidx == g])
            self.wes_sorted.append(v)
            if self.G == 1:
                self.frange.append((self.lo_tot, self.hi_tot))
            else:                      # fraction of THIS region's bins; total checked by penalty/eligibility
                n_g = max(len(v), 1)
                hi = min(1.0, self.hi_tot * self.N / n_g)
                lo = max(1.0 / n_g, 1e-6)
                if lo >= hi:
                    lo = hi / 10.0
                self.frange.append((lo, hi))
        self.wgs_sorted = np.sort(self.wgs)
        self.frange.append((self.lo_tot, self.hi_tot))
        self.sorted_axes = self.wes_sorted + [self.wgs_sorted]

        # per-bin empirical quantile rank within its own axis (WES: within its region; WGS: all bins).
        # The surrogate works in this rank space, where "fraction called positive" is linear.
        self.u_wes = np.empty(self.N)
        for g in range(self.G):
            m = self.gidx == g
            sv = self.wes_sorted[g]
            self.u_wes[m] = np.searchsorted(sv, self.wes[m], side="right") / max(len(sv), 1)
        self.u_wgs = np.searchsorted(self.wgs_sorted, self.wgs, side="right") / max(self.N, 1)
        # rank of the CN-like floor on each axis (the floor caps how low a threshold can go)
        if a.min_cn_like is not None and scale == "cn":
            self.u_floor = np.array([np.searchsorted(sv, a.min_cn_like, side="right") / max(len(sv), 1)
                                     for sv in self.sorted_axes])
        else:
            self.u_floor = np.zeros(len(self.sorted_axes))

        # bins that can ever be positive given the floor (caps the min-count penalty, as in the original)
        self.n_avail_wes = int(self.gate_w.sum())
        self.n_avail_wgs = int(self.gate_g.sum())

    # ---- parameter <-> threshold ----
    def fracs(self, z):
        out = []
        for zi, (lo, hi) in zip(z, self.frange):
            out.append(float(np.exp(np.log(lo) + (np.log(hi) - np.log(lo)) * expit(zi))))
        return out

    def thresholds(self, z):
        t = [quantile_sorted(sv, 1.0 - f) if len(sv) else np.inf for sv, f in zip(self.sorted_axes, self.fracs(z))]
        if self.floor is not None:
            t = [max(x, self.floor) for x in t]
        return t

    # ---- smooth surrogate ----
    def soft_loss(self, z):
        # Rank-space sigmoid membership: a bin is "soft-positive" by how far its quantile rank sits
        # above the cutoff rank c = max(1 - f, rank of floor), with width --temp x (1 - c), i.e. a
        # fixed fraction of the bins currently called positive. Scale-free on every axis (raw or
        # CN-like, on- or off-target), so soft counts track hard counts and penalties behave.
        f = np.asarray(self.fracs(z))
        c = np.maximum(1.0 - f, self.u_floor)
        w = np.maximum(self.a.temp * (1.0 - c), 1e-9)
        s_w = expit((self.u_wes - c[:-1][self.gidx]) / w[:-1][self.gidx]) * self.gate_w
        s_g = expit((self.u_wgs - c[-1]) / w[-1]) * self.gate_g
        s_min, s_max = np.minimum(s_w, s_g), np.maximum(s_w, s_g)
        w_sum, g_sum = float(s_w.sum()), float(s_g.sum())
        if self.aggregate == "micro":
            inter, union = float(s_min.sum()), float(s_max.sum())
            if self.metric == "jaccard":
                score = inter / union if union > 0 else 0.0
            else:
                mn = min(w_sum, g_sum)
                score = inter / mn if mn > 0 else 0.0
        else:
            pw = np.add.reduceat(s_w, self.starts)
            pg = np.add.reduceat(s_g, self.starts)
            pi = np.add.reduceat(s_min, self.starts)
            pu = np.add.reduceat(s_max, self.starts)
            pmn = np.minimum(pw, pg)
            with np.errstate(divide="ignore", invalid="ignore"):
                per = (np.where(pu > 1e-9, pi / pu, 0.0) if self.metric == "jaccard"
                       else np.where(pmn > 1e-9, pi / pmn, 0.0))
            weight = np.tanh(pmn)
            wsum = float(weight.sum())
            score = float((per * weight).sum() / wsum) if wsum > 1e-9 else 0.0
        a = self.a
        score -= count_penalty_term(w_sum, min(self.min_bins, self.n_avail_wes), self.max_bins, a.count_penalty)
        score -= count_penalty_term(g_sum, min(self.min_bins, self.n_avail_wgs), self.max_bins, a.count_penalty)
        return -score

    # ---- true hard score ----
    def hard(self, z):
        t = self.thresholds(z)
        W = (self.wes > np.asarray(t[:-1])[self.gidx]) & self.gate_w
        Gp = (self.wgs > t[-1]) & self.gate_g
        A_s = np.add.reduceat(W.astype(np.int64), self.starts)
        B_s = np.add.reduceat(Gp.astype(np.int64), self.starts)
        I_s = np.add.reduceat((W & Gp).astype(np.int64), self.starts)
        A, B, I = int(A_s.sum()), int(B_s.sum()), int(I_s.sum())
        s_val, _ = score_from_counts(np.float64(I), np.float64(A), np.float64(B), self.metric)
        micro = float(np.nan_to_num(s_val))
        per, valid = score_from_counts(I_s.astype(float), A_s.astype(float), B_s.astype(float), self.metric)
        macro = float(per[valid].mean()) if valid.any() else 0.0
        n_contrib = int(((A_s > 0) & (B_s > 0)).sum())
        a = self.a
        lo_n = max(a.min_bins, a.min_frac * self.N)
        eligible = bool(lo_n <= A <= self.max_bins and lo_n <= B <= self.max_bins and n_contrib >= a.min_samples)
        return dict(thresholds=t, score=macro if self.aggregate == "macro" else micro,
                    micro=micro, macro=macro, A=A, B=B, I=I, n_contrib=n_contrib, eligible=eligible)

    # ---- Adam over all starts ----
    def run(self):
        a = self.a
        D = self.G + 1
        rng = np.random.default_rng(a.seed)
        beta1, beta2, eps, fd = 0.9, 0.999, 1e-8, 1e-3
        rows = []
        for si, u0 in enumerate(np.linspace(0.1, 0.9, max(1, a.n_starts))):
            z = np.full(D, logit(u0)) + rng.normal(0, 0.1, size=D)
            m, v = np.zeros(D), np.zeros(D)
            for step in range(a.n_steps + 1):
                h = self.hard(z)
                row = dict(start=si, step=step, soft_objective=-self.soft_loss(z), hard_score=h["score"],
                           micro=h["micro"], macro=h["macro"], n_wes_pos=h["A"], n_wgs_pos=h["B"],
                           n_both=h["I"], n_contributing_samples=h["n_contrib"], eligible=h["eligible"])
                row.update({f"frac_{n}": f for n, f in zip(self.names + ["wgs"], self.fracs(z))})
                row.update({f"wes_threshold_{n}": t for n, t in zip(self.names, h["thresholds"][:-1])})
                row["wgs_threshold"] = h["thresholds"][-1]
                rows.append(row)
                if step == a.n_steps:
                    break
                grad = np.zeros(D)
                for i in range(D):
                    zp, zm = z.copy(), z.copy()
                    zp[i] += fd
                    zm[i] -= fd
                    grad[i] = (self.soft_loss(zp) - self.soft_loss(zm)) / (2 * fd)
                t_ = step + 1
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad ** 2
                z = z - a.lr * (m / (1 - beta1 ** t_)) / (np.sqrt(v / (1 - beta2 ** t_)) + eps)
        return pd.DataFrame(rows)


def run_experiment(pool, exp, metric, a, combos=({},)):
    """Gradient-descent search on `pool` for every constraint combo (the constraints shape the
    search itself -- fraction ranges and penalties -- so each combo gets its own run).
    Returns [(combo, best or None, top_df or None, history)]."""
    scale, separate = EXPERIMENTS[exp]
    out = []
    for combo in combos:
        ac = with_overrides(a, combo)
        try:
            gs = GradSearch(pool, scale, separate, metric, a.aggregate, ac)
        except ValueError as e:
            print(f"   combo {combo or '(base)'}: {e}", flush=True)
            out.append((combo, None, None, None))
            continue
        hist = gs.run()
        thr_cols = [f"wes_threshold_{n}" for n in group_names(separate)] + ["wgs_threshold"]
        elig = hist[hist["eligible"]]
        if elig.empty:
            if len(combos) == 1:
                print(f"   no eligible step: WES+ ranged {hist.n_wes_pos.min()}-{hist.n_wes_pos.max()}, "
                      f"WGS+ {hist.n_wgs_pos.min()}-{hist.n_wgs_pos.max()} "
                      f"(allowed {max(ac.min_bins, ac.min_frac * pool.n):.0f}-{ac.max_frac * pool.n:.0f}); "
                      f"contributing samples max {hist.n_contributing_samples.max()} (need {ac.min_samples})",
                      flush=True)
            out.append((combo, None, None, hist))
            continue
        b = elig.loc[elig["hard_score"].idxmax()]
        best = dict(thresholds=[float(b[c]) for c in thr_cols], score=float(b["hard_score"]))
        top = (elig.sort_values("hard_score", ascending=False).drop_duplicates(thr_cols).head(50)
               .rename(columns={"hard_score": "score"})
               [thr_cols + ["score", "n_wes_pos", "n_wgs_pos", "n_both", "n_contributing_samples", "start", "step"]])
        out.append((combo, best, top, hist))
    return out


def group_names(separate):
    return ["ontarget", "offtarget"] if separate else ["shared"]


def evaluate(pool, scale, separate, wes_thr, wgs_thr, cn_floor=None):
    """Hard-threshold per-sample table + pooled micro/macro scores at one combo."""
    wes, wgs = pool.wes(scale), pool.wgs(scale)
    thr = np.where(pool.region == 0, wes_thr[0], wes_thr[1]) if separate else np.full(pool.n, wes_thr[0])
    W, Gp = wes > thr, wgs > wgs_thr
    if cn_floor is not None:          # CN-like gate (see GradSearch): low-CN bins are never positive
        W &= pool.wes("cn") > cn_floor
        Gp &= pool.wgs("cn") > cn_floor
    ns = len(pool.samples)
    cnt = lambda m: np.bincount(pool.sample[m], minlength=ns)
    tab = pd.DataFrame(dict(sample=pool.samples,
                            n_bins=cnt(np.ones(pool.n, bool)),
                            n_ontarget_bins=cnt(pool.region == 0), n_offtarget_bins=cnt(pool.region == 1),
                            n_wes_pos=cnt(W), n_wgs_pos=cnt(Gp), n_both=cnt(W & Gp),
                            n_wes_pos_ontarget=cnt(W & (pool.region == 0)),
                            n_wes_pos_offtarget=cnt(W & (pool.region == 1))))
    union = tab.n_wes_pos + tab.n_wgs_pos - tab.n_both
    mn = np.minimum(tab.n_wes_pos, tab.n_wgs_pos)
    with np.errstate(divide="ignore", invalid="ignore"):
        tab["jaccard"] = np.where(union > 0, tab.n_both / union.where(union > 0, 1), np.nan)
        # one side calls bins, the other none -> overlap 0; nothing called on either side -> NaN (not averaged)
        tab["overlap"] = np.where(union > 0, np.where(mn > 0, tab.n_both / mn.where(mn > 0, 1), 0.0), np.nan)
    Ia, Aa, Ba = int(tab.n_both.sum()), int(tab.n_wes_pos.sum()), int(tab.n_wgs_pos.sum())
    pooled = dict(
        micro_jaccard=Ia / (Aa + Ba - Ia) if Aa + Ba - Ia else np.nan,
        micro_overlap=Ia / min(Aa, Ba) if min(Aa, Ba) else np.nan,
        macro_jaccard=float(tab["jaccard"].mean(skipna=True)),
        macro_overlap=float(tab["overlap"].mean(skipna=True)),
        n_wes_pos=Aa, n_wgs_pos=Ba, n_both=Ia, n_bins=pool.n, n_samples=ns,
    )
    return tab, pooled


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True,
                   help="TSV: sample, ontarget_input, offtarget_input [, offtarget_clusters] -- search is fit on these")
    p.add_argument("--eval-manifest", default=None,
                   help="Same columns. Never searched on; used to CHOOSE among --tune combos "
                        "(the combo with the best eval score wins)")
    p.add_argument("--test-manifest", default=None,
                   help="Same columns. Never searched on and never used for choosing; scored once, "
                        "at the chosen combo only -- the unbiased final number")
    p.add_argument("--holdout-manifest", default=None, help="(legacy) extra split, scored like --test-manifest")
    p.add_argument("--tune", nargs="*", default=[],
                   help="Constraint values to tune on the eval split, e.g. "
                        "--tune max_frac=0.01,0.05,0.10 min_samples=10,50 min_cn_like=3,4 . Every "
                        f"combination is searched on train and scored on eval. Tunable: {', '.join(TUNABLE)}")
    p.add_argument("--ontarget-wes-column", default="raw_wes_depth")
    p.add_argument("--ontarget-wgs-column", default="wgs_tumor_depth")
    p.add_argument("--offtarget-wes-column", default="predicted_loess_upscale_depth")
    p.add_argument("--offtarget-wgs-column", default="wgs_tumor_depth")
    p.add_argument("--offtarget-qc-column", default="mask_rejected",
                   help="Off-target rows with a non-zero value here are dropped BEFORE rebinning and "
                        "CN-like normalization ('none' to keep all rows). Default mask_rejected, the "
                        "same QC column the recurrent-bin search used as --mask-col")
    p.add_argument("--ontarget-qc-column", default="none")
    p.add_argument("--offtarget-cn-mean", choices=["positive", "all"], default="positive",
                   help="Off-target CN-like reference mean: 'positive' = mean of off-target bins with depth > 0 "
                        "(default), 'all' = all off-target bins. Applied to WES and WGS separately.")
    p.add_argument("--ontarget-cn-mean", choices=["all", "positive"], default="all",
                   help="On-target CN-like reference mean: 'all' = mean of all on-target bins (default).")
    p.add_argument("--target-bin-size", type=int, default=25000)
    p.add_argument("--mask-regions", default=None)

    p.add_argument("--clusters-column", default="offtarget_clusters")
    p.add_argument("--offtarget-cluster-glob", default=None)
    p.add_argument("--cluster-label-column", default=None)
    p.add_argument("--cluster-exclude-labels", default="-1,noise")
    p.add_argument("--cluster-keep-labels", default=None)
    p.add_argument("--cluster-min-overlap-bp", type=int, default=1)
    p.add_argument("--missing-clusters", choices=["skip", "keep"], default="skip")
    p.add_argument("--keep-masked-in-clusters", choices=["off", "offtarget", "both"], default="off",
                   help="Do NOT apply the recurrent-bin mask to bins inside the sample's clusters. "
                        "'both' = on- and off-target bins overlapping a cluster interval keep despite the mask; "
                        "'offtarget' = only off-target bins (equivalent to no mask for scored off-target bins, "
                        "since only clustered off-target bins are scored); 'off' = mask everything (default)")

    p.add_argument("--experiments", default="exp1_raw,exp1_cnlike,exp2_cnlike")
    p.add_argument("--metrics", default="jaccard,overlap")
    p.add_argument("--aggregate", choices=["micro", "macro"], default="micro")
    p.add_argument("--n-steps", type=int, default=100, help="Adam steps per starting point")
    p.add_argument("--n-starts", type=int, default=5,
                   help="Number of starting fractions (log-spaced across each axis' allowed range)")
    p.add_argument("--lr", type=float, default=0.2, help="Adam learning rate in the unconstrained (logit) space")
    p.add_argument("--temp", type=float, default=0.1,
                   help="Sigmoid width, as a fraction of the bins currently called positive on each axis "
                        "(rank-space surrogate; scale-free across raw/CN-like and on/off-target)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--count-penalty", type=float, default=1.0,
                   help="Weight of the surrogate penalty keeping total WES+/WGS+ inside the allowed bin range")
    p.add_argument("--min-frac", type=float, default=0.0005)
    p.add_argument("--max-frac", type=float, default=0.01)
    p.add_argument("--min-bins", type=int, default=1000)
    p.add_argument("--min-samples", type=int, default=10)
    p.add_argument("--min-cn-like", type=float, default=4.0,
                   help="CN-like floor. CN-like experiments: thresholds >= this. RAW experiments: a bin can only "
                        "be called positive if its CN-like value exceeds this (raw thresholds are still what is "
                        "optimised; CN-like is computed only to apply this floor). Tunable with --tune.")
    p.add_argument("--no-raw-cn-floor", dest="raw_cn_floor", action="store_false",
                   help="Do NOT apply the --min-cn-like floor to raw experiments")
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--outdir", default="pooled_onoff_clustered_experiments")
    a = p.parse_args()

    exps = [e.strip() for e in a.experiments.split(",") if e.strip()]
    bad = set(exps) - set(EXPERIMENTS)
    if bad:
        raise SystemExit(f"Unknown experiments {sorted(bad)}; choose from {sorted(EXPERIMENTS)}")
    metrics = [m.strip() for m in a.metrics.split(",") if m.strip()]
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cache_dir = a.cache_dir or str(outdir / "cache")

    split = lambda v: tuple(x.strip() for x in v.split(",") if x.strip()) if v else ()
    cluster_opts = dict(label_col=a.cluster_label_column, exclude_labels=split(a.cluster_exclude_labels),
                        keep_labels=split(a.cluster_keep_labels) or None, min_overlap_bp=a.cluster_min_overlap_bp,
                        keep_masked_in_clusters=a.keep_masked_in_clusters)
    cols = dict(on_wes=a.ontarget_wes_column, on_wgs=a.ontarget_wgs_column,
                off_wes=a.offtarget_wes_column, off_wgs=a.offtarget_wgs_column,
                on_qc=a.ontarget_qc_column, off_qc=a.offtarget_qc_column,
                on_mean=a.ontarget_cn_mean, off_mean=a.offtarget_cn_mean)

    def prepare(manifest_path, tag):
        man = pd.read_csv(manifest_path, sep="\t")
        miss = {"sample", "ontarget_input", "offtarget_input"} - set(man.columns)
        if miss:
            raise SystemExit(f"{manifest_path} missing columns {sorted(miss)}")
        jobs = []
        for r in man.to_dict("records"):
            s = str(r["sample"])
            jobs.append(dict(sample=s, ontarget_input=str(r["ontarget_input"]), offtarget_input=str(r["offtarget_input"]),
                             cluster_file=resolve_cluster_file(s, r.get(a.clusters_column), a.offtarget_cluster_glob),
                             cols=cols, bin_size=a.target_bin_size, mask_path=a.mask_regions, cache_dir=cache_dir,
                             cluster_opts=cluster_opts, missing_clusters=a.missing_clusters))
        print(f"[{tag}] {len(jobs)} samples, {sum(j['cluster_file'] is not None for j in jobs)} with cluster files",
              flush=True)
        if a.n_jobs > 1:
            with ProcessPoolExecutor(max_workers=a.n_jobs) as ex:
                results = list(ex.map(preprocess_sample, jobs, chunksize=1))
        else:
            results = [preprocess_sample(j) for j in jobs]
        used = [r for r in results if r["error"] is None and len(r["region"]) > 0]
        skipped = [dict(sample=r["sample"], reason=r["error"] or "no usable bins")
                   for r in results if not (r["error"] is None and len(r["region"]) > 0)]
        pd.DataFrame(skipped, columns=["sample", "reason"]).to_csv(
            outdir / f"samples_skipped_{tag}.tsv", sep="\t", index=False)
        pd.DataFrame([dict(sample=r["sample"],
                           **{k: int(r[k]) for k in ("n_on_aligned", "n_off_aligned", "n_on_masked", "n_off_masked",
                                                     "n_on_masked_kept_in_cluster", "n_off_masked_kept_in_cluster",
                                                     "n_off_wes_positive", "n_on_used", "n_off_used",
                                                     "n_cluster_intervals")},
                           **{k: float(r[k]) for k in ("on_wes_mean", "on_wgs_mean", "off_wes_mean", "off_wgs_mean")})
                      for r in used]).to_csv(outdir / f"samples_used_{tag}.tsv", sep="\t", index=False)
        if not used:
            raise SystemExit(f"[{tag}] no usable samples")
        pool = Pool(used)
        n_off = int((pool.region == 1).sum())
        print(f"[{tag}] using {len(used)} samples ({len(skipped)} skipped): {pool.n} bins "
              f"({pool.n - n_off} on-target, {n_off} clustered off-target)", flush=True)
        return pool

    print(f"CN-like = {CN_SCALE:g} x depth / region mean; on-target mean over "
          f"{'all bins' if a.ontarget_cn_mean == 'all' else 'bins > 0'}, off-target mean over "
          f"{'all bins' if a.offtarget_cn_mean == 'all' else 'bins > 0'} (WES and WGS separately; computed "
          f"before mask/cluster filtering)", flush=True)
    train = prepare(a.manifest, "train")
    if a.min_bins > a.max_frac * train.n:
        print(f"[warn] base --min-bins {a.min_bins} exceeds --max-frac x pooled bins = {a.max_frac * train.n:.0f}: "
              f"combos with these values cannot be eligible.", flush=True)
    print(f"Recurrent mask: {a.mask_regions or 'none'}; masked bins inside clusters "
          f"{'are KEPT (' + a.keep_masked_in_clusters + ')' if a.keep_masked_in_clusters != 'off' else 'are removed'}",
          flush=True)
    print(f"CN-like floor --min-cn-like {a.min_cn_like}: applied to CN-like thresholds"
          f"{' and, as a per-bin gate, to RAW experiments' if a.raw_cn_floor else ' only (raw ungated)'}", flush=True)
    print(f"Search: Adam on a sigmoid surrogate ({a.n_starts} starts x {a.n_steps} steps, lr {a.lr}, "
          f"temp {a.temp}); best = highest HARD score among eligible steps", flush=True)
    evalp = prepare(a.eval_manifest, "eval") if a.eval_manifest else None
    scored = {k: prepare(m, k) for k, m in (("test", a.test_manifest), ("holdout", a.holdout_manifest)) if m}

    combos = [{}]
    if a.tune:
        import itertools
        keys, vals = [], []
        for tok in a.tune:
            k, _, v = tok.partition("=")
            k = k.strip().replace("-", "_")
            if k not in TUNABLE:
                raise SystemExit(f"--tune: '{k}' is not tunable; choose from {TUNABLE}")
            caster = int if k in ("min_bins", "min_samples", "n_steps", "n_starts") else float
            keys.append(k); vals.append([caster(x) for x in v.split(",") if x.strip()])
        combos = [dict(zip(keys, c)) for c in itertools.product(*vals)]
        print(f"Tuning {len(combos)} constraint combos on "
              f"{'the eval split' if evalp is not None else 'TRAIN (no --eval-manifest: picks by train score)'}",
              flush=True)

    def gate_of(combo, scale):
        ac = with_overrides(a, combo)
        return ac.min_cn_like if (scale == "cn" or ac.raw_cn_floor) else None

    summary = []
    key = lambda metric: f"{a.aggregate}_{metric}"
    for exp in exps:
        scale, separate = EXPERIMENTS[exp]
        names = group_names(separate)
        for metric in metrics:
            print(f"== {exp} / {metric} ({a.aggregate})", flush=True)
            results = run_experiment(train, exp, metric, a, combos)

            tuning_rows = []
            for combo, best, top, hist in results:
                trow = {**{k: getattr(with_overrides(a, combo), k) for k in TUNABLE}}
                if best is None:
                    tuning_rows.append({**trow, "status": "no_eligible_combo"})
                    continue
                wes_thr, wgs_thr = best["thresholds"][:-1], best["thresholds"][-1]
                trow.update({f"wes_threshold_{n}": t for n, t in zip(names, wes_thr)}, wgs_threshold=wgs_thr,
                            train_score=best["score"], status="ok")
                if evalp is not None:
                    _, ep = evaluate(evalp, scale, separate, wes_thr, wgs_thr, gate_of(combo, scale))
                    trow["eval_score"] = ep[key(metric)]
                    trow.update({f"eval_{k}": v for k, v in ep.items()})
                tuning_rows.append(trow)
            tuning = pd.DataFrame(tuning_rows)
            if len(combos) > 1:
                tuning.to_csv(outdir / f"{exp}_{metric}_tuning.tsv", sep="\t", index=False)

            ok_rows = [i for i, r in enumerate(results) if r[1] is not None]
            row = dict(experiment=exp, scale=scale, metric=metric, aggregate=a.aggregate)
            if not ok_rows:
                print("   no eligible combo -- loosen --min-frac/--max-frac/--min-bins/--min-samples", flush=True)
                summary.append({**row, "status": "no_eligible_combo"})
                continue
            sel_col = "eval_score" if evalp is not None else "train_score"
            pick = max(ok_rows, key=lambda i: (np.nan_to_num(tuning.loc[i, sel_col], nan=-np.inf),
                                               tuning.loc[i, "train_score"]))
            combo, best, top, hist = results[pick]
            hist.to_csv(outdir / f"{exp}_{metric}_search_history.tsv", sep="\t", index=False)
            wes_thr, wgs_thr = best["thresholds"][:-1], best["thresholds"][-1]
            row.update({f"chosen_{k}": getattr(with_overrides(a, combo), k) for k in TUNABLE})
            row.update({f"wes_threshold_{n}": t for n, t in zip(names, wes_thr)}, wgs_threshold=wgs_thr,
                       train_score=best["score"], n_combos_tried=len(combos))
            top.to_csv(outdir / f"{exp}_{metric}_top_eligible.tsv", sep="\t", index=False)

            splits = {"train": train, **({"eval": evalp} if evalp is not None else {}), **scored}
            for tag, pool_ in splits.items():
                tab, pooled = evaluate(pool_, scale, separate, wes_thr, wgs_thr, gate_of(combo, scale))
                tab.to_csv(outdir / f"{exp}_{metric}_per_sample_{tag}.tsv", sep="\t", index=False)
                row.update({f"{tag}_{k}": v for k, v in pooled.items()})
            row["status"] = "ok"
            pd.DataFrame([row]).to_csv(outdir / f"{exp}_{metric}_best.tsv", sep="\t", index=False)
            summary.append(row)

            thr_txt = ", ".join(f"WES {n}={t:.4g}" for n, t in zip(names, wes_thr))
            sc_txt = "; ".join(f"{tag} {row.get(f'{tag}_{key(metric)}', np.nan):.4f}" for tag in splits)
            ch_txt = (" | chosen " + ", ".join(f"{k}={v}" for k, v in combo.items())) if combo else ""
            print(f"   best: {thr_txt}, WGS={wgs_thr:.4g}; {a.aggregate} {metric}: {sc_txt}{ch_txt}", flush=True)

    pd.DataFrame(summary).to_csv(outdir / "summary.tsv", sep="\t", index=False)
    print(f"Saved: {outdir / 'summary.tsv'}", flush=True)


if __name__ == "__main__":
    main()
