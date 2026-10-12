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
  4. Drop masked bins, then drop off-target bins outside the sample's clusters.
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
  threshold. Scores: jaccard = |W&G| / |W|G|, overlap = |W&G| / min(|W|,|G|).
  --aggregate micro pools bin counts across samples; macro averages per-sample
  scores. In macro, a sample with nothing called on EITHER side is left out of the
  mean; a sample where only one side calls bins scores 0 (for both metrics).

SEARCH
------
  Exact search over a threshold grid (no gradients/softening): each axis gets
  --grid-size candidate thresholds at upper-tail quantiles of that axis' values
  (from --q-low to --q-high, denser toward the tail), every combination is
  scored exactly with 2-D cumulative count tables, then one refinement pass
  re-grids between the neighbours of the best point. All three experiments use
  the same engine, so their scores are directly comparable.

  Eligibility (a combo can only be "best" if all hold):
    - WES-positive and WGS-positive pooled bins each >= --min-bins and each a
      fraction of all pooled bins within [--min-frac, --max-frac]
    - >= --min-samples samples with at least one WES-positive AND one
      WGS-positive bin
    - CN-like thresholds (exp1_cnlike, exp2_cnlike) >= --min-cn-like

TRAIN / HOLDOUT
---------------
  The search is fit on --manifest only. If --holdout-manifest is given, those
  samples are preprocessed identically and scored at each best threshold.

OUTPUT (--outdir)
-----------------
  summary.tsv                                one row per experiment x metric:
                                             thresholds, train + holdout scores
  <exp>_<metric>_best.tsv                    full detail for the best combo
  <exp>_<metric>_top_eligible.tsv            top 50 eligible grid combos
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
                              cfile, _mtime(cfile), job["cluster_opts"], "v4"], sort_keys=True)
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
        on, off = on[~on["masked"]], off[~off["masked"]]

        n_iv = 0
        if cfile is not None:
            o = job["cluster_opts"]
            iv = load_cluster_intervals(cfile, job["bin_size"], o["label_col"], o["exclude_labels"], o["keep_labels"])
            n_iv = len(iv)
            keep = bins_in_intervals(off["chrom"].to_numpy(), off["start"].to_numpy(), off["end"].to_numpy(),
                                     iv, o["min_overlap_bp"])
            off = off[keep]

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


def tail_grid(values, n, q_low, q_high, extra=()):
    v = values[np.isfinite(values)]
    if not len(v):
        return np.asarray(sorted(set(extra)), dtype=float)
    qs = 1.0 - np.logspace(np.log10(1 - q_low), np.log10(1 - q_high), n)
    return np.unique(np.concatenate([np.quantile(v, qs), np.asarray(extra, dtype=float)]))


def count_tables(w, y, tw, ty):
    """C[i,k] = #(w>tw[i] & y>ty[k]), A[i] = #(w>tw[i]), B[k] = #(y>ty[k])."""
    n, m = len(tw), len(ty)
    iw = np.searchsorted(tw, w, side="left")       # w > tw[j]  <=>  j < iw
    iy = np.searchsorted(ty, y, side="left")
    H = np.bincount(iw.astype(np.int64) * (m + 1) + iy, minlength=(n + 1) * (m + 1)).reshape(n + 1, m + 1)
    R = H[::-1, ::-1].cumsum(0).cumsum(1)[::-1, ::-1]
    return R[1:, 1:], R[1:, 0], R[0, 1:]


def combine(tables):
    """Broadcast per-group tables into full grids: I, A over (WES axes..., WGS axis), B."""
    G = len(tables)
    I = A = Bsum = 0
    for g, (C, Ag, Bg) in enumerate(tables):
        shp = [1] * (G + 1)
        shp[g], shp[-1] = C.shape[0], C.shape[1]
        I = I + C.reshape(shp)
        shp_a = [1] * (G + 1)
        shp_a[g] = len(Ag)
        A = A + Ag.reshape(shp_a)
        Bsum = Bsum + Bg
    return I, A, np.asarray(Bsum).reshape([1] * G + [-1])


def score_from_counts(I, A, B, metric):
    """Score + validity. A sample/pool is scored whenever anything is called on either side
    (union > 0). For overlap, if one side calls bins and the other calls none, the score is 0
    (nothing confirmed) rather than undefined -- otherwise macro averages would silently drop
    exactly the samples where WES and WGS disagree completely."""
    union = A + B - I
    with np.errstate(divide="ignore", invalid="ignore"):
        if metric == "jaccard":
            s = I / union
        else:
            mn = np.minimum(A, B)
            s = np.where(mn > 0, I / np.where(mn > 0, mn, 1), 0.0)
    return s, union > 0


def grid_scores(pool, scale, separate, metric, aggregate, wes_grids, wgs_grid):
    wes, wgs = pool.wes(scale), pool.wgs(scale)
    masks = pool.groups(separate)
    tables = [count_tables(wes[m], wgs[m], tg, wgs_grid) for m, tg in zip(masks, wes_grids)]
    I, A, B = combine(tables)
    micro, _ = score_from_counts(I, A, B, metric)
    micro = np.nan_to_num(micro, nan=0.0)

    # per-sample pass: macro scores + number of samples with >=1 WES+ and >=1 WGS+
    G = len(masks)
    full = np.broadcast_shapes(I.shape, A.shape, B.shape)
    n_contrib = np.zeros(full, dtype=np.int32)
    macro_sum = np.zeros(full) if aggregate == "macro" else None
    macro_cnt = np.zeros(full, dtype=np.int32) if aggregate == "macro" else None
    order = np.argsort(pool.sample, kind="stable")
    bounds = np.searchsorted(pool.sample[order], np.arange(len(pool.samples) + 1))
    for s in range(len(pool.samples)):
        idx = order[bounds[s]:bounds[s + 1]]
        if not len(idx):
            continue
        ys = wgs[idx]
        has_b = (ys.max() > wgs_grid).reshape([1] * G + [-1])
        has_a = np.zeros([len(g) for g in wes_grids] + [1], dtype=bool)
        for g, (m, tg) in enumerate(zip(masks, wes_grids)):
            sel = idx[m[idx]]
            if len(sel):
                shp = [1] * (G + 1)
                shp[g] = len(tg)
                has_a = has_a | (wes[sel].max() > tg).reshape(shp)
        n_contrib += has_a & has_b
        if aggregate == "macro":
            t = [count_tables(wes[idx[m[idx]]], ys[m[idx]], tg, wgs_grid) for m, tg in zip(masks, wes_grids)]
            s_sc, valid = score_from_counts(*combine(t), metric)
            macro_sum += np.where(valid, s_sc, 0.0)
            macro_cnt += valid
    if aggregate == "macro":
        with np.errstate(invalid="ignore", divide="ignore"):
            primary = np.nan_to_num(macro_sum / macro_cnt, nan=0.0)
    else:
        primary = micro
    return dict(score=primary, micro=micro, A=np.broadcast_to(A, full), B=np.broadcast_to(B, full),
                I=np.broadcast_to(I, full), n_contrib=n_contrib)


def eligibility(res, pool, scale, wes_grids, wgs_grid, a):
    A, B = res["A"], res["B"]
    ok = (A >= a.min_bins) & (B >= a.min_bins)
    ok &= (A / pool.n >= a.min_frac) & (A / pool.n <= a.max_frac)
    ok &= (B / pool.n >= a.min_frac) & (B / pool.n <= a.max_frac)
    ok &= res["n_contrib"] >= a.min_samples
    if scale == "cn":
        G = len(wes_grids)
        for g, tg in enumerate(wes_grids):
            shp = [1] * (G + 1)
            shp[g] = len(tg)
            ok &= (tg >= a.min_cn_like).reshape(shp)
        ok &= (wgs_grid >= a.min_cn_like).reshape([1] * G + [-1])
    return ok


def diagnose(res, pool, scale, wes_grids, wgs_grid, a):
    """Explain which constraint(s) leave no eligible combo."""
    A, B, N = res["A"], res["B"], pool.n
    lo, hi = a.min_frac * N, a.max_frac * N
    print(f"   pooled bins N={N}: allowed positives per side = [{max(a.min_bins, lo):.0f}, {hi:.0f}] "
          f"(--min-bins {a.min_bins}, --min-frac {a.min_frac} -> {lo:.0f}, --max-frac {a.max_frac} -> {hi:.0f})")
    if a.min_bins > hi:
        print(f"   INFEASIBLE: --min-bins {a.min_bins} > --max-frac x N = {hi:.0f}; "
              f"lower --min-bins or raise --max-frac")
    checks = {
        "WES+ bins within limits": (A >= max(a.min_bins, lo)) & (A <= hi),
        "WGS+ bins within limits": (B >= max(a.min_bins, lo)) & (B <= hi),
        f">= {a.min_samples} contributing samples": res["n_contrib"] >= a.min_samples,
    }
    for name, m in checks.items():
        print(f"   {name}: {100 * m.mean():.1f}% of grid combos pass")
    print(f"   WES+ range over grid: {int(A.min())}-{int(A.max())}, WGS+ range: {int(B.min())}-{int(B.max())}")


def refine_axis(grid, i, n):
    lo = grid[max(i - 1, 0)]
    hi = grid[min(i + 1, len(grid) - 1)]
    return np.unique(np.concatenate([np.linspace(lo, hi, n), [grid[i]]]))


def run_experiment(pool, exp, metric, a):
    scale, separate = EXPERIMENTS[exp]
    wes, wgs = pool.wes(scale), pool.wgs(scale)
    extra = [a.min_cn_like] if scale == "cn" else []
    wes_grids = [tail_grid(wes[m], a.grid_size, a.q_low, a.q_high, extra) for m in pool.groups(separate)]
    wgs_grid = tail_grid(wgs, a.grid_size, a.q_low, a.q_high, extra)

    best = None
    for rnd in range(1 + a.refine_rounds):
        res = grid_scores(pool, scale, separate, metric, a.aggregate, wes_grids, wgs_grid)
        ok = eligibility(res, pool, scale, wes_grids, wgs_grid, a)
        if not ok.any():
            if best is None:
                diagnose(res, pool, scale, wes_grids, wgs_grid, a)
                return None, None
            break
        cand = np.where(ok, res["score"], -np.inf)
        flat = int(np.argmax(cand))
        ijk = np.unravel_index(flat, cand.shape)
        if rnd == 0:
            top_idx = np.argsort(cand, axis=None)[::-1][:50]
            top_idx = [t for t in top_idx if np.isfinite(cand.flat[t])]
            top = []
            for t in top_idx:
                u = np.unravel_index(t, cand.shape)
                row = {f"wes_threshold_{n}": float(g[u[k]]) for k, (n, g) in enumerate(zip(group_names(separate), wes_grids))}
                row.update(wgs_threshold=float(wgs_grid[u[-1]]), score=float(cand.flat[t]),
                           n_wes_pos=int(res["A"][u]), n_wgs_pos=int(res["B"][u]), n_both=int(res["I"][u]),
                           n_contributing_samples=int(res["n_contrib"][u]))
                top.append(row)
            top_df = pd.DataFrame(top)
        best = dict(thresholds=[float(g[ijk[k]]) for k, g in enumerate(wes_grids)] + [float(wgs_grid[ijk[-1]])],
                    score=float(cand.flat[flat]))
        if rnd < a.refine_rounds:
            wes_grids = [refine_axis(g, ijk[k], a.refine_size) for k, g in enumerate(wes_grids)]
            wgs_grid = refine_axis(wgs_grid, ijk[-1], a.refine_size)
    return best, top_df


def group_names(separate):
    return ["ontarget", "offtarget"] if separate else ["shared"]


def evaluate(pool, scale, separate, wes_thr, wgs_thr):
    """Hard-threshold per-sample table + pooled micro/macro scores at one combo."""
    wes, wgs = pool.wes(scale), pool.wgs(scale)
    thr = np.where(pool.region == 0, wes_thr[0], wes_thr[1]) if separate else np.full(pool.n, wes_thr[0])
    W, Gp = wes > thr, wgs > wgs_thr
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
    p.add_argument("--holdout-manifest", default=None, help="Same columns; scored only, never searched on")
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

    p.add_argument("--experiments", default="exp1_raw,exp1_cnlike,exp2_cnlike")
    p.add_argument("--metrics", default="jaccard,overlap")
    p.add_argument("--aggregate", choices=["micro", "macro"], default="micro")
    p.add_argument("--grid-size", type=int, default=100, help="Candidate thresholds per axis")
    p.add_argument("--q-low", type=float, default=0.5, help="Lowest quantile tried per axis")
    p.add_argument("--q-high", type=float, default=0.9999, help="Highest quantile tried per axis")
    p.add_argument("--refine-rounds", type=int, default=1)
    p.add_argument("--refine-size", type=int, default=25)
    p.add_argument("--min-frac", type=float, default=0.0005)
    p.add_argument("--max-frac", type=float, default=0.01)
    p.add_argument("--min-bins", type=int, default=1000)
    p.add_argument("--min-samples", type=int, default=10)
    p.add_argument("--min-cn-like", type=float, default=4.0)
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
                        keep_labels=split(a.cluster_keep_labels) or None, min_overlap_bp=a.cluster_min_overlap_bp)
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
        print(f"[warn] --min-bins {a.min_bins} exceeds --max-frac x pooled bins = {a.max_frac * train.n:.0f}: "
              f"no combo can be eligible. Lower --min-bins or raise --max-frac.", flush=True)
    holdout = prepare(a.holdout_manifest, "holdout") if a.holdout_manifest else None

    summary = []
    for exp in exps:
        scale, separate = EXPERIMENTS[exp]
        names = group_names(separate)
        for metric in metrics:
            print(f"== {exp} / {metric} ({a.aggregate})", flush=True)
            best, top = run_experiment(train, exp, metric, a)
            row = dict(experiment=exp, scale=scale, metric=metric, aggregate=a.aggregate)
            if best is None:
                print("   no eligible combo -- loosen --min-frac/--max-frac/--min-bins/--min-samples", flush=True)
                summary.append({**row, "status": "no_eligible_combo"})
                continue
            wes_thr, wgs_thr = best["thresholds"][:-1], best["thresholds"][-1]
            for n, t in zip(names, wes_thr):
                row[f"wes_threshold_{n}"] = t
            row["wgs_threshold"] = wgs_thr
            row["train_score"] = best["score"]
            top.to_csv(outdir / f"{exp}_{metric}_top_eligible.tsv", sep="\t", index=False)

            tab, pooled = evaluate(train, scale, separate, wes_thr, wgs_thr)
            tab.to_csv(outdir / f"{exp}_{metric}_per_sample_train.tsv", sep="\t", index=False)
            row.update({f"train_{k}": v for k, v in pooled.items()})
            if holdout is not None:
                htab, hpooled = evaluate(holdout, scale, separate, wes_thr, wgs_thr)
                htab.to_csv(outdir / f"{exp}_{metric}_per_sample_holdout.tsv", sep="\t", index=False)
                row.update({f"holdout_{k}": v for k, v in hpooled.items()})
            row["status"] = "ok"
            pd.DataFrame([row]).to_csv(outdir / f"{exp}_{metric}_best.tsv", sep="\t", index=False)
            summary.append(row)
            thr_txt = ", ".join(f"WES {n}={t:.4g}" for n, t in zip(names, wes_thr))
            hold_txt = (f"; holdout {a.aggregate} {metric}={row.get(f'holdout_{a.aggregate}_{metric}', np.nan):.4f}"
                        if holdout is not None else "")
            print(f"   best: {thr_txt}, WGS={wgs_thr:.4g}; train {a.aggregate} {metric}={best['score']:.4f}{hold_txt}",
                  flush=True)

    pd.DataFrame(summary).to_csv(outdir / "summary.tsv", sep="\t", index=False)
    print(f"Saved: {outdir / 'summary.tsv'}", flush=True)


if __name__ == "__main__":
    main()
