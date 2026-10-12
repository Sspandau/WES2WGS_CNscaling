#!/usr/bin/env python3
"""
pooled_grad_on_off_plots.py

Diagnostics + figures for a finished run of compare_wes_wgs_overlap_grad_pooled_on_off_tuning.py
(train / eval / test). Every sample is re-preprocessed WITHOUT dropping recurrent bins, so the
effect of the mask can be measured; everything else (QC drop, CN-like rules, cluster filter,
finite filter) is identical to the run, by importing the run's own functions.

OUTPUT (--outdir, default <run-dir>/figures)
  1. raw_threshold_cnlike_equivalents.tsv / .png
       Each RAW-model threshold is one cohort-wide depth; per sample it corresponds to
       CN-like = scale x threshold / that sample's region mean. Distribution across samples
       (median, IQR, 5-95%) per axis (WES on, WES off, WGS on-target, WGS off-target), plus the
       CN-like gate (--min-cn-like) that also applied to raw calls.
  2. amplified_bin_counts.tsv / .png
       Best models: WES+, WGS+, both, WES-only, WGS-only bins per split (totals and per sample),
       on the bins the run actually scored (after clustering + recurrent-bin mask).
  3. mask_effect.tsv / .png
       Bins removed by the recurrent mask that WOULD have been called at the chosen thresholds:
       how many were WES-only (no WGS support) vs confirmed by WGS -- i.e. did masking remove
       WES calls with little WGS intersection. --compare-mask adds a second mask file (e.g. the
       older / smaller one) side by side.
  4. score_ranges_per_sample.png  -- per-sample Jaccard and overlap, by split, for each best model
     score_ranges_tuning.png      -- train and eval scores across ALL tuned combos per model
  5. scatter_<exp>_<metric>.png
       WES vs WGS for every scored bin (on- and off-target panels, density-shaded), with the
       chosen thresholds; quadrant counts annotated.

  sbatch -c 8 --mem 64G -t 6:00:00 --wrap "python3 viz_pooled_onoff_results.py --run-dir <run> \\
        --search-script <scripts>/compare_wes_wgs_overlap_grad_pooled_on_off_tuning.py \\
        --mask-regions <mask.csv> --n-jobs 8"
"""

import argparse
import importlib.util
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

M = None          # the run's own module (set in main / worker init)
SPLITS = ("train", "eval", "test")
CATS = ("wes_pos", "wgs_pos", "both", "wes_only", "wgs_only")
CAT_LABEL = {"wes_pos": "WES+", "wgs_pos": "WGS+", "both": "both", "wes_only": "WES-only", "wgs_only": "WGS-only"}


def load_module(path):
    spec = importlib.util.spec_from_file_location("pooled_run", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(path).parent))
    spec.loader.exec_module(mod)
    return mod


def _init_worker(path):
    global M
    M = load_module(path)


# --------------------------------------------------------------------------
# Per-sample preprocessing (identical to the run, but recurrent bins are KEPT and flagged)
# --------------------------------------------------------------------------

def read_mask(path):
    m = pd.read_csv(path, sep="," if str(path).endswith(".csv") else "\t")
    cols = {c.lower(): c for c in m.columns}
    iv = pd.DataFrame({"chrom": M.norm_chrom(m[cols["chrom"]]),
                       "start": m[cols["start"]].astype(np.int64), "end": m[cols["end"]].astype(np.int64)})
    return iv.sort_values(["chrom", "start"]).reset_index(drop=True)


def masked_flag(df, iv):
    if iv is None:
        return np.zeros(len(df), dtype=bool)
    # merge mask intervals (bins_in_intervals expects merged, sorted, non-overlapping)
    merged = []
    for ch, g in iv.groupby("chrom", sort=False):
        cs = ce = None
        for s, e in zip(g["start"].to_numpy(), g["end"].to_numpy()):
            if cs is None:
                cs, ce = s, e
            elif s <= ce:
                ce = max(ce, e)
            else:
                merged.append((ch, cs, ce)); cs, ce = s, e
        merged.append((ch, cs, ce))
    miv = pd.DataFrame(merged, columns=["chrom", "start", "end"])
    return M.bins_in_intervals(df["chrom"].to_numpy(), df["start"].to_numpy(), df["end"].to_numpy(), miv, 1)


def prep_sample(job):
    try:
        c = job["cols"]
        # the run's own mask logic (apply_mask_regions_to_df) flags the bins for the RUN's mask, so the
        # scored bins here are exactly the run's; any extra comparison mask is flagged by interval overlap
        run_mask = M.load_mask_regions(job["mask_paths"][0])
        on, on_ref = M.load_region(job["ontarget_input"], c["on_wes"], c["on_wgs"], job["bin_size"], run_mask,
                                   c["on_qc"], c["on_mean"])
        off, off_ref = M.load_region(job["offtarget_input"], c["off_wes"], c["off_wgs"], job["bin_size"], run_mask,
                                     c["off_qc"], c["off_mean"])
        # off-target bins are KEPT and flagged by cluster membership (the run drops non-cluster bins);
        # on-target bins are never cluster-filtered, so they are all "in cluster"
        # in_cluster = bin overlaps one of the sample's cluster intervals (both regions; used for the
        # off-target cluster filter and for --keep-masked-in-clusters)
        if job["cluster_file"]:
            iv = M.load_cluster_intervals(job["cluster_file"], job["bin_size"], None, ("-1", "noise"), None)
            ov = lambda d: M.bins_in_intervals(d["chrom"].to_numpy(), d["start"].to_numpy(), d["end"].to_numpy(), iv, 1)
            on_clu, off_clu = ov(on), ov(off)
        else:
            on_clu, off_clu = np.zeros(len(on), dtype=bool), np.ones(len(off), dtype=bool)
        extra = [read_mask(p) for p in job["mask_paths"][1:]]
        parts = []
        for region, d, clu in ((0, on, on_clu), (1, off, off_clu)):
            vals = d[["wes_raw", "wgs_raw", "wes_cn", "wgs_cn"]].to_numpy(float)
            ok = np.isfinite(vals).all(axis=1)
            flags = [d["masked"].to_numpy(bool)[ok]] + [masked_flag(d, m)[ok] for m in extra]
            parts.append((region, vals[ok], flags, clu[ok]))
        return dict(sample=job["sample"], split=job["split"], error=None,
                    region=np.concatenate([np.full(len(v), r, np.int8) for r, v, _, _ in parts]),
                    values=np.concatenate([v for _, v, _, _ in parts]),
                    masked=[np.concatenate([f[k] for _, _, f, _ in parts]) for k in range(1 + len(extra))],
                    in_cluster=np.concatenate([c for _, _, _, c in parts]),
                    means=dict(on_wes=on_ref["wes_mean"], on_wgs=on_ref["wgs_mean"],
                               off_wes=off_ref["wes_mean"], off_wgs=off_ref["wgs_mean"]))
    except Exception as e:
        return dict(sample=job["sample"], split=job["split"], error=repr(e))


# --------------------------------------------------------------------------
# Calls at chosen thresholds
# --------------------------------------------------------------------------

def calls(values, region, model, scale_floor_gate):
    """WES-positive / WGS-positive booleans for one model (same rule as the run)."""
    exp = model["experiment"]
    raw = exp == "exp1_raw"
    wes = values[:, 0] if raw else values[:, 2]
    wgs = values[:, 1] if raw else values[:, 3]
    if "ontarget" in model["wes_thr"]:
        thr = np.where(region == 0, model["wes_thr"]["ontarget"], model["wes_thr"]["offtarget"])
    else:
        thr = np.full(len(region), model["wes_thr"]["shared"])
    W, G = wes > thr, wgs > model["wgs_thr"]
    floor = model["floor"]
    if floor is not None and (not raw or scale_floor_gate):
        W &= values[:, 2] > floor
        G &= values[:, 3] > floor
    return W, G


def category_counts(W, G):
    return dict(wes_pos=int(W.sum()), wgs_pos=int(G.sum()), both=int((W & G).sum()),
                wes_only=int((W & ~G).sum()), wgs_only=int((~W & G).sum()))


def per_sample_scores(W, G, sample_idx, n_samples):
    A = np.bincount(sample_idx[W], minlength=n_samples).astype(float)
    B = np.bincount(sample_idx[G], minlength=n_samples).astype(float)
    I = np.bincount(sample_idx[W & G], minlength=n_samples).astype(float)
    U = A + B - I
    mn = np.minimum(A, B)
    with np.errstate(divide="ignore", invalid="ignore"):
        jac = np.where(U > 0, I / np.where(U > 0, U, 1), np.nan)
        ovl = np.where(U > 0, np.where(mn > 0, I / np.where(mn > 0, mn, 1), 0.0), np.nan)
    return jac, ovl


# ==========================================================================
# Requested figures (raw-depth models): per-sample scatters, CN-like boxplots by
# focal-amplification category, and the chosen raw thresholds expressed as CN-like
# ==========================================================================

CATEGORY_ORDER = ["ecDNA", "BFB (no ecDNA)", "FAN (no ecDNA/BFB)", "other focal amp", "no focal amp"]
CATEGORY_COLOR = {"ecDNA": "#d62728", "BFB (no ecDNA)": "#ff7f0e", "FAN (no ecDNA/BFB)": "#9467bd",
                  "other focal amp": "#2ca02c", "no focal amp": "#7f7f7f"}
NON_FOCAL_CLASSES = {"noampinvalid", "noamp", "invalid", "virus", ""}


def sample_categories(classification_tsv, samples):
    """One category per sample, by priority ecDNA > BFB > FAN > other focal amp > none."""
    import re
    df = pd.read_csv(classification_tsv, sep="\t", dtype=str, keep_default_na=False)
    pos = lambda c: (df[c].str.strip() == "Positive") if c in df.columns else pd.Series(False, index=df.index)
    cls = df["amplicon_decomposition_class"].map(lambda x: re.sub(r"[\s_/-]+", "", str(x).strip().lower())) \
        if "amplicon_decomposition_class" in df.columns else pd.Series("", index=df.index)
    df = df.assign(_ec=pos("ecDNA+"), _bfb=pos("BFB+"), _fan=pos("FAN+"), _focal=~cls.isin(NON_FOCAL_CLASSES))
    g = df.groupby(df["sample_name"].str.strip())[["_ec", "_bfb", "_fan", "_focal"]].any()
    cat = {}
    for s in samples:
        if s not in g.index:
            cat[s] = "no focal amp"
            continue
        r = g.loc[s]
        cat[s] = ("ecDNA" if r["_ec"] else "BFB (no ecDNA)" if r["_bfb"] else "FAN (no ecDNA/BFB)" if r["_fan"]
                  else "other focal amp" if (r["_focal"] or r["_ec"] or r["_bfb"] or r["_fan"]) else "no focal amp")
    n_found = sum(s in g.index for s in samples)
    return cat, n_found


def requested_outputs(out, models, values, region, sample_idx, kept, meta, scale, a):
    req = out / "requested"
    req.mkdir(exist_ok=True)
    raw_models = [m for m in models if m["experiment"] == "exp1_raw"]
    if not models:
        print("[requested] no models in summary.tsv -- skipping")
        return
    samples = meta["sample"].tolist()
    cat, n_found = sample_categories(a.classification_tsv, samples)
    meta = meta.assign(category=[cat[s] for s in samples])
    print(f"\n[requested] categories: {n_found}/{len(samples)} samples found in {a.classification_tsv} "
          f"(others = 'no focal amp');  " +
          ", ".join(f"{c}: {int((meta.category == c).sum())}" for c in CATEGORY_ORDER))
    if n_found < 0.5 * len(samples):
        print("[requested] WARNING: fewer than half the samples are in the classification table -- "
              "check that its sample_name values are the WES sample names")
    meta[["sample", "split", "category"]].to_csv(req / "sample_categories.tsv", sep="\t", index=False)
    ns = len(samples)
    cnt = lambda m: np.bincount(sample_idx[m], minlength=ns)

    # ---------------- 3. chosen raw thresholds as CN-like (raw models only) ----------------
    trows = []
    for m in raw_models:
        axes = [("WES on-target", m["wes_thr"]["ontarget"], 0, 0, 2, "on_wes_mean"),
                ("WES off-target", m["wes_thr"]["offtarget"], 1, 0, 2, "off_wes_mean"),
                ("WGS (on-target bins)", m["wgs_thr"], 0, 1, 3, "on_wgs_mean"),
                ("WGS (off-target bins)", m["wgs_thr"], 1, 1, 3, "off_wgs_mean")]
        for name, thr, reg, raw_col, cn_col, mean_col in axes:
            means = meta[mean_col].to_numpy(float)
            means = means[np.isfinite(means) & (means > 0)]
            single = scale * thr / np.median(means)                       # cohort-level CN-like value
            per_sample = scale * thr / means                              # same threshold, each sample's scale
            sel = kept & (region == reg) & (np.abs(values[:, raw_col] - thr) <= 0.05 * thr)
            at = values[sel, cn_col]                                      # observed CN-like of bins AT the threshold
            q = lambda x, p: float(np.quantile(x, p)) if len(x) else np.nan
            row = dict(model=m["name"], axis=name, raw_threshold=thr,
                       cnlike_single=single,
                       samples_min=q(per_sample, 0), samples_p05=q(per_sample, .05), samples_q25=q(per_sample, .25),
                       samples_median=q(per_sample, .5), samples_q75=q(per_sample, .75),
                       samples_p95=q(per_sample, .95), samples_max=q(per_sample, 1),
                       at_threshold_n_bins=int(sel.sum()), at_threshold_median=q(at, .5),
                       at_threshold_q25=q(at, .25), at_threshold_q75=q(at, .75),
                       cn_floor=m["floor"],
                       effective_cnlike=max(single, m["floor"]) if m["floor"] is not None else single)
            row["samples_median_over_single"] = row["samples_median"] / single
            row["at_threshold_median_over_single"] = row["at_threshold_median"] / single
            trows.append(row)
    thr_tab = pd.DataFrame(trows, columns=["model", "axis", "cnlike_single"] if not trows else None)
    if trows:
        thr_tab.to_csv(req / "raw_thresholds_as_cnlike.tsv", sep="\t", index=False)
    print(f"\n== Chosen RAW thresholds expressed as CN-like (CN-like = {scale:g} x depth / region mean)")
    print("   single = threshold / cohort-median region mean; samples = same threshold on each sample's own scale;")
    print("   at-threshold = observed CN-like of bins whose raw depth is within +/-5% of the threshold")
    for _, r in thr_tab.iterrows():
        print(f"   {r['model']:22} {r['axis']:22} raw {r['raw_threshold']:8.4g} -> CN-like {r['cnlike_single']:.2f}"
              f"  | samples median {r['samples_median']:.2f} (IQR {r['samples_q25']:.2f}-{r['samples_q75']:.2f}, "
              f"5-95% {r['samples_p05']:.2f}-{r['samples_p95']:.2f})"
              f"  | at-threshold median {r['at_threshold_median']:.2f} (IQR {r['at_threshold_q25']:.2f}-"
              f"{r['at_threshold_q75']:.2f}, n={r['at_threshold_n_bins']})"
              + (f"  | CN-like gate {r['cn_floor']:g} -> effective {r['effective_cnlike']:.2f}" if r['cn_floor'] is not None else ""))

    fig, axs = plt.subplots(max(len(raw_models), 1), 4, figsize=(17, 4.2 * max(len(raw_models), 1)), squeeze=False)
    for i, m in enumerate(raw_models):
        for j, (_, r) in enumerate(thr_tab[thr_tab.model == m["name"]].iterrows()):
            ax = axs[i, j]
            ax.barh([2], [r["samples_q75"] - r["samples_q25"]], left=r["samples_q25"], height=0.5, color="#a6c8e8")
            ax.plot([r["samples_p05"], r["samples_p95"]], [2, 2], color="#4c72b0")
            ax.plot([r["samples_median"]], [2], "o", color="#4c72b0")
            ax.barh([1], [r["at_threshold_q75"] - r["at_threshold_q25"]], left=r["at_threshold_q25"], height=0.5,
                    color="#f2c08a")
            ax.plot([r["at_threshold_median"]], [1], "o", color="#dd8452")
            ax.axvline(r["cnlike_single"], color="k", lw=2, label=f"single CN-like {r['cnlike_single']:.2f}")
            if r["cn_floor"] is not None:
                ax.axvline(r["cn_floor"], color="#c44e52", ls="--", lw=1.4, label=f"CN-like gate {r['cn_floor']:g}")
            ax.set_yticks([1, 2]); ax.set_yticklabels(["bins at threshold\n(IQR, median)", "per-sample\n(5-95%, IQR, median)"])
            ax.set_xlabel(f"CN-like (scale {scale:g})")
            ax.set_title(f"{m['name']}\n{r['axis']}: raw {r['raw_threshold']:.4g}", fontsize=9)
            ax.legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout(); fig.savefig(req / "raw_thresholds_as_cnlike.png", dpi=150); plt.close(fig)

    # ---------------- 1 + 2. per split: scatters and boxplots (ALL models) ----------------
    for m in models:
        raw = m["experiment"] == "exp1_raw"
        gated = m["floor"] is not None and (not raw or a.raw_cn_floor)
        thr_off = m["wes_thr"].get("offtarget", m["wes_thr"].get("shared"))
        thr_on = m["wes_thr"].get("ontarget", m["wes_thr"].get("shared"))
        shared = "shared" in m["wes_thr"]
        unit = "raw" if raw else "CN-like"
        W, G = calls(values, region, m, a.raw_cn_floor)
        W, G = W & kept, G & kept
        on, off = region == 0, region == 1
        per = pd.DataFrame(dict(
            sample=samples, split=meta["split"], category=meta["category"],
            wes_pos_off=cnt(W & off), both_off=cnt(W & G & off),
            wgs_pos=cnt(G), both_total=cnt(W & G),
            wes_pos_on=cnt(W & on), both_on=cnt(W & G & on)))
        tag = f"{m['experiment']}_{m['metric']}"
        per.to_csv(req / f"per_sample_counts_{tag}.tsv", sep="\t", index=False)
        sh = " (shared)" if shared else ""
        panels = [("wes_pos_off", "both_off", "off-target WES bins > threshold", "off-target intersection bins",
                   f"WES off-target thr{sh} {thr_off:.4g} ({unit})"),
                  ("wgs_pos", "both_total", "WGS bins > threshold (on + off target)", "intersection bins (total)",
                   f"WGS thr {m['wgs_thr']:.4g} ({unit})"),
                  ("wes_pos_on", "both_on", "on-target WES bins > threshold", "on-target intersection bins",
                   f"WES on-target thr{sh} {thr_on:.4g} ({unit})")]
        for sp in SPLITS:
            d = per[per.split == sp]
            if d.empty:
                continue
            # --- scatter figure ---
            fig, axs = plt.subplots(1, 3, figsize=(19, 6))
            for ax, (xc, yc, xl, yl, ttl) in zip(axs, panels):
                for c in reversed(CATEGORY_ORDER):          # draw 'no focal amp' first, ecDNA on top
                    dd = d[d.category == c]
                    ax.scatter(dd[xc], dd[yc], s=28, alpha=.85, color=CATEGORY_COLOR[c], edgecolors="white",
                               linewidths=.4, label=f"{c} (n={len(dd)})")
                hi = max(d[xc].max(), d[yc].max(), 1)
                ax.plot([0, hi], [0, hi], color="k", lw=.8, ls=":", label="y = x (all confirmed)")
                pad = 0.03 * hi                                    # linear count axes, starting at 0
                ax.set_xlim(-pad, hi + pad); ax.set_ylim(-pad, hi + pad)
                ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(ttl, fontsize=10)
            h, l = axs[0].get_legend_handles_labels()
            order = [l.index(x) for x in l if not x.startswith("y = x")][::-1] + [l.index(x) for x in l if x.startswith("y = x")]
            axs[0].legend([h[i] for i in order], [l[i] for i in order], fontsize=8, frameon=False, loc="upper left")
            fig.suptitle(f"{m['name']} -- {sp} ({len(d)} samples): each dot = one sample "
                         f"(bins after clustering + recurrent mask"
                         + (f"; raw calls also need CN-like > {m['floor']:g})" if raw and gated else ")"))
            fig.tight_layout(); fig.savefig(req / f"sample_scatter_{tag}_{sp}.png", dpi=150); plt.close(fig)

            # --- boxplot figure: WES CN-like of bins above the WES threshold, by category ---
            in_split = (meta["split"].to_numpy() == sp)[sample_idx]
            cat_of_bin = meta["category"].to_numpy()[sample_idx]
            fig, axs = plt.subplots(1, 2, figsize=(15, 6), sharey=False)
            brows = []
            for ax, (reg, rname, thr_val) in zip(axs, ((1, "off-target", thr_off), (0, "on-target", thr_on))):
                sel = W & (region == reg) & in_split
                data, labels = [], []
                for c in CATEGORY_ORDER:
                    v = values[sel & (cat_of_bin == c), 2]
                    nsamp = len(np.unique(sample_idx[sel & (cat_of_bin == c)]))
                    data.append(v)
                    labels.append(f"{c}\n{len(v)} bins / {nsamp} samples")
                    brows.append(dict(split=sp, region=rname, category=c, n_bins=len(v), n_samples=nsamp,
                                      median_cnlike=float(np.median(v)) if len(v) else np.nan,
                                      q25=float(np.quantile(v, .25)) if len(v) else np.nan,
                                      q75=float(np.quantile(v, .75)) if len(v) else np.nan))
                bp = ax.boxplot([x if len(x) else [np.nan] for x in data], showfliers=False, patch_artist=True)
                for patch, c in zip(bp["boxes"], CATEGORY_ORDER):
                    patch.set_facecolor(CATEGORY_COLOR[c]); patch.set_alpha(.55)
                rng = np.random.default_rng(0)
                for k, x in enumerate(data):
                    if len(x):
                        xs = x if len(x) <= 3000 else rng.choice(x, 3000, replace=False)
                        ax.scatter(rng.normal(k + 1, .07, len(xs)), xs, s=3, alpha=.25, color=CATEGORY_COLOR[CATEGORY_ORDER[k]])
                if raw:
                    single = thr_tab[(thr_tab.model == m["name"]) &
                                     (thr_tab.axis == ("WES off-target" if reg == 1 else "WES on-target"))]["cnlike_single"].iloc[0]
                    ax.axhline(single, color="k", lw=1.2, ls="-", label=f"threshold as CN-like (single) {single:.2f}")
                else:
                    ax.axhline(thr_val, color="k", lw=1.2, ls="-", label=f"CN-like threshold {thr_val:.3g}")
                if gated and (raw or abs(thr_val - m["floor"]) > 1e-9):
                    ax.axhline(m["floor"], color="#c44e52", lw=1.2, ls="--", label=f"CN-like floor {m['floor']:g}")
                ax.set_yscale("log")
                ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels, fontsize=8)
                ax.set_ylabel(f"WES CN-like (scale {scale:g})")
                ax.set_title(f"{rname} WES bins > threshold ({unit} {thr_val:.4g}{', shared' if shared else ''})",
                             fontsize=10)
                ax.legend(fontsize=8, frameon=False)
            fig.suptitle(f"{m['name']} -- {sp}: CN-like of WES-amplified bins by sample category")
            fig.tight_layout(); fig.savefig(req / f"cnlike_boxplot_{tag}_{sp}.png", dpi=150); plt.close(fig)
            pd.DataFrame(brows).to_csv(req / f"cnlike_boxplot_{tag}_{sp}.tsv", sep="\t", index=False)
    print(f"[requested] figures and tables written to {req}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    global M
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--search-script", required=True,
                   help="The compare_wes_wgs_overlap_grad_pooled_on_off_tuning.py used for the run")
    p.add_argument("--mask-regions", required=True, help="Recurrent-bin mask the run used")
    p.add_argument("--compare-mask", default=None, help="Optional second mask to compare (e.g. older/smaller)")
    p.add_argument("--manifest-prefix", default=None, help="Default: <run-dir>/manifests/pooled")
    p.add_argument("--target-bin-size", type=int, default=25000)
    p.add_argument("--ontarget-wes-column", default="raw_wes_depth")
    p.add_argument("--ontarget-wgs-column", default="wgs_tumor_depth")
    p.add_argument("--offtarget-wes-column", default="predicted_loess_upscale_depth")
    p.add_argument("--offtarget-wgs-column", default="wgs_tumor_depth")
    p.add_argument("--offtarget-qc-column", default="mask_rejected")
    p.add_argument("--ontarget-qc-column", default="none")
    p.add_argument("--offtarget-cn-mean", default="positive")
    p.add_argument("--ontarget-cn-mean", default="all")
    p.add_argument("--no-raw-cn-floor", dest="raw_cn_floor", action="store_false")
    p.add_argument("--keep-masked-in-clusters", choices=["off", "offtarget", "both"], default="off",
                   help="Must match the training run: masked bins inside the sample's clusters are kept")
    p.add_argument("--scatter-splits", default="train,eval,test")
    p.add_argument("--max-scatter-points", type=int, default=5000000,
                   help="Hexbin panels with more bins than this are randomly subsampled for drawing "
                        "(counts in the corner box always use all bins)")
    p.add_argument("--classification-tsv",
                   default="/metropolis/projects/sspandau/depmap/recurrent_bins_inputs/amplicon_classification_profiles.merged.tsv",
                   help="AmpliconClassifier profiles with sample_name = WES sample names (merged table from the "
                        "recurrent-bin pipeline) -- used for the ecDNA / BFB / FAN / other / none categories")
    p.add_argument("--requested-only", action="store_true",
                   help="Only make the requested figures (figures/requested/), skip the general diagnostics")
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--outdir", default=None)
    a = p.parse_args()

    run = Path(a.run_dir)
    out = Path(a.outdir or run / "figures")
    out.mkdir(parents=True, exist_ok=True)
    M = load_module(a.search_script)
    scale = M.CN_SCALE
    prefix = Path(a.manifest_prefix or run / "manifests" / "pooled")
    mask_paths = [a.mask_regions] + ([a.compare_mask] if a.compare_mask else [])
    mask_names = ["run mask"] + (["compare mask"] if a.compare_mask else [])
    print(f"CN-like scale from the run's cn_like_from_track: {scale:g} (CN-like = {scale:g} x depth / region mean)")

    # ---- models from summary.tsv ----
    summ = pd.read_csv(run / "summary.tsv", sep="\t")
    summ = summ[summ["status"] == "ok"]
    models = []
    for _, r in summ.iterrows():
        wes = {k: r[f"wes_threshold_{k}"] for k in ("ontarget", "offtarget", "shared")
               if f"wes_threshold_{k}" in r and pd.notna(r[f"wes_threshold_{k}"])}
        floor = r.get("chosen_min_cn_like", np.nan)
        models.append(dict(experiment=r["experiment"], metric=r["metric"], wes_thr=wes,
                           wgs_thr=float(r["wgs_threshold"]),
                           floor=float(floor) if pd.notna(floor) else None,
                           name=f"{r['experiment']} / {r['metric']}"))

    # ---- preprocess all splits (cached) ----
    cache = out / "viz_bins_v3.npz"            # v3: all off-target bins + cluster overlap for both regions
    meta_path = out / "viz_samples.tsv"
    if cache.exists() and meta_path.exists() and np.load(cache)["n_masks"] == len(mask_paths):
        z = np.load(cache)
        region, values, sample_idx = z["region"], z["values"], z["sample_idx"]
        masked = [z[f"masked_{k}"] for k in range(len(mask_paths))]
        in_cluster = z["in_cluster"]
        meta = pd.read_csv(meta_path, sep="\t")
        print(f"loaded cached bins from {cache}")
    else:
        cols = dict(on_wes=a.ontarget_wes_column, on_wgs=a.ontarget_wgs_column,
                    off_wes=a.offtarget_wes_column, off_wgs=a.offtarget_wgs_column,
                    on_qc=a.ontarget_qc_column, off_qc=a.offtarget_qc_column,
                    on_mean=a.ontarget_cn_mean, off_mean=a.offtarget_cn_mean)
        jobs = []
        for sp in SPLITS:
            f = Path(f"{prefix}.{sp}.tsv")
            if not f.exists():
                continue
            for r in pd.read_csv(f, sep="\t").to_dict("records"):
                cf = r.get("offtarget_clusters")
                jobs.append(dict(sample=str(r["sample"]), split=sp, ontarget_input=r["ontarget_input"],
                                 offtarget_input=r["offtarget_input"],
                                 cluster_file=cf if isinstance(cf, str) and cf.strip() else None,
                                 cols=cols, bin_size=a.target_bin_size, mask_paths=mask_paths))
        print(f"preprocessing {len(jobs)} samples ({a.n_jobs} workers) ...", flush=True)
        if a.n_jobs > 1:
            with ProcessPoolExecutor(a.n_jobs, initializer=_init_worker, initargs=(a.search_script,)) as ex:
                res = list(ex.map(prep_sample, jobs, chunksize=1))
        else:
            res = [prep_sample(j) for j in jobs]
        bad = [r for r in res if r["error"]]
        for r in bad[:5]:
            print(f"  skipped {r['sample']}: {r['error']}")
        res = [r for r in res if not r["error"]]
        region = np.concatenate([r["region"] for r in res])
        values = np.concatenate([r["values"] for r in res]).astype(np.float32)
        sample_idx = np.concatenate([np.full(len(r["region"]), i, np.int32) for i, r in enumerate(res)])
        masked = [np.concatenate([r["masked"][k] for r in res]) for k in range(len(mask_paths))]
        in_cluster = np.concatenate([r["in_cluster"] for r in res])
        meta = pd.DataFrame([dict(sample=r["sample"], split=r["split"], **{f"{k}_mean": v for k, v in r["means"].items()})
                             for r in res])
        meta.to_csv(meta_path, sep="\t", index=False)
        np.savez(cache, region=region, values=values, sample_idx=sample_idx, n_masks=len(mask_paths),
                 in_cluster=in_cluster,
                 **{f"masked_{k}": masked[k] for k in range(len(mask_paths))})
    # --keep-masked-in-clusters (same rule as the training run): masked bins inside the sample's
    # clusters are NOT removed -- applied to every mask so the run mask and compare mask stay comparable
    if a.keep_masked_in_clusters != "off":
        rescue_zone = in_cluster & ((region == 1) if a.keep_masked_in_clusters == "offtarget" else True)
        n_resc = [int((mk & rescue_zone).sum()) for mk in masked]
        masked = [mk & ~rescue_zone for mk in masked]
        print(f"--keep-masked-in-clusters {a.keep_masked_in_clusters}: masked bins kept because they are in a "
              f"cluster: {n_resc[0]} (run mask)" + (f", {n_resc[1]} (compare mask)" if len(n_resc) > 1 else ""))
    # full set (all off-target bins, mask flagged) for the "before filtering" panel ...
    ALL = dict(region=region, values=values, sample_idx=sample_idx, masked=masked[0], in_cluster=in_cluster,
               split_of=meta["split"].to_numpy()[sample_idx])
    n_off_all = int((region == 1).sum()); n_off_clu = int(((region == 1) & in_cluster).sum())  # noqa
    n_off_mask = int(((region == 1) & masked[0]).sum())
    print(f"off-target bins: {n_off_all} total, {n_off_clu} in clusters, {n_off_mask} in the run's recurrent mask")
    # ... and the cluster-filtered set the run used, which every other analysis below works on
    scored_region = (region == 0) | in_cluster          # on-target: all bins; off-target: in clusters
    region, values, sample_idx = region[scored_region], values[scored_region], sample_idx[scored_region]
    masked = [mk[scored_region] for mk in masked]
    split_of = meta["split"].to_numpy()[sample_idx]
    kept = ~masked[0]                                  # bins the run actually scored
    print(f"{len(meta)} samples, {len(region)} bins after clustering; "
          f"{int(masked[0].sum())} in the run's recurrent mask"
          + (f", {int(masked[1].sum())} in the compare mask" if len(masked) > 1 else ""))

    requested_outputs(out, models, values, region, sample_idx, kept, meta, scale, a)
    if a.requested_only:
        print(f"\nSaved to {out / 'requested'}")
        return

    # =================== 1. raw thresholds -> CN-like equivalents ===================
    rows = []
    raw_models = [m for m in models if m["experiment"] == "exp1_raw"]
    for m in raw_models:
        axes = [("WES on-target", m["wes_thr"]["ontarget"], "on_wes_mean"),
                ("WES off-target", m["wes_thr"]["offtarget"], "off_wes_mean"),
                ("WGS (on-target bins)", m["wgs_thr"], "on_wgs_mean"),
                ("WGS (off-target bins)", m["wgs_thr"], "off_wgs_mean")]
        for axis, thr, col in axes:
            eq = scale * thr / meta[col].to_numpy(float)
            eq = eq[np.isfinite(eq)]
            rows.append(dict(model=m["name"], axis=axis, raw_threshold=thr,
                             cnlike_median=np.median(eq), cnlike_q25=np.quantile(eq, .25),
                             cnlike_q75=np.quantile(eq, .75), cnlike_p05=np.quantile(eq, .05),
                             cnlike_p95=np.quantile(eq, .95), cn_floor=m["floor"],
                             frac_samples_floor_binding=float((eq < (m["floor"] or -np.inf)).mean()),
                             _eq=eq))
    if rows:
        t = pd.DataFrame(rows)
        t.drop(columns="_eq").to_csv(out / "raw_threshold_cnlike_equivalents.tsv", sep="\t", index=False)
        fig, axs = plt.subplots(len(raw_models), 4, figsize=(16, 3.6 * len(raw_models)), squeeze=False)
        for i, m in enumerate(raw_models):
            for j, (_, r) in enumerate(t[t["model"] == m["name"]].iterrows()):
                ax = axs[i, j]
                ax.hist(r["_eq"], bins=40, color="#4c72b0", alpha=.8)
                ax.axvline(r["cnlike_median"], color="k", lw=1.2, label=f"median {r['cnlike_median']:.2f}")
                if m["floor"] is not None:
                    ax.axvline(m["floor"], color="#c44e52", ls="--", lw=1.2, label=f"CN-like gate {m['floor']:g}")
                ax.set_title(f"{m['name']}\n{r['axis']}: raw {r['raw_threshold']:.4g}", fontsize=9)
                ax.set_xlabel(f"CN-like equivalent (scale {scale:g})"); ax.set_ylabel("samples")
                ax.legend(fontsize=7, frameon=False)
        fig.tight_layout(); fig.savefig(out / "raw_threshold_cnlike_equivalents.png", dpi=150); plt.close(fig)
        print("\n== raw thresholds as CN-like (per-sample distribution)")
        print(t.drop(columns="_eq")[["model", "axis", "raw_threshold", "cnlike_median", "cnlike_q25", "cnlike_q75",
                                      "cn_floor", "frac_samples_floor_binding"]].round(3).to_string(index=False))

    # =================== 2. amplified bin counts per split ===================
    n_samp = len(meta)
    crow = []
    for m in models:
        W, G = calls(values, region, m, a.raw_cn_floor)
        for sp in SPLITS:
            sel = kept & (split_of == sp)
            n_s = int((meta["split"] == sp).sum())
            if not n_s:
                continue
            c = category_counts(W[sel], G[sel])
            crow.append(dict(model=m["name"], split=sp, n_samples=n_s, n_bins=int(sel.sum()),
                             **c, **{f"{k}_per_sample": v / n_s for k, v in c.items()},
                             wes_only_frac_of_wes=c["wes_only"] / c["wes_pos"] if c["wes_pos"] else np.nan))
    counts = pd.DataFrame(crow)
    counts.to_csv(out / "amplified_bin_counts.tsv", sep="\t", index=False)
    n_ok = n_chk = 0
    for _, r in summ.iterrows():
        for sp in SPLITS:
            col = f"{sp}_n_wes_pos"
            v = counts[(counts.model == f"{r['experiment']} / {r['metric']}") & (counts.split == sp)]
            if col in r and len(v):
                n_chk += 1
                n_ok += int(v.iloc[0]["wes_pos"] == r[col] and v.iloc[0]["wgs_pos"] == r[f"{sp}_n_wgs_pos"]
                            and v.iloc[0]["both"] == r[f"{sp}_n_both"])
    print(f"consistency with the run's own counts: {n_ok}/{n_chk} model x split combinations match exactly"
          + ("" if n_ok == n_chk else
             ("  (EXPECTED: --keep-masked-in-clusters differs from how this run was trained, so a different set "
              "of bins is scored at the run's thresholds)" if a.keep_masked_in_clusters != "off" else
              "  <-- check that --mask-regions / column / CN-like options match the run")))
    fig, axs = plt.subplots(2, 3, figsize=(16, 8), sharey=False)
    for ax, m in zip(axs.flat, models):
        d = counts[counts["model"] == m["name"]]
        x = np.arange(len(CATS)); wbar = 0.8 / max(len(d), 1)
        for k, (_, r) in enumerate(d.iterrows()):
            ax.bar(x + k * wbar, [r[f"{c}_per_sample"] for c in CATS], wbar, label=f"{r['split']} (n={r['n_samples']})")
        ax.set_xticks(x + wbar * (len(d) - 1) / 2); ax.set_xticklabels([CAT_LABEL[c] for c in CATS])
        ax.set_title(m["name"], fontsize=10); ax.set_ylabel("bins per sample")
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Amplified bins at the chosen thresholds (scored bins: clustered, recurrent-masked)")
    fig.tight_layout(); fig.savefig(out / "amplified_bin_counts.png", dpi=150); plt.close(fig)

    # =================== 3. mask effect ===================
    mrow = []
    for m in models:
        W, G = calls(values, region, m, a.raw_cn_floor)
        for k, name in enumerate(mask_names):
            for label, sel in ((f"removed by {name}", masked[k]), (f"kept by {name}", ~masked[k])):
                c = category_counts(W[sel], G[sel])
                mrow.append(dict(model=m["name"], mask=name, bins=label, n_bins=int(sel.sum()), **c,
                                 wes_only_frac_of_wes=c["wes_only"] / c["wes_pos"] if c["wes_pos"] else np.nan,
                                 wgs_support_of_wes=c["both"] / c["wes_pos"] if c["wes_pos"] else np.nan))
    meff = pd.DataFrame(mrow)
    meff.to_csv(out / "mask_effect.tsv", sep="\t", index=False)
    fig, axs = plt.subplots(2, 3, figsize=(17, 8))
    for ax, m in zip(axs.flat, models):
        d = meff[meff["model"] == m["name"]]
        labels = d["bins"].tolist()
        both, wonly = d["both"].to_numpy(), d["wes_only"].to_numpy()
        ax.barh(labels, both, color="#55a868", label="WES+ with WGS support")
        ax.barh(labels, wonly, left=both, color="#c44e52", label="WES-only")
        for y, (b, w) in enumerate(zip(both, wonly)):
            if b + w:
                ax.text(b + w, y, f"  {w / (b + w):.0%} WES-only", va="center", fontsize=8)
        ax.set_title(m["name"], fontsize=10); ax.set_xlabel("WES-positive bins (all splits)")
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.suptitle("Did the recurrent mask remove WES calls without WGS support?")
    fig.tight_layout(); fig.savefig(out / "mask_effect.png", dpi=150); plt.close(fig)
    print("\n== WES-positive bins inside vs outside the mask (all splits)")
    print(meff[["model", "bins", "wes_pos", "both", "wes_only", "wes_only_frac_of_wes"]].round(3).to_string(index=False))

    # =================== 4a. per-sample score ranges ===================
    fig, axs = plt.subplots(2, 6, figsize=(24, 8), sharey=True)
    srow = []
    for j, m in enumerate(models):
        W, G = calls(values, region, m, a.raw_cn_floor)
        jac, ovl = per_sample_scores(W & kept, G & kept, sample_idx, n_samp)
        for i, (sc, nm) in enumerate(((jac, "Jaccard"), (ovl, "overlap"))):
            data = [sc[(meta["split"] == sp).to_numpy() & np.isfinite(sc)] for sp in SPLITS]
            ax = axs[i, j]
            ax.boxplot(data, showfliers=False); ax.set_xticks(range(1, len(SPLITS) + 1), SPLITS)
            for k, dd in enumerate(data):
                ax.scatter(np.random.default_rng(k).normal(k + 1, 0.06, len(dd)), dd, s=6, alpha=.4)
                srow.append(dict(model=m["name"], score=nm, split=SPLITS[k], n=len(dd),
                                 mean=np.mean(dd) if len(dd) else np.nan, median=np.median(dd) if len(dd) else np.nan,
                                 q25=np.quantile(dd, .25) if len(dd) else np.nan, q75=np.quantile(dd, .75) if len(dd) else np.nan))
            ax.set_title(f"{m['name']}\nper-sample {nm}", fontsize=9)
            if j == 0:
                ax.set_ylabel(nm)
    fig.tight_layout(); fig.savefig(out / "score_ranges_per_sample.png", dpi=150); plt.close(fig)
    pd.DataFrame(srow).to_csv(out / "score_ranges_per_sample.tsv", sep="\t", index=False)

    # =================== 4b. score ranges across tuned combos ===================
    tun = []
    for m in models:
        f = run / f"{m['experiment']}_{m['metric']}_tuning.tsv"
        if f.exists():
            d = pd.read_csv(f, sep="\t")
            d = d[d["status"] == "ok"]
            tun.append((m, d))
    if tun:
        fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
        for ax, metric in zip(axs, ("jaccard", "overlap")):
            sel = [(m, d) for m, d in tun if m["metric"] == metric]
            pos = np.arange(len(sel))
            ax.boxplot([d["train_score"] for _, d in sel], positions=pos - 0.18, widths=0.3, showfliers=False,
                       patch_artist=True, boxprops=dict(facecolor="#a6c8e8"))
            if all("eval_score" in d for _, d in sel):
                ax.boxplot([d["eval_score"] for _, d in sel], positions=pos + 0.18, widths=0.3, showfliers=False,
                           patch_artist=True, boxprops=dict(facecolor="#f2c08a"))
            for k, (m, d) in enumerate(sel):
                chosen = summ[(summ.experiment == m["experiment"]) & (summ.metric == metric)]
                if len(chosen):
                    ax.scatter([k + 0.18], [chosen[f"eval_macro_{metric}"].iloc[0]
                                            if f"eval_macro_{metric}" in chosen else np.nan],
                               marker="*", s=150, color="#c44e52", zorder=5)
            ax.set_xticks(pos); ax.set_xticklabels([m["experiment"] for m, _ in sel])
            ax.set_title(f"{metric}: train (blue) vs eval (orange) across {len(sel[0][1]) if sel else 0} tuned combos; "
                         f"* = chosen", fontsize=9)
            ax.set_ylabel(f"macro {metric}")
        fig.tight_layout(); fig.savefig(out / "score_ranges_tuning.png", dpi=150); plt.close(fig)

    # =================== 5. scatter plots ===================
    sc_splits = [s_.strip() for s_ in a.scatter_splits.split(",")]
    rng = np.random.default_rng(0)

    def panel(fig, ax, m, vals, reg_arr, sel, W, G, wes_thr, title):
        raw = m["experiment"] == "exp1_raw"
        wes = vals[:, 0] if raw else vals[:, 2]
        wgs = vals[:, 1] if raw else vals[:, 3]
        idx = np.flatnonzero(sel)
        if len(idx) > a.max_scatter_points:
            idx = rng.choice(idx, a.max_scatter_points, replace=False)
        x, y = np.clip(wgs[idx], 1e-2, None), np.clip(wes[idx], 1e-2, None)     # WGS on x, WES on y
        hb = ax.hexbin(x, y, xscale="log", yscale="log", gridsize=90, mincnt=1, norm=LogNorm(), cmap="viridis")
        fig.colorbar(hb, ax=ax, label="bins" + (" (subsampled)" if sel.sum() > a.max_scatter_points else ""))
        ax.axhline(wes_thr, color="#c44e52", lw=1.4, label=f"WES threshold {wes_thr:.4g}")
        ax.axvline(m["wgs_thr"], color="#dd8452", lw=1.4, label=f"WGS threshold {m['wgs_thr']:.4g}")
        c = category_counts(W[sel], G[sel])
        ax.text(0.02, 0.98, f"both {c['both']}\nWES-only {c['wes_only']}\nWGS-only {c['wgs_only']}\n"
                            f"bins {int(sel.sum())}", transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(facecolor="white", alpha=.8, edgecolor="none"))
        gate = m["floor"] is not None and (not raw or a.raw_cn_floor)
        ax.set_title(title + (f"; calls also need CN-like > {m['floor']:g}" if raw and gate else ""), fontsize=9)
        ax.set_xlabel("WGS " + ("raw depth" if raw else "CN-like")); ax.set_ylabel("WES " + ("raw depth" if raw else "CN-like"))
        ax.legend(fontsize=7, frameon=False, loc="lower right")

    in_sp = np.isin(split_of, sc_splits)
    in_sp_all = np.isin(ALL["split_of"], sc_splits)
    sp_txt = ", ".join(sc_splits)
    for m in models:
        thr_on = m["wes_thr"].get("ontarget", m["wes_thr"].get("shared"))
        thr_off = m["wes_thr"].get("offtarget", m["wes_thr"].get("shared"))
        W, G = calls(values, region, m, a.raw_cn_floor)
        Wa, Ga = calls(ALL["values"], ALL["region"], m, a.raw_cn_floor)

        # (a) as before: the scored bins only
        fig, axs = plt.subplots(1, 2, figsize=(14, 6.2))
        panel(fig, axs[0], m, values, region, kept & (region == 0) & in_sp, W, G, thr_on, f"on-target ({sp_txt})")
        panel(fig, axs[1], m, values, region, kept & (region == 1) & in_sp, W, G, thr_off,
              f"clustered off-target ({sp_txt})")
        fig.suptitle(f"{m['name']}: scored bins after clustering + recurrent-bin mask")
        fig.tight_layout(); fig.savefig(out / f"scatter_{m['experiment']}_{m['metric']}.png", dpi=150); plt.close(fig)

        # (b) same two panels + ALL off-target bins: no cluster filter, recurrent bins NOT removed
        fig, axs = plt.subplots(1, 3, figsize=(21, 6.2))
        panel(fig, axs[0], m, values, region, kept & (region == 0) & in_sp, W, G, thr_on,
              f"on-target, scored ({sp_txt})")
        off_desc = ("in clusters (mask not applied inside clusters)" if a.keep_masked_in_clusters != "off"
                    else "in clusters, not masked")
        panel(fig, axs[1], m, values, region, kept & (region == 1) & in_sp, W, G, thr_off,
              f"off-target, scored: {off_desc} ({sp_txt})")
        panel(fig, axs[2], m, ALL["values"], ALL["region"], (ALL["region"] == 1) & in_sp_all, Wa, Ga, thr_off,
              f"off-target, ALL bins: no cluster filter, recurrent bins kept ({sp_txt})")
        fig.suptitle(f"{m['name']}: scored bins vs all off-target bins before cluster + mask filtering")
        fig.tight_layout()
        fig.savefig(out / f"scatter_unfiltered_{m['experiment']}_{m['metric']}.png", dpi=150); plt.close(fig)

    print(f"\nSaved figures and tables to {out}")


if __name__ == "__main__":
    main()
