#!/usr/bin/env python3
"""
myc_amplicon_bins.py

Compile every sample with a focal amplicon overlapping a gene (default MYC) and plot the
WES and WGS CN-like values of the bins around it, one dot per sample per bin.

  1. Amplicons: AmpliconClassifier interval BEDs (<sample>_amplicon<N>_<class>_<k>_intervals.bed)
     in --classification-bed-dir. An amplicon "has MYC" if any of its intervals overlaps the gene.
     Sample names are matched to the manifest's WES names (longest prefix before "_amplicon").
  2. Bins: each MYC sample is preprocessed with the training script's own load_region (same
     QC drop, rebinning, CN-like = scale x depth / region mean). All on- and off-target bins in
     the window are kept and flagged: recurrent-masked, and (off-target) inside a cluster.
  3. Figures (written twice: ALL bins, and SCORED bins = the bins the models used, i.e.
     on-target unmasked + off-target in clusters and unmasked):
       myc_cnlike_<set>.png   top: WES CN-like, bottom: WGS CN-like, x = bin position;
                              one dot per sample per bin (circles off-target, triangles on-target),
                              coloured by the class of the sample's MYC amplicon
       myc_ratio_<set>.png    log2(WES CN-like / WGS CN-like) per sample per bin
     and the same pair per amplicon class: myc_cnlike_<set>_<class>.png, myc_ratio_<set>_<class>.png
     (each sample in ONE class: ecDNA > BFB > complex non-cyclic > linear > other, if it has several)
     plus myc_bins.tsv (every bin, every sample) and myc_samples.tsv (sample, amplicons, classes).
"""

import argparse
import glob
import importlib.util
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

M = None
BED_RE = re.compile(r"^(?P<prefix>.+)_amplicon(?P<num>\d+)_(?P<cls>.+?)_(?P<k>\d+)_intervals\.bed$")
CLASS_ORDER = ["ecDNA", "BFB", "complex non-cyclic", "linear", "other"]
CLASS_COLOR = {"ecDNA": "#d62728", "BFB": "#ff7f0e", "complex non-cyclic": "#2ca02c",
               "linear": "#1f77b4", "other": "#7f7f7f"}


def load_module(path):
    spec = importlib.util.spec_from_file_location("pooled_run", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(path).parent))
    spec.loader.exec_module(mod)
    return mod


def _init(path):
    global M
    M = load_module(path)


def norm_chrom(c):
    return re.sub(r"^CHR", "", str(c).strip().upper())


def norm_class(c):
    k = re.sub(r"[\s_/-]+", "", c.lower())
    if "ecdna" in k or k == "cyclic":
        return "ecDNA"
    if "bfb" in k:
        return "BFB"
    if "complex" in k or "noncyclic" in k:
        return "complex non-cyclic"
    if "linear" in k:
        return "linear"
    return "other"


def find_gene_amplicons(bed_dir, samples, chrom, start, end):
    names = sorted(samples, key=len, reverse=True)           # longest match first
    rows = []
    for f in glob.glob(os.path.join(bed_dir, "*_intervals.bed")):
        m = BED_RE.match(os.path.basename(f))
        if not m:
            continue
        sample = next((s for s in names if m["prefix"] == s), None)
        if sample is None:
            continue
        try:
            b = pd.read_csv(f, sep="\t", header=None, comment="#", usecols=[0, 1, 2])
        except Exception:
            continue
        b.columns = ["chrom", "start", "end"]
        hit = (b["chrom"].map(norm_chrom) == chrom) & (b["start"] < end) & (b["end"] > start)
        if hit.any():
            on_chr = b[b["chrom"].map(norm_chrom) == chrom]
            rows.append(dict(sample=sample, amplicon=int(m["num"]), raw_class=m["cls"],
                             cls=norm_class(m["cls"]), amp_start=int(on_chr["start"].min()),
                             amp_end=int(on_chr["end"].max()), bed=f))
    return pd.DataFrame(rows)


def prep(job):
    try:
        c = job["cols"]
        mask_df = M.load_mask_regions(job["mask"]) if job["mask"] else None
        out = []
        for reg, inp, wcol, gcol, qc, mean in ((0, job["on"], c["on_wes"], c["on_wgs"], c["on_qc"], c["on_mean"]),
                                               (1, job["off"], c["off_wes"], c["off_wgs"], c["off_qc"], c["off_mean"])):
            d, _ = M.load_region(inp, wcol, gcol, job["bin_size"], mask_df, qc, mean)
            d = d[(d["chrom"] == job["chrom"]) & (d["end"] > job["w0"]) & (d["start"] < job["w1"])].copy()
            if reg == 1 and job["clusters"]:
                iv = M.load_cluster_intervals(job["clusters"], job["bin_size"], None, ("-1", "noise"), None)
                d["in_cluster"] = M.bins_in_intervals(d["chrom"].to_numpy(), d["start"].to_numpy(),
                                                      d["end"].to_numpy(), iv, 1)
            else:
                d["in_cluster"] = reg == 0 or not job["clusters"]
            d["region"] = "on-target" if reg == 0 else "off-target"
            out.append(d)
        d = pd.concat(out, ignore_index=True)
        d["sample"] = job["sample"]
        return d, None
    except Exception as e:
        return None, f"{job['sample']}: {e!r}"


def main():
    global M
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--search-script", required=True, help="compare_wes_wgs_overlap_grad_pooled_on_off_tuning.py")
    p.add_argument("--manifest", required=True, help="pooled.all.tsv from the run (all splits)")
    p.add_argument("--classification-bed-dir", required=True,
                   help="AmpliconClassifier *_intervals.bed files, named with the WES sample names")
    p.add_argument("--mask-regions", default=None, help="recurrent-bin mask the run used (bins are flagged)")
    p.add_argument("--gene", default="MYC")
    p.add_argument("--gene-locus", default="chr8:127735434-127742951", help="GRCh38 MYC by default")
    p.add_argument("--window-mb", type=float, default=2.5, help="plot gene +/- this many Mb")
    p.add_argument("--target-bin-size", type=int, default=25000)
    p.add_argument("--ontarget-wes-column", default="raw_wes_depth")
    p.add_argument("--ontarget-wgs-column", default="wgs_tumor_depth")
    p.add_argument("--offtarget-wes-column", default="predicted_loess_upscale_depth")
    p.add_argument("--offtarget-wgs-column", default="wgs_tumor_depth")
    p.add_argument("--offtarget-qc-column", default="mask_rejected")
    p.add_argument("--ontarget-qc-column", default="none")
    p.add_argument("--offtarget-cn-mean", default="positive")
    p.add_argument("--ontarget-cn-mean", default="all")
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--outdir", required=True)
    a = p.parse_args()

    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    M = load_module(a.search_script)
    g_chr, g_rng = a.gene_locus.split(":")
    g_chr = norm_chrom(g_chr)
    g0, g1 = (int(x.replace(",", "")) for x in g_rng.split("-"))
    w0, w1 = int(g0 - a.window_mb * 1e6), int(g1 + a.window_mb * 1e6)

    man = pd.read_csv(a.manifest, sep="\t")
    amps = find_gene_amplicons(a.classification_bed_dir, man["sample"].astype(str).tolist(), g_chr, g0, g1)
    if amps.empty:
        sys.exit(f"No amplicon in {a.classification_bed_dir} overlaps {a.gene} ({a.gene_locus}) for the "
                 f"manifest's samples -- check the BED names use the WES sample names")
    rank = {c: i for i, c in enumerate(CLASS_ORDER)}
    samp = (amps.assign(r=amps["cls"].map(rank)).sort_values("r")
            .groupby("sample").agg(cls=("cls", "first"), classes=("raw_class", lambda x: ",".join(sorted(set(x)))),
                                   amplicons=("amplicon", lambda x: ",".join(map(str, sorted(set(x))))),
                                   amp_start=("amp_start", "min"), amp_end=("amp_end", "max")).reset_index())
    split = man.set_index(man["sample"].astype(str)).get("split")
    if split is not None:
        samp["split"] = samp["sample"].map(split)
    samp.to_csv(out / f"{a.gene.lower()}_samples.tsv", sep="\t", index=False)
    print(f"{len(samp)} samples with a {a.gene}-overlapping amplicon: "
          + ", ".join(f"{c} {int((samp.cls == c).sum())}" for c in CLASS_ORDER if (samp.cls == c).any()))

    cols = dict(on_wes=a.ontarget_wes_column, on_wgs=a.ontarget_wgs_column, off_wes=a.offtarget_wes_column,
                off_wgs=a.offtarget_wgs_column, on_qc=a.ontarget_qc_column, off_qc=a.offtarget_qc_column,
                on_mean=a.ontarget_cn_mean, off_mean=a.offtarget_cn_mean)
    mrow = man.set_index(man["sample"].astype(str))
    jobs = []
    for s in samp["sample"]:
        r = mrow.loc[s]
        cf = r.get("offtarget_clusters")
        jobs.append(dict(sample=s, on=r["ontarget_input"], off=r["offtarget_input"],
                         clusters=cf if isinstance(cf, str) and cf.strip() else None, cols=cols,
                         bin_size=a.target_bin_size, mask=a.mask_regions, chrom=g_chr, w0=w0, w1=w1))
    if a.n_jobs > 1:
        with ProcessPoolExecutor(a.n_jobs, initializer=_init, initargs=(a.search_script,)) as ex:
            res = list(ex.map(prep, jobs))
    else:
        res = [prep(j) for j in jobs]
    for _, err in res:
        if err:
            print("  skipped", err)
    bins = pd.concat([d for d, e in res if d is not None], ignore_index=True)
    bins = bins.merge(samp[["sample", "cls"]], on="sample")
    bins["mid_mb"] = (bins["start"] + bins["end"]) / 2e6
    with np.errstate(divide="ignore", invalid="ignore"):
        bins["log2_wes_over_wgs"] = np.log2(bins["wes_cn"] / bins["wgs_cn"])
    bins["scored"] = ~bins["masked"] & bins["in_cluster"]
    bins.drop(columns=[c for c in ("masked_x",) if c in bins]).to_csv(out / f"{a.gene.lower()}_bins.tsv",
                                                                      sep="\t", index=False)
    print(f"{len(bins)} bins in {a.gene} +/- {a.window_mb} Mb across {bins['sample'].nunique()} samples "
          f"({int(bins['scored'].sum())} scored by the models)")

    def decorate(ax, ylabel):
        ax.axvspan(g0 / 1e6, g1 / 1e6, color="gold", alpha=.35, lw=0)
        ax.text((g0 + g1) / 2e6, 1.0, a.gene, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=9, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xlim(w0 / 1e6, w1 / 1e6)

    handles = ([Line2D([], [], ls="", marker="o", color=CLASS_COLOR[c], label=f"{c} (n={int((samp.cls == c).sum())})")
                for c in CLASS_ORDER if (samp.cls == c).any()]
               + [Line2D([], [], ls="", marker="o", color="k", mfc="none", label="off-target bin"),
                  Line2D([], [], ls="", marker="^", color="k", mfc="none", label="on-target bin")])

    def plot_set(subset, name, tag, who):
        """Two figures (CN-like WES/WGS, and the ratio) for one bin set and one group of samples."""
        if subset.empty:
            return
        desc = ("all bins (no cluster filter, recurrent bins kept)" if name == "all"
                else "scored bins only (on-target unmasked; off-target in clusters, unmasked)")
        present = [c for c in CLASS_ORDER if (subset.cls == c).any()]
        hd = ([Line2D([], [], ls="", marker="o", color=CLASS_COLOR[c],
                      label=f"{c} (n={subset.loc[subset.cls == c, 'sample'].nunique()})") for c in present]
              + [Line2D([], [], ls="", marker="o", color="k", mfc="none", label="off-target bin"),
                 Line2D([], [], ls="", marker="^", color="k", mfc="none", label="on-target bin")])
        # ---- CN-like: WES (top) and WGS (bottom) ----
        fig, axs = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
        for ax, col, lab in ((axs[0], "wes_cn", "WES CN-like"), (axs[1], "wgs_cn", "WGS CN-like")):
            for reg, mk in (("off-target", "o"), ("on-target", "^")):
                for c in present:
                    d = subset[(subset.region == reg) & (subset.cls == c)]
                    ax.scatter(d["mid_mb"], d[col].clip(lower=1e-2), s=10 if mk == "o" else 26, marker=mk,
                               alpha=.45, color=CLASS_COLOR[c], edgecolors="none")
            ax.set_yscale("log")
            ax.axhline(1, color="k", lw=.6, ls=":")
            decorate(ax, f"{lab} (one dot per sample)")
        axs[1].set_xlabel(f"chr{g_chr} position (Mb, bin midpoint)")
        axs[0].legend(handles=hd, fontsize=8, frameon=False, loc="upper left", ncol=2)
        fig.suptitle(f"{a.gene} amplicons, {who} (n={subset['sample'].nunique()} samples): {desc}")
        fig.tight_layout(); fig.savefig(out / f"{a.gene.lower()}_cnlike_{name}{tag}.png", dpi=150); plt.close(fig)
        # ---- ratio ----
        fig, ax = plt.subplots(figsize=(14, 5.5))
        for reg, mk in (("off-target", "o"), ("on-target", "^")):
            for c in present:
                d = subset[(subset.region == reg) & (subset.cls == c) & np.isfinite(subset.log2_wes_over_wgs)]
                ax.scatter(d["mid_mb"], d["log2_wes_over_wgs"], s=10 if mk == "o" else 26, marker=mk, alpha=.45,
                           color=CLASS_COLOR[c], edgecolors="none")
        ax.axhline(0, color="k", lw=.8)
        decorate(ax, "log2( WES CN-like / WGS CN-like )")
        ax.set_xlabel(f"chr{g_chr} position (Mb, bin midpoint)")
        ax.legend(handles=hd, fontsize=8, frameon=False, loc="upper left", ncol=2)
        fig.suptitle(f"{a.gene} amplicons, {who} (n={subset['sample'].nunique()} samples): "
                     f"WES / WGS CN-like ratio -- {desc}")
        fig.tight_layout(); fig.savefig(out / f"{a.gene.lower()}_ratio_{name}{tag}.png", dpi=150); plt.close(fig)

    for name, sel in (("all", np.ones(len(bins), bool)), ("scored", bins["scored"].to_numpy())):
        plot_set(bins[sel], name, "", "all classes")                       # combined, as before
        for c in CLASS_ORDER:                                               # one set of figures per class
            sub = bins[sel & (bins["cls"] == c).to_numpy()]
            if not sub.empty:
                plot_set(sub, name, "_" + c.replace(" ", "_").replace("-", "_"), c)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
