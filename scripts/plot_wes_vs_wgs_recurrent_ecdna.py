#!/usr/bin/env python3
"""
plot_wes_vs_wgs_recurrent_ecdna.py

Standalone WES-vs-WGS scatter, built from:
  1. the <OUTDIR>/inputs/ folder written by Step 1 of
     run_recurrent_bin_opt_DepMap_CCLE_slurm.sh
        inputs/wes_flat/<sample>/<sample>_off_target_copy_ratios.tsv
        inputs/classification_bed_merged/<sample>_amplicon<N>_<feature>_<i>_intervals.bed
        inputs/wes_to_aa_sample_map.tsv   (only used with --wgs-long-csv)
  2. a recurrent-bin list such as best_combo_recurrent_orange_overlap.csv
        chrom,start,end,n_samples_high,overlaps_orange

No re-run of the grid search is needed.

One point per (sample, bin):
    x = that sample's predicted loess WES depth (--column) for the bin
    y = that sample's WGS depth for the bin
Both are averaged into the same fixed-width bins as the recurrent list
(bin width inferred from end - start), using the same floor(start / width)
binning and mask_rejected filtering as the grid-search script, so bins line
up exactly with the recurrent list. The same genomic bin therefore appears
once per sample, at that sample's own coordinates.

Marker shape:
    grey dot         other bins
    purple triangle  bin location is in the recurrent list
    orange square    bin overlaps an AC ecDNA feature called in THIS
                     point's own source sample
    red diamond      both

WGS depth source -- give exactly one:
    --wgs-depth-col COL     a column in each sample's copy-ratio TSV
    --wgs-long-csv PATH     a long-format file (e.g. cv_predictions.csv:
                            meta_sample, meta_chrom, meta_start,
                            y_wgs_tumor_depth). Read in chunks and averaged
                            into the same bins. Its sample names are matched
                            to the WES sample id first, then to the AA sample
                            name from inputs/wes_to_aa_sample_map.tsv.

Outputs (--outdir):
    wes_vs_wgs_scatter.png                 all samples pooled
    wes_vs_wgs_scatter_<sample>.png        one per --highlight-samples entry
    wes_vs_wgs_points.tsv.gz               only if --points-out is given
"""

import argparse
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


ECDNA_BED_RE = re.compile(r"^(?P<sample>.+)_amplicon(?P<num>\d+)_ecDNA_", re.IGNORECASE)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def normalize_chrom_series(s):
    s = s.astype(str).str.strip()
    return s.where(s.str.startswith("chr"), "chr" + s)


def sep_for(path):
    p = str(path).lower()
    return "\t" if p.endswith((".tsv", ".txt", ".tsv.gz", ".txt.gz", ".bed")) else ","


def discover_wes_tsvs(wes_flat, pattern):
    out = {}
    for d in sorted(Path(wes_flat).iterdir()):
        if not d.is_dir():
            continue
        hits = sorted(d.glob(pattern))
        if hits:
            out[d.name] = hits[0]
    return out


def index_ecdna_beds(bed_dir):
    index = {}
    for p in sorted(Path(bed_dir).glob("*.bed")):
        m = ECDNA_BED_RE.match(p.name)
        if m:
            index.setdefault(m.group("sample"), []).append(str(p))
    return index


def load_recurrent(path):
    df = pd.read_csv(path)
    missing = {"chrom", "start", "end"} - set(df.columns)
    if missing:
        sys.exit(f"Error: {path} is missing columns {missing}")
    df["chrom"] = normalize_chrom_series(df["chrom"])
    df["start"] = df["start"].astype(np.int64)
    df["end"] = df["end"].astype(np.int64)
    widths = (df["end"] - df["start"]).value_counts()
    return df, int(widths.index[0]), widths


# --------------------------------------------------------------------------
# Per-sample worker (top-level so it can run in a process pool)
# --------------------------------------------------------------------------

def process_sample(job):
    (sample, tsv, ecdna_beds, value_col, mask_col, raw_depth_col, wgs_col,
     bin_size, rec_chroms, rec_starts) = job

    header = pd.read_csv(tsv, sep=sep_for(tsv), nrows=0).columns
    for col in ("chrom", "start", value_col):
        if col not in header:
            return sample, None, f"column '{col}' not in {tsv}"
    wanted = {"chrom", "start", value_col, mask_col, raw_depth_col}
    if wgs_col:
        wanted.add(wgs_col)
    df = pd.read_csv(tsv, sep=sep_for(tsv), usecols=[c for c in header if c in wanted])

    if mask_col.lower() != "none" and mask_col in df.columns:
        df = df[df[mask_col].astype(float).fillna(0) == 0]

    df = df.dropna(subset=[value_col, "start"])
    out = pd.DataFrame({
        "chrom": normalize_chrom_series(df["chrom"]).to_numpy(),
        "start": (df["start"].astype(np.int64) // bin_size * bin_size).to_numpy(),
        "value_raw": df[value_col].astype(float).to_numpy(),
        "raw_depth": df[raw_depth_col].astype(float).to_numpy() if raw_depth_col in df.columns else np.nan,
        "wgs_depth": (df[wgs_col].astype(float).to_numpy()
                      if wgs_col and wgs_col in df.columns else np.nan),
    })
    wgs_missing = bool(wgs_col) and wgs_col not in df.columns
    binned = out.groupby(["chrom", "start"], sort=True)[["value_raw", "raw_depth", "wgs_depth"]].mean().reset_index()

    # ecDNA flag: every fixed-width bin touched by an ecDNA interval of THIS sample
    ecdna_keys = set()
    for bed in ecdna_beds:
        try:
            iv = pd.read_csv(bed, sep="\t", header=None, comment="#", usecols=[0, 1, 2])
        except (pd.errors.EmptyDataError, ValueError):
            continue
        iv.columns = ["chrom", "start", "end"]
        iv["chrom"] = normalize_chrom_series(iv["chrom"])
        for c, s, e in iv.itertuples(index=False):
            first, last = int(s) // bin_size, (int(e) - 1) // bin_size
            ecdna_keys.update((c, b * bin_size) for b in range(first, last + 1))
    keys = pd.MultiIndex.from_arrays([binned["chrom"], binned["start"]])
    binned["is_ecdna"] = keys.isin(list(ecdna_keys)) if ecdna_keys else False

    rec_index = pd.MultiIndex.from_arrays([rec_chroms, rec_starts])
    binned["is_recurrent"] = keys.isin(rec_index)

    for c in ("value_raw", "raw_depth", "wgs_depth"):
        binned[c] = binned[c].astype(np.float32)
    binned["sample"] = sample
    note = f"--wgs-depth-col '{wgs_col}' not in {tsv}" if wgs_missing else None
    return sample, binned, note


# --------------------------------------------------------------------------
# Optional: WGS depth from one long-format file, read in chunks
# --------------------------------------------------------------------------

def load_wgs_long(path, sample_col, chrom_col, start_col, value_col, bin_size, wanted_names, chunksize):
    name_to_idx = {n: i for i, n in enumerate(wanted_names)}
    parts, n_rows = [], 0
    reader = pd.read_csv(path, sep=sep_for(path), usecols=[sample_col, chrom_col, start_col, value_col],
                         chunksize=chunksize)
    for chunk in reader:
        n_rows += len(chunk)
        sid = chunk[sample_col].astype(str).str.strip().map(name_to_idx)
        keep = sid.notna().to_numpy()
        if not keep.any():
            continue
        c = chunk.loc[keep]
        part = pd.DataFrame({
            "sid": sid[keep].astype(np.int32).to_numpy(),
            "chrom": normalize_chrom_series(c[chrom_col]).to_numpy(),
            "start": (c[start_col].astype(np.int64) // bin_size * bin_size).to_numpy(),
            "depth": c[value_col].astype(float).to_numpy(),
        }).dropna(subset=["depth"])
        parts.append(part.groupby(["sid", "chrom", "start"])["depth"].agg(["sum", "count"]).reset_index())
        print(f"[wgs-long] read {n_rows:,} rows", flush=True)
    if not parts:
        return {}
    agg = pd.concat(parts, ignore_index=True).groupby(["sid", "chrom", "start"])[["sum", "count"]].sum().reset_index()
    agg["wgs_depth"] = (agg["sum"] / agg["count"]).astype(np.float32)
    return {wanted_names[i]: g[["chrom", "start", "wgs_depth"]].reset_index(drop=True)
            for i, g in agg.groupby("sid")}


# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------

def apply_min_depth(df, min_depth, label):
    if min_depth is None:
        return df
    missing = df["raw_depth"].isna()
    if missing.any():
        print(f"[warn] {label}{int(missing.sum()):,} bin-instances have no raw depth -- kept, not filtered")
    return df[missing | (df["raw_depth"] > min_depth)]


DEFAULT_COLORS = {"other": "#bdbdbd", "recurrent": "#1f5fbf", "ecdna": "#e07b00", "both": "#c0143c"}


def plot_scatter(points, out_path, x_label, y_label, title, min_depth, label="", colors=None,
                 other_size=2.5):
    colors = {**DEFAULT_COLORS, **(colors or {})}
    points = apply_min_depth(points, min_depth, label)
    x = points["value_raw"].to_numpy(dtype=float)
    y = points["wgs_depth"].to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        print(f"[warn] {label}no points with both WGS and WES depth -- skipping {out_path}")
        return
    keep = finite & (x > 0) & (y > 0)
    n_nan, n_nonpos = int((~finite).sum()), int((finite & ~keep).sum())
    if n_nan or n_nonpos:
        print(f"[info] {label}dropped {n_nan:,} points with missing depth and {n_nonpos:,} with depth <= 0")

    is_rec = points["is_recurrent"].to_numpy(dtype=bool) & keep
    is_ec = points["is_ecdna"].to_numpy(dtype=bool) & keep
    groups = [
        ("Other bins", keep & ~is_rec & ~is_ec, "o", other_size, colors["other"], 0.25),
        ("Recurrent bins", is_rec & ~is_ec, "^", 2.5, colors["recurrent"], 0.6),
        ("ecDNA bins (ecDNA in source sample)", is_ec & ~is_rec, "s", 2.5, colors["ecdna"], 0.75),
        ("ecDNA AND recurrent", is_ec & is_rec, "D", 3.0, colors["both"], 0.9),
    ]

    fig, ax = plt.subplots(figsize=(9, 8))
    for name, mask, marker, size, color, alpha in groups:
        n = int(mask.sum())
        print(f"[info] {label}{name}: {n:,} points")
        if n:
            ax.plot(x[mask], y[mask], linestyle="none", marker=marker, markersize=size,
                    markeredgewidth=0, color=color, alpha=alpha, rasterized=True,
                    label=f"{name} (n={n:,})")
    lo, hi = min(x[keep].min(), y[keep].min()), max(x[keep].max(), y[keep].max())
    ax.plot([lo, hi], [lo, hi], color="black", lw=1, ls="--", label="y = x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    leg = ax.legend(loc="upper left", markerscale=4, fontsize=8)
    for h in getattr(leg, "legend_handles", None) or getattr(leg, "legendHandles", []):
        h.set_alpha(1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs-dir", required=True,
                   help="The <OUTDIR>/inputs folder from the SLURM script's Step 1")
    p.add_argument("--recurrent-csv", required=True,
                   help="Recurrent bin list (chrom,start,end,...), e.g. best_combo_recurrent_orange_overlap.csv")
    p.add_argument("--outdir", default=None, help="Default: the recurrent CSV's folder")
    p.add_argument("--wes-pattern", default="*_off_target_copy_ratios.tsv")
    p.add_argument("--column", default="predicted_loess_upscale_depth", help="y-axis value column")
    p.add_argument("--mask-col", default="mask_rejected", help="'none' to disable")
    p.add_argument("--raw-depth-col", default="raw_wes_depth", help="Used only for --min-depth")
    p.add_argument("--min-depth", type=float, default=0.1,
                   help="Keep bins with raw depth above this (same default as the grid script)")
    p.add_argument("--rebin-to", type=int, default=None,
                   help="Bin width in bp. Default: inferred from the recurrent CSV (end - start)")

    p.add_argument("--wgs-depth-col", default=None, help="WGS depth column inside each copy-ratio TSV")
    p.add_argument("--wgs-long-csv", default=None, help="Long-format WGS depth file, e.g. cv_predictions.csv")
    p.add_argument("--wgs-sample-col", default="meta_sample")
    p.add_argument("--wgs-chrom-col", default="meta_chrom")
    p.add_argument("--wgs-start-col", default="meta_start")
    p.add_argument("--wgs-value-col", default="y_wgs_tumor_depth")
    p.add_argument("--wgs-chunksize", type=int, default=5_000_000)

    p.add_argument("--other-color", default=DEFAULT_COLORS["other"], help="Any matplotlib color")
    p.add_argument("--other-size", type=float, default=2.5,
                   help="Marker size (points) of the grey 'other' circles")
    p.add_argument("--recurrent-color", default=DEFAULT_COLORS["recurrent"])
    p.add_argument("--ecdna-color", default=DEFAULT_COLORS["ecdna"])
    p.add_argument("--both-color", default=DEFAULT_COLORS["both"])
    p.add_argument("--highlight-samples", nargs="*", default=[],
                   help="Also make a single-sample plot for each of these WES sample ids")
    p.add_argument("--points-out", action="store_true",
                   help="Also write every plotted (sample, bin) point to wes_vs_wgs_points.tsv.gz")
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    args = p.parse_args()

    if bool(args.wgs_depth_col) == bool(args.wgs_long_csv):
        sys.exit("Error: give exactly one of --wgs-depth-col or --wgs-long-csv")

    inputs = Path(args.inputs_dir)
    wes_flat = inputs / "wes_flat"
    bed_dir = inputs / "classification_bed_merged"
    map_path = inputs / "wes_to_aa_sample_map.tsv"
    for d in (wes_flat, bed_dir):
        if not d.is_dir():
            sys.exit(f"Error: {d} not found -- is --inputs-dir the Step 1 inputs folder?")
    outdir = Path(args.outdir) if args.outdir else Path(args.recurrent_csv).resolve().parent
    outdir.mkdir(parents=True, exist_ok=True)

    rec, inferred, widths = load_recurrent(args.recurrent_csv)
    bin_size = args.rebin_to or inferred
    if len(widths) > 1 or (args.rebin_to and args.rebin_to != inferred):
        print(f"[warn] recurrent bin widths {dict(widths)}; using {bin_size} bp. Bins of other widths "
              f"won't line up with the rebinned data.")
    print(f"Recurrent list: {len(rec):,} bins of {bin_size:,} bp from {args.recurrent_csv}")

    wes = discover_wes_tsvs(wes_flat, args.wes_pattern)
    if not wes:
        sys.exit(f"Error: no {args.wes_pattern} found under {wes_flat}/<sample>/")
    ecdna = index_ecdna_beds(bed_dir)
    n_ec_samples = sum(1 for s in wes if s in ecdna)
    print(f"Found {len(wes)} WES samples; {n_ec_samples} have >=1 ecDNA feature BED in {bed_dir}")

    rec_chroms = rec["chrom"].to_numpy()
    rec_starts = rec["start"].to_numpy()
    jobs = [(s, str(t), ecdna.get(s, []), args.column, args.mask_col, args.raw_depth_col,
             args.wgs_depth_col, bin_size, rec_chroms, rec_starts) for s, t in wes.items()]

    frames, problems = [], []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for i, (sample, df, note) in enumerate(pool.map(process_sample, jobs, chunksize=4), 1):
            if df is None:
                problems.append((sample, note))
                continue
            if note:
                problems.append((sample, note))
            frames.append(df)
            if i % 25 == 0 or i == len(jobs):
                print(f"[load] {i}/{len(jobs)} samples", flush=True)
    for s, note in problems[:10]:
        print(f"[warn] {s}: {note}")
    if len(problems) > 10:
        print(f"[warn] ... and {len(problems) - 10} more")
    if not frames:
        sys.exit("Error: no samples loaded")

    if args.wgs_long_csv:
        aa_of = {}
        if map_path.exists():
            m = pd.read_csv(map_path, sep="\t", dtype=str).fillna("")
            aa_of = dict(zip(m["wes_sample"], m["aa_sample"]))
        candidates = sorted({s for s in wes} | {aa_of[s] for s in wes if aa_of.get(s)})
        wgs = load_wgs_long(args.wgs_long_csv, args.wgs_sample_col, args.wgs_chrom_col, args.wgs_start_col,
                            args.wgs_value_col, bin_size, candidates, args.wgs_chunksize)
        merged, n_missing = [], 0
        for df in frames:
            s = df["sample"].iat[0]
            w = wgs.get(s)
            if w is None and aa_of.get(s):
                w = wgs.get(aa_of[s])
            if w is None:
                n_missing += 1
                continue
            merged.append(df.drop(columns="wgs_depth").merge(w, on=["chrom", "start"], how="left"))
        print(f"[wgs-long] matched WGS depth for {len(merged)}/{len(frames)} samples "
              f"({n_missing} had no rows under either their WES id or AA sample name)")
        frames = merged
        if not frames:
            sys.exit("Error: no WES sample matched a sample in --wgs-long-csv")

    points = pd.concat(frames, ignore_index=True)
    points["sample"] = points["sample"].astype("category")
    points["chrom"] = points["chrom"].astype("category")
    print(f"Total: {len(points):,} (sample, bin) points from {points['sample'].nunique()} samples; "
          f"{int(points['is_recurrent'].sum()):,} at recurrent locations, "
          f"{int(points['is_ecdna'].sum()):,} in own-sample ecDNA")
    n_rec_locs_seen = points.loc[points["is_recurrent"], ["chrom", "start"]].drop_duplicates().shape[0]
    if n_rec_locs_seen < len(rec):
        print(f"[info] {len(rec) - n_rec_locs_seen:,} of {len(rec):,} recurrent bins had no data in any "
              f"sample after masking (check --rebin-to / --mask-col match the grid run)")

    bin_kb = f"{bin_size / 1000:g} kb"
    wgs_name = args.wgs_depth_col or args.wgs_value_col
    x_label = f"Predicted loess WES depth ({args.column}, mean per {bin_kb} bin)"
    y_label = f"WGS depth ({wgs_name}, mean per {bin_kb} bin)"
    rec_name = Path(args.recurrent_csv).name
    colors = {"other": args.other_color, "recurrent": args.recurrent_color,
              "ecdna": args.ecdna_color, "both": args.both_color}

    plot_scatter(points, outdir / "wes_vs_wgs_scatter.png", x_label, y_label,
                 f"Recurrent bins: {rec_name} ({len(rec):,} bins)\n"
                 f"all samples pooled, one point per (sample, bin)",
                 args.min_depth, label="[pooled] ", colors=colors, other_size=args.other_size)

    for s in args.highlight_samples:
        sub = points[points["sample"] == s]
        if sub.empty:
            print(f"[warn] --highlight-samples '{s}' not found; e.g. {list(wes)[:10]}")
            continue
        plot_scatter(sub, outdir / f"wes_vs_wgs_scatter_{s}.png", x_label, y_label,
                     f"Recurrent bins: {rec_name}\nsample: {s}", args.min_depth, label=f"[{s}] ",
                     colors=colors, other_size=args.other_size)

    if args.points_out:
        out = outdir / "wes_vs_wgs_points.tsv.gz"
        apply_min_depth(points, args.min_depth, "[points-out] ").to_csv(out, sep="\t", index=False)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
