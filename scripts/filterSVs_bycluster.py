#!/usr/bin/env python3
"""
filter_svs_by_cluster_and_target.py

Filter a clustered-SV TSV file down to SVs that are either:
  - inside an overall genomic cluster region (from clusters.csv), OR
  - "on target"      : both breakpoints overlap the target BED, OR
  - "half on target" : exactly one breakpoint overlaps the target BED

SVs that are fully off-target AND not inside any genomic cluster are dropped.

Every kept (and, optionally, every) SV is annotated with the columns:
  on_target_A, on_target_B, target_status, in_cluster_A, in_cluster_B,
  in_genomic_cluster, keep

USAGE
-----
1) Build a manifest (sample <tab> clusters_csv <tab> svs_tsv), one row per sample.
   See build_manifest_example() below / the README section printed with --help
   for a template you can adapt to your directory layout.

2) Run:
   python filter_svs_by_cluster_and_target.py \
       --manifest manifest.tsv \
       --target-bed v5.target.bed \
       --outdir filtered_svs/ \
       --combined-output all_samples_filtered_svs.tsv

Each sample's filtered file is written to <outdir>/<sample>_clustered_svs_final.tsv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Interval loading / lookup
# --------------------------------------------------------------------------

def build_interval_index(df: pd.DataFrame, chrom_col: str, start_col: str, end_col: str):
    """
    Build a per-chromosome sorted, merged interval index: {chrom: (starts, ends)}
    Merging overlapping/adjacent intervals keeps lookups correct and fast.
    """
    index = {}
    for chrom, grp in df.groupby(chrom_col):
        grp_sorted = grp.sort_values(start_col)
        starts = grp_sorted[start_col].to_numpy()
        ends = grp_sorted[end_col].to_numpy()

        merged_starts, merged_ends = [], []
        for s, e in zip(starts, ends):
            if merged_starts and s <= merged_ends[-1]:
                merged_ends[-1] = max(merged_ends[-1], e)
            else:
                merged_starts.append(s)
                merged_ends.append(e)

        index[chrom] = (np.array(merged_starts), np.array(merged_ends))
    return index


def annotate_point_overlap(df: pd.DataFrame, chrom_col: str, pos_col: str, interval_index: dict) -> np.ndarray:
    """
    Vectorized (per-chrom) point-in-interval test.
    Returns a boolean numpy array aligned to df's original row order.
    """
    result = np.zeros(len(df), dtype=bool)
    for chrom, grp in df.groupby(chrom_col):
        if chrom not in interval_index:
            continue
        starts, ends = interval_index[chrom]
        if len(starts) == 0:
            continue
        positions = grp[pos_col].to_numpy()
        idx = np.searchsorted(starts, positions, side="right") - 1
        valid = idx >= 0
        hit = np.zeros(len(positions), dtype=bool)
        hit[valid] = (positions[valid] >= starts[idx[valid]]) & (positions[valid] <= ends[idx[valid]])
        result[grp.index.to_numpy()] = hit
    return result


# --------------------------------------------------------------------------
# Loaders for your specific file formats
# --------------------------------------------------------------------------

def load_target_bed(path: str) -> dict:
    bed = pd.read_csv(
        path, sep=r"\s+", header=None,
        names=["chrom", "start", "end", "strand"],
        usecols=[0, 1, 2, 3],
    )
    return build_interval_index(bed, "chrom", "start", "end")


def load_cluster_index(path: str) -> dict:
    clusters = pd.read_csv(path)
    return build_interval_index(clusters, "chrom", "start", "end")


def load_svs(path: str) -> pd.DataFrame:
    # Falls back to whitespace-splitting if the file isn't actually tab-delimited.
    df = pd.read_csv(path, sep="\t")
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=r"\s+")
    return df


# --------------------------------------------------------------------------
# Core annotation / filter logic
# --------------------------------------------------------------------------

def annotate_and_filter(sv_df: pd.DataFrame, target_index: dict, cluster_index: dict) -> pd.DataFrame:
    sv_df = sv_df.reset_index(drop=True)

    on_target_A = annotate_point_overlap(sv_df, "chrom_A", "pos_A_clustered", target_index)
    on_target_B = annotate_point_overlap(sv_df, "chrom_B", "pos_B_clustered", target_index)
    in_cluster_A = annotate_point_overlap(sv_df, "chrom_A", "pos_A_clustered", cluster_index)
    in_cluster_B = annotate_point_overlap(sv_df, "chrom_B", "pos_B_clustered", cluster_index)

    n_on_target = on_target_A.astype(int) + on_target_B.astype(int)
    target_status = np.select(
        [n_on_target == 2, n_on_target == 1, n_on_target == 0],
        ["on_target", "half_target", "off_target"],
    )

    in_genomic_cluster = in_cluster_A | in_cluster_B
    keep = in_genomic_cluster | (target_status != "off_target")

    sv_df = sv_df.assign(
        on_target_A=on_target_A,
        on_target_B=on_target_B,
        target_status=target_status,
        in_cluster_A=in_cluster_A,
        in_cluster_B=in_cluster_B,
        in_genomic_cluster=in_genomic_cluster,
        keep=keep,
    )
    return sv_df


# --------------------------------------------------------------------------
# Manifest-driven batch run
# --------------------------------------------------------------------------

def load_manifest(path: str) -> pd.DataFrame:
    sep = "\t" if str(path).endswith((".tsv", ".txt")) else ","
    manifest = pd.read_csv(path, sep=sep)
    required = {"sample", "clusters_csv", "svs_tsv"}
    missing = required - set(manifest.columns)
    if missing:
        sys.exit(f"Manifest is missing required column(s): {missing}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="TSV/CSV with columns: sample, clusters_csv, svs_tsv")
    parser.add_argument("--target-bed", required=True, help="Path to v5.target.bed")
    parser.add_argument("--outdir", required=True, help="Directory to write per-sample filtered files")
    parser.add_argument("--combined-output", default=None,
                         help="Optional path to also write one concatenated filtered TSV across all samples")
    parser.add_argument("--keep-annotated-unfiltered", action="store_true",
                         help="Also write the FULL annotated (unfiltered) table per sample, for auditing")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.manifest)
    target_index = load_target_bed(args.target_bed)

    combined_frames = []

    for _, row in manifest.iterrows():
        sample = row["sample"]
        clusters_path = row["clusters_csv"]
        svs_path = row["svs_tsv"]

        if not Path(clusters_path).exists():
            print(f"[{sample}] WARNING: clusters file not found: {clusters_path}", file=sys.stderr)
            continue
        if not Path(svs_path).exists():
            print(f"[{sample}] WARNING: SVs file not found: {svs_path}", file=sys.stderr)
            continue

        cluster_index = load_cluster_index(clusters_path)
        sv_df = load_svs(svs_path)

        annotated = annotate_and_filter(sv_df, target_index, cluster_index)
        filtered = annotated[annotated["keep"]].drop(columns=["keep"])

        out_path = outdir / f"{sample}_clustered_svs_final.tsv"
        filtered.to_csv(out_path, sep="\t", index=False)

        if args.keep_annotated_unfiltered:
            full_path = outdir / f"{sample}_clustered_svs_annotated_all.tsv"
            annotated.to_csv(full_path, sep="\t", index=False)

        n_total, n_kept = len(sv_df), len(filtered)
        print(f"[{sample}] kept {n_kept}/{n_total} SVs -> {out_path}")

        if args.combined_output:
            filtered = filtered.copy()
            filtered.insert(0, "sample", sample)
            combined_frames.append(filtered)

    if args.combined_output and combined_frames:
        combined = pd.concat(combined_frames, ignore_index=True)
        combined.to_csv(args.combined_output, sep="\t", index=False)
        print(f"\nCombined output ({len(combined)} SVs across {len(combined_frames)} samples) -> {args.combined_output}")


if __name__ == "__main__":
    main()