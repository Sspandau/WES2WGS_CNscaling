import os
import sys
import argparse
import pandas as pd
import pybedtools

'''
Computes per-off-target-window covariates used by the pooled multi-covariate
bias model (train_pooled_bias_model.py / apply_pooled_bias_model.py):

  - gc_pct              : GC fraction of the window (bedtools nuc)
  - mappability         : mean mappability score in the window (0-1), from a
                           mappability bedgraph/bed track (e.g. Umap/Bismap
                           k100 single-read mappability, or GEM mappability
                           converted to bedgraph)
  - dist_to_target      : distance in bp from window midpoint to the nearest
                           bait/target interval (0 if overlapping, though
                           off-target windows should not overlap targets
                           given Step 1's exclusion mask)

Output is a single TSV keyed on window_id (chrom:start-end), so it can be
merged into the training table or the per-sample apply step by index.

python3 compute_window_covariates.py \
  --windows_bed v5_offtargets.bed \
  --reference GRCh38_no_alt.fa \
  --mappability_bedgraph k100.umap.bedgraph \
  --targets original_targets.bed \
  --output v5_offtargets.covariates.tsv
'''

def compute_gc(windows_bed_path, reference_fasta_path):
    print(f"[+] Computing GC%% via bedtools nuc...")
    windows_bed = pybedtools.BedTool(windows_bed_path)
    nuc_bed = windows_bed.nucleotide_content(fi=reference_fasta_path)
    df_nuc = pd.read_csv(nuc_bed.fn, sep='\t', header=0)
    gc = df_nuc['6_pct_gc'].astype(float).reset_index(drop=True)
    print(f"    -> Parsed GC for {len(gc)} windows.")
    return gc

def compute_mappability(windows_bed_path, mappability_bedgraph_path):
    """
    Mean mappability per window via bedtools map (mean of overlapping
    bedgraph intervals, weighted by overlap length).
    """
    print(f"[+] Computing mean mappability per window from: {mappability_bedgraph_path}")
    windows = pybedtools.BedTool(windows_bed_path).sort()
    mapp_track = pybedtools.BedTool(mappability_bedgraph_path).sort()
    mapped = windows.map(mapp_track, c=4, o="mean", null="NA")
    df = mapped.to_dataframe(names=['chrom', 'start', 'end', 'name', 'mappability'])
    df['mappability'] = pd.to_numeric(df['mappability'], errors='coerce')
    n_missing = df['mappability'].isna().sum()
    if n_missing > 0:
        print(f"    [!] Warning: {n_missing} windows had no mappability track overlap; "
              f"filling with track-wide median.")
        df['mappability'] = df['mappability'].fillna(df['mappability'].median())
    print(f"    -> Parsed mappability for {len(df)} windows.")
    return df[['name', 'mappability']].rename(columns={'name': 'window_id'})

def compute_dist_to_target(windows_bed_path, targets_bed_path):
    """
    Distance (bp) from each window to the nearest original target/bait
    interval, using bedtools closest -d. Uses the ORIGINAL targets file
    (not the flank-buffered version from Step 1) so distances reflect true
    capture-bleed geometry.
    """
    print(f"[+] Computing distance-to-nearest-target via bedtools closest...")
    windows = pybedtools.BedTool(windows_bed_path).sort()
    targets = pybedtools.BedTool(targets_bed_path).sort()
    closest = windows.closest(targets, d=True)

    # bedtools closest -d appends ALL target-file columns before the final
    # distance column, so total width = (window cols) + (target cols) + 1.
    # Rather than hardcode target column count, read headerless and grab
    # window cols [0:4] + the last column (always distance) by position.
    raw_df = pd.read_csv(closest.fn, sep='\t', header=None)
    df = raw_df.iloc[:, [0, 1, 2, 3, -1]].copy()
    df.columns = ['chrom', 'start', 'end', 'name', 'distance']
    df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
    # -1 from bedtools closest means no feature on that chromosome at all
    df.loc[df['distance'] < 0, 'distance'] = df['distance'].max()
    print(f"    -> Parsed distances for {len(df)} windows.")
    return df[['name', 'distance']].rename(columns={'name': 'window_id', 'distance': 'dist_to_target'})

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--windows_bed", metavar="FILE", required=True,
                   help="Off-target windows BED from Step 1 (4th col = window_id)")
    p.add_argument("--reference", metavar="FILE", required=True,
                   help="Reference FASTA (for GC calc)")
    p.add_argument("--mappability_bedgraph", metavar="FILE", required=True,
                   help="Mappability track as bedgraph/BED (e.g. Umap/Bismap or GEM, "
                        "converted with bigWigToBedGraph if starting from bigWig)")
    p.add_argument("--targets", metavar="FILE", required=True,
                   help="Original (un-buffered) target/bait BED file")
    p.add_argument("--output", metavar="FILE", required=True,
                   help="Output covariates TSV, indexed by window_id")
    args = p.parse_args()

    for f in [args.windows_bed, args.reference, args.mappability_bedgraph, args.targets]:
        if not os.path.exists(f):
            print(f"[-] Error: input file not found: {f}")
            sys.exit(1)

    print("[+] Building per-window covariate table...")

    gc_series = compute_gc(args.windows_bed, args.reference)
    windows_df = pybedtools.BedTool(args.windows_bed).to_dataframe(
        names=['chrom', 'start', 'end', 'window_id'])
    windows_df['gc_pct'] = gc_series.values

    mapp_df = compute_mappability(args.windows_bed, args.mappability_bedgraph)
    dist_df = compute_dist_to_target(args.windows_bed, args.targets)

    out_df = windows_df.merge(mapp_df, on='window_id', how='left') \
                        .merge(dist_df, on='window_id', how='left')

    out_df.to_csv(args.output, sep='\t', index=False)

    print("=" * 60)
    print(f"[+] COVARIATES SUCCESSFUL!")
    print(f"    -> Saved {len(out_df)} windows x "
          f"[gc_pct, mappability, dist_to_target] to: {args.output}")
    print("=" * 60)

if __name__ == "__main__":
    main()