import os
import sys
import argparse
import pandas as pd
import numpy as np
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
  - gap_size            : span (bp) of the target-free interval the window
                           sits inside -- distance between the nearest
                           upstream target's end and the nearest downstream
                           target's start
  - gap_fraction        : dist_to_target / gap_size -- the window's relative
                           position within its flanking-target gap. Raw
                           dist_to_target conflates "how deep into a desert"
                           with "how big is the desert"; this decouples them,
                           so e.g. 5kb from a target means something very
                           different in a 6kb gap than in a 200kb one.
  - log_gap_size         : log1p(gap_size) -- desert size on its own, which
                           may correlate with mappability/repeat content
                           independent of where the window sits within it.
                           gap_size/gap_fraction/log_gap_size are NaN where a
                           window has no flanking target on one side (e.g.
                           near a chromosome end) -- there's no bounded gap
                           to normalize against there.

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

def load_chrom_sizes(reference_fasta_path):
    """
    Reads chrom -> length from a samtools .fai index next to the reference
    FASTA (reference_fasta_path + '.fai'). Used as a virtual boundary for
    windows with no flanking target on one side (near a chromosome end) --
    treats the chromosome start/end as if it were another target edge,
    rather than leaving the gap undefined.
    """
    fai_path = reference_fasta_path + '.fai'
    if not os.path.exists(fai_path):
        print(f"    [-] Warning: no .fai index found at {fai_path}; windows missing a "
              f"flanking target on one side will be left as NaN instead of falling back "
              f"to the chromosome boundary. (Index with: samtools faidx {reference_fasta_path})")
        return None

    sizes = {}
    with open(fai_path) as fh:
        for line in fh:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 2:
                continue
            chrom, length = parts[0], parts[1]
            try:
                sizes[chrom] = int(length)
            except ValueError:
                continue
    return sizes


def compute_gap_covariates(windows_bed_path, targets_bed_path, chrom_sizes=None):
    """
    For each off-target window, finds the nearest flanking target on each
    side using two directional bedtools closest passes:
      -id ("ignore downstream")  -> nearest target strictly UPSTREAM
      -iu ("ignore upstream")    -> nearest target strictly DOWNSTREAM
    and derives:
      gap_size     = downstream_target_start - upstream_target_end
      gap_fraction = min(upstream_distance, downstream_distance) / gap_size
      log_gap_size = log1p(gap_size)

    If chrom_sizes is given (see load_chrom_sizes), a window missing a
    flanking target on ONE side (e.g. near a chromosome end) has that side
    substituted with the chromosome boundary itself -- position 0 standing
    in for a missing upstream target's end, or the chromosome length
    standing in for a missing downstream target's start. This treats "off
    the end of the chromosome" as just another kind of boundary, the same
    way a real target edge would be.

    Still NaN if: chrom_sizes isn't provided, a window is missing targets
    on BOTH sides (its whole chromosome has no targets at all -- no bounded
    region of any kind to normalize against), or its chromosome isn't found
    in chrom_sizes (naming mismatch between the FASTA index and the BED
    files, e.g. 'chr1' vs '1' -- reported via a warning, not silently NaN'd).
    """
    print(f"[+] Computing flanking-target gap covariates (gap_size / gap_fraction)...")
    windows = pybedtools.BedTool(windows_bed_path).sort()
    targets = pybedtools.BedTool(targets_bed_path).sort()

    upstream = windows.closest(targets, D="ref", id=True)
    downstream = windows.closest(targets, D="ref", iu=True)

    def parse_closest(bedtool_result):
        raw = pd.read_csv(bedtool_result.fn, sep='\t', header=None)
        a_ncols = 4  # windows_bed: chrom, start, end, window_id
        total_cols = raw.shape[1]
        dist_col = total_cols - 1
        b_start_col = a_ncols + 1   # target's own start column, right after a-block + b_chrom
        b_end_col = a_ncols + 2     # target's own end column

        df = raw.iloc[:, [0, 1, 2, 3, b_start_col, b_end_col, dist_col]].copy()
        df.columns = ['window_chrom', 'window_start', 'window_end', 'window_id',
                       'target_start', 'target_end', 'distance']
        for c in ['window_start', 'window_end', 'target_start', 'target_end', 'distance']:
            df[c] = pd.to_numeric(df[c], errors='coerce')

        # -D ref reports upstream matches as negative distance (lower
        # start/stop in B = upstream, per bedtools' "ref" orientation) --
        # gap math below only needs magnitude, so take abs() here.
        df['distance'] = df['distance'].abs()

        # bedtools closest emits a placeholder b-interval (target_start/end
        # == -1) and distance == -1 when nothing is found in the requested
        # direction on that chromosome -- treat all of those as missing.
        no_match = (df['target_start'] < 0) | (df['target_end'] < 0)
        df.loc[no_match, ['target_start', 'target_end', 'distance']] = np.nan
        return df

    up_df = parse_closest(upstream).rename(
        columns={'target_end': 'upstream_target_end', 'distance': 'upstream_distance'}
    )[['window_id', 'window_chrom', 'window_start', 'window_end',
       'upstream_target_end', 'upstream_distance']]

    down_df = parse_closest(downstream).rename(
        columns={'target_start': 'downstream_target_start', 'distance': 'downstream_distance'}
    )[['window_id', 'downstream_target_start', 'downstream_distance']]

    merged = up_df.merge(down_df, on='window_id', how='outer')

    n_missing_upstream_before = merged['upstream_target_end'].isna().sum()
    n_missing_downstream_before = merged['downstream_target_start'].isna().sum()

    if chrom_sizes is not None:
        # Missing upstream side -> chromosome start (position 0) stands in
        # for a target ending right at the start of the chromosome.
        missing_up = merged['upstream_target_end'].isna()
        merged.loc[missing_up, 'upstream_target_end'] = 0
        merged.loc[missing_up, 'upstream_distance'] = merged.loc[missing_up, 'window_start']

        # Missing downstream side -> chromosome end stands in for a target
        # starting right at the end of the chromosome.
        missing_down = merged['downstream_target_start'].isna()
        chrom_len = merged['window_chrom'].map(chrom_sizes)
        rescued_down = missing_down & chrom_len.notna()
        merged.loc[rescued_down, 'downstream_target_start'] = chrom_len[rescued_down]
        merged.loc[rescued_down, 'downstream_distance'] = (
            chrom_len[rescued_down] - merged.loc[rescued_down, 'window_end']
        )

        unmapped_chrom = missing_down & chrom_len.isna()
        n_unmapped = int(unmapped_chrom.sum())
        if n_unmapped > 0:
            bad_chroms = sorted(merged.loc[unmapped_chrom, 'window_chrom'].unique().tolist())
            print(f"    [-] Warning: {n_unmapped} windows on chromosomes not found in the "
                  f".fai index ({bad_chroms[:10]}{'...' if len(bad_chroms) > 10 else ''}) -- "
                  f"likely a naming mismatch (e.g. 'chr1' vs '1'); these stay NaN for the "
                  f"downstream side.")

        n_rescued_up = int(missing_up.sum())
        n_rescued_down = int(rescued_down.sum())
        print(f"    -> Chromosome-boundary fallback: rescued {n_rescued_up} windows missing "
              f"an upstream target and {n_rescued_down} missing a downstream target "
              f"(of {n_missing_upstream_before} and {n_missing_downstream_before} originally "
              f"missing, respectively).")

    merged['gap_size'] = merged['downstream_target_start'] - merged['upstream_target_end']
    merged.loc[merged['gap_size'] <= 0, 'gap_size'] = np.nan

    nearest_distance = merged[['upstream_distance', 'downstream_distance']].min(axis=1, skipna=True)
    merged['gap_fraction'] = nearest_distance / merged['gap_size']
    merged['log_gap_size'] = np.log1p(merged['gap_size'])

    n_total = len(merged)
    n_missing = merged['gap_fraction'].isna().sum()
    print(f"    -> Computed gap_fraction for {n_total - n_missing}/{n_total} windows "
          f"({n_missing} missing a flanking target on at least one side, e.g. near a "
          f"chromosome end -- left as NaN)")

    return merged[['window_id', 'gap_size', 'gap_fraction', 'log_gap_size']]

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
    chrom_sizes = load_chrom_sizes(args.reference)
    gap_df = compute_gap_covariates(args.windows_bed, args.targets, chrom_sizes=chrom_sizes)

    out_df = windows_df.merge(mapp_df, on='window_id', how='left') \
                        .merge(dist_df, on='window_id', how='left') \
                        .merge(gap_df, on='window_id', how='left')

    out_df.to_csv(args.output, sep='\t', index=False)

    print("=" * 60)
    print(f"[+] COVARIATES SUCCESSFUL!")
    print(f"    -> Saved {len(out_df)} windows x "
          f"[gc_pct, mappability, dist_to_target, gap_size, gap_fraction, log_gap_size] "
          f"to: {args.output}")
    print("=" * 60)

if __name__ == "__main__":
    main()
