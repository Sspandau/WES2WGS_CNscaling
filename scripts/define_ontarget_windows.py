import os
import sys
import argparse
import pandas as pd
import pybedtools

'''
Script that defines on-target genome windows by tiling each target/bait interval
into fixed-size windows, mirroring define_offtarget_windows.py's Step 1 logic:

  - Windows are generated within each target interval at --window_size (default
    5000bp, matching the off-target default), via pybedtools.makewindows -- the
    last window in a target shorter than window_size, or the leftover tail of a
    target not evenly divisible by window_size, is naturally shorter than the
    rest. This is the "smaller if the on-target region is smaller than the
    window size" behavior: no target is padded or dropped just because it's
    below the full window size.
  - Any resulting window fragment below --min_frac * window_size is dropped
    (default 0.5, identical cutoff to the off-target script's
    MIN_REMAINING_LEN = window_size * 0.5), so very short slivers at the end of
    a target don't enter the depth extraction step.
  - Centromere regions are optionally subtracted first (targets should not
    normally sit in centromeric sequence, but this keeps the two pipelines
    consistent and protects against edge cases in the target BED).

Output is a 4-column BED (chrom, start, end, window_id) with the same
window_id convention ("chrom:start-end") as the off-target windows BED, so it
can be used interchangeably by mosdepth --by and by anything downstream that
expects that naming.

python3 define_ontarget_windows.py \
  --targets original_targets.bed \
  --centromeres centromeres.bed \
  --window_size 5000 \
  --min_frac 0.5 \
  --output_bed v5_ontargets.bed
'''


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--targets", metavar="FILE", required=True,
                   help="Target/bait BED file (the original, un-buffered targets -- "
                        "the same file used as --targets in define_offtarget_windows.py "
                        "and as --targets in compute_window_covariates.py)")
    p.add_argument("--centromeres", metavar="FILE", default=None,
                   help="Optional centromere regions BED; overlapping window fragments "
                        "are excluded before the min-length filter")
    p.add_argument("--window_size", metavar="INT", type=int,
                   default=5000, help="genomic window size (5kb default, matching off-target)")
    p.add_argument("--min_frac", metavar="FLOAT", type=float, default=0.5,
                   help="minimum fraction of window_size a leftover fragment must retain "
                        "to be kept (default 0.5, matching off-target's 50%% cutoff)")
    p.add_argument("--output_bed", metavar="FILE", required=True,
                   help="Output on-target windows bed")
    args = p.parse_args()

    TARGET_BED_PATH = args.targets
    CENTROMERE_BED_PATH = args.centromeres
    WINDOW_SIZE = args.window_size
    MIN_FRAC = args.min_frac
    OUTPUT_BED = args.output_bed

    if not os.path.exists(TARGET_BED_PATH):
        print(f"[-] Error: Targets file '{TARGET_BED_PATH}' not found.")
        sys.exit(1)
    if CENTROMERE_BED_PATH is not None and not os.path.exists(CENTROMERE_BED_PATH):
        print(f"[-] Error: Centromeres file '{CENTROMERE_BED_PATH}' not found.")
        sys.exit(1)

    print("[+] Initializing Step 1 (on-target): Tiling target intervals into fixed-size windows...")

    # 1. OPTIONALLY REMOVE CENTROMERE-OVERLAPPING TARGET SEQUENCE FIRST
    targets = pybedtools.BedTool(TARGET_BED_PATH).sort().merge()
    print(f"    -> Merged target intervals: {targets.count()}")

    if CENTROMERE_BED_PATH is not None:
        centromeres = pybedtools.BedTool(CENTROMERE_BED_PATH).sort().merge()
        print(f"    -> Centromere regions loaded: {centromeres.count()}")
        targets = targets.subtract(centromeres)
        print(f"    -> Target intervals after centromere exclusion: {targets.count()}")

    # 2. TILE EACH TARGET INTERVAL AT window_size
    #    makewindows' final window in each input interval is naturally shorter than
    #    window_size whenever the interval isn't an exact multiple of window_size --
    #    this is exactly the "smaller if the target region is smaller than the
    #    window size" behavior, with no extra logic needed here.
    print(f"[+] Creating {WINDOW_SIZE/1000:.1f}kb windows inside target intervals...")
    windows = targets.makewindows(b=targets, w=WINDOW_SIZE)
    print(f"    -> Generated raw on-target windows (pre-length-filter): {windows.count()}")

    # 3. DROP FRAGMENTS SHORTER THAN min_frac * window_size
    MIN_REMAINING_LEN = int(WINDOW_SIZE * MIN_FRAC)
    print(f"[+] Filtering out fragmented windows shorter than {MIN_REMAINING_LEN} bp...")

    df = windows.to_dataframe(names=['chrom', 'start', 'end'])
    df['length'] = df['end'] - df['start']

    final_df = df[df['length'] >= MIN_REMAINING_LEN].copy()
    n_dropped = len(df) - len(final_df)

    final_df['name'] = final_df['chrom'] + ":" + final_df['start'].astype(str) + "-" + final_df['end'].astype(str)
    final_bed_df = final_df[['chrom', 'start', 'end', 'name']].sort_values(['chrom', 'start'])

    final_bed_df.to_csv(OUTPUT_BED, sep='\t', index=False, header=False)

    print("=" * 60)
    print(f"[+] STEP 1 (on-target) SUCCESSFUL!")
    print(f"    -> Dropped {n_dropped} fragments below {MIN_REMAINING_LEN} bp "
          f"({MIN_FRAC:.0%} of window_size)")
    print(f"    -> Final on-target windows saved to: {OUTPUT_BED}")
    print(f"    -> Total usable windows: {len(final_bed_df)}")
    if len(final_bed_df):
        print(f"    -> Window length range: {final_bed_df['end'].sub(final_bed_df['start']).min()}"
              f"-{final_bed_df['end'].sub(final_bed_df['start']).max()} bp")
    print("=" * 60)


if __name__ == "__main__":
    main()