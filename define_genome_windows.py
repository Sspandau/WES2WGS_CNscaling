import os
import sys
import argparse
import pandas as pd
import pybedtools

''' 
Script that defines the off target genome windows 
To add: log file instead of print statements
'''

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--targets", metavar="FILE", required=True,
                   help="Inpput target bed file")
    p.add_argument("--blacklist", metavar="FILE", required=True,
                   help="Blasklisted regions")
    p.add_argument("--centromeres", metavar="FILE", required=True,
                   help="Centromere regions bed file")
    p.add_argument("--chr_sizes", metavar="FILE", required=True,
                   help="reference file containing chromosome sizes (Tab-separated: chrom \t size)")
    p.add_argument("--window_size", metavar="INT", type=int,
                   default=5000, help="genomic window size (5kb default)")
    p.add_argument("--flank_buffer", metavar="INT", type=int,
                   default=200, help="flanking buffer (default 200 bp)")
    p.add_argument("--output_bed", metavar="FILE", required=True,
                   help="Output off target windows bed")
    args = p.parse_args()
    
    # Reference file containing chromosome sizes (Tab-separated: chrom \t size)
    CHROM_SIZES = args.chr_sizes
    
    # Input BED files
    TARGET_BED_PATH = args.targets
    BLACKLIST_BED_PATH = args.blacklist
    CENTROMERE_BED_PATH = args.centromeres
    # MAPPABILITY_BED_PATH = "low_mappability_regions.bed" # Optional pre-filtered low-map file
    
    # Settings
    WINDOW_SIZE = args.window_size       # 5 kb fixed-width bins
    FLANK_BUFFER = args.flank_buffer      # Target flanking buffer (+/- 200 bp)
    
    # Output file
    OUTPUT_BED = args.output_bed

    # Verify genome size file exists
    if not os.path.exists(CHROM_SIZES):
        print(f"[-] Error: Chrom_sizes file '{CHROM_SIZES}' not found.")
        print("    Generate it via: cut -f1,2 reference.fa.fai > hg38.chrom.sizes")
        sys.exit(1)

    print("[+] Initializing Step 1: Defining Genome-Wide Off-Target Windows...")


    # 1. GENERATE INITIAL GENOME-WIDE WINDOWS

    print(f"[+] Creating {WINDOW_SIZE/1000:.1f}kb fixed-width windows across genome...")
    # Equivalent to: bedtools makewindows -g chrom.sizes -w 5000
    windows = pybedtools.BedTool.makewindows(g=CHROM_SIZES, w=WINDOW_SIZE)
    print(f"    -> Generated initial windows: {windows.count()}")

    # 2. PREPARE THE EXCLUSION/SUBTRACTION MASK

    print("[+] Loading and processing exclusion masks...")
    
    # Load targets and add a flanking buffer (+/- 200bp)
    # Equivalent to: bedtools slop -i target.bed -g chrom.sizes -b 200
    targets = pybedtools.BedTool(TARGET_BED_PATH)
    buffered_targets = targets.slop(g=CHROM_SIZES, b=FLANK_BUFFER)
    print(f"    -> Buffered target regions loaded: {buffered_targets.count()}")

    # Load alternative exclusion masks
    blacklist = pybedtools.BedTool(BLACKLIST_BED_PATH)
    centromeres = pybedtools.BedTool(CENTROMERE_BED_PATH)
    
    # Combine all exclusions into a single file/stream to speed up downstream subtraction
    # Equivalent to: cat buffered_targets.bed blacklist.bed centromeres.bed | bedtools sort
    combined_exclusions = buffered_targets.cat(blacklist, postmerge=False).cat(centromeres, postmerge=False).sort()
    
    # Optional: merge overlapping exclusions to optimize subtraction step
    combined_exclusions = combined_exclusions.merge()
    print(f"    -> Total unique merged exclusion zones: {combined_exclusions.count()}")


    # 3. SUBTRACT EXCLUSIONS FROM WINDOWS

    print("[+] Subtracting exclusion zones from genome windows...")
    # Equivalent to: bedtools subtract -a windows.bed -b combined_exclusions.bed
    filtered_windows = windows.subtract(combined_exclusions)
    

    # 4. POST-FILTER: CLEANUP SHORT FRAGMENTS

    # Subtraction can cut a 5kb window down into a tiny fragment (e.g. 12 bp remaining).
    # We drop any windows left with less than 50% of original window size to avoid noise.
    MIN_REMAINING_LEN = int(WINDOW_SIZE * 0.50) 
    print(f"[+] Filtering out fragmented windows shorter than {MIN_REMAINING_LEN} bp...")

    # Convert the bed stream to a Pandas Dataframe for fast length filtering
    df = filtered_windows.to_dataframe(names=['chrom', 'start', 'end'])
    df['length'] = df['end'] - df['start']
    
    final_df = df[df['length'] >= MIN_REMAINING_LEN].copy()
    
    # Generate clean window identifiers/names for downstream count matrices
    final_df['name'] = final_df['chrom'] + ":" + final_df['start'].astype(str) + "-" + final_df['end'].astype(str)
    
    # Reorder columns to match standard BED format: chrom, start, end, name
    final_bed_df = final_df[['chrom', 'start', 'end', 'name']]
    

    # 5. SAVE COMPLETED WINDOW TRACK

    final_bed_df.to_csv(OUTPUT_BED, sep='\t', index=False, header=False)
    print("="*60)
    print(f"[+] STEP 1 SUCCESSFUL!")
    print(f"    -> Final actionable off-target windows saved to: {OUTPUT_BED}")
    print(f"    -> Total usable windows: {len(final_bed_df)}")
    print("="*60)

if __name__ == "__main__":
    main()