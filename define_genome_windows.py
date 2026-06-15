import os
import sys
import argparse
import pandas as pd
import pybedtools

''' 
Script that defines the off target genome windows using an access file
and explicitly filters out centromeres and buffered targets.
'''

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--targets", metavar="FILE", required=True,
                   help="Input target bed file")
    p.add_argument("--access", metavar="FILE", required=True,
                   help="Accessible/mappable regions bed file (e.g., access-5kb.GRCh38_no_alt.bed)")
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
    
    # Reference file containing chromosome sizes
    CHROM_SIZES = args.chr_sizes
    
    # Input BED files
    TARGET_BED_PATH = args.targets
    ACCESS_BED_PATH = args.access
    CENTROMERE_BED_PATH = args.centromeres
    
    # Settings
    WINDOW_SIZE = args.window_size       
    FLANK_BUFFER = args.flank_buffer      
    
    # Output file
    OUTPUT_BED = args.output_bed

    # Verify genome size file exists
    if not os.path.exists(CHROM_SIZES):
        print(f"[-] Error: Chrom_sizes file '{CHROM_SIZES}' not found.")
        sys.exit(1)

    print("[+] Initializing Step 1: Defining Off-Target Windows via Access File & Centromere Filtering...")


    # 1. GENERATE INITIAL WINDOWS WITHIN ACCESSIBLE REGIONS
    print(f"[+] Creating {WINDOW_SIZE/1000:.1f}kb windows inside accessible regions...")
    windows = pybedtools.BedTool.makewindows(b=ACCESS_BED_PATH, w=WINDOW_SIZE)
    print(f"    -> Generated initial accessible windows: {windows.count()}")


    # 2. PREPARE THE EXCLUSION MASK (TARGETS + CENTROMERES)
    print("[+] Loading and processing exclusion masks...")
    
    # Load targets and add the flanking buffer
    targets = pybedtools.BedTool(TARGET_BED_PATH)
    buffered_targets = targets.slop(g=CHROM_SIZES, b=FLANK_BUFFER)
    print(f"    -> Buffered target regions loaded: {buffered_targets.count()}")

    # Load centromeres explicitly
    centromeres = pybedtools.BedTool(CENTROMERE_BED_PATH)
    print(f"    -> Centromere regions loaded: {centromeres.count()}")
    
    # Combine buffered targets and centromeres into one exclusion track
    combined_exclusions = buffered_targets.cat(centromeres, postmerge=False).sort()
    
    # Merge overlapping exclusions to speed up the subtraction step
    combined_exclusions = combined_exclusions.merge()
    print(f"    -> Total unique merged exclusion zones (Targets + Centromeres): {combined_exclusions.count()}")


    # 3. SUBTRACT EXCLUSIONS FROM ACCESSIBLE WINDOWS
    print("[+] Subtracting exclusion zones from accessible windows...")
    filtered_windows = windows.subtract(combined_exclusions)
    

    # 4. POST-FILTER: CLEANUP SHORT FRAGMENTS
    MIN_REMAINING_LEN = int(WINDOW_SIZE * 0.50) 
    print(f"[+] Filtering out fragmented windows shorter than {MIN_REMAINING_LEN} bp...")

    df = filtered_windows.to_dataframe(names=['chrom', 'start', 'end'])
    df['length'] = df['end'] - df['start']
    
    final_df = df[df['length'] >= MIN_REMAINING_LEN].copy()
    
    # Generate window names
    final_df['name'] = final_df['chrom'] + ":" + final_df['start'].astype(str) + "-" + final_df['end'].astype(str)
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