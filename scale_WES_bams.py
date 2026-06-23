import os
import glob
import random
import pandas as pd
import numpy as np
import pysam

def load_scaling_factors(copy_ratio_tsv):
    df = pd.read_csv(copy_ratio_tsv, sep='\t')
    
    # Secure safe division boundaries
    safe_ratio = np.where(df['depth_scaled_ratio'] <= 1e-4, 1e-4, df['depth_scaled_ratio'])
    
    # Invert to turn (WES / Raw WGS) into the absolute Multiplier: (Raw WGS / WES)
    upscale_factor = 1.0 / safe_ratio
    
    # High-quality windows get upscaled; rejected ones stay at baseline 1.0
    df['final_scale'] = np.where(df['mask_rejected'] == 1, 1.0, upscale_factor)
    
    # Enforce safe structural constraints
    df['final_scale'] = np.clip(df['final_scale'], a_min=1.0, a_max=200.0)
    
    # Inside load_scaling_factors(copy_ratio_tsv):
    windows_by_chrom = {}
    for chrom, group in df.groupby('chrom'):
        # Force a standardized string layout (e.g., stripping 'chr' prefixes if they vary)
        chrom_str = str(chrom).replace('chr', '') 
        sorted_group = group.sort_values('start')
        windows_by_chrom[chrom_str] = list(zip(
            sorted_group['start'].astype(int), 
            sorted_group['end'].astype(int), 
            sorted_group['final_scale'].astype(float)
        ))
    return windows_by_chrom

def get_scaling_factor(chrom, pos, windows_by_chrom, _cache=[None, None, None, 1.0]):
    """
    Fast sequential lookup optimization utilizing a persistent mutable list 
    as an instantaneous single-element lookbehind cache.
    """
    chrom_clean = str(chrom).replace('chr', '')
    
    # 1. Micro-cache hit: If we are in the same chrom and within the same window bounds, return instantly
    if _cache[0] == chrom_clean and _cache[1] <= pos < _cache[2]:
        return _cache[3]
        
    if chrom_clean not in windows_by_chrom:
        return 1.0
        
    # 2. Cache miss: Fall back to sequential loop search
    for start, end, scale in windows_by_chrom[chrom_clean]:
        if start <= pos < end:
            # Update cache element state
            _cache[0], _cache[1], _cache[2], _cache[3] = chrom_clean, start, end, scale
            return scale
        if start > pos:
            break
            
    return 1.0

def write_scaled_pair(read1, read2, scale, outfile, stats):
    """
    Symmetrically duplicates a valid paired-end template match
    while maintaining clean duplicate flags and modified tracking qnames.
    """
    floor_scale = int(np.floor(scale))
    fractional_part = scale - floor_scale
    
    num_copies = floor_scale
    if random.random() < fractional_part:
        num_copies += 1
        
    # Safeguard check (Since input scale is clamped >= 1.0, this should always be >= 1)
    if num_copies < 1:
        num_copies = 1
        
    base_qname = read1.query_name
    
# Inside write_scaled_pair loop:
    for i in range(num_copies):
        # Force flag states off explicitly across objects
        read1.is_duplicate = False
        read2.is_duplicate = False
        
        # Explicit bitwise clearing of duplicate flag bit (0x0400) from the raw field
        read1.flag &= ~1024
        read2.flag &= ~1024
        
        if i == 0:
            read1.query_name = base_qname
            read2.query_name = base_qname
            outfile.write(read1)
            outfile.write(read2)
            stats['kept_reads'] += 2
        else:
            # Append unique suffix to prevent downstream duplicate filtering
            dup_name = f"{base_qname}:upscale_{i}"
            read1.query_name = dup_name
            read2.query_name = dup_name
            outfile.write(read1)
            outfile.write(read2)
            stats['duplicated_reads'] += 2

def process_bam_mate_aware(input_bam_path, output_bam_path, windows_by_chrom):
    """
    Streams a BAM file sequentially and synchronizes pairs via a sliding lookahead cache
    to ensure mate-pair continuity for downstream breakpoint/SV calling.
    """
    random.seed(42) # Reproducible fractional scaling distributions
    infile = pysam.AlignmentFile(input_bam_path, "rb")
    outfile = pysam.AlignmentFile(output_bam_path, "wb", header=infile.header)
    
    # In-memory sliding lookahead hash map cache: { query_name: AlignedSegment }
    mate_cache = {}
    
    stats = {'kept_reads': 0, 'duplicated_reads': 0, 'processed': 0}
    
    print(f"[+] Streaming and pairing alignments from: {os.path.basename(input_bam_path)}")
    
    for read in infile:
        stats['processed'] += 1
        if stats['processed'] % 2000000 == 0:
            print(f"    -> Parsed {stats['processed']:,} alignments. Active Cache Size: {len(mate_cache):,}")
            
        # Instantly bypass unmapped, secondary, or supplementary records (passed through unscaled)
        if read.is_unmapped or read.is_secondary or read.is_supplementary or read.reference_name is None:
            outfile.write(read)
            stats['kept_reads'] += 1
            continue
            
        qname = read.query_name
        
        # Check if mate was pre-cached in our sliding window
        if qname in mate_cache:
            mate = mate_cache.pop(qname)
            
            # Determine the structural scale factor for the pair.
            # If coordinates cross windows, take the mean factor to stay balanced.
            scale1 = get_scaling_factor(read.reference_name, read.reference_start, windows_by_chrom)
            scale2 = get_scaling_factor(mate.reference_name, mate.reference_start, windows_by_chrom)
            chosen_scale = (scale1 + scale2) / 2.0
            
            # Write pairs together symmetrically
            if read.is_read1:
                write_scaled_pair(read, mate, chosen_scale, outfile, stats)
            else:
                write_scaled_pair(mate, read, chosen_scale, outfile, stats)
        else:
            # First mate encountered; hold in lookahead cache
            mate_cache[qname] = read
            
        # Cache Protection Safety Valve: prevent memory leaks from singletons or ultra-wide inserts
        if len(mate_cache) > 200000:
            # Flush oldest 50,000 orphan records unscaled to protect the system RAM pool
            oldest_keys = list(mate_cache.keys())[:50000]
            for k in oldest_keys:
                orphan = mate_cache.pop(k)
                outfile.write(orphan)
                stats['kept_reads'] += 1

    # Empty any remaining reads left in cache at EOF
    if mate_cache:
        print(f"[~] Flushing final {len(mate_cache):,} cache records...")
        for orphan in mate_cache.values():
            outfile.write(orphan)
            stats['kept_reads'] += 1
        mate_cache.clear()
        
    infile.close()
    outfile.close()
    
    print(f"[+] Generation Complete for {os.path.basename(output_bam_path)}:")
    print(f"    -> Input Alignments Analyzed:  {stats['processed']:,}")
    print(f"    -> Baseline Alignments Dumped: {stats['kept_reads']:,}")
    print(f"    -> Synthetic Duplicates Added: {stats['duplicated_reads']:,}")

def main():
    # ----------------------------------------------------
    # PATH CONFIGURATIONS
    # ----------------------------------------------------
    SCALED_RATIO_DIR = "./tumor_scaled_tracks" # Input text metrics from Step 5/6
    ORIGINAL_BAM_DIR = "/path/to/your/tumor_WES_bams/"
    OUTPUT_BAM_DIR = "./tumor_upscaled_bams"
    
    os.makedirs(OUTPUT_BAM_DIR, exist_ok=True)
    
    ratio_files = glob.glob(os.path.join(SCALED_RATIO_DIR, "*_off_target_copy_ratios.tsv"))
    if not ratio_files:
        print(f"[-] Error: No scaling factor files found in target folder: {SCALED_RATIO_DIR}")
        return

    print(f"[+] Found {len(ratio_files)} matching scaling tracks to process.")

    # ----------------------------------------------------
    # PROCESSING RUNTIME
    # ----------------------------------------------------
    for ratio_path in ratio_files:
        sample_name = os.path.basename(ratio_path).replace("_off_target_copy_ratios.tsv", "")
        
        # Locate corresponding source raw alignment track
        input_bam = os.path.join(ORIGINAL_BAM_DIR, f"{sample_name}.bam")
        if not os.path.exists(input_bam):
            potential_bams = glob.glob(os.path.join(ORIGINAL_BAM_DIR, f"{sample_name}*.bam"))
            if potential_bams: 
                input_bam = potential_bams[0]
            else:
                print(f"[!] Warning: Skipping '{sample_name}', source file missing in {ORIGINAL_BAM_DIR}")
                continue

        output_bam = os.path.join(OUTPUT_BAM_DIR, f"{sample_name}.additive_scaled.bam")
        
        print("\n" + "="*80)
        print(f"[+] EXECUTING MATE-AWARE ADDITIVE-ONLY SCALE RUN: {sample_name}")
        print(f"    -> Track Source: {os.path.basename(ratio_path)}")
        print(f"    -> Input BAM:    {input_bam}")
        print(f"    -> Output BAM:   {output_bam}")
        print("="*80)
        
        # 1. Parse genomic scaling coordinates with strict additive constraint adjustments
        windows_by_chrom = load_scaling_factors(ratio_path)
        
        # 2. Process alignment stream mapping
        process_bam_mate_aware(input_bam, output_bam, windows_by_chrom)
        
        # 3. Post-Process: Automatically index new bam track for immediate IGV / tool usage
        print(f"[+] Generating coordinate tracker index (.bai) for {os.path.basename(output_bam)}...")
        pysam.index(output_bam)
        print(f"    -> Finished creating index: {output_bam}.bai")

    print("\n" + "="*80 + "\n[+] ADDITIVE ALIGNMENT PROCESSING RUN SUCCESSFULLY COMPLETED!\n" + "="*80)

if __name__ == "__main__":
    main()                                   