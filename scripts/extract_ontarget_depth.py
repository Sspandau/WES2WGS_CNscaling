import os
import sys
import argparse
import subprocess
import pandas as pd
import numpy as np

'''
Extracts raw depth over on-target windows (from define_ontarget_windows.py) for a
matched tumor WES/WGS BAM pair, mirroring scale_WES_tracks.py's mosdepth calling
convention but WITHOUT the PoN/LOESS scaling step -- on-target depth is driven by
capture efficiency, not the low-signal off-target regime that scaling was built
for, so this intentionally stops at raw (and CN-like, computed downstream by
compare_wes_wgs_overlap.py's make_cn_like_version, using a mean specific to the
on-target track) rather than trying to rescale WES depth onto a WGS-comparable
absolute depth like the off-target script does.

Unlike scale_WES_tracks.py (one --wgs_bam applied across every BAM in --wes_dir),
this script takes one matched WES/WGS BAM pair per invocation -- one sample per
call -- matching how the wrapper scripts and compare_wes_wgs_overlap*.py are
already invoked per sample. Use the accompanying wrapper to loop over a cohort.

Output columns are named to match the off-target script's output where the same
thing is being measured (chrom, start, end, raw_wes_depth, wgs_tumor_depth), so
the two on/off-target TSVs can be concatenated or compared directly. There is no
predicted_loess_upscale_depth column here since no LOESS/PoN correction is
applied -- pass --wes-column raw_wes_depth (not predicted_loess_upscale_depth)
when pointing compare_wes_wgs_overlap*.py at this output.

python3 extract_ontarget_depth.py \
  --ontarget_windows v5_ontargets.bed \
  --wes_bam SAMPLE_tumor_wes.bam \
  --wgs_bam SAMPLE_tumor_wgs.bam \
  --output_dir ontarget_depth_output/ \
  --temp_dir ./tmp
'''


def run_mosdepth(bam_path, bed_path, output_prefix, threads=4, min_mapq=15):
    """Same mosdepth invocation as scale_WES_tracks.py: MAPQ filter, excludes
    duplicates/supplementary/non-primary/QC-fail reads (--flag 3844), no per-base output."""
    cmd = [
        "mosdepth",
        "--threads", str(threads),
        "--by", bed_path,
        "--mapq", str(min_mapq),
        "--flag", "3844",
        "--no-per-base",
        output_prefix,
        bam_path
    ]
    subprocess.run(cmd, check=True)


def parse_mosdepth_regions(output_prefix):
    regions_file = f"{output_prefix}.regions.bed.gz"
    df = pd.read_csv(regions_file, sep='\t', compression='gzip',
                     header=None, names=['chrom', 'start', 'end', 'name', 'depth'])
    df['chrom'] = df['chrom'].astype(str)
    df['start'] = df['start'].astype(int)
    df['end'] = df['end'].astype(int)
    return df


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ontarget_windows", metavar="FILE", required=True,
                   help="On-target windows BED from define_ontarget_windows.py")
    p.add_argument("--wes_bam", metavar="FILE", required=True,
                   help="Tumor WES BAM")
    p.add_argument("--wgs_bam", metavar="FILE", default=None,
                   help="Matched tumor WGS BAM. If omitted, wgs_tumor_depth is filled with NaN "
                        "(mirrors scale_WES_tracks.py's behavior when --wgs_bam is not given).")
    p.add_argument("-t", metavar="INT", type=int,
                   default=4, help="Threads per mosdepth run")
    p.add_argument("--min_mapq", metavar="INT", type=int, default=15,
                   help="Minimum per-read MAPQ passed to mosdepth (default 15, matching "
                        "scale_WES_tracks.py)")
    p.add_argument("--sample_name", default=None,
                   help="Sample name for the output filename; defaults to the WES BAM's "
                        "basename up to the first '.'")
    p.add_argument("--output_dir", required=True,
                   help="Output directory for the per-sample on-target depth TSV")
    p.add_argument("--temp_dir", required=True,
                   help="Temp directory for mosdepth intermediary files")
    args = p.parse_args()

    ONTARGET_BED = args.ontarget_windows
    WES_BAM = args.wes_bam
    WGS_BAM = args.wgs_bam
    THREADS = args.t
    MIN_MAPQ = args.min_mapq
    OUTPUT_DIR = args.output_dir
    TMP_DIR = args.temp_dir

    if not os.path.exists(ONTARGET_BED):
        print(f"[-] Error: On-target windows BED not found: {ONTARGET_BED}")
        sys.exit(1)
    if not os.path.exists(WES_BAM):
        print(f"[-] Error: WES BAM not found: {WES_BAM}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)

    sample_name = args.sample_name or os.path.basename(WES_BAM).split('.')[0]
    print("=" * 55)
    print(f"[+] Processing tumor: {sample_name}")
    print("=" * 55)

    # 1. LOAD WINDOW COORDINATES (also gives us window length for reference/diagnostics)
    windows_df = pd.read_csv(ONTARGET_BED, sep='\t', header=None,
                              names=['chrom', 'start', 'end', 'window_id'])
    windows_df['chrom'] = windows_df['chrom'].astype(str)
    n_windows = len(windows_df)
    print(f"[+] Loaded {n_windows} on-target windows from {ONTARGET_BED}")

    # 2. WES ON-TARGET DEPTH
    prefix_wes = os.path.join(TMP_DIR, f"{sample_name}_on_target_wes")
    print(f"    -> Running mosdepth on on-target windows (WES), "
          f"excluding reads with MAPQ < {MIN_MAPQ}...")
    run_mosdepth(WES_BAM, ONTARGET_BED, prefix_wes, threads=THREADS, min_mapq=MIN_MAPQ)
    df_wes = parse_mosdepth_regions(prefix_wes)

    if len(df_wes) != n_windows:
        print(f"    [-] Error: Window count mismatch -- "
              f"mosdepth={len(df_wes)}, windows_bed={n_windows}. Aborting sample.")
        sys.exit(1)

    raw_wes_depth = df_wes['depth'].values.astype(float)

    # 3. MATCHED WGS ON-TARGET DEPTH (same windows, so the two tracks are directly
    #    comparable bin-for-bin -- no separate alignment step needed downstream)
    if WGS_BAM:
        if not os.path.exists(WGS_BAM):
            print(f"    [-] Warning: WGS BAM not found at {WGS_BAM}. wgs_tumor_depth will be NaN.")
            wgs_tumor_depth = np.full(n_windows, np.nan)
        else:
            prefix_wgs = os.path.join(TMP_DIR, f"{sample_name}_on_target_wgs")
            print(f"    -> Running mosdepth on matched tumor WGS BAM: {os.path.basename(WGS_BAM)} "
                  f"(excluding reads with MAPQ < {MIN_MAPQ})...")
            run_mosdepth(WGS_BAM, ONTARGET_BED, prefix_wgs, threads=THREADS, min_mapq=MIN_MAPQ)
            df_wgs = parse_mosdepth_regions(prefix_wgs)
            if len(df_wgs) != n_windows:
                print(f"    [-] Warning: WGS window count mismatch -- "
                      f"mosdepth={len(df_wgs)}, windows_bed={n_windows}. wgs_tumor_depth will be NaN.")
                wgs_tumor_depth = np.full(n_windows, np.nan)
            else:
                wgs_tumor_depth = df_wgs['depth'].values.astype(float)
    else:
        wgs_tumor_depth = np.full(n_windows, np.nan)

    # 4. ASSEMBLE OUTPUT (columns named to match scale_WES_tracks.py's off-target output
    #    where the same quantity is being measured, so both can share a downstream
    #    --wes-column of "raw_wes_depth" if you want a raw-vs-raw comparison, or the
    #    two tables can simply be concatenated on shared column names)
    df_output = windows_df[['chrom', 'start', 'end']].copy()
    df_output['length'] = df_output['end'] - df_output['start']
    df_output['raw_wes_depth'] = raw_wes_depth
    df_output['wgs_tumor_depth'] = wgs_tumor_depth
    df_output['flag_zero_wes'] = (raw_wes_depth == 0).astype(int)

    output_path = os.path.join(OUTPUT_DIR, f"{sample_name}_on_target_depth.tsv")
    df_output.to_csv(output_path, sep='\t', index=True)

    n_zero_wes = df_output['flag_zero_wes'].sum()
    wgs_covered = (~np.isnan(wgs_tumor_depth)).sum()

    print(f"    [-] Saved: {output_path}")
    print(f"    -> Windows:                          {n_windows}")
    print(f"    -> Window length range:              {df_output['length'].min()}-{df_output['length'].max()} bp")
    print(f"    -> Zero-depth WES windows:            {n_zero_wes} / {n_windows}")
    print(f"    -> WGS windows with depth:            {wgs_covered} / {n_windows}")

    print("\n" + "=" * 55)
    print("[+] extract_ontarget_depth.py COMPLETE")
    print("=" * 55)


if __name__ == "__main__":
    main()