import os
import sys
import argparse
import pandas as pd
import numpy as np

'''
Extracts on-target and off-target (antitarget) depth and CN estimates from a
CNVkit .cnr file (the output of `cnvkit.py fix` / `cnvkit.py batch`), for one
WES sample and, optionally, a matched WGS sample run through the SAME reference
(so on/off-target bins line up between the two tracks bin-for-bin, the same way
the off-target mosdepth pipeline uses one shared windows BED for WES and WGS).

ASSUMED .cnr SCHEMA (standard CNVkit output; override with --chrom-col /
--gene-col / --depth-col / --log2-col / --weight-col if your files differ):
    chromosome, start, end, gene, depth, log2, weight
On-target vs antitarget rows are distinguished by the gene column: CNVkit
labels every antitarget (off-target) bin's gene as "Antitarget" by convention
(--antitarget-label lets you override this if your reference used something
else). Everything else (real gene names, "-", intergenic on-target-only
labels, etc.) is treated as on-target.

CN is derived from log2 as a simple relative estimate: cn = ploidy * 2^log2
(--ploidy, default 2.0). This is NOT purity- or absolute-copy-number-corrected
-- it's the same kind of relative estimate CNVkit's own log2 already encodes,
just converted out of log-space. If you have per-sample purity/ploidy calls
(e.g. from CNVkit's --purity or a downstream caller), a proper absolute CN
conversion needs those and is not done here.

Because on-target and off-target bins are extracted from the SAME rows of the
SAME file, splitting after loading (rather than needing separate on/off-target
files), the two subsets are written to two separate per-sample TSVs so they
can go through different downstream handling (as planned: separate WES
thresholds for on vs off target now, and eventually a single CN-like scale for
WES that uses a different mean for on- vs off-target bins).

If a WGS .cnr is given, WES and WGS are matched bin-for-bin via an outer merge
on (chrom, start, end) -- this assumes the WGS .cnr was generated against the
SAME reference/bin set as the WES .cnr (e.g. WGS reads processed with
`cnvkit.py batch --reference <the WES reference.cnn>`), which is what makes a
bin-for-bin WES/WGS comparison meaningful in the first place. If your WGS .cnr
instead came from an independent WGS-only CNVkit run with its own bin set, the
merge will mostly fail to line up and you'll see very few matched rows -- the
script warns if the matched fraction looks low.

Output columns (both on-target and off-target TSVs):
    chrom, start, end, gene,
    wes_depth, wes_log2, wes_cn,
    wgs_depth, wgs_log2, wgs_cn      (all-NaN if no --wgs_cnr given)

This schema plugs directly into compare_wes_wgs_overlap.py's prepare_track /
rebin_track (chrom + start columns are already present), so e.g.
    --wes-column wes_depth --wgs-column wgs_depth
or
    --wes-column wes_cn --wgs-column wgs_cn
work with the existing overlap-search scripts unchanged.

python3 extract_cnvkit_on_off_target.py \
  --wes_cnr SAMPLE_wes.cnr \
  --wgs_cnr SAMPLE_wgs.cnr \
  --sample_name SAMPLE \
  --output_dir cnvkit_extracted/
'''


def load_cnr(path, chrom_col, gene_col, depth_col, log2_col, weight_col, ploidy, label):
    if not os.path.exists(path):
        print(f"[-] Error: .cnr file not found: {path}")
        sys.exit(1)

    df = pd.read_csv(path, sep='\t')

    # Resolve the chromosome column name (CNVkit has used both 'chromosome' and
    # 'chrom' across versions).
    resolved_chrom = chrom_col if chrom_col in df.columns else (
        'chromosome' if 'chromosome' in df.columns else ('chrom' if 'chrom' in df.columns else None)
    )
    required = {resolved_chrom, 'start', 'end', gene_col, depth_col}
    missing = [c for c in required if c is None or c not in df.columns]
    if missing or resolved_chrom is None:
        print(f"[-] Error ({label}): expected columns not found in {path}.")
        print(f"    Looking for: chrom-like, start, end, '{gene_col}', '{depth_col}'.")
        print(f"    Actual columns present: {list(df.columns)}")
        print(f"    If your .cnr uses different names, pass --chrom-col/--gene-col/--depth-col/--log2-col "
              f"to match.")
        sys.exit(1)

    df = df.rename(columns={resolved_chrom: 'chrom'})
    df['chrom'] = df['chrom'].astype(str)
    df['start'] = df['start'].astype(int)
    df['end'] = df['end'].astype(int)
    df['depth'] = pd.to_numeric(df[depth_col], errors='coerce')

    if log2_col in df.columns:
        df['log2'] = pd.to_numeric(df[log2_col], errors='coerce')
        df['cn'] = ploidy * np.power(2.0, df['log2'])
    else:
        print(f"    [!] Warning ({label}): log2 column '{log2_col}' not found; "
              f"cn will be NaN for this track.")
        df['log2'] = np.nan
        df['cn'] = np.nan

    n_total = len(df)
    is_antitarget = df[gene_col].astype(str) == str(antitarget_label_global)
    n_antitarget = int(is_antitarget.sum())
    print(f"    -> {label}: {n_total} bins loaded ({n_antitarget} antitarget / {n_total - n_antitarget} on-target)")

    return df[['chrom', 'start', 'end', gene_col, 'depth', 'log2', 'cn']].rename(
        columns={gene_col: 'gene'}
    ), is_antitarget


def main():
    global antitarget_label_global

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wes_cnr", metavar="FILE", required=True, help="WES CNVkit .cnr file")
    p.add_argument("--wgs_cnr", metavar="FILE", default=None,
                   help="Matched WGS CNVkit .cnr file (same reference/bin set as --wes_cnr). "
                        "If omitted, WGS columns are filled with NaN.")
    p.add_argument("--sample_name", required=True, help="Sample name, used in output filenames")
    p.add_argument("--antitarget-label", default="Antitarget",
                   help="Value in the gene column marking an off-target/antitarget bin "
                        "(CNVkit's default convention; override if your reference used something else)")
    p.add_argument("--chrom-col", default="chromosome", help="Chromosome column name")
    p.add_argument("--gene-col", default="gene", help="Gene column name")
    p.add_argument("--depth-col", default="depth", help="Depth column name")
    p.add_argument("--log2-col", default="log2", help="log2 ratio column name")
    p.add_argument("--weight-col", default="weight", help="Weight column name (currently carried "
                        "through only if present; not required)")
    p.add_argument("--ploidy", type=float, default=2.0,
                   help="Ploidy used for the relative CN conversion cn = ploidy * 2^log2 (default 2.0)")
    p.add_argument("--output_dir", required=True, help="Output directory for the extracted TSVs")
    args = p.parse_args()

    antitarget_label_global = args.antitarget_label
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 55)
    print(f"[+] Extracting CNVkit on/off-target tracks for: {args.sample_name}")
    print("=" * 55)

    print(f"[+] Loading WES .cnr: {args.wes_cnr}")
    wes_df, wes_is_anti = load_cnr(args.wes_cnr, args.chrom_col, args.gene_col, args.depth_col,
                                    args.log2_col, args.weight_col, args.ploidy, "WES")

    if args.wgs_cnr:
        print(f"[+] Loading WGS .cnr: {args.wgs_cnr}")
        wgs_df, wgs_is_anti = load_cnr(args.wgs_cnr, args.chrom_col, args.gene_col, args.depth_col,
                                        args.log2_col, args.weight_col, args.ploidy, "WGS")
    else:
        print("[+] No --wgs_cnr given; WGS columns will be NaN.")
        wgs_df, wgs_is_anti = None, None

    for track_name, is_anti_wes in (("ontarget", ~wes_is_anti), ("offtarget", wes_is_anti)):
        wes_sub = wes_df[is_anti_wes].copy()
        merged = wes_sub[['chrom', 'start', 'end', 'gene', 'depth', 'log2', 'cn']].rename(
            columns={'depth': 'wes_depth', 'log2': 'wes_log2', 'cn': 'wes_cn'}
        )

        if wgs_df is not None:
            # Match on the SAME gene-based label within the WGS track, then merge on
            # (chrom, start, end) so both tracks line up bin-for-bin.
            is_anti_wgs = wgs_is_anti if track_name == "offtarget" else ~wgs_is_anti
            wgs_sub = wgs_df[is_anti_wgs][['chrom', 'start', 'end', 'depth', 'log2', 'cn']].rename(
                columns={'depth': 'wgs_depth', 'log2': 'wgs_log2', 'cn': 'wgs_cn'}
            )
            merged = merged.merge(wgs_sub, on=['chrom', 'start', 'end'], how='outer')

            n_wes_only = merged['wes_depth'].notna().sum()
            n_both = (merged['wes_depth'].notna() & merged['wgs_depth'].notna()).sum()
            match_frac = n_both / n_wes_only if n_wes_only else 0.0
            print(f"    -> {track_name}: {n_both}/{n_wes_only} WES bins matched a WGS bin "
                  f"({match_frac:.1%})")
            if match_frac < 0.5:
                print(f"    [!] Warning: less than half of {track_name} WES bins matched a WGS bin. "
                      f"This usually means the WGS .cnr was NOT generated against the same "
                      f"reference/bin set as the WES .cnr -- double check both were run with "
                      f"the same CNVkit reference.")
        else:
            merged['wgs_depth'] = np.nan
            merged['wgs_log2'] = np.nan
            merged['wgs_cn'] = np.nan

        merged = merged.sort_values(['chrom', 'start']).reset_index(drop=True)
        output_path = os.path.join(args.output_dir, f"{args.sample_name}_{track_name}_cnvkit.tsv")
        merged.to_csv(output_path, sep='\t', index=False)
        print(f"    [-] Saved: {output_path} ({len(merged)} bins)")

    print("\n" + "=" * 55)
    print("[+] extract_cnvkit_on_off_target.py COMPLETE")
    print("=" * 55)


if __name__ == "__main__":
    main()