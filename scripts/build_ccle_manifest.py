import os
import glob
import argparse
import csv

'''
Builds the sample_id / wes_bam / wgs_bam manifest for train_pooled_bias_model.py
from your two CCLE directory trees:

  WGS: /nucleus/projects/cancer_cell_lines/WGS/ccle_bams/<SAMPLE>/<SAMPLE>.cs.rmdup.bam
  WES: /pedigree2/cui/CCLE_WXS/<SAMPLE>/<SRR>.GRCh38_realigned.sorted.bam

Only includes a sample if BOTH a WES and WGS bam are found. Non-sample
entries in the WES directory (GRCh37/, logs/, 1000genomes_highcov_WGS/,
loose files like wrapper.sh, *.csv, *.txt, *.ngc) are skipped automatically
since they either aren't directories or contain no matching bam pattern.

If a sample has more than one WES bam (multiple SRR runs), all are listed
and flagged -- you must resolve this manually (e.g. merge with samtools
merge, or pick the highest-depth run) before using the manifest, since the
manifest format assumes exactly one WES bam per sample.

python3 build_manifest.py \
  --wes_dir /pedigree2/cui/CCLE_WXS \
  --wgs_dir /nucleus/projects/cancer_cell_lines/WGS/ccle_bams \
  --output matched_cohort_manifest.tsv
'''

def find_wgs_bam(wgs_dir, sample_id):
    candidate = os.path.join(wgs_dir, sample_id, f"{sample_id}.cs.rmdup.bam")
    return candidate if os.path.exists(candidate) else None

def find_wes_bams(wes_dir, sample_id):
    pattern = os.path.join(wes_dir, sample_id, "*.GRCh38_realigned.sorted.bam")
    return sorted(glob.glob(pattern))

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wes_dir", required=True, help="Root dir of WES sample subfolders")
    p.add_argument("--wgs_dir", required=True, help="Root dir of WGS sample subfolders")
    p.add_argument("--output", required=True, help="Output manifest TSV path")
    args = p.parse_args()

    wes_sample_dirs = sorted([
        d for d in os.listdir(args.wes_dir)
        if os.path.isdir(os.path.join(args.wes_dir, d))
    ])

    rows = []
    multi_wes = []
    missing_wgs = []
    missing_wes = []

    for sample_id in wes_sample_dirs:
        wes_bams = find_wes_bams(args.wes_dir, sample_id)
        wgs_bam = find_wgs_bam(args.wgs_dir, sample_id)

        if not wes_bams:
            missing_wes.append(sample_id)
            continue
        if wgs_bam is None:
            missing_wgs.append(sample_id)
            continue
        if len(wes_bams) > 1:
            multi_wes.append((sample_id, wes_bams))
            continue  # skip until resolved -- see warning below

        rows.append((sample_id, wes_bams[0], wgs_bam))

    with open(args.output, 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['sample_id', 'wes_bam', 'wgs_bam'])
        w.writerows(rows)

    print("=" * 60)
    print(f"[+] Wrote {len(rows)} matched WGS/WES pairs to: {args.output}")
    if missing_wgs:
        print(f"[!] {len(missing_wgs)} WES samples had NO matching WGS bam (skipped): "
              f"{missing_wgs}")
    if missing_wes:
        print(f"[!] {len(missing_wes)} WES sample dirs had no *.GRCh38_realigned.sorted.bam "
              f"file (skipped, likely non-sample dirs or unprocessed): {missing_wes}")
    if multi_wes:
        print(f"[!] {len(multi_wes)} samples had MULTIPLE WES bams -- excluded from manifest, "
              f"resolve manually (merge or pick one) and add by hand:")
        for sample_id, bams in multi_wes:
            print(f"      {sample_id}:")
            for b in bams:
                print(f"        - {b}")
    print("=" * 60)

if __name__ == "__main__":
    main()
