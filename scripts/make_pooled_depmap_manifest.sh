#!/usr/bin/env python3
"""
build_pooled_manifest.py

Builds the manifests for run_pooled_cnlike_shared_clustered.py from the DepMap
layout:

  on-target : <ontarget-root>/<S>/ontarget_depth_output/<S>_on_target_depth.tsv
              (needs chrom, start, raw_wes_depth, wgs_tumor_depth)
  off-target: <offtarget-root>/<S>/offtarget_scaled_output/<S>_off_target_copy_ratios.tsv
              (needs chrom, start, predicted_loess_upscale_depth, wgs_tumor_depth)
  clusters  : <offtarget-root>/<S>/offtarget_scaled_output/<S>_cluster_results/<table>
              -- the first tabular file (.tsv/.csv/.bed/.txt) in that folder whose
                 header has a chrom-like AND start-like column; plots etc. ignored

Every file's header is checked; samples missing anything go to
<prefix>.problems.tsv with the reason instead of into the manifests.

Complete samples are split at random (fixed --seed, so reproducible) into
  <prefix>.train.tsv    (--train-frac, default 0.6)  -> run the search on these
  <prefix>.holdout.tsv  (the rest, default 0.4)       -> evaluated at the found thresholds
plus <prefix>.all.tsv with every complete sample and its split label.

Example:
  python3 build_pooled_manifest.py --out-prefix /home/sspandau/WES2WGS_CNscaling/manifests/pooled
"""

import argparse
import random
from pathlib import Path

import pandas as pd

CHROM_NAMES = {"chrom", "chr", "chromosome", "seqnames", "contig", "seqname"}
START_NAMES = {"start", "bin_start", "chromstart", "start_pos", "pos"}
TABULAR = (".tsv", ".csv", ".bed", ".txt", ".tsv.gz", ".csv.gz", ".bed.gz")


def shallowest(sample_dir, pattern):
    hits = [p for p in sample_dir.rglob(pattern) if p.is_file()]
    hits.sort(key=lambda p: (len(p.relative_to(sample_dir).parts), str(p)))
    return hits[0] if hits else None


def header(path):
    sep = "," if str(path).lower().endswith((".csv", ".csv.gz")) else "\t"
    return {str(c).strip() for c in pd.read_csv(path, sep=sep, nrows=0).columns}


def find_cluster_table(sample_dir, pattern):
    cands = sorted((p for p in sample_dir.rglob(pattern)
                    if p.is_file() and p.name.lower().endswith(TABULAR)),
                   key=lambda p: (len(p.relative_to(sample_dir).parts), str(p)))
    for p in cands:
        try:
            cols = {c.lower() for c in header(p)}
            if cols & CHROM_NAMES and cols & START_NAMES:
                return p, len(cands)
            # headerless BED: chrom<TAB>int<TAB>...
            first = p.open().readline().split("\t")
            if len(first) >= 3 and first[1].strip().isdigit():
                return p, len(cands)
        except Exception:
            continue
    return None, len(cands)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ontarget-root", default="/metropolis/projects/sspandau/depmap/wes/ontarget_output")
    ap.add_argument("--ontarget-pattern", default="*_on_target_depth.tsv")
    ap.add_argument("--offtarget-root", default="/metropolis/projects/sspandau/depmap/wes/scaled_output")
    ap.add_argument("--offtarget-pattern", default="*_off_target_copy_ratios.tsv")
    ap.add_argument("--cluster-pattern", default="*_cluster_results/*",
                    help="Searched under each off-target sample dir")
    ap.add_argument("--ontarget-wes-column", default="raw_wes_depth")
    ap.add_argument("--ontarget-wgs-column", default="wgs_tumor_depth")
    ap.add_argument("--offtarget-wes-column", default="predicted_loess_upscale_depth")
    ap.add_argument("--offtarget-wgs-column", default="wgs_tumor_depth")
    ap.add_argument("--allow-missing-clusters", action="store_true",
                    help="Keep samples without a cluster table (search must then use --missing-clusters keep)")
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-prefix", required=True)
    a = ap.parse_args()

    need_on = {"chrom", "start", a.ontarget_wes_column, a.ontarget_wgs_column}
    need_off = {"chrom", "start", a.offtarget_wes_column, a.offtarget_wgs_column}
    on_root, off_root = Path(a.ontarget_root), Path(a.offtarget_root)

    samples = sorted({d.name for root in (on_root, off_root) for d in root.iterdir()
                      if d.is_dir() and not d.name.startswith(".")})
    rows, problems = [], []
    for s in samples:
        issues = []
        on = shallowest(on_root / s, a.ontarget_pattern) if (on_root / s).is_dir() else None
        off = shallowest(off_root / s, a.offtarget_pattern) if (off_root / s).is_dir() else None
        if on is None:
            issues.append("no on-target file")
        elif need_on - header(on):
            issues.append(f"on-target missing {sorted(need_on - header(on))}")
        if off is None:
            issues.append("no off-target file")
        elif need_off - header(off):
            issues.append(f"off-target missing {sorted(need_off - header(off))}")
        cl, n_cand = find_cluster_table(off_root / s, a.cluster_pattern) if (off_root / s).is_dir() else (None, 0)
        if cl is None and not a.allow_missing_clusters:
            issues.append("no cluster table yet" if n_cand == 0
                          else f"{n_cand} tabular files in cluster results, none with chrom/start columns")
        if issues:
            problems.append(dict(sample=s, problem="; ".join(issues)))
        else:
            rows.append(dict(sample=s, ontarget_input=str(on), offtarget_input=str(off),
                             offtarget_clusters=str(cl) if cl else ""))

    prefix = Path(a.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(problems, columns=["sample", "problem"]).to_csv(f"{prefix}.problems.tsv", sep="\t", index=False)
    if not rows:
        raise SystemExit(f"No complete samples ({len(problems)} with problems) -- see {prefix}.problems.tsv")

    order = sorted(r["sample"] for r in rows)
    random.Random(a.seed).shuffle(order)
    n_train = round(a.train_frac * len(order))
    train = set(order[:n_train])
    df = pd.DataFrame(rows)
    df["split"] = ["train" if s in train else "holdout" for s in df["sample"]]
    df.to_csv(f"{prefix}.all.tsv", sep="\t", index=False)
    cols = ["sample", "ontarget_input", "offtarget_input", "offtarget_clusters"]
    df.loc[df["split"] == "train", cols].to_csv(f"{prefix}.train.tsv", sep="\t", index=False)
    df.loc[df["split"] == "holdout", cols].to_csv(f"{prefix}.holdout.tsv", sep="\t", index=False)

    print(f"{len(rows)} complete samples -> train {n_train}, holdout {len(rows) - n_train} "
          f"(seed {a.seed}); {len(problems)} with problems")
    print(f"  {prefix}.train.tsv / .holdout.tsv / .all.tsv / .problems.tsv")
    for pr in problems[:5]:
        print(f"  problem: {pr['sample']}: {pr['problem']}")
    print(f"  example cluster table: {rows[0]['offtarget_clusters'] or '(none)'}")


if __name__ == "__main__":
    main()
