#!/usr/bin/env python3
"""Pooled (cohort-wide) on/off-target overlap search, Experiment 2: ONE shared
WES threshold on a region-normalized CN-like scale, across the whole cohort.

Same idea as run_grad_search_combined_cnlike.py (the per-sample version): since
combine_ontarget_offtarget_for_overlap.py already normalizes on-target bins by
the on-target mean and off-target bins by the off-target mean (never mixed),
the resulting wes_cn_like/wgs_cn_like columns are on a comparable scale across
BOTH regions -- so this needs no new search logic at all. It's the exact same
2-parameter pooled search already implemented in
compare_wes_wgs_overlap_pooled_grad.py (imported directly, not reimplemented),
just with a preprocessing step that pools each sample's COMBINED on+off-target
CN-like columns instead of a single off-target-only track.

For the pooled version of Experiment 1 (two separate WES thresholds, on vs.
off target), see compare_wes_wgs_overlap_tworegion_pooled.py instead -- that
one genuinely needs new (3-parameter) pooled search logic, since a
region-aware threshold isn't reducible to the existing 2-parameter search.

python3 run_pooled_cnlike_shared.py \
  --manifest manifest.tsv \
  --pooled-grad-script-path /path/to/compare_wes_wgs_overlap_pooled_grad.py \
  --outdir pooled_cnlike_shared_output
"""

import argparse
import hashlib
import importlib.util
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from compare_wes_wgs_overlap import (
    align_tracks,
    apply_mask_regions_to_df,
    load_data,
    load_mask_regions,
    prepare_track,
    rebin_track,
    cn_like_from_track,
)


def load_grad_module(script_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def region_cn_like(path, wes_col, wgs_col, bin_size, mask_df):
    """Same per-region logic as combine_ontarget_offtarget_for_overlap.load_region,
    returning just the finite-joint region-normalized CN-like arrays (this script
    doesn't need the raw-scale columns or the region tag, since both regions get
    pooled together on the CN-like scale)."""
    df = load_data(path)
    wes_df = prepare_track(df, wes_col, None, "chrom", "start")
    wgs_df = prepare_track(df, wgs_col, None, "chrom", "start")
    wes_df = rebin_track(wes_df, wes_col, target_bin_size=bin_size)
    wgs_df = rebin_track(wgs_df, wgs_col, target_bin_size=bin_size)
    aligned = align_tracks(wes_df, wgs_df, wes_col, wgs_col)
    n_masked = 0
    if mask_df is not None:
        aligned = apply_mask_regions_to_df(aligned, mask_df)
        n_masked = int(aligned["masked"].sum()) if "masked" in aligned.columns else 0
    wes_cn = cn_like_from_track(aligned["wes_value"])
    wgs_cn = cn_like_from_track(aligned["wgs_value"])
    return wes_cn.to_numpy(dtype=float), wgs_cn.to_numpy(dtype=float), n_masked, len(aligned)


def preprocess_sample(job):
    """Returns the same dict shape compare_wes_wgs_overlap_pooled_grad.PooledData
    expects (sample, wes, wgs, n_total, n_masked, error), pooling BOTH regions'
    CN-like values together for this one sample."""
    try:
        cache_file = None
        if job["cache_dir"] is not None:
            key = json.dumps(
                [job["ontarget_input"], _mtime(job["ontarget_input"]),
                 job["offtarget_input"], _mtime(job["offtarget_input"]),
                 job["ontarget_wes_col"], job["ontarget_wgs_col"],
                 job["offtarget_wes_col"], job["offtarget_wgs_col"],
                 job["bin_size"], job["mask_path"],
                 _mtime(job["mask_path"]) if job["mask_path"] else None],
                sort_keys=True,
            )
            digest = hashlib.md5(key.encode()).hexdigest()[:16]
            safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in job["sample"])
            cache_file = Path(job["cache_dir"]) / f"{safe}.{digest}.npz"
            if cache_file.exists():
                z = np.load(cache_file)
                return dict(sample=job["sample"], wes=z["wes"], wgs=z["wgs"],
                           n_total=int(z["n_total"]), n_masked=int(z["n_masked"]), error=None)

        mask_df = load_mask_regions(job["mask_path"]) if job["mask_path"] else None

        on_wes, on_wgs, on_masked, on_total = region_cn_like(
            job["ontarget_input"], job["ontarget_wes_col"], job["ontarget_wgs_col"], job["bin_size"], mask_df)
        off_wes, off_wgs, off_masked, off_total = region_cn_like(
            job["offtarget_input"], job["offtarget_wes_col"], job["offtarget_wgs_col"], job["bin_size"], mask_df)

        wes = np.concatenate([on_wes, off_wes])
        wgs = np.concatenate([on_wgs, off_wgs])
        ok = np.isfinite(wes) & np.isfinite(wgs)
        out = dict(sample=job["sample"], wes=wes[ok].astype(np.float32), wgs=wgs[ok].astype(np.float32),
                  n_total=int(on_total + off_total), n_masked=int(on_masked + off_masked), error=None)
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_file, wes=out["wes"], wgs=out["wgs"], n_total=out["n_total"], n_masked=out["n_masked"])
        return out
    except Exception as exc:
        return dict(sample=job["sample"], wes=None, wgs=None, n_total=0, n_masked=0, error=repr(exc))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True,
                  help="TSV with columns: sample, ontarget_input, offtarget_input")
    p.add_argument("--pooled-grad-script-path", required=True,
                  help="Path to compare_wes_wgs_overlap_pooled_grad.py, imported directly so this "
                       "reuses its exact, already-tested PooledData/run_pooled_search implementation")
    p.add_argument("--ontarget-wes-column", default="raw_wes_depth")
    p.add_argument("--ontarget-wgs-column", default="wgs_tumor_depth")
    p.add_argument("--offtarget-wes-column", default="predicted_loess_upscale_depth")
    p.add_argument("--offtarget-wgs-column", default="wgs_tumor_depth")
    p.add_argument("--target-bin-size", type=int, default=25000)
    p.add_argument("--mask-regions", default=None)
    p.add_argument("--metric", choices=["jaccard", "overlap", "both"], default="both")
    p.add_argument("--aggregate", choices=["micro", "macro"], default="micro")
    p.add_argument("--n-steps", type=int, default=100)
    p.add_argument("--n-starts", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--temp", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-frac", type=float, default=0.0005)
    p.add_argument("--max-frac", type=float, default=0.01)
    p.add_argument("--min-bins", type=int, default=1000)
    p.add_argument("--min-samples", type=int, default=10)
    p.add_argument("--min-cn-like", type=float, default=4.0)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--outdir", default="wes_wgs_overlap_cnlike_shared_pooled_output")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or str(outdir / "cache")

    grad_pooled = load_grad_module(args.pooled_grad_script_path, "compare_wes_wgs_overlap_pooled_grad")

    manifest = pd.read_csv(args.manifest, sep="\t")
    missing = {"sample", "ontarget_input", "offtarget_input"} - set(manifest.columns)
    if missing:
        raise SystemExit(f"Manifest is missing columns: {sorted(missing)}")

    jobs = [dict(sample=str(r.sample), ontarget_input=str(r.ontarget_input), offtarget_input=str(r.offtarget_input),
                ontarget_wes_col=args.ontarget_wes_column, ontarget_wgs_col=args.ontarget_wgs_column,
                offtarget_wes_col=args.offtarget_wes_column, offtarget_wgs_col=args.offtarget_wgs_column,
                bin_size=args.target_bin_size, mask_path=args.mask_regions, cache_dir=cache_dir)
           for r in manifest.itertuples(index=False)]
    print(f"Preprocessing {len(jobs)} samples with {args.n_jobs} worker(s); cache: {cache_dir}", flush=True)

    results = []
    if args.n_jobs > 1:
        with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
            for i, r in enumerate(ex.map(preprocess_sample, jobs, chunksize=1), 1):
                results.append(r)
                if i % 25 == 0:
                    print(f"  preprocessed {i}/{len(jobs)}", flush=True)
    else:
        for i, job in enumerate(jobs, 1):
            results.append(preprocess_sample(job))
            if i % 25 == 0:
                print(f"  preprocessed {i}/{len(jobs)}", flush=True)

    used = [r for r in results if r["error"] is None and r["wes"] is not None and len(r["wes"]) > 0]
    skipped = [dict(sample=r["sample"], reason=r["error"] or "no jointly valid bins") for r in results
              if not (r["error"] is None and r["wes"] is not None and len(r["wes"]) > 0)]
    pd.DataFrame(skipped, columns=["sample", "reason"]).to_csv(outdir / "pooled_skipped_samples.tsv", sep="\t", index=False)
    if not used:
        raise SystemExit("No usable samples after preprocessing.")
    pd.DataFrame(dict(sample=[r["sample"] for r in used],
                      n_bins_total=[r["n_total"] for r in used],
                      n_bins_masked=[r["n_masked"] for r in used],
                      n_bins_used=[len(r["wes"]) for r in used])).to_csv(
        outdir / "pooled_samples_used.tsv", sep="\t", index=False)
    print(f"Using {len(used)} samples ({len(skipped)} skipped)", flush=True)

    data = grad_pooled.PooledData(used)
    del results, used
    print(f"Pooled bins (on+off target combined): {data.n_total}", flush=True)

    floor = args.min_cn_like
    metrics = ["jaccard", "overlap"] if args.metric == "both" else [args.metric]
    for i, metric in enumerate(metrics):
        best, history = grad_pooled.run_pooled_search(
            data, metric, args.aggregate, args.n_steps, args.n_starts, args.lr, args.seed + i,
            args.temp, args.min_frac, args.max_frac, args.min_bins, args.min_samples, floor,
        )
        pd.DataFrame([best]).to_csv(outdir / f"cnlike_shared_pooled_best_{metric}.tsv", sep="\t", index=False)
        history.to_csv(outdir / f"cnlike_shared_pooled_history_{metric}.tsv", sep="\t", index=False)
        grad_pooled.per_sample_table(data, best["wes_threshold"], best["wgs_threshold"]).to_csv(
            outdir / f"cnlike_shared_pooled_per_sample_{metric}.tsv", sep="\t", index=False)
        grad_pooled.plot_search(history, best, outdir / f"cnlike_shared_pooled_search_{metric}.png", metric, args.aggregate)
        print(f"Best pooled combined-CN-like {args.aggregate} {metric}: {best}", flush=True)


if __name__ == "__main__":
    main()