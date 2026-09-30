#!/bin/bash
#SBATCH --job-name=recurrent_bins_search
#SBATCH --output=/home/sspandau/logs/%x_%j.out
#SBATCH --error=/home/sspandau/logs/%x_%j.err
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

# NOTE: SLURM does not create the log directory. Run once before submitting:
#   mkdir -p /home/sspandau/logs
#
# Submit from the directory that contains scripts/ (or set PROJECT_DIR below):
#   sbatch run_bayes_threshold_search.sh

set -euo pipefail

# ==========================================================================
# CONFIG
# ==========================================================================
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPTS_DIR="${PROJECT_DIR}/scripts"

WES_ROOT="/metropolis/projects/sspandau/depmap/wes/scaled_output"

# Every directory holding per-sample AA results (one subdir per sample).
# Independent of the classification lists below -- list as many as needed.
AA_ROOTS=(
    "/nucleus/projects/cancer_cell_lines/AA/CCLE/AA"     # classified by CCLE_AC_update (6a5a432e...)
    "/metropolis/projects/jluebeck/depmap/stage2_aa"      # classified by DepMap_CCLE/WGS (6aa45edc...)
)

# AmpliconClassifier runs. These two lists ARE parallel (TSV i and BED dir i
# come from the same run). Each AA sample's classification is looked up by
# name across all runs, so it doesn't matter which AA root a sample came from.
CLASSIFICATION_TSVS=(
    "/home/sspandau/CCLE_AC_update/results/consolidated_classification/6a5a432efaab0afcb5586742_amplicon_classification_profiles.tsv"
    "/home/sspandau/DepMap_CCLE/WGS/results/consolidated_classification/6aa45edcef17405eac5bbf51_amplicon_classification_profiles.tsv"
)
CLASSIFICATION_BED_DIRS=(
    "/home/sspandau/CCLE_AC_update/results/consolidated_classification/6a5a432efaab0afcb5586742_classification_bed_files"
    "/home/sspandau/DepMap_CCLE/WGS/results/consolidated_classification/6aa45edcef17405eac5bbf51_classification_bed_files"
)

# Optional: explicit WES-id -> cell-line table (2 columns, tab or comma,
# header optional). Only needed if some WES names can't be matched by name.
ID_MAP_IN=""

OUTDIR="/data/analysis/bayesian_threshold_search"
INPUT_DIR="${OUTDIR}/inputs"          # merged/renamed inputs live here
MERGED_AA="${INPUT_DIR}/aa_merged"
MERGED_BED="${INPUT_DIR}/classification_bed_merged"
MERGED_TSV="${INPUT_DIR}/amplicon_classification_profiles.merged.tsv"
SAMPLE_MAP="${INPUT_DIR}/wes_to_aa_sample_map.tsv"
MATCH_REPORT="${INPUT_DIR}/sample_matching_report.tsv"

# ==========================================================================
# ENVIRONMENT
# ==========================================================================
# module load python/3.11          # <- adjust for your cluster
# source activate ecdna_env        # <- or conda activate ...

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export PYTHONUNBUFFERED=1

cd "${PROJECT_DIR}"

echo "== Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
echo "== Project dir: ${PROJECT_DIR}"

BASE_SCRIPT="${SCRIPTS_DIR}/find_recurrent_novel_amplifications_binlevel.py"
GRID_SCRIPT="${SCRIPTS_DIR}/optimize_recurrent_amplification_threshold.py"
BAYES_SCRIPT="${SCRIPTS_DIR}/optimize_recurrent_amplification_threshold_wgstreemodel.py"
for f in "${BAYES_SCRIPT}" "${BASE_SCRIPT}" "${GRID_SCRIPT}"; do
    [[ -f "$f" ]] || { echo "ERROR: missing $f" >&2; exit 1; }
done

if (( ${#CLASSIFICATION_TSVS[@]} != ${#CLASSIFICATION_BED_DIRS[@]} )); then
    echo "ERROR: CLASSIFICATION_TSVS and CLASSIFICATION_BED_DIRS must have the same length" >&2
    exit 1
fi

python3 - <<'EOF'
import importlib.util, sys
missing = [m for m in ("numpy", "pandas", "scipy", "sklearn", "skopt", "matplotlib")
           if importlib.util.find_spec(m) is None]
if missing:
    sys.exit(f"ERROR: missing python packages: {missing} "
             "(skopt = pip install scikit-optimize)")
print("== Python deps OK")
EOF

mkdir -p "${OUTDIR}" "${INPUT_DIR}"

# ==========================================================================
# STEP 1: match WES samples to AA samples, then rebuild the AA root, the
# classification TSV and the classification BED dir so that ALL of them use
# the WES sample id. With one shared namespace, no --sample-map is needed.
# ==========================================================================
echo "== Matching WES samples to AA samples and renaming inputs"
export WES_ROOT MERGED_AA MERGED_BED MERGED_TSV SAMPLE_MAP MATCH_REPORT ID_MAP_IN
export AA_ROOTS_JOINED="$(IFS=:; echo "${AA_ROOTS[*]}")"
export TSVS_JOINED="$(IFS=:; echo "${CLASSIFICATION_TSVS[*]}")"
export BEDS_JOINED="$(IFS=:; echo "${CLASSIFICATION_BED_DIRS[*]}")"

python3 - <<'EOF'
import csv, os, re, shutil, sys
from collections import defaultdict
from pathlib import Path
import pandas as pd

env = os.environ
wes_root  = Path(env["WES_ROOT"])
aa_roots  = [Path(p) for p in env["AA_ROOTS_JOINED"].split(":")]
tsvs      = [Path(p) for p in env["TSVS_JOINED"].split(":")]
bed_dirs  = [Path(p) for p in env["BEDS_JOINED"].split(":")]
merged_aa, merged_bed = Path(env["MERGED_AA"]), Path(env["MERGED_BED"])
merged_tsv, map_out, report = Path(env["MERGED_TSV"]), Path(env["SAMPLE_MAP"]), Path(env["MATCH_REPORT"])
id_map_in = env.get("ID_MAP_IN", "").strip()

for p in aa_roots + bed_dirs:
    if not p.is_dir():
        sys.exit(f"ERROR: directory not found: {p}")
for p in tsvs:
    if not p.is_file():
        sys.exit(f"ERROR: file not found: {p}")

def norm(s):
    return re.sub(r"[^A-Z0-9]", "", s.upper())

# --- AA samples: name -> (AA root index, path); earlier root wins ---
aa = {}
for b, root in enumerate(aa_roots):
    n = 0
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        if d.name in aa:
            print(f"   WARN: {d.name} in AA roots {aa[d.name][0]} and {b}; using root {aa[d.name][0]}",
                  file=sys.stderr)
            continue
        aa[d.name] = (b, d)
        n += 1
    print(f"   AA root {b}: {n:4d} samples in {root}")

def aliases(k):
    # common vendor-prefix variants: NCIH1048 <-> H1048, NIHOVCAR3 <-> OVCAR3
    out = {k}
    for pre in ("NCI", "NIH"):
        if k.startswith(pre) and len(k) > len(pre) + 1:
            out.add(k[len(pre):])
    return out

index = defaultdict(set)
for name in aa:
    for k in aliases(norm(name)) | aliases(norm(name.split("_", 1)[0])):
        index[k].add(name)

tissues = sorted({n.split("_", 1)[1].upper() for n in aa if "_" in n}, key=len, reverse=True)
def strip_tissue(q):
    for t in tissues:
        if q.upper().endswith("_" + t):
            return q[: -len(t) - 1]
    return q

# --- WES samples ---
wes_ids = []
for e in sorted(wes_root.iterdir()):
    if e.name.startswith("."):
        continue
    sid = e.name if e.is_dir() else e.name.split(".", 1)[0]
    if sid and sid not in wes_ids:
        wes_ids.append(sid)
print(f"   {len(wes_ids):4d} WES samples in {wes_root}")

id_map = {}
if id_map_in:
    with open(id_map_in) as fh:
        head = fh.read(4096); fh.seek(0)
        for row in csv.reader(fh, delimiter="\t" if "\t" in head else ","):
            if len(row) >= 2 and row[0].strip():
                id_map[row[0].strip()] = row[1].strip()

# --- match WES -> AA ---
for d in (merged_aa, merged_bed):
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)

rows, pairs = [], []                   # pairs: (wes_id, aa_name, aa_root_index)
for wid in wes_ids:
    query = id_map.get(wid, wid)
    hits = set()
    for k in (norm(query), norm(strip_tissue(query))):
        hits = set().union(*(index.get(a, set()) for a in aliases(k)))
        if hits:
            break
    if len(hits) == 1:
        aname = next(iter(hits)); b, path = aa[aname]
        (merged_aa / wid).symlink_to(path.resolve())
        pairs.append((wid, aname, b))
        rows.append([wid, query, aname, str(b), "matched"])
    elif len(hits) > 1:
        rows.append([wid, query, ";".join(sorted(hits)), "", "ambiguous"])
    else:
        rows.append([wid, query, "", "", "unmatched"])

if not pairs:
    sys.exit("ERROR: no WES samples matched an AA sample.")

# --- classification: find each matched AA sample in whichever AC run has
#     it (first listed run wins if several do), rename sample_name -> WES id ---
dfs, src_of = [], {}
for i, tsv in enumerate(tsvs):
    df = pd.read_csv(tsv, sep="\t", dtype=str, keep_default_na=False)
    if "sample_name" not in df.columns:
        sys.exit(f"ERROR: no 'sample_name' column in {tsv}; columns = {list(df.columns)[:10]}")
    dfs.append(df)
    for name in df["sample_name"].unique():
        src_of.setdefault(name, i)

wids_of = defaultdict(list)
for wid, aname, _ in pairs:
    wids_of[aname].append(wid)

frames, n_rows, src_used = [], {}, {}
for aname, wids in wids_of.items():
    i = src_of.get(aname)
    if i is None:
        continue
    sub = dfs[i][dfs[i]["sample_name"] == aname]
    for wid in wids:
        frames.append(sub.assign(sample_name=wid))
        n_rows[wid], src_used[wid] = len(sub), i

for i, tsv in enumerate(tsvs):
    n_names = dfs[i]["sample_name"].nunique()
    n_hit = sum(1 for a in wids_of if src_of.get(a) == i)
    print(f"   AC run {i}: {n_names} samples in TSV, {n_hit} used for matched WES samples  ({tsv.name})")
no_class = sorted(a for a in wids_of if a not in src_of)
if no_class:
    print(f"   {len(no_class)} matched AA samples are in no classification TSV "
          f"(no amplicons, or a name mismatch), e.g. {no_class[:5]}")

merged = pd.concat(frames, ignore_index=True, sort=False).fillna("") if frames else pd.DataFrame()
merged.to_csv(merged_tsv, sep="\t", index=False)
print(f"   wrote {len(merged)} amplicon rows -> {merged_tsv}")

# --- classification BEDs, from the same AC run the TSV rows came from ---
n_beds = defaultdict(int)
pat = re.compile(r"^(.+?)_amplicon\d+")
for i, bdir in enumerate(bed_dirs):
    for f in sorted(bdir.iterdir()):
        m = pat.match(f.name)
        if not m:
            continue
        aname = m.group(1)
        if src_of.get(aname) != i or aname not in wids_of:
            continue
        for wid in wids_of[aname]:
            (merged_bed / (wid + f.name[len(aname):])).symlink_to(f.resolve())
            n_beds[wid] += 1
print(f"   linked {sum(n_beds.values())} classification BEDs -> {merged_bed}")

# --- map + report ---
with open(map_out, "w") as fh:
    fh.write("wes_sample\taa_sample\taa_root\tac_run\n")
    fh.writelines(f"{w}\t{a}\t{b}\t{src_used.get(w, '')}\n" for w, a, b in pairs)
with open(report, "w") as fh:
    fh.write("wes_sample\tlookup_name\taa_candidates\taa_root\tstatus\tac_run\tn_amplicon_rows\tn_beds\n")
    for r in rows:
        w = r[0]
        fh.write("\t".join(r) + f"\t{src_used.get(w, '')}\t{n_rows.get(w, 0)}\t{n_beds.get(w, 0)}\n")

status = [r[4] for r in rows]
print(f"   matched={status.count('matched')}  ambiguous={status.count('ambiguous')}  "
      f"unmatched={status.count('unmatched')}  (details: {report})")
used = {a for _, a, _ in pairs}
unused = sorted(set(aa) - used)
if unused:
    print(f"   {len(unused)} AA samples had no WES match, e.g. {unused[:5]}")
EOF

# ==========================================================================
# STEP 2: pre-flight -- confirm the pipeline's own loaders pair the inputs
# ==========================================================================
echo "== Pre-flight: checking loaders on renamed inputs"
export BASE_SCRIPT GRID_SCRIPT
python3 - <<'EOF'
import importlib.util, os, sys
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod
base = load(os.environ["BASE_SCRIPT"], "base")
gs = load(os.environ["GRID_SCRIPT"], "gs")

table, skips = base.discover_samples(os.environ["WES_ROOT"], os.environ["MERGED_AA"])
print(f"   discover_samples: {len(table)} usable samples, {len(skips)} skipped")
for s in list(skips)[:10]:
    print(f"     skip: {s}")
if len(table) == 0:
    sys.exit("ERROR: 0 samples paired. The base script may expect seed BEDs named by the "
             "AA sample name rather than the directory name -- inspect a skip entry above.")

qual = gs.load_qualifying_amplicons_by_sample(os.environ["MERGED_TSV"])
col = next((c for c in ("sample", "sample_id", "sample_name") if c in table.columns), table.columns[0])
ids = set(table[col].astype(str))
print(f"   classification: {len(qual)} samples with ecDNA/BFB/CNC amplicons, "
      f"{len(ids & set(map(str, qual)))} of them in the discovered sample table (col '{col}')")
EOF

# ==========================================================================
# STEP 3: run the search
# ==========================================================================
echo "== Starting OPT Search at $(date)"
python3 "${OPT_SCRIPT}" \
  --wes-root "${WES_ROOT}" \
  --aa-root "${MERGED_AA}" \
  --classification-tsv "${MERGED_TSV}" \
  --classification-bed-dir "${MERGED_BED}" \
  --column predicted_loess_upscale_depth \
  --mask-col mask_rejected \
  --rebin-to 25000 \
  --quantiles "0.80:0.98:0.02" \
  --min-samples-ratios "0.5:1:0.05" \
  --min-bin-instances-for-selection 30 \
  --max-recurrent-fraction 0.15 \
  --outdir "${OUTDIR}"

echo "== Done at $(date). Results in ${OUTDIR}"