"""Build cifs_pool and labels for the csd2023 11-class dataset.

- Copy 6644 cifs from cifs_nodf (names unchanged).
- Copy nuclearity-1 cifs from csd_structures/cifs renaming 1_X.cif -> 1X.cif.
  By default ALL automatic-rule mononuclears are taken (162 files = 74 manual
  1_* + 88 relabelled 2_/p_ per labels/csd_labels_auto_manifest.csv, auto
  prefix '1'); pass --manual-only to reproduce the original 74-file variant.
- Write labels/csd2023_labels.txt (order = sorted(*.dat) glob order, i.e. sorted by
  dat filename) and labels/csd2023_labels_manifest.csv.
Idempotent: skips copy if same-size file exists.

2026-09-08: switched to the automatic 162-file set (label consistency with the
nodf pool, which is entirely auto-labelled).
"""
import os, shutil, csv, argparse
from collections import Counter

DATA = "/workspace/home/pdf-nn-data"
POOL = f"{DATA}/csd2023/cifs_pool"
NODF = f"{DATA}/csd2023/cifs_nodf"
CSD1 = f"{DATA}/csd_structures/cifs"
LABELS = f"{DATA}/labels"

ap = argparse.ArgumentParser()
ap.add_argument("--manual-only", action="store_true",
                help="use only the 74 manual 1_* files (original variant)")
args = ap.parse_args()

os.makedirs(POOL, exist_ok=True)

# --- copy nodf files ---
nodf = sorted(os.listdir(NODF))
copied = skipped = 0
for f in nodf:
    src, dst = f"{NODF}/{f}", f"{POOL}/{f}"
    if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
        skipped += 1
        continue
    shutil.copy2(src, dst)
    copied += 1
print(f"nodf: copied {copied}, already present {skipped}, total {len(nodf)}")

# --- copy nuclearity-1 files with rename ---
# automatic-rule mononuclears from the Task-1 relabel manifest; fallback to
# filename-based selection if the manifest is absent
manifest = f"{LABELS}/csd_labels_auto_manifest.csv"
if args.manual_only or not os.path.exists(manifest):
    n1 = sorted(f for f in os.listdir(CSD1) if f.startswith("1_") and f.endswith(".cif"))
else:
    with open(manifest) as fh:
        n1 = sorted(r["filename"] for r in csv.DictReader(fh)
                    if r["auto_prefix"] == "1")
import re as _re
for f in n1:
    dst = f"{POOL}/1{_re.sub(r'^(\d+|p)_', '', f)}"  # <pfx>_REFCODE.cif -> 1REFCODE.cif
    src = f"{CSD1}/{f}"
    if not (os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src)):
        shutil.copy2(src, dst)
print(f"nuc1: {len(n1)} files")
pool = sorted(os.listdir(POOL))
assert len(pool) == 6644 + len(n1), len(pool)

# --- labels ---
# map cif filename -> label string (3C-style numeric strings)
cif2label = {}
with open(f"{DATA}/csd2023/labels_nodf_pool.csv") as fh:
    for row in csv.DictReader(fh):
        lab = row["label_binned"]
        s = "10" if lab == "10+" else ("11" if lab == "polymer" else lab)
        cif2label[row["out_filename"]] = s
for f in n1:
    cif2label[f"1{_re.sub(r'^(\d+|p)_', '', f)}"] = "1"

missing = [f for f in pool if f not in cif2label]
assert not missing, missing[:5]

# the 88 relabelled files previously sat in the pool under a different prefix;
# remove those stale copies so pool count and labels stay consistent
stale = [f for f in os.listdir(POOL)
         if f not in cif2label and not f.startswith(("10", "12", "18", "20", "22", "26"))
         and f[0] != "p"]
for f in stale:
    os.remove(f"{POOL}/{f}")
if stale:
    print(f"removed stale pre-switch copies: {len(stale)}")
    pool = sorted(os.listdir(POOL))
    assert len(pool) == 6644 + len(n1), len(pool)

# order must match sorted(glob('calculated_pdfs/*.dat')); dat name = cif stem + .dat
dat_sorted = sorted(f[:-4] + ".dat" for f in pool)
with open(f"{LABELS}/csd2023_labels.txt", "w") as fh:
    for d in dat_sorted:
        fh.write(cif2label[d[:-4] + ".cif"] + "\n")
with open(f"{LABELS}/csd2023_labels_manifest.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["filename", "label", "source"])
    for f in pool:
        w.writerow([f, cif2label[f], "nuc1" if f[0] == "1" and f[1].isalpha() else "nodf"])

print(f"pool total: {len(pool)}")

dist = Counter(cif2label[f] for f in pool)
for k in sorted(dist, key=int):
    print(k, dist[k])
