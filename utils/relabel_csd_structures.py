#!/usr/bin/env python3
"""
relabel_csd_structures.py — Reclassify the 700 original csd_structures CIFs
under the automatic prepare_cifs.py rules and write:

  1. labels/csd_labels_auto_manifest.csv  (per-structure record)
  2. labels/csd_labels_auto.txt           (labels for the 645 .dat files,
                                           in sorted-glob order of the .dat files)

Run with the SYSTEM python3 (has gemmi):
    /workspace/miniconda3/bin/python3 relabel_csd_structures.py

Label rule for the labels file: auto_prefix ('1'..'9' or 'p'), except any
recomputed finite prefix >= 10 is folded into class 10 ('p'-equivalent, written
as '10' to match the original csd_labels.txt convention of digits 1-9 + '10').
"""

import csv
import glob
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/workspace/home/pdf-nn-data/csd2023")
from prepare_cifs import build_structure, analyse_structure, classify, parse_tag, DEFAULT_BRIDGES

DATA = Path("/workspace/home/pdf-nn-data")
CIF_DIR = DATA / "csd_structures" / "cifs"
PDF_DIR = DATA / "csd_structures" / "calculated_pdfs"
LABELS_DIR = DATA / "labels"
MANIFEST = LABELS_DIR / "csd_labels_auto_manifest.csv"
LABELS_FILE = LABELS_DIR / "csd_labels_auto.txt"

BOND_TOL = 0.4


def reclassify(cif_path):
    """Return dict with auto classification of one CIF, or raise."""
    text = cif_path.read_text()
    log = []
    struct, reason = build_structure(text, log)
    if struct is None:
        raise RuntimeError(reason)
    analysis = analyse_structure(struct, {"O", "N", "F", "Cl"}, BOND_TOL, log)
    if analysis is None:
        raise RuntimeError("no Ln atoms found")
    prefix, klass, nuc, arch, ambiguous = classify(analysis)
    return {
        "prefix": prefix,
        "architecture": arch,
        "entity_nuclearities": ";".join(map(str, analysis["entity_nuclearities"])),
        "formula": parse_tag(text, "_chemical_formula_sum"),
    }


def main():
    cif_paths = sorted(CIF_DIR.glob("*.cif"))
    print(f"Reclassifying {len(cif_paths)} CIFs ...")

    rows = []
    failures = []
    t0 = time.time()
    for k, path in enumerate(cif_paths, 1):
        filename = path.name
        refcode = filename.split("_", 1)[1].removesuffix(".cif")
        manual_prefix = filename.split("_", 1)[0]
        try:
            res = reclassify(path)
            auto_prefix = res["prefix"]
        except Exception as exc:  # noqa: BLE001
            failures.append((filename, str(exc)))
            continue
        rows.append({
            "filename": filename,
            "refcode": refcode,
            "manual_prefix": manual_prefix,
            "auto_prefix": auto_prefix,
            "changed": manual_prefix != auto_prefix,
            "architecture": res["architecture"],
            "entity_nuclearities": res["entity_nuclearities"],
            "formula": res["formula"],
        })
        if k % 100 == 0 or k == len(cif_paths):
            print(f"  [{k}/{len(cif_paths)}] elapsed {time.time() - t0:6.1f} s, "
                  f"failures: {len(failures)}", flush=True)

    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- summary ----
    n_agree = sum(1 for r in rows if not r["changed"])
    print(f"\nManifest: {MANIFEST}")
    print(f"Classified: {len(rows)}  failed: {len(failures)}  "
          f"elapsed: {time.time() - t0:.1f} s")
    print(f"Agreement manual vs auto: {n_agree}/{len(rows)} "
          f"({100 * n_agree / len(rows):.1f}%)")
    if failures:
        for fn, reason in failures:
            print(f"  FAILED {fn}: {reason}")

    confusion = Counter((r["manual_prefix"], r["auto_prefix"]) for r in rows)
    print("\nConfusion tally (manual -> auto):")
    for (m, a), c in sorted(confusion.items()):
        print(f"  {m:>3} -> {a:<3} : {c}")

    print("\nAuto prefix distribution (all classified CIFs):",
          dict(sorted(Counter(r["auto_prefix"] for r in rows).items())))

    # ---- labels file for the 645 .dat files ----
    auto_by_refcode = {r["refcode"]: r["auto_prefix"] for r in rows}

    def dat_label(dat_path):
        name = dat_path.name  # e.g. 'p_X.dat' or '2_X.dat'
        refcode = name.split("_", 1)[1].removesuffix(".cif").removesuffix(".dat")
        prefix = auto_by_refcode[refcode]
        if prefix == "p":
            return "10"
        if prefix.isdigit() and int(prefix) >= 10:
            return "10"
        return prefix

    dat_files = sorted(glob.glob(str(PDF_DIR / "*.dat")))
    labels = [dat_label(Path(p)) for p in dat_files]
    LABELS_FILE.write_text("\n".join(labels) + "\n")
    dist = Counter(labels)
    print(f"\nLabels file: {LABELS_FILE}")
    print(f"Wrote {len(labels)} labels for {len(dat_files)} .dat files "
          f"(sorted-glob order)")
    print("Label distribution (645 .dat subset):")
    for key in sorted(dist, key=lambda x: (len(x), x)):
        print(f"  {key:>3}: {dist[key]}")
    print("Raw counter:", dict(sorted(dist.items())))


if __name__ == "__main__":
    main()
