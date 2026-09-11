#!/usr/bin/env python3
"""calculate_pdfs_dedup.py — PDF calculation with coincident-site dedup (2026-09-09).

Root cause of the unphysical PDFs (|G| up to ~2300, 141/6806 files): the CSD
ConQuest export lists multiple symmetry-equivalent copies of the same site as
separate _atom_site rows (labels like Nd1, Nd1N, Nd1J...) with slightly different
coordinate digits; pyobjcryst/diffpy apply all symmetry ops to every row without
merging the resulting coincident images, and the Debye sum then puts huge weight
on zero-distance pairs.

Fix implemented here, after loadStructure and before PDFCalculator:
  1. fold all sites into the unit cell (mod 1)
  2. union-merge sites closer than DEDUP_TOL Å — direct in-cell pairs AND pairs
     across the 26 neighboring lattice images (cKDTree)
  3. recompute the PDF on the deduplicated Structure

Calculator convention (user decision, 2026-09-09): ALL CSD structures are
periodic crystals, so PDFCalculator (periodic summation) is used for everything,
regardless of assigned nuclearity. DebyePDFCalculator remains the convention for
the finite CeO2-/Ce40-based cluster datasets only.

Also sanitizes the CIF text before parsing:
  * deuterium 'D' sites -> 'H' (recovers the 6 previously-crashed files)
  * 'Unknown'-element site rows dropped (they crash the parser)

Validated on: 2PADBAZ (3675->561 atoms, heavy-atom counts match C54H108Br6N6Nd6O54
x Z=3 exactly), 2ZERPAP, pBEWWAB01 (183->39 = formula count), pEBODUY; PDFCalculator
sanity: p_AKOROJ max|G| 5.4, 2_HICZEA 11.1 (old Debye pipeline: 39.2).

Idempotent: skips existing non-empty .dat unless --force.
Usage: python calculate_pdfs_dedup.py --cif-dir DIR --out-dir DIR [--workers 96] [--force]
"""
import argparse
import os
import re
import time
from collections import Counter
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

DEDUP_TOL = 0.05      # Å — coincident-image tolerance (real bonds are > 0.7 Å)
G_MAX_THRESHOLD = 100.0
QMIN, QMAX, RMAX = 0.3, 11.0, 20.0


def sanitize_cif_text(text):
    """D->H rename and Unknown-element row drop in the _atom_site loop.

    Returns (new_text, n_renamed, n_dropped). Operates on the loop whose header
    starts with _atom_site_label; rows are whitespace-split token lists.
    """
    lines = text.splitlines(keepends=True)
    out = []
    n_ren = n_drop = 0
    i = 0
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if s.startswith("_atom_site_label"):
            # collect this loop: header tags then rows
            tags = []
            j = i
            while j < len(lines) and lines[j].strip().startswith("_"):
                tags.append(lines[j].strip())
                j += 1
            out.extend(lines[i:j])
            type_i = tags.index("_atom_site_type_symbol") if "_atom_site_type_symbol" in tags else None
            label_i = 0  # _atom_site_label is first by construction
            while j < len(lines):
                row = lines[j].strip()
                if not row or row.startswith("_") or row.startswith("loop_") or row.startswith("#"):
                    break
                parts = row.split()
                sym = parts[type_i if type_i is not None else label_i]
                el = re.match(r"[A-Za-z]{1,2}", sym).group(0)
                el = el[0].upper() + el[1:].lower()
                if el == "D":
                    if type_i is not None:
                        parts[type_i] = "H"
                        out.append(" ".join(parts) + "\n")
                        n_ren += 1
                    else:
                        # diffpy derives element from label; D-prefixed label -> H
                        parts[label_i] = re.sub(r"^[Dd](?=[A-Z0-9])|^[Dd]$", "H", parts[label_i], count=1)
                        if parts[label_i][0] not in "DH":
                            parts[label_i] = "H" + parts[label_i]
                        out.append(" ".join(parts) + "\n")
                        n_ren += 1
                    j += 1
                    continue
                if el in ("Unknown", "Un", "Xx", "Q", "Qq"):
                    n_drop += 1
                    j += 1
                    continue
                out.append(lines[j])
                j += 1
            i = j
            continue
        out.append(ln)
        i += 1
    return "".join(out), n_ren, n_drop


def dedup_structure(st, tol=DEDUP_TOL):
    """Union-merge coincident sites (in-cell + 26 neighbor images); return new Structure."""
    from diffpy.structure import Structure
    from scipy.spatial import cKDTree
    import itertools

    n = len(st)
    xyz = np.mod(np.array([a.xyz for a in st]), 1.0)
    L = st.lattice
    cart = np.array([L.cartesian(f) for f in xyz])

    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    tree = cKDTree(cart)
    for i, j in tree.query_pairs(tol):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
    for s in itertools.product((-1, 0, 1), repeat=3):
        if s == (0, 0, 0):
            continue
        img = cKDTree(cart + L.cartesian(np.array(s)))
        for i, jl in enumerate(tree.query_ball_tree(img, tol)):
            for j in jl:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
    keep = sorted({find(i) for i in range(n)})
    return Structure([st[i] for i in keep], L), n, len(keep)


def process_one(args):
    cif_path, out_dir, force = args
    from pathlib import Path
    cif_path, out_dir = Path(cif_path), Path(out_dir)
    out = out_dir / (cif_path.stem + ".dat")
    if out.exists() and out.stat().st_size > 0 and not force:
        return ("exists", cif_path.name, None, None)
    try:
        import tempfile
        from diffpy.structure import loadStructure
        from diffpy.srreal.pdfcalculator import PDFCalculator

        raw = cif_path.read_text()
        text, n_ren, n_drop = sanitize_cif_text(raw)
        if text != raw:
            with tempfile.NamedTemporaryFile("w", suffix=".cif", delete=False) as tf:
                tf.write(text)
                tmp = tf.name
            try:
                st = loadStructure(tmp)
            finally:
                os.unlink(tmp)
        else:
            st = loadStructure(str(cif_path))

        st_dedup, n_before, n_after = dedup_structure(st)
        # The CSD export carries no ADPs; loadStructure leaves U=0 with
        # anisotropy=True, which makes PDFCalculator's peak engine yield
        # baseline-only output. A tiny isotropic U unblocks it without
        # meaningful peak broadening (validated: peak heights change <10%
        # vs Uiso=1e-6, and match Debye reference well).
        for a in st_dedup:
            a.anisotropy = False
            a.Uisoequiv = 1e-4
        dpc = PDFCalculator()
        dpc.qmax = QMAX
        dpc.rmax = RMAX
        r, g = dpc(st_dedup, qmin=QMIN)
        g = np.asarray(g)
        max_g = float(np.max(np.abs(g)))
        if max_g > G_MAX_THRESHOLD:
            return ("skipped", cif_path.name, max_g, (n_before, n_after))
        np.savetxt(out, np.column_stack([r, g]), header="r g")
        return ("ok", cif_path.name, max_g, (n_before, n_after))
    except Exception as exc:  # noqa: BLE001
        return ("failed", cif_path.name, f"{type(exc).__name__}: {exc}", None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cif-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    from pathlib import Path
    cif_dir, out_dir = Path(args.cif_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cifs = sorted(cif_dir.glob("*.cif"))
    print(f"total={len(cifs)} workers={args.workers} out={out_dir}", flush=True)

    t0 = time.time()
    with Pool(args.workers) as pool:
        results = pool.map(process_one,
                           [(str(c), str(out_dir), args.force) for c in cifs],
                           chunksize=8)
    stats = Counter(r[0] for r in results)
    print(f"elapsed={time.time()-t0:.0f}s {dict(stats)}", flush=True)

    dedup_counts = [r[3] for r in results if r[3] and r[0] == "ok"]
    changed = sum(1 for b, a in dedup_counts if a < b)
    tot_removed = sum(b - a for b, a in dedup_counts)
    print(f"dedup changed {changed} structures; {tot_removed} coincident sites removed total")

    report = Path(args.report) if args.report else out_dir.parent / f"dedup_calc_report_{out_dir.name}.txt"
    with open(report, "w") as fh:
        fh.write(f"# dedup PDF calc: cifs={len(cifs)} tol={DEDUP_TOL}A "
                 f"q=({QMIN},{QMAX}) rmax={RMAX}\n")
        fh.write(f"# {dict(stats)}; dedup changed {changed}, removed {tot_removed} sites\n\n")
        for status, name, extra, counts in results:
            if status in ("skipped", "failed"):
                fh.write(f"{status}: {name}  # {extra}\n")
            elif status == "ok" and counts and counts[0] != counts[1]:
                fh.write(f"dedup: {name}  # {counts[0]}->{counts[1]} atoms, max|G|={extra:.1f}\n")
    print(f"report -> {report}")
    print(f".dat files in {out_dir}: {len(sorted(out_dir.glob('*.dat')))}")


if __name__ == "__main__":
    main()
