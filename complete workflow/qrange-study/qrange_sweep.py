#!/usr/bin/env python3
"""
Q-range sweep driver for the PDF-NN qrange study.

Replaces the manual PAIR_INDEX workflow: instead of editing PAIR_INDEX at the
top of each notebook and re-running it 5 times, this script automates the full
sweep over all (qmin, qmax) pairs defined in config.QRANGE_PAIRS.

Usage
-----
Run from the project root (the directory containing config.py):

    python "complete workflow/qrange-study/qrange_sweep.py" prepare
    python "complete workflow/qrange-study/qrange_sweep.py" train --dataset all
    python "complete workflow/qrange-study/qrange_sweep.py" compare
    python "complete workflow/qrange-study/qrange_sweep.py" all

Or from inside the qrange-study directory:

    python qrange_sweep.py prepare
    python qrange_sweep.py train --dataset ceo2
    python qrange_sweep.py compare

Commands
--------
prepare   Recalculate cluster + CSD PDFs for all pairs (idempotent: skips
          pairs whose output directories already contain .dat files).
train     Train models for each pair.  --dataset selects which of
          ceo2 / ce40 / csd / all to train (default: all).
compare   Aggregate metrics across all completed pairs and produce the
          comparison table + multi-panel figure.
all       prepare -> train -> compare, in sequence.

Flags
-----
--pairs INDEX [INDEX ...]   Only process the given pair indices (0-based into
                            QRANGE_PAIRS).  Useful for adding one new pair
                            without re-running everything.
--dry-run                   For 'prepare': estimate disk and report what would
                            be done without writing anything.
--force                     Recompute even if outputs already exist.

Design notes
------------
- PDF computation uses diffpy.srreal DebyePDFCalculator (Debye formula route),
  matching the notebooks exactly.
- All output paths go through config.qrange_paths() so they are identical to
  what the notebooks produce — the notebooks and this script are
  interchangeable.
- The script is importable: ``from qrange_sweep import sweep_prepare`` etc.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: find config.py at the project root.
# ---------------------------------------------------------------------------
# When run as a script, __file__ is .../complete workflow/qrange-study/qrange_sweep.py
# Project root is two levels up.
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent  # complete workflow/qrange-study -> project root

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config
from config import QRANGE_PAIRS, qrange_tag, qrange_paths


# ---------------------------------------------------------------------------
# Constants matching the notebooks
# ---------------------------------------------------------------------------
RMAX = 20          # PDF calculation rmax (matches all notebooks)
RMIN = 0           # PDF calculation rmin
RSTEP = 0.01       # PDF calculation rstep (implicit in DebyePDFCalculator)
G_MAX_THRESHOLD = 100  # CSD PDF sanity threshold (matches cif-preparePDF notebook)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _select_pairs(indices):
    """Return list of (index, (qmin, qmax)) for the given indices."""
    pairs = []
    for i in indices:
        if i < 0 or i >= len(QRANGE_PAIRS):
            raise ValueError(f"Pair index {i} out of range (0..{len(QRANGE_PAIRS) - 1})")
        pairs.append((i, QRANGE_PAIRS[i]))
    return pairs


def _pair_has_pdfs(qmin, qmax, kind="ceo2"):
    """Check if a pair's PDF output directory already has .dat files."""
    paths = qrange_paths(qmin, qmax)
    key_map = {"ceo2": "ceo2_calculated_pdfs",
               "ce40": "ce_calculated_pdfs",
               "csd": "csd_calculated_pdfs"}
    d = paths[key_map[kind]]
    return d.exists() and len(list(d.glob("*.dat"))) > 0


def _count_input_files(kind):
    """Count available input structure files for each cluster type."""
    if kind == "ceo2":
        d = config.get_path("ceo2_model_clusters")
        return len(glob.glob(str(d / "*.xyz"))) if d.exists() else 0
    elif kind == "ce40":
        d = config.get_path("ce_model_clusters")
        return len(glob.glob(str(d / "*.xyz"))) if d.exists() else 0
    elif kind == "csd":
        d = config.get_path("csd_cifs")
        return len(glob.glob(str(d / "*.cif"))) if d.exists() else 0
    return 0


# ---------------------------------------------------------------------------
# PDF recalculation (prepare step)
# ---------------------------------------------------------------------------
def _calc_pdf_debye(structure, qmin, qmax):
    """Calculate PDF via DebyePDFCalculator, matching the notebooks."""
    from diffpy.srreal.pdfcalculator import DebyePDFCalculator
    import numpy as np

    dpc = DebyePDFCalculator()
    dpc.qmax = qmax
    dpc.rmax = RMAX
    r, g = dpc(structure, qmin=qmin)
    return np.column_stack([r, g])


def _prepare_clusters(qmin, qmax, kind="ceo2", force=False, dry_run=False):
    """Recalculate cluster PDFs for one pair (CeO2 or Ce40)."""
    kind_long = "CeO2" if kind == "ceo2" else "Ce40"
    paths = qrange_paths(qmin, qmax)
    out_key = "ceo2_calculated_pdfs" if kind == "ceo2" else "ce_calculated_pdfs"
    in_key = "ceo2_model_clusters" if kind == "ceo2" else "ce_model_clusters"

    in_dir = paths[in_key]
    out_dir = paths[out_key]

    if not in_dir.exists():
        print(f"  [{kind_long}] SKIP — input dir not found: {in_dir}")
        return 0

    files = sorted(in_dir.glob("*.xyz"))
    if not files:
        print(f"  [{kind_long}] SKIP — no .xyz files in {in_dir}")
        return 0

    if not force and _pair_has_pdfs(qmin, qmax, kind=kind):
        n_existing = len(list(out_dir.glob("*.dat")))
        print(f"  [{kind_long}] SKIP — {n_existing} .dat files already exist "
              f"(use --force to recompute)")
        return n_existing

    if dry_run:
        est_mb = len(files) * 2001 * 16 / 1e6  # ~2001 rows × ~16 bytes/row
        print(f"  [{kind_long}] DRY-RUN — would compute {len(files)} PDFs "
              f"(~{est_mb:.0f} MB) -> {out_dir}")
        return 0

    from diffpy.Structure import loadStructure
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for j, f in enumerate(files):
        model = loadStructure(str(f))
        data = _calc_pdf_debye(model, qmin, qmax)
        out_file = out_dir / f.name.replace(".xyz", ".dat")
        np.savetxt(out_file, data, header="r g")
        if (j + 1) % 100 == 0:
            print(f"    [{kind_long}] {j + 1}/{len(files)}...")

    dt = time.time() - t0
    print(f"  [{kind_long}] Wrote {len(files)} PDFs to {out_dir} ({dt:.0f}s)")
    return len(files)


def _prepare_csd(qmin, qmax, force=False, dry_run=False):
    """Recalculate CSD PDFs for one pair."""
    paths = qrange_paths(qmin, qmax)
    cif_dir = paths["csd_cifs"]
    out_dir = paths["csd_calculated_pdfs"]
    labels_dir = paths["labels"]
    tag = qrange_tag(qmin, qmax)

    if not cif_dir.exists():
        print(f"  [CSD] SKIP — CIF dir not found: {cif_dir}")
        return 0

    cifs = sorted(cif_dir.glob("*.cif"))
    if not cifs:
        print(f"  [CSD] SKIP — no .cif files in {cif_dir}")
        return 0

    if not force and _pair_has_pdfs(qmin, qmax, kind="csd"):
        n_existing = len(list(out_dir.glob("*.dat")))
        print(f"  [CSD] SKIP — {n_existing} .dat files already exist "
              f"(use --force to recompute)")
        return n_existing

    if dry_run:
        est_mb = len(cifs) * 2001 * 16 / 1e6
        print(f"  [CSD] DRY-RUN — would compute {len(cifs)} PDFs "
              f"(~{est_mb:.0f} MB) -> {out_dir}")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    from diffpy.Structure import loadStructure
    try:
        from pyobjcryst import loadCrystal
    except ImportError:
        loadCrystal = None

    t0 = time.time()
    successful = 0
    skipped = []
    failed = []

    for cif_path in cifs:
        try:
            if loadCrystal is not None:
                model = loadCrystal(str(cif_path))
            else:
                model = loadStructure(str(cif_path))

            data = _calc_pdf_debye(model, qmin, qmax)
            max_g = float(np.max(np.abs(data[:, 1])))
            if max_g > G_MAX_THRESHOLD:
                skipped.append((cif_path.name, max_g))
                continue

            out_file = out_dir / cif_path.name.replace(".cif", ".dat")
            np.savetxt(out_file, data, header="r g")
            successful += 1

            if successful % 50 == 0:
                print(f"    [CSD] {successful} valid PDFs...")

        except Exception as e:
            failed.append((cif_path.name, str(e)))

    # Write exclusion list (matches cif-preparePDF_qrange.ipynb)
    if skipped:
        excl_path = labels_dir / f"csd_excluded_cifs_{tag}.txt"
        with open(excl_path, "w") as f:
            f.write("# CIF files excluded due to unreasonable G(r) values\n")
            f.write(f"# Threshold: |G(r)| > {G_MAX_THRESHOLD}\n")
            f.write(f"# Total excluded: {len(skipped)}\n#\n")
            for name, max_g in sorted(skipped, key=lambda x: x[1], reverse=True):
                f.write(f"{name}  # max|G(r)| = {max_g:.1f}\n")

    dt = time.time() - t0
    print(f"  [CSD] Wrote {successful} PDFs ({len(skipped)} skipped, "
          f"{len(failed)} failed) to {out_dir} ({dt:.0f}s)")
    return successful


def sweep_prepare(pairs_indices=None, force=False, dry_run=False):
    """Recalculate all PDFs for all (or selected) pairs."""
    if pairs_indices is None:
        pairs_indices = list(range(len(QRANGE_PAIRS)))
    pairs = _select_pairs(pairs_indices)

    print("=" * 70)
    print(f"SWEEP PREPARE — {len(pairs)} pair(s)")
    if dry_run:
        print("[DRY RUN — no files will be written]")
    print("=" * 70)

    for idx, (qmin, qmax) in pairs:
        tag = qrange_tag(qmin, qmax)
        print(f"\n--- Pair {idx}: qmin={qmin}, qmax={qmax}  [{tag}] ---")
        _prepare_clusters(qmin, qmax, kind="ceo2", force=force, dry_run=dry_run)
        _prepare_clusters(qmin, qmax, kind="ce40", force=force, dry_run=dry_run)
        _prepare_csd(qmin, qmax, force=force, dry_run=dry_run)

    print("\nPREPARE DONE.")


# ---------------------------------------------------------------------------
# Training (train step)
# ---------------------------------------------------------------------------
def sweep_train(pairs_indices=None, dataset="all"):
    """
    Train models for all (or selected) pairs.

    This is a thin orchestrator — the heavy training logic lives in the
    notebooks (3A/3B/3C).  In a notebook-driven workflow, this function
    prints the exact PAIR_INDEX to set for each pair and each notebook, and
    verifies that the required PDFs exist.

    For automated (headless) training, use papermill or nbconvert to execute
    the notebooks programmatically.  See README.md for the pattern.
    """
    if pairs_indices is None:
        pairs_indices = list(range(len(QRANGE_PAIRS)))
    pairs = _select_pairs(pairs_indices)

    datasets_map = {
        "ceo2": ("3A-train-model-ceo2_qrange.ipynb", "ceo2"),
        "ce40": ("3B-train-model-ce40_qrange.ipynb", "ce40"),
        "csd":  ("3C-train-model-CSD_qrange.ipynb", "csd"),
    }

    if dataset == "all":
        to_run = list(datasets_map.keys())
    else:
        if dataset not in datasets_map:
            raise ValueError(f"Unknown dataset '{dataset}'. Choose from: {list(datasets_map)}")
        to_run = [dataset]

    print("=" * 70)
    print(f"SWEEP TRAIN — {len(pairs)} pair(s) × {len(to_run)} dataset(s)")
    print("=" * 70)

    for idx, (qmin, qmax) in pairs:
        tag = qrange_tag(qmin, qmax)
        print(f"\n--- Pair {idx}: qmin={qmin}, qmax={qmax}  [{tag}] ---")

        for ds in to_run:
            nb_name, kind = datasets_map[ds]
            pdfs_ready = _pair_has_pdfs(qmin, qmax, kind=kind)

            results_dir = qrange_paths(qmin, qmax)["results"] / kind
            if pdfs_ready and results_dir.exists() and any(results_dir.rglob("*.csv")):
                print(f"  [{ds}] Results already exist in {results_dir} "
                      f"(remove to retrain)")

            if not pdfs_ready:
                print(f"  [{ds}] WARNING — PDFs not found for {tag}. "
                      f"Run 'prepare' first.")
            print(f"  [{ds}] -> Set PAIR_INDEX={idx} in {nb_name} and run.")
            print(f"           Or use papermill:")
            print(f"           papermill {nb_name} - -p PAIR_INDEX {idx} "
                  f"| jupyter nbconvert --to notebook --execute /dev/stdin")

    print("\nTRAIN instructions printed.")
    print("For fully automated training, install papermill:")
    print("  pip install papermill")
    print("Then run:")
    print('  for i in $(seq 0 %d); do' % (len(QRANGE_PAIRS) - 1))
    for ds, (nb_name, _) in datasets_map.items():
        if ds in to_run:
            print(f'    papermill "{nb_name}" output_{ds}_pair$i.ipynb -p PAIR_INDEX $i')
    print("  done")


# ---------------------------------------------------------------------------
# Compare step — delegates to qrange_compare module
# ---------------------------------------------------------------------------
def sweep_compare(pairs_indices=None):
    """Run the aggregated comparison across all completed pairs."""
    from qrange_compare import run_comparison

    if pairs_indices is None:
        pairs_indices = list(range(len(QRANGE_PAIRS)))
    pairs = _select_pairs(pairs_indices)
    pair_list = [(qmin, qmax) for _, (qmin, qmax) in pairs]

    print("=" * 70)
    print(f"SWEEP COMPARE — {len(pair_list)} pair(s)")
    print("=" * 70)

    run_comparison(pair_list)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Q-range sweep driver for PDF-NN qrange study.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "command",
        choices=["prepare", "train", "compare", "all"],
        help="What to do.",
    )
    parser.add_argument(
        "--dataset",
        default="all",
        choices=["ceo2", "ce40", "csd", "all"],
        help="Dataset to train (default: all). Only for 'train'.",
    )
    parser.add_argument(
        "--pairs",
        type=int,
        nargs="+",
        default=None,
        help="Pair indices (0-based into QRANGE_PAIRS) to process.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="For 'prepare': report what would be done without writing.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if outputs already exist.",
    )

    args = parser.parse_args()

    if args.command == "prepare":
        sweep_prepare(args.pairs, force=args.force, dry_run=args.dry_run)
    elif args.command == "train":
        sweep_train(args.pairs, dataset=args.dataset)
    elif args.command == "compare":
        sweep_compare(args.pairs)
    elif args.command == "all":
        sweep_prepare(args.pairs, force=args.force, dry_run=args.dry_run)
        sweep_train(args.pairs, dataset=args.dataset)
        sweep_compare(args.pairs)


if __name__ == "__main__":
    main()
