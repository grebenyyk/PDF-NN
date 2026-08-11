#!/usr/bin/env python3
"""
Stage 2: Calculate PDFs from structures and prepare experimental PDFs.

Script equivalent of ``complete workflow/2-prepare-PDF.ipynb``.

Three sub-commands:
  - ``ceo2``: Calculate PDFs from CeO2-based cluster .xyz files (dataset A).
  - ``ce40``: Calculate PDFs from Ce40-based cluster .xyz files (dataset B).
  - ``experimental``: Interpolate experimental .gr files to the model grid.

CSD PDF calculation from .cif files is handled by ``scripts/cif_prepare_pdf.py``
which includes quality control (G(r) validation).

Usage examples::

    python scripts/s2_prepare_pdf.py ceo2 --qmax 11 --qmin 0.3 --rmax 20
    python scripts/s2_prepare_pdf.py ce40 --qmax 11 --qmin 0.3 --rmax 20
    python scripts/s2_prepare_pdf.py experimental
    python scripts/s2_prepare_pdf.py ceo2 --force  # recalculate all
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import get_path
from scripts.pipeline_utils import ensure_dir, print_section


# ---------------------------------------------------------------------------
# xyz → PDF calculation  (notebook 2-prepare-PDF cells 3-5, 7-9)
# ---------------------------------------------------------------------------
def fix_xyz_atom_count(filepath):
    """Rewrite the first line of an xyz file with the correct atom count.

    The cluster-creation step (stage 1) may leave the count inaccurate;
    this recalculates it from the actual number of data lines.
    """
    with open(filepath, "r") as f:
        contents = f.readlines()
        new_first_line = str(len(contents) - 2) + "\n"
        contents[0] = new_first_line
    with open(filepath, "w") as f:
        f.writelines(contents)


def calculate_pdfs_from_xyz(input_dir, output_dir, qmax=11, qmin=0.3, rmax=20,
                            force=False):
    """Calculate PDFs for all .xyz files in *input_dir*.

    Uses DiffPy's DebyePDFCalculator — identical parameters to the notebook.
    Skips files whose output .dat already exists unless *force* is True.
    """
    from diffpy.Structure import loadStructure
    from diffpy.srreal.pdfcalculator import DebyePDFCalculator

    input_dir = Path(input_dir)
    output_dir = ensure_dir(output_dir)

    xyz_files = sorted(input_dir.glob("*.xyz"))
    if not xyz_files:
        print(f"No .xyz files found in {input_dir}")
        return

    print(f"Found {len(xyz_files)} .xyz files in {input_dir}")

    # Fix atom counts (notebook does this in a separate cell)
    print("Fixing atom counts...")
    for f in xyz_files:
        fix_xyz_atom_count(f)

    skipped = 0
    calculated = 0
    for i, f in enumerate(xyz_files):
        output_file = output_dir / f.name.replace(".xyz", ".dat")
        if output_file.exists() and not force:
            skipped += 1
            continue
        model = loadStructure(str(f))
        dpc = DebyePDFCalculator()
        dpc.qmax = qmax
        dpc.rmax = rmax
        r, g = dpc(model, qmin=qmin)
        datagcalc = np.column_stack([r, g])
        np.savetxt(output_file, datagcalc, header="r g")
        calculated += 1
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(xyz_files)}...")

    print(f"\nCalculated: {calculated}, Skipped (existing): {skipped}")
    print(f"PDFs saved to: {output_dir}")


# ---------------------------------------------------------------------------
# Experimental PDF interpolation  (notebook 2-prepare-PDF cells 11-12)
# ---------------------------------------------------------------------------
def process_experimental_pdfs(input_dir, output_dir, r_min=2.0, r_max=12.0,
                              r_step=0.01, force=False):
    """Interpolate experimental .gr files to the model grid.

    Experimental data may have irregular spacing; this interpolates onto
    a uniform grid matching the calculated PDFs.
    """
    input_dir = Path(input_dir)
    output_dir = ensure_dir(output_dir)

    gr_files = sorted(input_dir.glob("*.gr"))
    if not gr_files:
        print(f"No .gr files found in {input_dir}")
        return

    print(f"Found {len(gr_files)} .gr files to process")
    print(f"Processed files will be saved to: {output_dir}")

    x = np.arange(r_min, r_max + r_step, r_step)
    processed = 0
    skipped = 0

    for f in gr_files:
        output_file = output_dir / f.name.replace(".gr", "_processed.gr")
        if output_file.exists() and not force:
            skipped += 1
            continue
        # Notebook reads with header=1, delim_whitespace; using sep=r'\s+'
        data = pd.read_csv(f, header=1, sep=r"\s+")
        df = pd.DataFrame(data)
        df.columns = ["r", "g(r)"]
        df = df.loc[(df["r"] >= r_min) & (df["r"] <= r_max)]
        df["interpolated_data"] = np.interp(x, df["r"], df["g(r)"])
        df = df.drop(columns=["g(r)"])
        df.rename(columns={"interpolated_data": "g(r)"}, inplace=True)
        df.to_csv(output_file, index=False, sep="\t")
        processed += 1

    print(f"\nProcessed: {processed}, Skipped (existing): {skipped}")
    print(f"Processed experimental PDFs saved to: {output_dir}")


# ---------------------------------------------------------------------------
# Sub-command dispatch
# ---------------------------------------------------------------------------
def cmd_ceo2(args):
    print_section("Stage 2: Calculate PDFs — CeO2 clusters (Dataset A)")
    input_dir = get_path("ceo2_model_clusters")
    output_dir = get_path("ceo2_calculated_pdfs")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Qmax={args.qmax}, Qmin={args.qmin}, Rmax={args.rmax}")
    calculate_pdfs_from_xyz(input_dir, output_dir, qmax=args.qmax, qmin=args.qmin,
                            rmax=args.rmax, force=args.force)


def cmd_ce40(args):
    print_section("Stage 2: Calculate PDFs — Ce40 clusters (Dataset B)")
    input_dir = get_path("ce_model_clusters")
    output_dir = get_path("ce_calculated_pdfs")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Qmax={args.qmax}, Qmin={args.qmin}, Rmax={args.rmax}")
    calculate_pdfs_from_xyz(input_dir, output_dir, qmax=args.qmax, qmin=args.qmin,
                            rmax=args.rmax, force=args.force)


def cmd_experimental(args):
    print_section("Stage 2: Prepare experimental PDFs")
    input_dir = get_path("experimental_pdfs")
    output_dir = get_path("experimental_processed")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    process_experimental_pdfs(input_dir, output_dir, force=args.force)


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: Calculate PDFs and prepare experimental data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="dataset", required=True,
                                       help="Dataset to process")

    # CeO2
    p_ceo2 = subparsers.add_parser("ceo2", help="CeO2-based clusters (Dataset A)")
    p_ceo2.add_argument("--qmax", type=float, default=11, help="Qmax for PDF calculation (default: 11)")
    p_ceo2.add_argument("--qmin", type=float, default=0.3, help="Qmin for PDF calculation (default: 0.3)")
    p_ceo2.add_argument("--rmax", type=float, default=20, help="Rmax for PDF calculation (default: 20)")
    p_ceo2.add_argument("--force", action="store_true", help="Recalculate existing PDFs")
    p_ceo2.set_defaults(func=cmd_ceo2)

    # Ce40
    p_ce40 = subparsers.add_parser("ce40", help="Ce40-based clusters (Dataset B)")
    p_ce40.add_argument("--qmax", type=float, default=11, help="Qmax for PDF calculation (default: 11)")
    p_ce40.add_argument("--qmin", type=float, default=0.3, help="Qmin for PDF calculation (default: 0.3)")
    p_ce40.add_argument("--rmax", type=float, default=20, help="Rmax for PDF calculation (default: 20)")
    p_ce40.add_argument("--force", action="store_true", help="Recalculate existing PDFs")
    p_ce40.set_defaults(func=cmd_ce40)

    # Experimental
    p_exp = subparsers.add_parser("experimental", help="Interpolate experimental .gr files")
    p_exp.add_argument("--force", action="store_true", help="Reprocess existing files")
    p_exp.set_defaults(func=cmd_experimental)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
