#!/usr/bin/env python3
"""
CIF to PDF Utility — CIF validation, PDF calculation, and quality control.

Script equivalent of ``utils/cif-preparePDF.ipynb``.

Scans calculated PDFs from CSD .cif files for unreasonable G(r) values
(|G(r)| > threshold, default 100 Å⁻²).  Creates an exclusion list
(``labels/csd_excluded_cifs.txt``) and optionally recalculates PDFs for valid
CIF files.

Usage examples::

    # Scan existing CSD PDFs for unreasonable G(r) values
    python scripts/cif_prepare_pdf.py scan

    # Scan with custom threshold
    python scripts/cif_prepare_pdf.py scan --threshold 50.0

    # Recalculate PDFs from CIFs, skipping bad G(r) values
    python scripts/cif_prepare_pdf.py recalculate --qmax 11 --qmin 0.3 --rmax 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import get_path
from scripts.pipeline_utils import ensure_dir, print_section


def scan_pdfs(g_max_threshold: float = 100.0, save_exclusion: bool = True):
    """Scan CSD calculated PDFs for max |G(r)| > threshold.

    Ported verbatim from notebook cells 2-3 & 7.
    """
    cif_dir = get_path("csd_cifs")
    pdf_dir = get_path("csd_calculated_pdfs")

    print_section("CIF/PDF Quality Control — Scanning PDFs")
    print(f"CIF directory: {cif_dir}")
    print(f"PDF directory: {pdf_dir}")

    pdf_files = sorted(pdf_dir.glob("*.dat"))
    print(f"Found {len(pdf_files)} PDF files")
    print(f"Threshold: |G(r)| > {g_max_threshold}\n")

    problematic_files = []
    valid_files = []
    max_values = {}

    for pdf_file in pdf_files:
        try:
            data = np.loadtxt(pdf_file, skiprows=1)
            g = data[:, 1]
            max_g = float(np.max(np.abs(g)))
            max_values[pdf_file.name] = max_g

            if max_g > g_max_threshold:
                problematic_files.append((pdf_file.name, max_g))
            else:
                valid_files.append(pdf_file.name)
        except Exception as e:
            print(f"Error reading {pdf_file.name}: {e}")
            problematic_files.append((pdf_file.name, "READ_ERROR"))

    print(f"Valid PDFs:       {len(valid_files)}")
    print(f"Problematic PDFs: {len(problematic_files)}")

    if problematic_files:
        print("\n" + "=" * 70)
        print("PROBLEMATIC FILES - Unreasonable G(r) values detected")
        print("=" * 70)

        problematic_sorted = sorted(
            [p for p in problematic_files if isinstance(p[1], (int, float))],
            key=lambda x: x[1],
            reverse=True,
        )

        print(f"\n{'Filename':<40} {'Max |G(r)|':>15}")
        print("-" * 55)
        for filename, max_g in problematic_sorted:
            print(f"{filename:<40} {max_g:>15.1f}")

        problematic_cif_names = [
            f.replace(".dat", ".cif") for f, _ in problematic_files if isinstance(_, (int, float))
        ]
        print(f"\nCorresponding CIF files to exclude ({len(problematic_cif_names)} total):")
        for cif_name in problematic_cif_names:
            print(f"  {cif_name}")

        if save_exclusion:
            exclusion_list_path = get_path("labels") / "csd_excluded_cifs.txt"
            ensure_dir(exclusion_list_path.parent)

            with open(exclusion_list_path, "w") as f:
                f.write("# CIF files excluded due to unreasonable G(r) values\n")
                f.write(f"# Threshold: |G(r)| > {g_max_threshold}\n")
                f.write(f"# Total excluded: {len(problematic_cif_names)}\n#\n")
                for cif_name in problematic_cif_names:
                    dat_name = cif_name.replace(".cif", ".dat")
                    max_g = max_values.get(dat_name, "N/A")
                    if isinstance(max_g, (int, float)):
                        f.write(f"{cif_name}  # max|G(r)| = {max_g:.1f}\n")
                    else:
                        f.write(f"{cif_name}  # {max_g}\n")

            print(f"\nExclusion list saved to: {exclusion_list_path}")

    else:
        print("All PDFs have reasonable G(r) values!")

    return valid_files, problematic_files, max_values


def recalculate_pdfs(qmax: float = 11.0, qmin: float = 0.3, rmax: float = 20.0,
                     g_max_threshold: float = 100.0):
    """Recalculate PDFs from CIF files with G(r) threshold validation.

    Ported verbatim from notebook cell 8.
    """
    from diffpy.srreal.pdfcalculator import DebyePDFCalculator
    from pyobjcryst import loadCrystal

    cif_dir = get_path("csd_cifs")
    pdf_dir = ensure_dir(get_path("csd_calculated_pdfs"))

    cif_files = sorted(cif_dir.glob("*.cif"))
    print_section("CIF/PDF Quality Control — Recalculating PDFs")
    print(f"Processing {len(cif_files)} CIF files...")
    print(f"Threshold: |G(r)| <= {g_max_threshold}")
    print(f"Qmin={qmin}, Qmax={qmax}, Rmax={rmax}\n")

    successful = 0
    skipped_unreasonable = []
    failed = []

    for cif_path in cif_files:
        try:
            model = loadCrystal(str(cif_path))
            dpc = DebyePDFCalculator()
            dpc.qmax = qmax
            dpc.rmax = rmax
            r, g = dpc(model, qmin=qmin)

            max_g = float(np.max(np.abs(g)))
            if max_g > g_max_threshold:
                skipped_unreasonable.append((cif_path.name, max_g))
                continue

            output_path = pdf_dir / cif_path.name.replace(".cif", ".dat")
            datagcalc = np.column_stack([r, g])
            np.savetxt(output_path, datagcalc, header="r g")

            successful += 1
            if successful % 50 == 0:
                print(f"  Processed {successful} valid files...")
        except Exception as e:
            print(f"Failed: {cif_path.name} - {e}")
            failed.append((cif_path.name, str(e)))

    print("\n" + "=" * 60)
    print("RECALCULATION COMPLETE")
    print("=" * 60)
    print(f"Valid PDFs saved:     {successful}")
    print(f"Skipped (bad G(r)):   {len(skipped_unreasonable)}")
    print(f"Failed (errors):      {len(failed)}")

    if skipped_unreasonable:
        exclusion_list_path = get_path("labels") / "csd_excluded_cifs.txt"
        ensure_dir(exclusion_list_path.parent)
        with open(exclusion_list_path, "w") as f:
            f.write("# CIF files excluded due to unreasonable G(r) values\n")
            f.write(f"# Threshold: |G(r)| > {g_max_threshold}\n")
            f.write(f"# Total excluded: {len(skipped_unreasonable)}\n#\n")
            for cif_name, max_g in skipped_unreasonable:
                f.write(f"{cif_name}  # max|G(r)| = {max_g:.1f}\n")
        print(f"\nExclusion list updated: {exclusion_list_path}")


def main():
    parser = argparse.ArgumentParser(
        description="CIF/PDF quality control and calculation utility.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # scan
    p_scan = subparsers.add_parser("scan", help="Scan calculated PDFs for unreasonable G(r)")
    p_scan.add_argument("--threshold", type=float, default=100.0, help="Max |G(r)| threshold (default: 100)")
    p_scan.add_argument("--no-exclusion", action="store_true", help="Do not write csd_excluded_cifs.txt")

    # recalculate
    p_recalc = subparsers.add_parser("recalculate", help="Recalculate PDFs from CIFs")
    p_recalc.add_argument("--qmax", type=float, default=11.0)
    p_recalc.add_argument("--qmin", type=float, default=0.3)
    p_recalc.add_argument("--rmax", type=float, default=20.0)
    p_recalc.add_argument("--threshold", type=float, default=100.0)

    args = parser.parse_args()

    if args.command == "scan":
        scan_pdfs(g_max_threshold=args.threshold, save_exclusion=not args.no_exclusion)
    elif args.command == "recalculate":
        recalculate_pdfs(qmax=args.qmax, qmin=args.qmin, rmax=args.rmax, g_max_threshold=args.threshold)


if __name__ == "__main__":
    main()
