#!/usr/bin/env python3
"""
Stage 1: Create model polynuclear CexOy clusters from parent structures.

Script equivalent of ``complete workflow/1-create-clusters.ipynb``.

Generates .xyz files of clusters from a parent cluster structure (CeO2 or
Ce40).  The parent is "sliced" by randomly selecting a subset of metal
atoms (within a configurable nuclearity range) and retaining coordinated
oxygen atoms + the largest connected metal-metal fragment.

Usage examples::

    # Generate CeO2-based clusters (dataset A)
    python scripts/s1_create_clusters.py --parent ceo2 --n-structures 10000

    # Generate Ce40-based clusters (dataset B)
    python scripts/s1_create_clusters.py --parent ce40 --n-structures 10000

    # Force regeneration (overwrite existing files)
    python scripts/s1_create_clusters.py --parent ceo2 --force
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path so config is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import get_path
from scripts.pipeline_utils import ensure_dir, print_section, skip_if_exists


# ---------------------------------------------------------------------------
# Core functions (ported verbatim from notebook cells)
# ---------------------------------------------------------------------------
def format_xyz(xyz_file, allowed_atoms):
    """Read an xyz file and count the number of atoms of specified types."""
    atom_count = 0
    with open(xyz_file, "r") as f:
        lines = f.readlines()
    for line in lines[2:]:  # Skip first two lines (atom count and comment)
        fields = line.split()
        if fields and fields[0] in allowed_atoms:
            atom_count += 1
    print(f"Found {atom_count} {allowed_atoms} atoms in {xyz_file}")
    return atom_count


def structure_catalogue_maker(n_structures, n_atoms, lower_atom_number, higher_atom_number):
    """Create a shuffled catalogue of structure masks.

    Each entry is a list of length ``n_atoms + 1``: the first element is
    the atom count, the rest is a 0/1 mask indicating which metal atoms to
    keep.  The number of 1's (kept atoms) is uniformly random in
    [lower_atom_number, higher_atom_number].
    """
    print(f"Starting to make a structure catalogue with: {n_structures} structures.")
    print(f"The structures will have between {lower_atom_number} and {higher_atom_number} atoms.")
    structure_catalogue = []
    for _ in range(n_structures):
        one_count = random.randint(lower_atom_number, higher_atom_number)
        zero_count = n_atoms - one_count
        my_list = [0] * zero_count + [1] * one_count
        random.shuffle(my_list)
        my_list.insert(0, one_count)
        structure_catalogue.append(my_list)
    print("Permutations succeeded.")
    return structure_catalogue


def make_xyz_catalogue(catalogue, initial_xyz, output_dir, metal_atom, threshold_MO, threshold_MM):
    """Create xyz files for polynuclear clusters from a parent cluster structure.

    For each catalogue entry, the metal-atom mask is applied, then the
    largest connected metal-metal fragment (within ``threshold_MM``) is
    retained with all oxygens within ``threshold_MO`` of any retained metal.
    """
    with open(initial_xyz, "r") as f:
        lines = f.readlines()

    metal_lines = []
    metal_indices = []
    for i, line in enumerate(lines[2:]):
        fields = line.split()
        if fields and fields[0] == metal_atom:
            metal_lines.append(line)
            metal_indices.append(i)

    created_count = 0
    for k, entry in enumerate(catalogue):
        array = np.array(entry[1:])

        selected_metal_lines = []
        for i, line in enumerate(metal_lines):
            if i < len(array) and array[i] == 1:
                selected_metal_lines.append(line)

        metal_coordinates = []
        for line in selected_metal_lines:
            fields = line.split()
            if fields:
                metal_coordinates.append([float(fields[1]), float(fields[2]), float(fields[3])])

        metal_groups = []
        while metal_coordinates:
            metal_group = [metal_coordinates.pop()]
            for metal_coordinate in metal_coordinates:
                distance = np.linalg.norm(np.array(metal_group[0]) - np.array(metal_coordinate))
                if distance < threshold_MM:
                    metal_group.append(metal_coordinate)
            metal_groups.append(metal_group)

        final_result = []
        for element in metal_groups:
            found = False
            for result in final_result:
                if set(map(tuple, element)) & set(map(tuple, result)):
                    result.extend([e for e in element if e not in result])
                    found = True
                    break
            if not found:
                final_result.append(element)

        if metal_groups:
            largest_metal_group = max(final_result, key=lambda x: len(x))
            metal_count = len(largest_metal_group)
            output_file = output_dir / f"{metal_count}_{k}.xyz"
            with open(output_file, "w") as new_f:
                total_atoms = len(largest_metal_group)
                for line in lines[2:]:
                    fields = line.split()
                    if fields and fields[0] == "O":
                        O_coordinate = [float(fields[1]), float(fields[2]), float(fields[3])]
                        for metal_coordinate in largest_metal_group:
                            distance = np.linalg.norm(np.array(metal_coordinate) - np.array(O_coordinate))
                            if distance < threshold_MO:
                                total_atoms += 1
                                break
                new_f.write(f"{total_atoms}\n")
                new_f.write(lines[1])  # Comment line
                for metal_coordinate in largest_metal_group:
                    new_f.write(f"{metal_atom} " + " ".join([str(x) for x in metal_coordinate]) + "\n")
                for line in lines[2:]:
                    fields = line.split()
                    if fields and fields[0] == "O":
                        O_coordinate = [float(fields[1]), float(fields[2]), float(fields[3])]
                        for metal_coordinate in largest_metal_group:
                            distance = np.linalg.norm(np.array(metal_coordinate) - np.array(O_coordinate))
                            if distance < threshold_MO:
                                new_f.write(line)
                                break
            created_count += 1

    print(f"Created {created_count} xyz files in {output_dir}")


def verify_clusters(output_dir):
    """Print distribution of created clusters by nuclearity."""
    from collections import Counter

    files = [f for f in os.listdir(output_dir) if f.endswith(".xyz") and "_" in f and f.split("_")[0].isdigit()]
    print(f"Total xyz files created: {len(files)}")
    nuclearities = [int(f.split("_")[0]) for f in files]
    counts = Counter(nuclearities)
    print("\nDistribution by nuclearity:")
    for nuc in sorted(counts.keys()):
        print(f"  {nuc} Ce atoms: {counts[nuc]} structures")
    return counts


# ---------------------------------------------------------------------------
# Parent-structure configuration
# ---------------------------------------------------------------------------
PARENT_CONFIG = {
    "ceo2": {
        "clusters_path_key": "ceo2_clusters",        # input dir with ceo2.xyz
        "output_path_key": "ceo2_model_clusters",     # output dir
        "starting_model": "ceo2.xyz",
        "lower_atom_number": 2,
        "higher_atom_number": 9,
    },
    "ce40": {
        "clusters_path_key": "ce_clusters",
        "output_path_key": "ce_model_clusters",
        "starting_model": "ce40.xyz",
        "lower_atom_number": 2,
        "higher_atom_number": 9,
    },
}


def run(parent: str, n_structures: int = 10000, threshold_MO: float = 3.0,
        threshold_MM: float = 5.2, force: bool = False, seed: int | None = None):
    """Generate clusters for one parent structure."""
    parent = parent.lower()
    if parent not in PARENT_CONFIG:
        valid = ", ".join(sorted(PARENT_CONFIG))
        raise ValueError(f"Unknown parent '{parent}'. Valid: {valid}")

    pcfg = PARENT_CONFIG[parent]
    clusters_path = get_path(pcfg["clusters_path_key"])
    output_path = get_path(pcfg["output_path_key"])

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    print_section(f"Stage 1: Create clusters — {parent}")
    print(f"Working directory: {clusters_path}")
    print(f"Output directory:  {output_path}")
    print(f"Starting model:    {pcfg['starting_model']}")
    print(f"N structures:      {n_structures}")
    print(f"Thresholds:        MO={threshold_MO}, MM={threshold_MM}")

    input_file = clusters_path / pcfg["starting_model"]
    if not input_file.exists():
        print(f"\nERROR: Input file not found: {input_file}")
        print("Download data from Zenodo and set DATA_ROOT in config.py.")
        sys.exit(1)

    ensure_dir(output_path)

    existing = list(output_path.glob("*.xyz"))
    if existing and not force:
        print(f"\nFound {len(existing)} existing .xyz files. Use --force to regenerate.")
        verify_clusters(output_path)
        return

    # Run generation (note: notebook uses os.chdir; we pass absolute paths instead)
    os.chdir(str(clusters_path))  # Match notebook behaviour for relative file access

    n_metal = format_xyz(pcfg["starting_model"], ["Ce"])
    catalogue = structure_catalogue_maker(
        n_structures,
        n_atoms=n_metal,
        lower_atom_number=pcfg["lower_atom_number"],
        higher_atom_number=pcfg["higher_atom_number"],
    )
    make_xyz_catalogue(
        catalogue=catalogue,
        initial_xyz=pcfg["starting_model"],
        output_dir=output_path,
        metal_atom="Ce",
        threshold_MO=threshold_MO,
        threshold_MM=threshold_MM,
    )

    verify_clusters(output_path)
    print(f"\nDone. Clusters saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Create model clusters from parent structures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--parent", required=True, choices=["ceo2", "ce40"],
        help="Parent structure to generate clusters from.",
    )
    parser.add_argument(
        "--n-structures", type=int, default=10000,
        help="Number of cluster structures to generate (default: 10000).",
    )
    parser.add_argument(
        "--threshold-mo", type=float, default=3.0, dest="threshold_MO",
        help="Maximum metal-O distance for coordination (default: 3.0 Å).",
    )
    parser.add_argument(
        "--threshold-mm", type=float, default=5.2, dest="threshold_MM",
        help="Maximum metal-metal distance for connectivity (default: 5.2 Å).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate even if output files already exist.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()
    run(
        parent=args.parent,
        n_structures=args.n_structures,
        threshold_MO=args.threshold_MO,
        threshold_MM=args.threshold_MM,
        force=args.force,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
