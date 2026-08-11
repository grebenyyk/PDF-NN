#!/usr/bin/env python3
"""
Convenience runner: run the full PDF-NN pipeline end-to-end for a given dataset.

Usage examples::

    # Run stages 1, 2, and 3 for CeO2
    python scripts/run_pipeline.py --dataset ceo2

    # Run CSD pipeline (skips stage 1 since CSD structures come from CIFs)
    python scripts/run_pipeline.py --dataset csd

    # Force regeneration of all intermediate files
    python scripts/run_pipeline.py --dataset ceo2 --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.pipeline_utils import get_dataset_config, print_section


def run_pipeline(dataset: str, force: bool = False, skip_tuning: bool = False, hps_json: str | None = None):
    dataset = dataset.lower()
    config = get_dataset_config(dataset)

    print_section(f"PDF-NN End-to-End Pipeline: {config['display_name']}")

    # Stage 1: Create clusters (only for ceo2 and ce40)
    if dataset in ("ceo2", "ce40"):
        print_section("STAGE 1: Create Clusters")
        from scripts.s1_create_clusters import run as run_s1
        run_s1(parent=dataset, force=force)
    else:
        print_section("STAGE 1: Skipped (CSD structures come from CIF files)")

    # Stage 2: Calculate PDFs
    print_section("STAGE 2: Calculate PDFs")
    from scripts.s2_prepare_pdf import calculate_pdfs_from_xyz, process_experimental_pdfs
    from config import get_path

    if dataset in ("ceo2", "ce40"):
        parent_map = {"ceo2": ("ceo2_model_clusters", "ceo2_calculated_pdfs"),
                      "ce40": ("ce_model_clusters", "ce_calculated_pdfs")}
        inp_key, out_key = parent_map[dataset]
        calculate_pdfs_from_xyz(get_path(inp_key), get_path(out_key), force=force)
    elif dataset == "csd":
        from scripts.cif_prepare_pdf import scan_pdfs
        scan_pdfs()

    # Stage 3: Train model
    print_section("STAGE 3: Train Model")
    from scripts.s3_train_model import run as run_s3
    run_s3(dataset=dataset, skip_tuning=skip_tuning, hps_json=hps_json)

    print_section("PIPELINE COMPLETE")
    print(f"Run 'python scripts/s4_analyze_model.py --dataset {dataset} --model <model_path>' to analyze results.")


def main():
    parser = argparse.ArgumentParser(
        description="Run the end-to-end PDF-NN pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset", required=True, choices=["ceo2", "ce40", "csd"])
    parser.add_argument("--force", action="store_true", help="Force recalculation of intermediate files")
    parser.add_argument("--skip-tuning", action="store_true", help="Skip HP tuning in Stage 3")
    parser.add_argument("--hps", type=str, default=None, help="Hyperparameters JSON for Stage 3")
    args = parser.parse_args()

    run_pipeline(
        dataset=args.dataset,
        force=args.force,
        skip_tuning=args.skip_tuning,
        hps_json=args.hps,
    )


if __name__ == "__main__":
    main()
