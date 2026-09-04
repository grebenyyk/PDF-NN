#!/usr/bin/env python3
"""Generate whole-PDF neural-network training notebooks for all q ranges."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DATASETS = {
    "ceo2": {
        "display_name": "CeO2",
        "template": "3A-train-model-ceo2_qrange.ipynb",
        "filename_prefix": "3A-train-model-ceo2",
        "pdf_path_key": "ceo2_calculated_pdfs",
    },
    "ce40": {
        "display_name": "Ce40-based",
        "template": "3B-train-model-ce40_qrange.ipynb",
        "filename_prefix": "3B-train-model-ce40",
        "pdf_path_key": "ce_calculated_pdfs",
    },
}
PARAMETER_PATTERN = re.compile(
    r"^PAIR_INDEX\s*=.*$\n^QMIN,\s*QMAX\s*=\s*QRANGE_PAIRS\[PAIR_INDEX\]$",
    re.MULTILINE,
)

sys.path.insert(0, str(PROJECT_ROOT))
from config import BASELINE_QRANGE, QRANGE_PAIRS, qrange_tag  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create fixed, output-free whole-PDF training notebooks for the "
            "published baseline and configured q ranges."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=("all", *sorted(DATASETS)),
        default="all",
        help="Dataset notebooks to generate (default: all).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=SCRIPT_DIR / "generated",
        help="Root containing the ceo2/ and ce40/ output folders.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="TAG",
        help="Generate only selected tags, for example q0.3-11 q0-11.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing generated whole-PDF notebooks.",
    )
    return parser.parse_args()


def source_text(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def set_source(cell: dict[str, Any], source: str) -> None:
    cell["source"] = source.splitlines(keepends=True)


def available_pairs() -> list[tuple[int, float, float, str]]:
    configured_pairs = [(-1, BASELINE_QRANGE), *enumerate(QRANGE_PAIRS)]
    pairs = []
    seen_tags = set()
    for pair_index, (qmin, qmax) in configured_pairs:
        tag = qrange_tag(qmin, qmax)
        if tag not in seen_tags:
            pairs.append((pair_index, float(qmin), float(qmax), tag))
            seen_tags.add(tag)
    return pairs


def selected_pairs(only: list[str] | None) -> list[tuple[int, float, float, str]]:
    pairs = available_pairs()
    if not only:
        return pairs

    requested = set(only)
    known = {tag for _, _, _, tag in pairs}
    unknown = sorted(requested - known)
    if unknown:
        raise ValueError(
            f"Unknown q-range tag(s): {', '.join(unknown)}. "
            f"Available tags: {', '.join(sorted(known))}"
        )
    return [pair for pair in pairs if pair[3] in requested]


def normalize_notebook(notebook: dict[str, Any]) -> None:
    for cell_index, cell in enumerate(notebook["cells"], start=1):
        cell_id = f"cell-{cell_index:02d}"
        cell["id"] = cell_id
        metadata = cell.setdefault("metadata", {})
        metadata["id"] = cell_id
        metadata["language"] = "python" if cell["cell_type"] == "code" else "markdown"
        metadata.pop("execution", None)
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []


def parameterize_notebook(
    template: dict[str, Any],
    pair_index: int,
    qmin: float,
    qmax: float,
    tag: str,
    dataset: dict[str, str],
) -> dict[str, Any]:
    notebook = copy.deepcopy(template)
    parameter_cells = [
        cell
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and PARAMETER_PATTERN.search(source_text(cell))
    ]
    if len(parameter_cells) != 1:
        raise ValueError(
            "Expected one PAIR_INDEX parameter block in "
            f"{dataset['template']}; found {len(parameter_cells)}"
        )

    pair_assignment = (
        "PAIR_INDEX = None  # published baseline; QMIN/QMAX below are fixed"
        if pair_index < 0
        else f"PAIR_INDEX = {pair_index}  # index at generation time; QMIN/QMAX below are fixed"
    )
    parameter_source = PARAMETER_PATTERN.sub(
        f"{pair_assignment}\nQMIN, QMAX = ({qmin!r}, {qmax!r})",
        source_text(parameter_cells[0]),
    )
    if pair_index < 0:
        path_assignment = "_qrange = qrange_paths(QMIN, QMAX)"
        if parameter_source.count(path_assignment) != 1:
            raise ValueError("Expected one _qrange path assignment in the template")
        pdf_path_key = dataset["pdf_path_key"]
        parameter_source = parameter_source.replace(
            path_assignment,
            f"{path_assignment}\n_qrange[{pdf_path_key!r}] = get_path({pdf_path_key!r})",
        )
    set_source(parameter_cells[0], parameter_source)

    first_markdown = next(
        cell for cell in notebook["cells"] if cell["cell_type"] == "markdown"
    )
    set_source(
        first_markdown,
        (
            f"# {dataset['display_name']} whole-PDF training: `{tag}`\n\n"
            f"Fixed range: `QMIN = {qmin:g}`, `QMAX = {qmax:g}`. "
            f"Generated from `{dataset['template']}`. This notebook uses all "
            "10,000 clusters with deterministic stratified 80/10/10 splits "
            "and independently tunes this Q range.\n"
        ),
    )

    notebook.setdefault("metadata", {})["pdf_nn_qrange"] = {
        "dataset": dataset["display_name"],
        "workflow": "whole-pdf-neural-network",
        "qmin": qmin,
        "qmax": qmax,
        "tag": tag,
        "template": dataset["template"],
        "pdf_source": "baseline" if pair_index < 0 else "qrange",
    }
    normalize_notebook(notebook)
    return notebook


def main() -> int:
    args = parse_args()
    datasets = DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}
    pairs = selected_pairs(args.only)
    generated = 0

    for dataset_key, dataset in datasets.items():
        template_path = SCRIPT_DIR / dataset["template"]
        if not template_path.is_file():
            raise FileNotFoundError(f"Template notebook not found: {template_path}")
        template = json.loads(template_path.read_text(encoding="utf-8"))
        output_dir = args.output_root.expanduser().resolve() / dataset_key
        output_dir.mkdir(parents=True, exist_ok=True)

        for pair_index, qmin, qmax, tag in pairs:
            output_path = output_dir / f"{dataset['filename_prefix']}_{tag}.ipynb"
            if output_path.exists() and not args.overwrite:
                raise FileExistsError(
                    f"Generated notebook already exists: {output_path}\n"
                    "Use --overwrite only when replacement is intentional."
                )
            notebook = parameterize_notebook(
                template, pair_index, qmin, qmax, tag, dataset
            )
            output_path.write_text(
                json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(f"Prepared {output_path}")
            generated += 1

    print(f"Generated {generated} whole-PDF notebook(s) under {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())