#!/usr/bin/env python3
"""Generate and optionally execute one descriptor notebook per q-range."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
KERNEL_NAME = "pdf-nn-keras"
DATASETS = {
    "ceo2": {
        "display_name": "CeO2",
        "template": "A-descriptors-CeO2_qrange.ipynb",
        "output_dir": "ceo2",
        "filename_prefix": "A-descriptors-CeO2",
        "pdf_path_key": "ceo2_calculated_pdfs",
    },
    "ce40": {
        "display_name": "Ce40-based",
        "template": "B-descriptors-Ce40_qrange.ipynb",
        "output_dir": "ce40",
        "filename_prefix": "B-descriptors-Ce40",
        "pdf_path_key": "ce_calculated_pdfs",
    },
}
PARAMETER_PATTERN = re.compile(
    r"^PAIR_INDEX\s*=.*$\n^QMIN,\s*QMAX\s*=\s*QRANGE_PAIRS\[PAIR_INDEX\]$",
    re.MULTILINE,
)

sys.path.insert(0, str(PROJECT_ROOT))
from config import QRANGE_PAIRS, qrange_paths, qrange_tag  # noqa: E402


def parse_args(default_dataset: str = "ceo2") -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a fixed, self-describing descriptor notebook for each "
            "configured q-range. Add --execute to store all results in the copies."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default=default_dataset,
        help=f"Dataset template to generate (default: {default_dataset}).",
    )
    parser.add_argument("--template", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="TAG",
        help="Generate only these tags, for example --only q0-11 q1-11.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute each generated notebook and embed all outputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing generated notebooks.",
    )
    parser.add_argument(
        "--kernel-python",
        type=Path,
        help="Python executable used for notebook cells (defaults to the Keras environment).",
    )
    parser.add_argument(
        "--controller-python",
        type=Path,
        help="Python with nbclient and nbformat (defaults to the Miniforge base environment).",
    )
    parser.add_argument(
        "--cell-timeout",
        type=int,
        default=0,
        help="Maximum seconds per cell; 0 disables the timeout (default: 0).",
    )
    return parser.parse_args()


def source_text(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def set_source(cell: dict[str, Any], source: str) -> None:
    cell["source"] = source.splitlines(keepends=True)


def discover_python(explicit: Path | None, candidates: list[Path], purpose: str) -> Path:
    if explicit is not None:
        candidates.insert(0, explicit.expanduser())

    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return resolved

    formatted = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find {purpose} Python. Checked:\n  {formatted}")


def controller_python(explicit: Path | None) -> Path:
    if importlib.util.find_spec("nbclient") and importlib.util.find_spec("nbformat"):
        return Path(sys.executable).resolve()

    return discover_python(
        explicit,
        [
            Path(os.environ["PDF_NN_CONTROLLER_PYTHON"])
            if "PDF_NN_CONTROLLER_PYTHON" in os.environ
            else Path("/missing"),
            Path.home() / "miniforge3" / "bin" / "python",
            Path.home() / "anaconda3" / "bin" / "python",
        ],
        "notebook-controller",
    )


def kernel_python(explicit: Path | None) -> Path:
    return discover_python(
        explicit,
        [
            Path(os.environ["PDF_NN_KERNEL_PYTHON"])
            if "PDF_NN_KERNEL_PYTHON" in os.environ
            else Path("/missing"),
            Path.home() / "miniforge3" / "envs" / "keras" / "bin" / "python",
            Path.home() / "anaconda3" / "envs" / "keras" / "bin" / "python",
        ],
        "Keras-kernel",
    )


def ensure_controller(args: argparse.Namespace) -> None:
    if not args.execute:
        return

    controller = controller_python(args.controller_python)
    if controller == Path(sys.executable).resolve():
        return

    probe = subprocess.run(
        [str(controller), "-c", "import nbclient, nbformat"],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        raise RuntimeError(
            f"Controller {controller} cannot import nbclient and nbformat:\n{probe.stderr}"
        )

    forwarded_args = sys.argv[1:]
    if not any(
        arg == "--dataset" or arg.startswith("--dataset=")
        for arg in forwarded_args
    ):
        forwarded_args = ["--dataset", args.dataset, *forwarded_args]
    os.execv(
        str(controller),
        [str(controller), str(Path(__file__).resolve()), *forwarded_args],
    )


def selected_pairs(only: list[str] | None) -> list[tuple[int, float, float, str]]:
    available = [
        (index, float(qmin), float(qmax), qrange_tag(qmin, qmax))
        for index, (qmin, qmax) in enumerate(QRANGE_PAIRS)
    ]
    if not only:
        return available

    requested = set(only)
    known = {tag for _, _, _, tag in available}
    unknown = sorted(requested - known)
    if unknown:
        raise ValueError(
            f"Unknown q-range tag(s): {', '.join(unknown)}. "
            f"Available tags: {', '.join(sorted(known))}"
        )
    return [item for item in available if item[3] in requested]


def clear_outputs(notebook: dict[str, Any]) -> None:
    for cell_index, cell in enumerate(notebook["cells"], start=1):
        metadata = cell.setdefault("metadata", {})
        cell_id = f"cell-{cell_index:02d}"
        cell["id"] = cell_id
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
            f"Expected one PAIR_INDEX parameter block in the template; found {len(parameter_cells)}"
        )

    parameter_source = PARAMETER_PATTERN.sub(
        (
            f"PAIR_INDEX = {pair_index}  # index at generation time; QMIN/QMAX below are fixed\n"
            f"QMIN, QMAX = ({qmin!r}, {qmax!r})"
        ),
        source_text(parameter_cells[0]),
    )
    set_source(parameter_cells[0], parameter_source)

    first_markdown = next(cell for cell in notebook["cells"] if cell["cell_type"] == "markdown")
    set_source(
        first_markdown,
        (
            f"# {dataset['display_name']} q-range descriptor results: `{tag}`\n\n"
            f"Fixed range: `QMIN = {qmin:g}`, `QMAX = {qmax:g}`. "
            "This notebook was generated from "
            f"`{dataset['template']}`; outputs belong only to this range.\n"
        ),
    )

    notebook.setdefault("metadata", {})["pdf_nn_qrange"] = {
        "dataset": dataset["display_name"],
        "qmin": qmin,
        "qmax": qmax,
        "tag": tag,
        "template": dataset["template"],
    }
    clear_outputs(notebook)
    return notebook


def write_json_notebook(notebook: dict[str, Any], output_path: Path) -> None:
    output_path.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


@contextmanager
def temporary_kernel_spec(python_executable: Path) -> Iterator[None]:
    with tempfile.TemporaryDirectory(prefix="pdf-nn-kernel-") as temp_dir:
        data_dir = Path(temp_dir)
        spec_dir = data_dir / "kernels" / KERNEL_NAME
        spec_dir.mkdir(parents=True)
        (spec_dir / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [
                        str(python_executable),
                        "-m",
                        "ipykernel_launcher",
                        "-f",
                        "{connection_file}",
                    ],
                    "display_name": "PDF-NN Keras",
                    "language": "python",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        previous = os.environ.get("JUPYTER_PATH")
        os.environ["JUPYTER_PATH"] = os.pathsep.join(
            part for part in (str(data_dir), previous) if part
        )
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("JUPYTER_PATH", None)
            else:
                os.environ["JUPYTER_PATH"] = previous


def execute_notebook(output_path: Path, python_executable: Path, cell_timeout: int) -> None:
    import nbformat  # type: ignore[import-not-found]
    from nbclient import NotebookClient  # type: ignore[import-not-found]

    notebook = nbformat.read(output_path, as_version=4)
    total_cells = len(notebook.cells)

    def report_start(cell: Any, cell_index: int, **_: Any) -> None:
        if cell.cell_type == "code":
            print(f"  cell {cell_index + 1}/{total_cells}", flush=True)

    client = NotebookClient(
        notebook,
        kernel_name=KERNEL_NAME,
        timeout=None if cell_timeout == 0 else cell_timeout,
        allow_errors=False,
        resources={"metadata": {"path": str(output_path.parent)}},
        on_cell_start=report_start,
    )

    with temporary_kernel_spec(python_executable):
        try:
            client.execute()
        finally:
            nbformat.write(notebook, output_path)


def main(default_dataset: str = "ceo2") -> int:
    args = parse_args(default_dataset)
    ensure_controller(args)

    dataset = DATASETS[args.dataset]
    template_path = (args.template or SCRIPT_DIR / dataset["template"]).expanduser().resolve()
    output_dir = (
        args.output_dir or SCRIPT_DIR / "generated" / dataset["output_dir"]
    ).expanduser().resolve()
    if not template_path.is_file():
        raise FileNotFoundError(f"Template notebook not found: {template_path}")

    template = json.loads(template_path.read_text(encoding="utf-8"))
    pairs = selected_pairs(args.only)
    output_dir.mkdir(parents=True, exist_ok=True)

    python_executable = kernel_python(args.kernel_python) if args.execute else None
    if python_executable is not None:
        probe = subprocess.run(
            [str(python_executable), "-c", "import ipykernel, sklearn, tensorflow, shap"],
            capture_output=True,
            text=True,
        )
        if probe.returncode != 0:
            raise RuntimeError(
                f"Kernel {python_executable} is missing a required package:\n{probe.stderr}"
            )
        print(f"Notebook kernel: {python_executable}")

    failures: list[str] = []
    for pair_index, qmin, qmax, tag in pairs:
        output_path = output_dir / f"{dataset['filename_prefix']}_{tag}.ipynb"
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Generated notebook already exists: {output_path}\n"
                "Use --overwrite only when replacing its embedded results is intentional."
            )

        if args.execute:
            pdf_dir = qrange_paths(qmin, qmax)[dataset["pdf_path_key"]]
            if not pdf_dir.is_dir():
                raise FileNotFoundError(f"PDF directory does not exist for {tag}: {pdf_dir}")

        notebook = parameterize_notebook(
            template, pair_index, qmin, qmax, tag, dataset
        )
        write_json_notebook(notebook, output_path)
        print(f"Prepared {output_path}")

        if args.execute and python_executable is not None:
            print(f"Executing {tag} ...", flush=True)
            try:
                execute_notebook(output_path, python_executable, args.cell_timeout)
            except Exception as error:
                failures.append(f"{tag}: {error}")
                print(f"Execution failed for {tag}: {error}", file=sys.stderr, flush=True)
            else:
                print(f"Completed {tag}", flush=True)

    if failures:
        print("\nFailed notebooks:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    action = "Executed" if args.execute else "Generated"
    print(f"{action} {len(pairs)} notebook(s) in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
