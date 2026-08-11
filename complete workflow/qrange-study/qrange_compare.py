#!/usr/bin/env python3
"""
Aggregated comparison across (qmin, qmax) pairs for the PDF-NN qrange study.

Collects training metrics from all completed pairs, produces:
  1. A summary CSV + LaTeX table (accuracy / F1 per dataset per pair).
  2. A multi-panel bar chart comparing metrics across pairs.
  3. A PDF-shape overlay panel showing representative PDFs across q-pairs.

These outputs are the figure/table the paper needs to answer the referee's
question: "Did you check the variation of the calculated PDF as function of Q?"

Usage
-----
Run from the project root or from inside qrange-study/:

    python "complete workflow/qrange-study/qrange_compare.py"
    python qrange_compare.py --pairs 0 2 4       # only specific pairs

Or import as a module:

    from qrange_compare import run_comparison
    run_comparison([(0.0, 11.0), (0.3, 25.0)])

Outputs are written to the untagged results aggregation directory:
    DATA_ROOT / results / qrange_comparison/
"""

from __future__ import annotations

import sys
from pathlib import Path

# Bootstrap config import (same pattern as qrange_sweep.py)
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config
from config import QRANGE_PAIRS, qrange_tag, qrange_paths

# numpy/pandas imported lazily inside functions so that --help and module
# inspection work without the full compute environment.

# Okabe-Ito colorblind-safe palette (matches pdf_q_sweep.py)
PALETTE = [
    "#000000",  # black
    "#0072B2",  # blue
    "#009E73",  # bluish green
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
]

# Dataset display names
DATASET_NAMES = {
    "ceo2": "Dataset A (CeO$_2$)",
    "ce_clusters": "Dataset B (Ce$_{40}$)",
    "csd": "Dataset C (CSD)",
}

# Subdirectory names within results/q{tag}/ that each training notebook uses
DATASET_RESULT_DIRS = {
    "ceo2": "ceo2_clusters",
    "ce_clusters": "ce_clusters",
    "csd": "csd",
}


# ---------------------------------------------------------------------------
# Metric collection
# ---------------------------------------------------------------------------
def _find_metric_file(results_dir: Path, dataset: str):
    """
    Find the metrics/results CSV in a pair's results directory.

    The training notebooks save confusion matrices, accuracy, etc. to
    results/q{tag}/{dataset}/.  We look for common output filenames.
    """
    ds_subdir = DATASET_RESULT_DIRS.get(dataset, dataset)
    search_dir = results_dir / ds_subdir

    if not search_dir.exists():
        return None

    # Look for CSV files that might contain metrics
    candidates = [
        "metrics.csv",
        "results.csv",
        "classification_report.csv",
        "accuracy.csv",
    ]
    for name in candidates:
        p = search_dir / name
        if p.exists():
            return p

    # Fall back: any CSV in the directory
    csvs = sorted(search_dir.glob("*.csv"))
    if csvs:
        return csvs[0]

    return None


def _parse_metrics(csv_path: Path):
    """Extract accuracy and F1-macro from a results CSV.

    Returns dict with 'accuracy' and 'f1_macro' (or None if not found).
    Handles various CSV layouts the notebooks might produce.
    """
    metrics = {}
    import pandas as pd

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    # Try to find accuracy
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in ("accuracy", "acc", "test_accuracy", "val_accuracy"):
            metrics["accuracy"] = float(df[col].iloc[-1])
            break

    # Try to find F1-macro
    for col in df.columns:
        col_lower = col.lower().strip()
        if "f1" in col_lower and "macro" in col_lower:
            metrics["f1_macro"] = float(df[col].iloc[-1])
            break
    if "f1_macro" not in metrics:
        for col in df.columns:
            if col.lower().strip() == "f1":
                metrics["f1_macro"] = float(df[col].iloc[-1])
                break

    return metrics if metrics else None


def collect_metrics(pairs):
    """
    Scan all pair result directories and collect metrics into a DataFrame.

    Parameters
    ----------
    pairs : list of (qmin, qmax) tuples

    Returns
    -------
    pd.DataFrame with columns: pair, qmin, qmax, tag, dataset, accuracy, f1_macro
    """
    import pandas as pd

    rows = []

    for qmin, qmax in pairs:
        tag = qrange_tag(qmin, qmax)
        paths = qrange_paths(qmin, qmax)
        results_dir = paths["results"]

        for ds_key in DATASET_RESULT_DIRS:
            csv_path = _find_metric_file(results_dir, ds_key)
            if csv_path is None:
                rows.append({
                    "pair": f"({qmin}, {qmax})",
                    "qmin": qmin,
                    "qmax": qmax,
                    "tag": tag,
                    "dataset": ds_key,
                    "accuracy": None,
                    "f1_macro": None,
                    "csv_path": None,
                })
                continue

            metrics = _parse_metrics(csv_path)
            if metrics:
                rows.append({
                    "pair": f"({qmin}, {qmax})",
                    "qmin": qmin,
                    "qmax": qmax,
                    "tag": tag,
                    "dataset": ds_key,
                    "accuracy": metrics.get("accuracy"),
                    "f1_macro": metrics.get("f1_macro"),
                    "csv_path": str(csv_path),
                })
            else:
                rows.append({
                    "pair": f"({qmin}, {qmax})",
                    "qmin": qmin,
                    "qmax": qmax,
                    "tag": tag,
                    "dataset": ds_key,
                    "accuracy": None,
                    "f1_macro": None,
                    "csv_path": str(csv_path),
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Output: metrics table
# ---------------------------------------------------------------------------
def metrics_table(df, output_dir):
    """Render the comparison table as CSV and LaTeX."""
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)

    # Pivot: rows = pairs, columns = datasets × metrics
    if not df.empty and df["accuracy"].notna().any():
        pivot = df.pivot_table(
            index=["tag", "qmin", "qmax"],
            columns="dataset",
            values=["accuracy", "f1_macro"],
            aggfunc="first",
        )
        pivot.columns = [f"{m}_{d}" for m, d in pivot.columns]
        pivot = pivot.reset_index()
    else:
        pivot = df[["tag", "qmin", "qmax", "dataset", "accuracy", "f1_macro"]].copy()

    # CSV
    csv_path = output_dir / "qrange_metrics.csv"
    pivot.to_csv(csv_path, index=False)
    print(f"  Metrics CSV -> {csv_path}")

    # LaTeX
    latex_path = output_dir / "qrange_metrics.tex"
    with open(latex_path, "w") as f:
        f.write("% Auto-generated by qrange_compare.py\n")
        f.write("% Classification metrics across (qmin, qmax) pairs\n\n")
        if not pivot.empty:
            latex_str = pivot.to_latex(
                index=False,
                float_format="%.3f",
                caption=("Classification metrics across Q-range pairs. "
                         "None entries indicate pairs not yet trained."),
                label="tab:qrange_metrics",
            )
            f.write(latex_str)
        else:
            f.write("% No metrics found.\n")
    print(f"  Metrics LaTeX -> {latex_path}")

    return pivot


# ---------------------------------------------------------------------------
# Output: comparison figure (bar chart)
# ---------------------------------------------------------------------------
def comparison_figure(df, output_dir):
    """Multi-panel bar chart: accuracy and F1 across pairs, grouped by dataset."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter to rows that have accuracy data
    df_valid = df.dropna(subset=["accuracy"]).copy()
    if df_valid.empty:
        print("  [comparison_figure] No accuracy data found — skipping figure.")
        return None

    datasets = [d for d in DATASET_RESULT_DIRS if d in df_valid["dataset"].values]
    n_ds = len(datasets)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    x = np.arange(len(df_valid["tag"].unique()))
    n_datasets = len(datasets)
    width = 0.8 / max(n_datasets, 1)

    for metric_idx, metric in enumerate(["accuracy", "f1_macro"]):
        ax = axes[metric_idx]
        if metric not in df_valid.columns or df_valid[metric].isna().all():
            ax.set_title(f"{metric.replace('_', ' ').title()} — no data")
            continue

        tags = df_valid["tag"].unique()
        x = np.arange(len(tags))

        for i, ds in enumerate(datasets):
            sub = df_valid[df_valid["dataset"] == ds].set_index("tag")
            sub = sub.reindex(tags)
            vals = sub[metric].values if metric in sub.columns else [np.nan] * len(tags)
            ax.bar(
                x + i * width, vals, width,
                label=DATASET_NAMES.get(ds, ds),
                color=PALETTE[i % len(PALETTE)],
                alpha=0.85,
            )

        ax.set_xticks(x + width * (n_datasets - 1) / 2)
        ax.set_xticklabels(tags, rotation=30, ha="right", fontsize=8)
        ax.set_xlabel("(qmin, qmax) pair", fontsize=10)
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, frameon=False)
        ax.grid(axis="y", alpha=0.3)
        ax.set_title(metric.replace("_", " ").title(), fontsize=11)

    fig.suptitle(
        "Classification performance across Q-range pairs",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = output_dir / "qrange_comparison.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Comparison figure -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Output: PDF-shape overlay panel
# ---------------------------------------------------------------------------
def pdf_shape_panel(pairs, output_dir, max_pdfs_per_pair=5):
    """
    Overlay representative PDFs from each q-pair to show how PDF shape
    changes with Q-range.

    This uses the cluster PDFs (CeO2) if available; it complements the
    preliminary pdf_q_sweep.py figures (which used a standalone 9-cluster
    test set) with full-pipeline PDFs.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)

    # Try to find CeO2 PDFs for each pair
    found_pairs = []
    for qmin, qmax in pairs:
        paths = qrange_paths(qmin, qmax)
        pdf_dir = paths["ceo2_calculated_pdfs"]
        if pdf_dir.exists():
            dats = sorted(pdf_dir.glob("*.dat"))
            if dats:
                found_pairs.append((qmin, qmax, pdf_dir, dats))

    if not found_pairs:
        print("  [pdf_shape_panel] No cluster PDFs found for any pair — skipping.")
        return None

    fig, ax = plt.subplots(figsize=(8, 6))

    gmax_span = 1.0
    for idx, (qmin, qmax, pdf_dir, dats) in enumerate(found_pairs):
        tag = qrange_tag(qmin, qmax)
        # Take a few representative PDFs (middle nuclearity range)
        sample = dats[:max_pdfs_per_pair]
        for j, dat_path in enumerate(sample):
            try:
                data = np.loadtxt(dat_path, skiprows=1)
            except Exception:
                continue
            r, g = data[:, 0], data[:, 1]
            # Offset by pair index for readability
            offset = idx * 1.5 * gmax_span
            label = f"({qmin}, {qmax})" if j == 0 else None
            ax.plot(r, g + offset, color=PALETTE[idx % len(PALETTE)],
                    lw=0.6, alpha=0.8, label=label)
            gmax_span = max(gmax_span, float(np.ptp(g)))

        ax.axhline(idx * 1.5 * gmax_span, color="0.8", lw=0.5, zorder=0)

    ax.set_xlim(0, 20)
    ax.set_xlabel("r (Å)", fontsize=11)
    ax.set_ylabel("G(r) (offset per pair)", fontsize=11)
    ax.set_title("CeO$_2$ cluster PDFs across Q-range pairs", fontsize=12)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()

    out_path = output_dir / "qrange_pdf_overlay.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  PDF overlay -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_comparison(pairs=None):
    """
    Full comparison pipeline: collect metrics, produce table + figures.

    Parameters
    ----------
    pairs : list of (qmin, qmax) tuples, or None for all QRANGE_PAIRS.
    """
    if pairs is None:
        pairs = list(QRANGE_PAIRS)

    print(f"\nCollecting metrics for {len(pairs)} pair(s)...")
    df = collect_metrics(pairs)

    # Summary to console
    print("\n--- Metrics summary ---")
    if not df.empty:
        display_df = df[["tag", "dataset", "accuracy", "f1_macro"]].copy()
        print(display_df.to_string(index=False))
    else:
        print("  (no data)")

    # Output directory: untagged aggregation dir
    output_dir = config.PATHS["results"] / "qrange_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting outputs to: {output_dir}")

    # Table
    pivot = metrics_table(df, output_dir)

    # Comparison figure
    comparison_figure(df, output_dir)

    # PDF overlay panel
    pdf_shape_panel(pairs, output_dir)

    print("\nCOMPARE DONE.")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Aggregated comparison across Q-range pairs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pairs",
        type=int,
        nargs="+",
        default=None,
        help="Pair indices (0-based into QRANGE_PAIRS) to compare.",
    )

    args = parser.parse_args()

    if args.pairs is not None:
        pairs = [QRANGE_PAIRS[i] for i in args.pairs]
    else:
        pairs = list(QRANGE_PAIRS)

    run_comparison(pairs)


if __name__ == "__main__":
    main()
