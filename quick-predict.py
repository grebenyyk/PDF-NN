#!/usr/bin/env python3
"""
Predict nuclearity of lanthanide compounds from experimental PDF (.gr) files.

Usage:
    python quick-predict.py /path/to/gr/files
    python quick-predict.py /path/to/gr/files --model /path/to/model.h5
"""

import argparse
import sys
import os
from pathlib import Path

# Suppress TF logging before import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer

# Model grid: 1000 points from r=2.00 to r=11.99 Å, step 0.01
R_MIN = 2.0
R_MAX = 12.0
R_STEP = 0.01
N_POINTS = 1000
R_GRID = np.arange(R_MIN, R_MAX, R_STEP)

# Default model filename in the same directory as this script
SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_MODEL = SCRIPT_DIR / "csd-3.h5"


def parse_gr_file(filepath):
    """Parse a .gr file, auto-detecting where the two-column numeric data begins.

    Returns (r, gr) numpy arrays, or (None, None) on failure.
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    data_start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) >= 2:
            try:
                float(parts[0])
                float(parts[1])
                data_start = i
                break
            except ValueError:
                continue

    if data_start is None:
        return None, None

    r_vals = []
    gr_vals = []
    for line in lines[data_start:]:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        try:
            r_vals.append(float(parts[0]))
            gr_vals.append(float(parts[1]))
        except ValueError:
            continue

    if len(r_vals) < 2:
        return None, None

    return np.array(r_vals), np.array(gr_vals)


def interpolate_to_grid(r, gr):
    """Interpolate experimental (r, g(r)) onto the model's 1000-point grid.

    Returns the interpolated array of length 1000, or None if the input
    data doesn't cover the required [2.0, 12.0) Å range.
    """
    tol = 0.05  # small tolerance for edge coverage
    if r.min() > R_MIN + tol or r.max() < R_MAX - tol:
        return None
    return np.interp(R_GRID, r, gr)


def pred_to_nuclearity(index):
    """Map model output index to human-readable nuclearity label."""
    if index == 10:
        return "polymer"
    return str(index)


def main():
    parser = argparse.ArgumentParser(
        description="Predict nuclearity from experimental PDF (.gr) files."
    )
    parser.add_argument("input_dir", help="Directory containing .gr files")
    parser.add_argument(
        "--model",
        default=None,
        help="Path to trained .h5 model (default: csd-3.h5 in project root)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: predictions.csv in the input directory)",
    )
    args = parser.parse_args()

    # --- Resolve paths ---
    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        print(f"Error: '{args.input_dir}' is not a directory.")
        sys.exit(1)

    model_path = Path(args.model).resolve() if args.model else DEFAULT_MODEL
    if not model_path.exists():
        msg = f"Error: Model not found at '{model_path}'."
        if not args.model:
            msg += "\nProvide a model path with --model or place csd-3.h5 in the project root."
        print(msg)
        sys.exit(1)

    # --- Find .gr files (recursive) ---
    gr_files = sorted(input_dir.rglob("*.gr"))
    if not gr_files:
        print(f"No .gr files found in '{input_dir}'.")
        sys.exit(1)

    print(f"Found {len(gr_files)} .gr file(s) in {input_dir}\n")

    # --- Parse & interpolate ---
    filenames = []
    data_rows = []
    skipped = []

    for gr_file in gr_files:
        r, gr = parse_gr_file(gr_file)
        if r is None:
            skipped.append((str(gr_file.relative_to(input_dir)), "could not parse numeric data"))
            continue

        interpolated = interpolate_to_grid(r, gr)
        if interpolated is None:
            skipped.append(
                (
                    str(gr_file.relative_to(input_dir)),
                    f"r range [{r.min():.2f}, {r.max():.2f}] Å does not cover "
                    f"the required [{R_MIN:.1f}, {R_MAX:.1f}] Å",
                )
            )
            continue

        filenames.append(str(gr_file.relative_to(input_dir)))
        data_rows.append(interpolated)

    if skipped:
        print(f"Skipped {len(skipped)} file(s):")
        for name, reason in skipped:
            print(f"  {name} — {reason}")
        print()

    if not data_rows:
        print("No files could be processed. Exiting.")
        sys.exit(1)

    data_points = np.array(data_rows)

    # --- Normalize (L2, same as training) ---
    data_points = Normalizer().fit_transform(data_points)

    # --- Load model ---
    print(f"Loading model: {model_path.name}")
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")

    # Always register SeqSelfAttention if available, so models with or
    # without the attention layer can be loaded transparently.
    custom_objects = {}
    try:
        from keras_self_attention import SeqSelfAttention
        custom_objects["SeqSelfAttention"] = SeqSelfAttention
    except ImportError:
        pass

    if custom_objects:
        from keras.utils import custom_object_scope
        with custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(str(model_path))
    else:
        model = tf.keras.models.load_model(str(model_path))

    # --- Predict ---
    y_pred_prob = model.predict(data_points, verbose=0)

    top2_idx = np.argsort(y_pred_prob, axis=1)[:, -2:][:, ::-1]
    top2_prob = np.take_along_axis(y_pred_prob, top2_idx, axis=1)

    # --- Print results ---
    print(f"\nProcessed {len(filenames)} file(s)\n")

    header = f"{'File':<40} {'Prediction':>10} {'Conf.':>7}   {'2nd best':>10} {'Conf.':>7}"
    print(header)
    print("-" * len(header))

    results = []
    for i, fname in enumerate(filenames):
        nuc1 = pred_to_nuclearity(top2_idx[i, 0])
        nuc2 = pred_to_nuclearity(top2_idx[i, 1])
        conf1 = top2_prob[i, 0]
        conf2 = top2_prob[i, 1]

        display = fname if len(fname) <= 38 else fname[:35] + "..."
        print(f"{display:<40} {nuc1:>10} {conf1:>7.1%}   {nuc2:>10} {conf2:>7.1%}")

        results.append(
            {
                "filename": fname,
                "prediction_1": nuc1,
                "confidence_1": round(float(conf1), 4),
                "prediction_2": nuc2,
                "confidence_2": round(float(conf2), 4),
            }
        )

    # --- Save CSV ---
    output_path = Path(args.output) if args.output else input_dir / "predictions.csv"
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()