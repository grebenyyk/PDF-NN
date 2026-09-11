#!/usr/bin/env python3
"""
Predict nuclearity of lanthanide compounds from experimental PDF (.gr) files.

Usage:
    python quick-predict.py /path/to/gr/files                     # csd2023 5-fold ensemble (default)
    python quick-predict.py /path/to/gr/files --ensemble 3c       # 3C auto-labels ensemble (2-12 A)
    python quick-predict.py /path/to/gr/files --model csd-3.h5    # legacy single model

Model families differ in input grid, preprocessing and label mapping:
    3d (default)  models/csd2023_ensemble/      0-20 A, 2000 pts, per-sample z-score,
                                                 class = output index + 1 (10 = "10+", 11 = polymer)
    3c            models/csd_auto_labels_ensemble/  2-12 A, 1000 pts, per-sample z-score,
                                                 class = output index (10 = polymer; index 0 unused)
    legacy --model                             2-12 A, 1000 pts, L2 normalization,
                                                 class = output index (10 = polymer; index 0 unused)

Ensemble prediction = mean of the five folds' softmax outputs.
"""

import argparse
import sys
import os
from pathlib import Path

# Suppress TF logging before import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent.resolve()

# --- model families ---------------------------------------------------------
# grid:   interpolation grid (start, stop, step)
# cover:  r-range the input .gr must cover (below `cover[0]` the curve is
#         clamped to its first value by np.interp, which is harmless: there
#         are no structural peaks below ~1 A)
# norm:   preprocessing used at training time
# labels: how output indices map to classes
FAMILIES = {
    "3d": {
        "dir": SCRIPT_DIR / "models" / "csd2023_ensemble",
        "grid": (0.0, 20.0, 0.01),
        "cover": (1.0, 20.0),
        "norm": "z",
        "labels": "plus1",
    },
    "3c": {
        "dir": SCRIPT_DIR / "models" / "csd_auto_labels_ensemble",
        "grid": (2.0, 12.0, 0.01),
        "cover": (2.0, 12.0),
        "norm": "z",
        "labels": "index",
    },
    "legacy": {  # original csd-3.h5 via --model
        "grid": (2.0, 12.0, 0.01),
        "cover": (2.0, 12.0),
        "norm": "l2",
        "labels": "index",
    },
}


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


def preprocess(values, norm):
    """Per-sample preprocessing exactly as used at training time."""
    if norm == "z":
        return (values - values.mean(axis=1, keepdims=True)) / values.std(
            axis=1, keepdims=True
        )
    # legacy L2 row normalization
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def pred_to_nuclearity(index, labels):
    """Map model output index to human-readable nuclearity label."""
    if labels == "plus1":
        cls = index + 1
        if cls == 11:
            return "polymer"
        if cls == 10:
            return "10+"
        return str(cls)
    # 'index' families: labels 1..10, output index 0 was never trained
    if index == 0:
        return "none"
    if index == 10:
        return "polymer"
    return str(index)


def load_models(paths):
    """Load keras models, registering SeqSelfAttention when available."""
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")

    custom_objects = {}
    try:
        from keras_self_attention import SeqSelfAttention

        custom_objects["SeqSelfAttention"] = SeqSelfAttention
    except ImportError:
        pass

    from keras.utils import custom_object_scope

    models = []
    with custom_object_scope(custom_objects):
        for p in paths:
            models.append(tf.keras.models.load_model(str(p)))
    return models


def main():
    parser = argparse.ArgumentParser(
        description="Predict nuclearity from experimental PDF (.gr) files."
    )
    parser.add_argument("input_dir", help="Directory containing .gr files")
    parser.add_argument(
        "--ensemble",
        default="3d",
        help="'3d' (default, models/csd2023_ensemble), '3c' "
        "(models/csd_auto_labels_ensemble), or a directory of fold_*.h5 models",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Single legacy .h5 model (e.g. csd-3.h5); overrides --ensemble",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: predictions.csv in the input directory)",
    )
    args = parser.parse_args()

    # --- Resolve model family and paths ---
    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        print(f"Error: '{args.input_dir}' is not a directory.")
        sys.exit(1)

    if args.model:
        model_paths = [Path(args.model).resolve()]
        if not model_paths[0].exists():
            print(f"Error: Model not found at '{model_paths[0]}'.")
            sys.exit(1)
        family = dict(FAMILIES["legacy"])
    else:
        if args.ensemble in FAMILIES and args.ensemble != "legacy":
            family = dict(FAMILIES[args.ensemble])
        else:
            ens_dir = Path(args.ensemble).resolve()
            if not ens_dir.is_dir():
                print(
                    f"Error: --ensemble must be '3d', '3c' or a directory, "
                    f"got '{args.ensemble}'."
                )
                sys.exit(1)
            family = {"dir": ens_dir, "grid": None, "cover": None,
                      "norm": "z", "labels": None}
        model_paths = sorted(family["dir"].glob("fold_*.h5"))
        if not model_paths:
            print(f"Error: no fold_*.h5 models in '{family['dir']}'.")
            sys.exit(1)

    # --- Find .gr files (recursive) ---
    gr_files = sorted(input_dir.rglob("*.gr"))
    if not gr_files:
        print(f"No .gr files found in '{input_dir}'.")
        sys.exit(1)

    # --- Load models first (input shape determines the grid for custom ensembles) ---
    print(f"Loading {len(model_paths)} model(s) from:"
          f" {model_paths[0].parent if len(model_paths) > 1 else model_paths[0]}")
    models = load_models(model_paths)

    if family["grid"] is None:
        # infer from the model's input length: 2000 -> 3D convention, else 2-12 A
        n_in = models[0].input_shape[1]
        if n_in == 2000:
            family.update(grid=(0.0, 20.0, 0.01), cover=(1.0, 20.0), labels="plus1")
        else:
            family.update(grid=(2.0, 12.0, 0.01), cover=(2.0, 12.0), labels="index")

    r_grid = np.arange(*family["grid"])
    cover = family["cover"]
    print(
        f"Input grid: r = {family['grid'][0]:g}-{family['grid'][1]:g} A "
        f"({len(r_grid)} pts) | preprocessing: {family['norm']} | "
        f"labels: {'index+1' if family['labels'] == 'plus1' else 'index'}"
    )

    # --- Parse & interpolate ---
    filenames = []
    data_rows = []
    skipped = []

    for gr_file in gr_files:
        r, gr = parse_gr_file(gr_file)
        if r is None:
            skipped.append((str(gr_file.relative_to(input_dir)), "could not parse numeric data"))
            continue

        tol = 0.05  # small tolerance for edge coverage
        if r.min() > cover[0] + tol or r.max() < cover[1] - tol:
            skipped.append(
                (
                    str(gr_file.relative_to(input_dir)),
                    f"r range [{r.min():.2f}, {r.max():.2f}] Å does not cover "
                    f"the required [{cover[0]:g}, {cover[1]:g}] Å",
                )
            )
            continue

        filenames.append(str(gr_file.relative_to(input_dir)))
        data_rows.append(np.interp(r_grid, r, gr))

    if skipped:
        print(f"\nSkipped {len(skipped)} file(s):")
        for name, reason in skipped:
            print(f"  {name} — {reason}")
        print()

    if not data_rows:
        print("No files could be processed. Exiting.")
        sys.exit(1)

    data_points = preprocess(np.array(data_rows), family["norm"])[..., None].astype(
        "float32"
    )

    # --- Predict (ensemble = mean of softmax outputs) ---
    probs = [m.predict(data_points, verbose=0) for m in models]
    y_pred_prob = np.mean(probs, axis=0)

    if family["labels"] == "index":
        # output index 0 was never trained - mask it out of the ranking
        y_pred_prob[:, 0] = -1.0

    top2_idx = np.argsort(y_pred_prob, axis=1)[:, -2:][:, ::-1]
    top2_prob = np.take_along_axis(y_pred_prob, top2_idx, axis=1)

    # --- Print results ---
    print(f"\nProcessed {len(filenames)} file(s)\n")

    header = f"{'File':<40} {'Prediction':>10} {'Conf.':>7}   {'2nd best':>10} {'Conf.':>7}"
    print(header)
    print("-" * len(header))

    results = []
    for i, fname in enumerate(filenames):
        nuc1 = pred_to_nuclearity(top2_idx[i, 0], family["labels"])
        nuc2 = pred_to_nuclearity(top2_idx[i, 1], family["labels"])
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
