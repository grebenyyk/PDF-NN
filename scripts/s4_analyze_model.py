#!/usr/bin/env python3
"""
Stage 4: Evaluate and analyze trained models (confusion matrix, attention, saliency).

Script equivalent of the post-training analysis cells in notebooks 3A, 3B, 3C:
  - Classification metrics: confusion matrix, recall, precision, F1 score
  - Attention weights: extraction + PDF overlay plot (datasets A and B)
  - Saliency maps: input-gradient backpropagation per nuclearity class

Usage examples::

    # Evaluate CeO2 model and save figures to results/ceo2_clusters/
    python scripts/s4_analyze_model.py --dataset ceo2 --model /path/to/model.h5

    # Evaluate CSD model
    python scripts/s4_analyze_model.py --dataset csd --model /path/to/csd_fold_1_model.h5

    # Display figures interactively (also saves to disk)
    python scripts/s4_analyze_model.py --dataset ceo2 --model /path/to/model.h5 --show
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from config import get_path
from scripts.pipeline_utils import (
    DATASET_CONFIG,
    balance_dataset,
    ensure_dir,
    get_dataset_config,
    get_labels_dir,
    get_results_dir,
    load_pdf_data,
    plot_confusion_matrix,
    print_section,
)


# ---------------------------------------------------------------------------
# Classification evaluation  (notebook cells 12-13)
# ---------------------------------------------------------------------------
def evaluate_classification(model, X_test, y_test, config, figures_dir, show=False):
    """Compute and print evaluation metrics; plot confusion matrix."""
    from sklearn.metrics import confusion_matrix, recall_score, f1_score, precision_score

    print_section("Evaluation Metrics (Test Set)")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test loss:     {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    cm = confusion_matrix(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average=None)
    f1 = f1_score(y_test, y_pred, average=None)
    precision = precision_score(y_test, y_pred, average=None)

    print("\nConfusion Matrix:")
    print(cm)
    print(f"\nRecall score:    {recall}")
    print(f"F1 score:        {f1}")
    print(f"Precision score: {precision}")

    # Plot confusion matrix
    cm_path = figures_dir / "confusion_matrix.png"
    classes = np.unique(y_test)
    plot_confusion_matrix(
        cm, classes=classes, title="Confusion matrix",
        savepath=cm_path, show=show,
    )

    return {
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "confusion_matrix": cm.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "precision": precision.tolist(),
    }


# ---------------------------------------------------------------------------
# Attention weights visualization  (notebook cells 14-15)
# ---------------------------------------------------------------------------
def extract_and_plot_attention(model, X_test, data_points, labels, config, figures_dir, show=False):
    """Extract SeqSelfAttention weights and plot overlay on raw PDFs."""
    if not config["has_attention"]:
        print("\nNote: This model architecture does not include an attention mechanism. Skipping.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap
    from keras.models import Model

    print_section("Attention Weights Extraction")

    # Find the attention layer
    attention_layers = [layer for layer in model.layers if "SeqSelfAttention" in str(type(layer))]
    if not attention_layers:
        print("WARNING: 'has_attention=True' in config, but no SeqSelfAttention layer found in model.")
        return

    attention_layer = attention_layers[0]
    model_with_attentions = Model(
        inputs=model.input, outputs=[model.output, attention_layer.output]
    )

    total_samples = X_test.shape[0]
    batch_size = 32
    all_attention_weights = []
    for i in range(0, total_samples, batch_size):
        batch_input = X_test[i:i + batch_size]
        _, attention_weights = model_with_attentions.predict(batch_input, verbose=0)
        all_attention_weights.append(attention_weights)

    all_attention_weights = np.concatenate(all_attention_weights, axis=0)
    average_attention_weights = np.mean(all_attention_weights, axis=0)
    summed_attention_weights = np.sum(average_attention_weights, axis=0)

    attention_dim = len(summed_attention_weights)
    print(f"Attention dimension: {attention_dim}")

    # Plot overlay (notebook cell 15)
    min_val = np.min(labels)
    max_val = np.max(labels)
    normalized_labels = (labels - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(labels)
    cmap = get_cmap(config.get("attention_cmap", "Greens"))

    attention_r = np.linspace(2, 12, attention_dim)

    fig, ax = plt.subplots(figsize=(10, 5))
    n_curves = min(300, data_points.shape[0])
    for i in range(1, n_curves):
        color = cmap(normalized_labels[i])
        pdf_r = np.arange(len(data_points[i, :])) / 100 + 2
        max_pdf = data_points[i, :].max()
        if max_pdf > 0:
            plt.plot(pdf_r, data_points[i, :] / max_pdf, alpha=0.7, linewidth=0.3, color=color)

    max_att = summed_attention_weights.max()
    if max_att > 0:
        ax.bar(
            attention_r,
            summed_attention_weights / max_att,
            width=(10.0 / attention_dim),
            color="black",
            alpha=0.8,
        )
    ax.set_facecolor("#ebedf0")
    plt.xlabel("r, Å")
    plt.ylabel("Attention Weights and G(r)")
    ax.set_xlim(2, 12)

    att_path = figures_dir / "attention_weights_overlay.png"
    plt.savefig(att_path, dpi=300, bbox_inches="tight")
    print(f"Attention plot saved to: {att_path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Saliency maps (input-gradient backprop)  (notebook cells 16-17)
# ---------------------------------------------------------------------------
def plot_saliency_maps(model, X_train, y_train, config, figures_dir, show=False):
    """Compute and plot saliency maps per nuclearity class using GradientTape."""
    import tensorflow as tf
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print_section("Saliency Maps (Input-Gradient Backpropagation)")

    n_classes = config["n_classes"]
    sample_size = 500 if config["display_name"].startswith("CeO2") else 400

    # Layout: 3A/3B uses 3x3 (labels 1-9); 3C uses 4x3 (labels 1-10)
    if n_classes == 11:
        n_rows, n_cols = 4, 3
        max_label = 11  # range(1, 11)
    else:
        n_rows, n_cols = 3, 3
        max_label = 10  # range(1, 10)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if config.get("saliency_title"):
        fig.suptitle(config["saliency_title"])

    angstrom_range = 2 + np.linspace(0, 1000, 1000) / 100

    for label in range(1, max_label):
        ax = axes[(label - 1) // n_cols, (label - 1) % n_cols]
        gradients = []

        for idx in range(min(sample_size, len(y_train))):
            if y_train[idx] == label:
                sample_input = tf.convert_to_tensor(
                    X_train[idx].reshape(1, 1000, 1), dtype=tf.float32
                )
                with tf.GradientTape() as tape:
                    tape.watch(sample_input)
                    prediction = model(sample_input)
                    target_idx = label - 1  # 0-based
                    loss = prediction[0][target_idx]

                grad_values = tape.gradient(loss, sample_input)
                if grad_values is not None and hasattr(grad_values, "numpy"):
                    grad_numpy = grad_values.numpy().reshape(1000)
                    gradients.append(grad_numpy)
                    ax.plot(
                        angstrom_range, grad_numpy, alpha=0.2,
                        color=config["saliency_color"],
                    )

        if gradients:
            mean_gradient = np.mean(gradients, axis=0)
            ax.plot(angstrom_range, mean_gradient, color="red")

        label_str = "10 (Polymer)" if label == 10 else str(label)
        ax.set_title(f"Label {label_str}")
        ax.set_xlabel("r, Å")
        ax.set_ylabel("Gradient Value")
        ax.grid(True)

    # Delete unused subplots (CSD uses 10 panels in a 4x3 grid = 2 empty)
    if n_classes == 11:
        fig.delaxes(axes[3, 1])
        fig.delaxes(axes[3, 2])

    plt.tight_layout()
    saliency_path = figures_dir / config["saliency_filename"]
    plt.savefig(saliency_path, dpi=400, bbox_inches="tight")
    print(f"Saliency maps saved to: {saliency_path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main analysis runner
# ---------------------------------------------------------------------------
def run(dataset: str, model_path: str | Path, show: bool = False):
    config = get_dataset_config(dataset)
    mpath = Path(model_path).resolve()
    if not mpath.exists():
        print(f"ERROR: Model file not found at: {mpath}")
        sys.exit(1)

    results_dir = ensure_dir(get_results_dir(dataset, config))
    figures_dir = ensure_dir(results_dir / "figures")

    print_section(f"Stage 4: Analyze model — {config['display_name']}")
    print(f"Model path:  {mpath}")
    print(f"Results dir: {results_dir}")
    print(f"Figures dir: {figures_dir}")

    # Load data
    pdfs_dir = get_path(config["pdfs_path_key"])
    files_calc = sorted(glob.glob(str(pdfs_dir / "*.dat")))
    files_calc = balance_dataset(files_calc, config)
    raw_data_points, labels, _ = load_pdf_data(files_calc, dataset, config)

    from sklearn.preprocessing import Normalizer
    data_points = Normalizer().fit_transform(raw_data_points)

    # Train/test split matching notebook random_state
    from sklearn.model_selection import train_test_split
    if config.get("use_stratified_kfold"):
        X_train, X_test, y_train, y_test = train_test_split(
            data_points, labels, test_size=config["test_size"], random_state=42,
            stratify=labels,
        )
    else:
        X_train, X_temp, y_train, y_temp = train_test_split(
            data_points, labels, test_size=0.2, random_state=42
        )
        X_test, X_val, y_test, y_val = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )

    # Load model with custom_object_scope if attention is present
    import tensorflow as tf
    import keras

    custom_objects = {}
    if config["has_attention"]:
        try:
            from keras_self_attention import SeqSelfAttention
            custom_objects["SeqSelfAttention"] = SeqSelfAttention
        except ImportError:
            pass

    print(f"Loading model: {mpath.name}")
    if custom_objects:
        from keras.utils import custom_object_scope
        with custom_object_scope(custom_objects):
            model = keras.models.load_model(str(mpath))
    else:
        model = keras.models.load_model(str(mpath))

    # Re-compile if needed
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # 1. Classification metrics + confusion matrix
    metrics = evaluate_classification(model, X_test, y_test, config, figures_dir, show=show)

    # 2. Attention weights visualization
    extract_and_plot_attention(model, X_test, data_points, labels, config, figures_dir, show=show)

    # 3. Saliency maps
    plot_saliency_maps(model, X_train, y_train, config, figures_dir, show=show)

    # Save metrics JSON
    metrics_path = results_dir / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        import json
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to: {metrics_path}")
    print("Analysis complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 4: Evaluate and analyze trained models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset", required=True, choices=["ceo2", "ce40", "csd"],
        help="Dataset to analyze.",
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to trained .h5 model file.",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display figures interactively (in addition to saving).",
    )
    args = parser.parse_args()
    run(dataset=args.dataset, model_path=args.model, show=args.show)


if __name__ == "__main__":
    main()
