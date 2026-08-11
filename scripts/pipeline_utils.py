"""
Shared utilities for the PDF-NN pipeline scripts.

This module centralizes:
  - Config / path management (reuses the project's config.py)
  - Dataset configuration (per-dataset parameters that differ across A/B/C)
  - Data loading and preprocessing (PDF .dat files → normalized arrays)
  - Model building (CNN architectures from the training notebooks)
  - Plotting helpers (confusion matrix, training curves, saliency maps)
  - Checkpoint helpers (skip-if-exists, training-log JSON)

All scientific logic is lifted verbatim from the original Jupyter notebooks
in ``complete workflow/`` — only notebook-isms (display, plt.show, os.chdir)
are replaced with file-based equivalents.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Python 3.7 compatibility shim (keras_tuner uses math.prod internally).
# Safe to leave in place for newer Pythons — math.prod already exists.
# ---------------------------------------------------------------------------
if not hasattr(math, "prod"):
    def _prod(values):
        result = 1
        for value in values:
            result *= int(value)
        return result
    math.prod = _prod


# ---------------------------------------------------------------------------
# Config bootstrap
# ---------------------------------------------------------------------------
def bootstrap_project_root(start: Path | None = None) -> Path:
    """Return the project root directory (parent of ``scripts/``).

    Also inserts it into ``sys.path`` so that ``import config`` works
    regardless of the caller's current working directory.
    """
    here = Path(__file__).resolve().parent
    root = here.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


PROJECT_ROOT = bootstrap_project_root()


# ---------------------------------------------------------------------------
# Dataset configuration — captures every difference between notebooks
# 3A (CeO2), 3B (Ce40), and 3C (CSD).
# ---------------------------------------------------------------------------

DATASET_CONFIG: dict[str, dict[str, Any]] = {
    "ceo2": {
        "display_name": "CeO2-based clusters (Dataset A)",
        "pdfs_path_key": "ceo2_calculated_pdfs",
        "labels_subdir": "ceo2_clusters",
        "models_path_key": "ce_models",           # DATA_ROOT/models/ce_clusters
        "results_subdir": "ceo2_clusters",
        "labels_file": "ceo2_labels.txt",
        "has_attention": True,
        "n_classes": 10,
        "nuclearities": list(range(1, 10)),       # 1–9
        "per_class_cap": 1500,
        "use_stratified_kfold": False,
        "test_size": 0.2,
        "hyperband_max_epochs": 30,
        "tuner_epochs": 50,
        "retrain_epochs": 50,
        "dropout_low": 0.2,
        "dropout_high": 0.6,
        "dropout_step": 0.1,
        "tuner_project": "hyperband_ceo2_2-12",
        "best_model_name": "tuned_sliced_ceo2_calc_2-12.h5",
        "retrain_model_name": "tuned_sliced_ceo2_calc_2-12_retrain.h5",
        "saliency_filename": "ceo2_saliency_maps.png",
        "saliency_title": "Saliency Maps for Different Labels (CeO2 clusters)",
        "saliency_color": "seagreen",
        "attention_cmap": "Greens",
    },
    "ce40": {
        "display_name": "Ce40-based clusters (Dataset B)",
        "pdfs_path_key": "ce_calculated_pdfs",
        "labels_subdir": None,                    # notebook writes labels directly under labels/
        "models_path_key": "ce_models",
        "results_subdir": "ce_clusters",
        "labels_file": "ce_labels.txt",
        "has_attention": True,
        "n_classes": 10,
        "nuclearities": list(range(1, 10)),
        "per_class_cap": 1500,
        "use_stratified_kfold": False,
        "test_size": 0.2,
        "hyperband_max_epochs": 30,
        "tuner_epochs": 50,
        "retrain_epochs": 50,
        "dropout_low": 0.2,
        "dropout_high": 0.6,
        "dropout_step": 0.1,
        "tuner_project": "hyperband_ce_2-12",
        "best_model_name": "tuned_sliced_ce_calc_2-12.h5",
        "retrain_model_name": "tuned_sliced_ce_calc_2-12_retrain.h5",
        "saliency_filename": "ce_saliency_maps.png",
        "saliency_title": "Saliency Maps for Different Labels (Ce clusters)",
        "saliency_color": "seagreen",
        "attention_cmap": "Greens",
    },
    "csd": {
        "display_name": "CSD crystal structures (Dataset C)",
        "pdfs_path_key": "csd_calculated_pdfs",
        "labels_subdir": None,
        "models_path_key": "csd_models",          # notebook uses csd_no_attention subdir
        "results_subdir": "csd",
        "labels_file": "csd_labels.txt",
        "has_attention": False,
        "n_classes": 11,
        "nuclearities": list(range(1, 11)),       # 1–10, 10 = polymer
        "per_class_cap": None,                    # no cap — uses all CSD PDFs
        "use_stratified_kfold": True,
        "n_folds": 5,
        "test_size": 0.2,
        "hyperband_max_epochs": 200,
        "tuner_epochs": 200,
        "retrain_epochs": 200,
        "dropout_low": 0.1,
        "dropout_high": 0.9,
        "dropout_step": 0.1,
        "tuner_project": "hyperband_csd_2-12",
        "best_model_name": "tuned_csd_2-12.h5",
        # CSD uses cross-validation; the "retrain model" concept maps to the
        # fold models directory:
        "retrain_model_name": "cv_folds/csd_fold_{fold}_model.h5",
        "saliency_filename": "csd_saliency_maps_no_attention.png",
        "saliency_title": "",  # CSD notebook has no suptitle
        "saliency_color": "royalblue",
        "attention_cmap": None,
    },
}


def get_dataset_config(dataset: str) -> dict[str, Any]:
    """Look up dataset configuration.  Raises KeyError for unknown datasets."""
    dataset = dataset.lower()
    if dataset not in DATASET_CONFIG:
        valid = ", ".join(sorted(DATASET_CONFIG))
        raise KeyError(f"Unknown dataset '{dataset}'. Valid options: {valid}")
    return DATASET_CONFIG[dataset]


# ---------------------------------------------------------------------------
# Path helpers (thin wrappers around config.py)
# ---------------------------------------------------------------------------
def _import_config():
    """Import the project config module, ensuring it's in sys.path."""
    import config  # noqa: F401  — re-imported below for clarity
    return config


def get_path(key: str) -> Path:
    """Proxy to ``config.get_path`` so callers don't need a separate import."""
    cfg = _import_config()
    return cfg.get_path(key)


def ensure_dir(path: Path | str) -> Path:
    """Create *path* (and parents) if it doesn't exist.  Returns the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_models_dir(dataset: str, config: dict[str, Any] | None = None) -> Path:
    """Resolve the models directory for *dataset*.

    Mirrors the notebook path logic:
      - ceo2 → models/ceo2_clusters
      - ce40 → models/ce_clusters   (the ce_models key)
      - csd  → models/csd_no_attention  (notebook uses .parent / 'csd_no_attention')
    """
    if config is None:
        config = get_dataset_config(dataset)
    if dataset == "csd":
        base = get_path("csd_models")           # DATA_ROOT/models/csd
        return base.parent / "csd_no_attention"
    return get_path(config["models_path_key"])


def get_labels_dir(dataset: str, config: dict[str, Any] | None = None) -> Path:
    """Resolve the labels directory for *dataset*.

    - ceo2 → labels/ceo2_clusters/
    - ce40 → labels/
    - csd  → labels/
    """
    if config is None:
        config = get_dataset_config(dataset)
    labels_root = get_path("labels")
    if config.get("labels_subdir"):
        return labels_root / config["labels_subdir"]
    return labels_root


def get_results_dir(dataset: str, config: dict[str, Any] | None = None) -> Path:
    """Resolve the results directory for *dataset*."""
    if config is None:
        config = get_dataset_config(dataset)
    results_root = get_path("results")
    return results_root / config["results_subdir"]


# ---------------------------------------------------------------------------
# Data loading & preprocessing  (from notebook cells: 3A-c3/4/5, 3B-*, 3C-*)
# ---------------------------------------------------------------------------
def analyze_filenames(filenames):
    """Count PDF files by nuclearity (first character of filename).

    For CSD files starting with 'p' (polymer, nuclearity 10), maps to key '10'.
    """
    counts: dict[str, int] = defaultdict(int)
    for filepath in filenames:
        filename = os.path.basename(filepath)
        if filename[0].isdigit() and filename[0] != "0":
            counts[filename[0]] += 1
        elif filename[0] == "p":
            counts["10"] += 1
    return counts


def create_output_array(filenames, selected_counts):
    """Select up to ``selected_counts[digit]`` files per nuclearity.

    Used by 3A/3B to balance the dataset (cap of 1500 per class).
    """
    output_array = []
    current_counts: dict[str, int] = defaultdict(int)
    for filepath in filenames:
        filename = os.path.basename(filepath)
        digit = filename[0]
        if digit in selected_counts and current_counts[digit] < selected_counts[digit]:
            output_array.append(filepath)
            current_counts[digit] += 1
    return output_array


def load_pdf_data(files_calc, dataset: str, config: dict[str, Any] | None = None,
                  labels_dir: Path | None = None):
    """Load PDF .dat files and extract labels (nuclearity from filename).

    Mirrors notebook cells 3A/3B/3C cell-4: reads column 1, skipping 201
    header rows and 800 footer rows, producing a 1000-point array per file.

    Parameters
    ----------
    files_calc : list of str/Path
        PDF .dat files to load.
    dataset : str
        Dataset key ('ceo2', 'ce40', 'csd').
    config : dict, optional
        Pre-resolved dataset config.
    labels_dir : Path, optional
        Where to write the labels .txt file.  Defaults to dataset labels dir.

    Returns
    -------
    raw_data_points : np.ndarray, shape (N, 1000)
    labels : np.ndarray, shape (N,)
    labels_file : Path
    """
    if config is None:
        config = get_dataset_config(dataset)
    if labels_dir is None:
        labels_dir = get_labels_dir(dataset, config)
    ensure_dir(labels_dir)
    labels_file = labels_dir / config["labels_file"]

    raw_data_points = []
    with open(labels_file, "w") as labels_out:
        for f in files_calc:
            df = pd.read_csv(
                f,
                usecols=[1],
                skiprows=201,
                header=None,
                sep=r"\s+",
                skipfooter=800,
                engine="python",
            )
            raw_data_points.append(df.values.ravel())
            filename = os.path.basename(f)
            # CSD: polymer files start with 'p' → label '10'
            if filename[0] == "p":
                labels_out.write("10\n")
            else:
                labels_out.write(filename[0] + "\n")

    raw_data_points = np.array(raw_data_points)
    labels = pd.read_csv(labels_file, header=None).values.ravel()
    print(f"Labels saved to: {labels_file}")
    return raw_data_points, labels, labels_file


def balance_dataset(files_calc, config: dict[str, Any]):
    """Apply per-class cap (3A/3B only).  Returns (possibly subsampled) file list."""
    if config.get("per_class_cap") is None:
        return files_calc  # CSD: no cap
    cap = config["per_class_cap"]
    selected_counts = {str(d): cap for d in config["nuclearities"]}
    return create_output_array(files_calc, selected_counts)


# ---------------------------------------------------------------------------
# Model building (from notebook build_model / create_model functions)
# ---------------------------------------------------------------------------
def build_model_with_attention(hp, config: dict[str, Any]):
    """CNN with SeqSelfAttention — used by datasets A and B.

    Layer-for-layer identical to the notebook ``build_model`` function.
    """
    from keras.layers import (
        Conv1D, BatchNormalization, MaxPooling1D, Dropout,
        Flatten, Dense,
    )
    from keras.models import Sequential
    from keras.optimizers import Adam
    from keras_self_attention import SeqSelfAttention
    import keras

    model = Sequential()
    model.add(Conv1D(
        filters=hp.Choice("filters1", [8, 16, 32]),
        kernel_size=hp.Choice("kernel_size1", [64, 128, 256]),
        activation="relu", input_shape=(1000, 1),
    ))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(hp.Float("dropout", config["dropout_low"], config["dropout_high"], step=config["dropout_step"])))
    model.add(Conv1D(
        filters=hp.Choice("filters2", [32, 64, 128]),
        kernel_size=hp.Choice("kernel_size2", [16, 32, 64]),
        activation="relu",
    ))
    model.add(Dropout(hp.Float("dropout", config["dropout_low"], config["dropout_high"], step=config["dropout_step"])))
    model.add(SeqSelfAttention(attention_width=16, attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL))
    model.add(Flatten())
    model.add(Dense(
        units=hp.Choice("dense_units", [64, 128, 256]),
        activation="relu", kernel_regularizer=keras.regularizers.l2(0.01),
    ))
    model.add(Dropout(hp.Float("dropout", config["dropout_low"], config["dropout_high"], step=config["dropout_step"])))
    model.add(Dense(units=config["n_classes"], activation="softmax", kernel_regularizer=keras.regularizers.l2(0.01)))
    optimizer = Adam(learning_rate=hp.Float("learning_rate", 1e-4, 1e-1, sampling="log"))
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_model_no_attention(hp, config: dict[str, Any]):
    """CNN without attention — used by dataset C (CSD).

    Layer-for-layer identical to the notebook ``build_model`` function.
    """
    from keras.layers import (
        Conv1D, BatchNormalization, MaxPooling1D, Dropout,
        Flatten, Dense,
    )
    from keras.models import Sequential
    from keras.optimizers import Adam
    import keras

    model = Sequential()
    model.add(Conv1D(
        filters=hp.Choice("filters1", [8, 16, 32]),
        kernel_size=hp.Choice("kernel_size1", [64, 128, 256]),
        activation="relu", input_shape=(1000, 1),
    ))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(hp.Float("dropout", config["dropout_low"], config["dropout_high"], step=config["dropout_step"])))
    model.add(Conv1D(
        filters=hp.Choice("filters2", [32, 64, 128]),
        kernel_size=hp.Choice("kernel_size2", [16, 32, 64]),
        activation="relu",
    ))
    model.add(Dropout(hp.Float("dropout", config["dropout_low"], config["dropout_high"], step=config["dropout_step"])))
    model.add(Flatten())
    model.add(Dense(
        units=hp.Choice("dense_units", [64, 128, 256]),
        activation="relu", kernel_regularizer=keras.regularizers.l2(0.01),
    ))
    model.add(Dropout(hp.Float("dropout", config["dropout_low"], config["dropout_high"], step=config["dropout_step"])))
    model.add(Dense(units=config["n_classes"], activation="softmax", kernel_regularizer=keras.regularizers.l2(0.01)))
    optimizer = Adam(learning_rate=hp.Float("learning_rate", 1e-4, 1e-1, sampling="log"))
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def make_build_fn(config: dict[str, Any]):
    """Return a ``build_model(hp)`` closure for keras_tuner."""
    if config["has_attention"]:
        return lambda hp: build_model_with_attention(hp, config)
    return lambda hp: build_model_no_attention(hp, config)


def create_model_from_hps(best_hps: dict[str, Any], config: dict[str, Any]):
    """Build a compiled model from resolved hyperparameters (post-tuning retrain).

    Equivalent to notebook ``create_model`` (CSD) or the inline Sequential
    construction in 3A/3B cell-11.
    """
    from keras.layers import (
        Conv1D, BatchNormalization, MaxPooling1D, Dropout,
        Flatten, Dense,
    )
    from keras.models import Sequential
    from keras.optimizers import Adam
    import keras
    if config["has_attention"]:
        from keras_self_attention import SeqSelfAttention

    model = Sequential()
    model.add(Conv1D(
        filters=best_hps["filters1"],
        kernel_size=best_hps["kernel_size1"],
        activation="relu", input_shape=(1000, 1),
    ))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(best_hps["dropout"]))
    model.add(Conv1D(
        filters=best_hps["filters2"],
        kernel_size=best_hps["kernel_size2"],
        activation="relu",
    ))
    model.add(Dropout(best_hps["dropout"]))
    if config["has_attention"]:
        model.add(SeqSelfAttention(attention_width=16, attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL))
    model.add(Flatten())
    model.add(Dense(
        units=best_hps["dense_units"],
        activation="relu", kernel_regularizer=keras.regularizers.l2(0.01),
    ))
    model.add(Dropout(best_hps["dropout"]))
    model.add(Dense(units=config["n_classes"], activation="softmax", kernel_regularizer=keras.regularizers.l2(0.01)))
    optimizer = Adam(learning_rate=best_hps["learning_rate"])
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_confusion_matrix(cm, classes, normalize=False, title="Confusion matrix",
                          cmap_name="Blues", savepath=None, show=False):
    """Plot (and optionally save) a confusion matrix.

    Lifted verbatim from the notebook ``plot_confusion_matrix`` function.
    """
    import itertools
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for scripts
    import matplotlib.pyplot as plt

    cmap = plt.cm.get_cmap(cmap_name)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")
    print(cm)

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix saved to: {savepath}")
    if show:
        plt.show()
    plt.close()


def save_training_curves(history, savepath, title="Training History"):
    """Plot loss/accuracy curves from a Keras History object."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss
    axes[0].plot(history.history["loss"], label="train loss")
    if "val_loss" in history.history:
        axes[0].plot(history.history["val_loss"], label="val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].set_title("Loss")

    # Accuracy
    axes[0].set_title("Loss")
    axes[1].plot(history.history["accuracy"], label="train acc")
    if "val_accuracy" in history.history:
        axes[1].plot(history.history["val_accuracy"], label="val acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].set_title("Accuracy")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches="tight")
    print(f"Training curves saved to: {savepath}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Checkpoint / serialization helpers
# ---------------------------------------------------------------------------
def save_training_log(log: dict[str, Any], path: Path):
    """Write a JSON training log (hyperparameters, metrics, paths)."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(log, f, indent=2, default=str)
    print(f"Training log saved to: {path}")


def load_training_log(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def skip_if_exists(path: Path | str, force: bool = False) -> bool:
    """Return True if *path* exists and *force* is False (i.e. caller should skip)."""
    exists = Path(path).exists()
    if exists and not force:
        return True
    return False


def print_section(title: str):
    """Print a visible section header (helps log readability in CI/agent output)."""
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}")
