#!/usr/bin/env python3
"""
Stage 3: Train neural network models on PDF data.

Script equivalent of notebooks 3A/3B/3C-train-model-*.ipynb.

Handles all three datasets via ``--dataset``:
  - ``ceo2``: attention CNN, 10-class, Hyperband + single retrain
  - ``ce40``: attention CNN, 10-class, Hyperband + single retrain
  - ``csd``:  no-attention CNN, 11-class, Hyperband + 5-fold stratified CV

The per-dataset differences (architecture, class count, dropout range,
split strategy, etc.) are encoded in ``DATASET_CONFIG`` in pipeline_utils.py.

Usage examples::

    # Full pipeline for CeO2 (tune + retrain + evaluate)
    python scripts/s3_train_model.py --dataset ceo2

    # CSD with 5-fold cross-validation
    python scripts/s3_train_model.py --dataset csd

    # Skip hyperparameter tuning, use known good hyperparameters
    python scripts/s3_train_model.py --dataset ceo2 --skip-tuning \\
        --hps '{"filters1": 32, "kernel_size1": 128, "filters2": 64, ...}'

    # Only run hyperparameter tuning (skip retraining)
    python scripts/s3_train_model.py --dataset ceo2 --tune-only

This script does NOT do analysis plots (confusion matrix, attention, saliency).
Use ``scripts/s4_analyze_model.py`` for post-training visualization.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Suppress TF logging before import
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from config import get_path
from scripts.pipeline_utils import (
    DATASET_CONFIG,
    analyze_filenames,
    balance_dataset,
    create_model_from_hps,
    ensure_dir,
    get_dataset_config,
    get_labels_dir,
    get_models_dir,
    get_results_dir,
    load_pdf_data,
    make_build_fn,
    print_section,
    save_training_log,
)


def load_and_prepare_data(dataset: str, config: dict):
    """Load PDF .dat files, balance (if applicable), normalize, and split.

    Returns (data_points, labels) or, for CSD with CV, the full dataset.
    The caller decides how to split.
    """
    print_section(f"Stage 3: Train model — {config['display_name']}")

    # Gather PDF files
    pdfs_dir = get_path(config["pdfs_path_key"])
    files_calc = sorted(glob.glob(str(pdfs_dir / "*.dat")))
    if not files_calc:
        print(f"ERROR: No .dat files found in {pdfs_dir}")
        print("Run scripts/s2_prepare_pdf.py first.")
        sys.exit(1)
    print(f"Found {len(files_calc)} PDF files")

    # Show nuclearity distribution
    counts = analyze_filenames(files_calc)
    if dataset == "csd":
        sorted_counts = {str(d): counts.get(str(d), 0) for d in range(1, 10)}
        sorted_counts["polymer"] = counts.get("10", 0)
    else:
        sorted_counts = {str(d): counts.get(str(d), 0) for d in range(1, 10)}
    print(f"Counts by nuclearity: {sorted_counts}")

    # Balance (3A/3B only)
    files_calc = balance_dataset(files_calc, config)
    print(f"After balancing: {len(files_calc)} files")

    # Load data
    labels_dir = get_labels_dir(dataset, config)
    raw_data_points, labels, labels_file = load_pdf_data(
        files_calc, dataset, config, labels_dir
    )
    print(f"Data shape: {raw_data_points.shape}, Labels shape: {labels.shape}")

    # Normalize (L2, same as notebooks)
    from sklearn.preprocessing import Normalizer
    normalize = Normalizer()
    data_points = normalize.fit_transform(raw_data_points)

    return data_points, labels


def hyperparameter_tuning(data_points, labels, config, models_dir):
    """Run Hyperband hyperparameter search.

    Returns the best hyperparameters as a plain dict.
    """
    from sklearn.model_selection import train_test_split
    from keras.callbacks import EarlyStopping
    from keras_tuner import Hyperband

    print_section("Hyperparameter Tuning (Hyperband)")

    X_train, X_val, y_train, y_val = train_test_split(
        data_points, labels, test_size=config["test_size"], random_state=42,
        stratify=labels if config.get("use_stratified_kfold") else None,
    )

    build_fn = make_build_fn(config)
    tuner = Hyperband(
        build_fn,
        objective="val_accuracy",
        max_epochs=config["hyperband_max_epochs"],
        factor=3,
        directory=str(models_dir),
        project_name=config["tuner_project"],
        overwrite=True,
    )

    tuner.search(
        X_train, y_train,
        epochs=config["tuner_epochs"],
        validation_data=(X_val, y_val),
        callbacks=[EarlyStopping(monitor="val_loss", patience=3)],
    )

    tuner.search_space_summary()
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_hps = {
        "filters1": best_hp.get("filters1"),
        "kernel_size1": best_hp.get("kernel_size1"),
        "filters2": best_hp.get("filters2"),
        "kernel_size2": best_hp.get("kernel_size2"),
        "dropout": best_hp.get("dropout"),
        "dense_units": best_hp.get("dense_units"),
        "learning_rate": best_hp.get("learning_rate"),
    }
    print(f"Best hyperparameters: {best_hps}")

    # Save best model from tuning
    best_model = tuner.get_best_models(num_models=1)[0]
    best_model_path = models_dir / config["best_model_name"]
    best_model.save(best_model_path)
    print(f"Best tuned model saved to: {best_model_path}")

    return best_hps


def train_single(data_points, labels, best_hps, config, models_dir):
    """Retrain with best hyperparameters (3A/3B approach).

    Uses train/test/val split (60/20/20), retrains for ``retrain_epochs``,
    saves the best checkpoint.
    """
    from sklearn.model_selection import train_test_split
    from keras.callbacks import ModelCheckpoint

    print_section("Retraining with Best Hyperparameters")

    # Split: train (60%) / temp (40%) → test (20%) / val (20%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        data_points, labels, test_size=0.2, random_state=42
    )
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    model = create_model_from_hps(best_hps, config)
    model.summary()

    retrain_path = models_dir / config["retrain_model_name"]
    checkpoint = ModelCheckpoint(
        str(retrain_path), monitor="val_accuracy", mode="max",
        verbose=1, save_best_only=True,
    )

    history = model.fit(
        X_train, y_train,
        epochs=config["retrain_epochs"],
        validation_data=(X_val, y_val),
        callbacks=[checkpoint],
    )

    val_loss, val_acc = model.evaluate(X_val, y_val)
    print(f"Validation accuracy: {val_acc}")
    print(f"Retrained model saved to: {retrain_path}")

    log = {
        "dataset": config["display_name"],
        "hyperparameters": best_hps,
        "val_accuracy": float(val_acc),
        "val_loss": float(val_loss),
        "retrain_model": str(retrain_path),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "retrain_epochs": config["retrain_epochs"],
    }
    return log, (X_test, y_test), retrain_path


def train_cross_validated(data_points, labels, best_hps, config, models_dir):
    """5-fold stratified cross-validation (3C/CSD approach).

    Trains one model per fold, saves each checkpoint.
    """
    from sklearn.model_selection import StratifiedKFold
    from keras.callbacks import ModelCheckpoint

    print_section(f"Cross-Validation ({config.get('n_folds', 5)}-fold stratified)")

    num_folds = config.get("n_folds", 5)
    num_epochs = config["retrain_epochs"]
    kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    folds_dir = ensure_dir(models_dir / "cv_folds")
    fold_num = 1
    all_fold_results = []
    histories = []

    for train_index, val_index in kf.split(data_points, labels):
        X_train, X_val = data_points[train_index], data_points[val_index]
        y_train, y_val = labels[train_index], labels[val_index]

        model = create_model_from_hps(best_hps, config)

        checkpoint_path = folds_dir / f"csd_fold_{fold_num}_model.h5"
        checkpoint = ModelCheckpoint(
            str(checkpoint_path), monitor="val_accuracy", mode="max",
            verbose=1, save_best_only=True,
        )

        print(f"Training fold {fold_num}...")
        history = model.fit(
            X_train, y_train,
            epochs=num_epochs,
            validation_data=(X_val, y_val),
            callbacks=[checkpoint],
        )
        histories.append(history)

        # Load the best model saved by checkpoint and evaluate
        import keras
        loaded = keras.models.load_model(str(checkpoint_path))
        val_loss, val_acc = loaded.evaluate(X_val, y_val)
        print(f"Fold {fold_num} accuracy: {val_acc}")
        all_fold_results.append(float(val_acc))
        fold_num += 1

    print(f"\nAll fold accuracies: {all_fold_results}")
    print(f"Mean accuracy: {np.mean(all_fold_results):.4f}")
    print(f"Std deviation: {np.std(all_fold_results):.4f}")
    print(f"Fold models saved to: {folds_dir}")

    best_fold = int(np.argmax(all_fold_results) + 1)
    best_model_path = folds_dir / f"csd_fold_{best_fold}_model.h5"

    # Also create a held-out test set for final evaluation (matching notebook cell 13)
    from sklearn.model_selection import train_test_split
    X_train_eval, X_val_eval, y_train_eval, y_val_eval = train_test_split(
        data_points, labels, test_size=config["test_size"], random_state=42
    )

    log = {
        "dataset": config["display_name"],
        "hyperparameters": best_hps,
        "fold_accuracies": all_fold_results,
        "mean_accuracy": float(np.mean(all_fold_results)),
        "std_accuracy": float(np.std(all_fold_results)),
        "best_fold": best_fold,
        "best_model": str(best_model_path),
        "n_folds": num_folds,
        "retrain_epochs": num_epochs,
    }
    return log, (X_val_eval, y_val_eval), best_model_path


def run(dataset: str, skip_tuning: bool = False, tune_only: bool = False,
        hps_json: str | None = None):
    """Full training pipeline for one dataset."""
    config = get_dataset_config(dataset)
    models_dir = ensure_dir(get_models_dir(dataset, config))
    results_dir = ensure_dir(get_results_dir(dataset, config))

    # Load data
    data_points, labels = load_and_prepare_data(dataset, config)

    # Hyperparameter tuning
    if skip_tuning:
        if not hps_json:
            print("ERROR: --skip-tuning requires --hps")
            sys.exit(1)
        best_hps = json.loads(hps_json)
        print(f"Using provided hyperparameters: {best_hps}")
    else:
        best_hps = hyperparameter_tuning(data_points, labels, config, models_dir)

    if tune_only:
        print("\n--tune-only specified. Stopping after hyperparameter search.")
        save_training_log(
            {"dataset": config["display_name"], "hyperparameters": best_hps,
             "tune_only": True},
            results_dir / "training_log.json",
        )
        return

    # Training (single retrain or cross-validation)
    if config.get("use_stratified_kfold"):
        log, eval_data, model_path = train_cross_validated(
            data_points, labels, best_hps, config, models_dir
        )
    else:
        log, eval_data, model_path = train_single(
            data_points, labels, best_hps, config, models_dir
        )

    # Save training log
    save_training_log(log, results_dir / "training_log.json")
    print(f"\nTraining complete. Model: {model_path}")
    print(f"Run scripts/s4_analyze_model.py --dataset {dataset} --model {model_path} for analysis.")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3: Train neural network models on PDF data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset", required=True, choices=["ceo2", "ce40", "csd"],
        help="Dataset to train on.",
    )
    parser.add_argument(
        "--skip-tuning", action="store_true",
        help="Skip hyperparameter tuning; use --hps instead.",
    )
    parser.add_argument(
        "--hps", type=str, default=None,
        help='Hyperparameters JSON (use with --skip-tuning). '
             'Example: \'{"filters1": 32, "kernel_size1": 128, ...}\'',
    )
    parser.add_argument(
        "--tune-only", action="store_true",
        help="Only run hyperparameter tuning, skip retraining.",
    )
    args = parser.parse_args()
    run(
        dataset=args.dataset,
        skip_tuning=args.skip_tuning,
        tune_only=args.tune_only,
        hps_json=args.hps,
    )


if __name__ == "__main__":
    main()
