"""Train and evaluate a normal-only transaction autoencoder chronologically."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import RobustScaler

from .data import LABEL_COLUMN, RAW_FEATURES, RAW_NAME, chronological_splits, load_dataset, sha256_file
from .features import FEATURE_COLUMNS, engineer_features
from .metrics import calibrate_threshold, evaluate_scores
from .model import build_autoencoder, per_feature_errors, reconstruction_errors


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _save_plots(output: Path, labels: np.ndarray, scores: np.ndarray, threshold: float) -> None:
    predictions = scores >= threshold
    matrix = np.array([
        [np.sum((labels == 0) & ~predictions), np.sum((labels == 0) & predictions)],
        [np.sum((labels == 1) & ~predictions), np.sum((labels == 1) & predictions)],
    ])
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Transactions")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["No alert", "Review"],
           yticklabels=["Genuine", "Fraud"], xlabel="Autoencoder decision", ylabel="Benchmark label",
           title="Chronological holdout confusion matrix")
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, f"{int(value):,}", ha="center", va="center", color="black", fontsize=12)
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    log_scores = np.log10(np.maximum(scores, 1e-12))
    normal = log_scores[labels == 0]
    fraud = log_scores[labels == 1]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.hist(normal, bins=70, alpha=0.72, density=True, label="Genuine", color="#3973ac")
    if fraud.size:
        ax.hist(fraud, bins=70, alpha=0.55, density=True, label="Fraud", color="#d35b48")
    ax.axvline(np.log10(max(threshold, 1e-12)), color="#a22d1c", linestyle="--", linewidth=2,
               label="Validation threshold")
    ax.set(title="Holdout reconstruction error (log10 scale)", xlabel="log10(mean squared reconstruction error)",
           ylabel="Density")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "score_distribution.png", dpi=160)
    plt.close(fig)


def train_project(root: Path, config_path: Path | None = None) -> dict:
    root = Path(root).resolve()
    config_path = config_path or root / "configs" / "default.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    raw_path = root / "data" / "raw" / RAW_NAME
    output = root / "artifacts"
    output.mkdir(parents=True, exist_ok=True)

    seed = int(config["seed"])
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass

    frame = load_dataset(raw_path)
    labels = frame[LABEL_COLUMN].to_numpy(dtype=np.int32)
    train_idx, validation_idx, test_idx = chronological_splits(
        frame, float(config["train_fraction"]), float(config["validation_fraction"])
    )
    features = engineer_features(frame).to_numpy(dtype=np.float32)
    train_normal_idx = train_idx[labels[train_idx] == 0]
    validation_normal_idx = validation_idx[labels[validation_idx] == 0]
    if not len(train_normal_idx) or not len(validation_normal_idx):
        raise ValueError("Normal-only training and threshold calibration each require genuine rows.")

    scaler = RobustScaler(quantile_range=(10.0, 90.0))
    x_train = scaler.fit_transform(features[train_normal_idx]).astype(np.float32)
    x_validation_normal = scaler.transform(features[validation_normal_idx]).astype(np.float32)
    x_validation = scaler.transform(features[validation_idx]).astype(np.float32)
    x_test = scaler.transform(features[test_idx]).astype(np.float32)

    model = build_autoencoder(
        input_dimension=features.shape[1],
        latent_dimension=int(config["latent_dimension"]),
        hidden_dimension=int(config["hidden_dimension"]),
        learning_rate=float(config["learning_rate"]),
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=int(config["patience"]), restore_best_weights=True,
            min_delta=1e-6,
        )
    ]
    history = model.fit(
        x_train, x_train,
        validation_data=(x_validation_normal, x_validation_normal),
        epochs=int(config["epochs"]),
        batch_size=int(config["batch_size"]),
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )
    pd.DataFrame(history.history).assign(epoch=np.arange(1, len(history.history["loss"]) + 1)).to_csv(
        output / "training_history.csv", index=False
    )

    validation_normal_scores = reconstruction_errors(model, x_validation_normal)
    threshold = calibrate_threshold(
        validation_normal_scores, float(config["target_validation_fpr"])
    )
    validation_scores = reconstruction_errors(model, x_validation)
    validation_labels = labels[validation_idx]
    validation_result = evaluate_scores(validation_labels, validation_scores, threshold)
    validation_normal_fpr = float(np.mean(validation_normal_scores >= threshold))

    test_scores = reconstruction_errors(model, x_test)
    test_labels = labels[test_idx]
    test_result = evaluate_scores(test_labels, test_scores, threshold)
    test_feature_errors = per_feature_errors(model, x_test)
    test_alerts = test_scores >= threshold

    test_rows = frame.iloc[test_idx]
    predictions = pd.DataFrame({
        "transaction_index": test_idx,
        "time_seconds_from_first_transaction": test_rows["Time"].to_numpy(),
        "amount_source_currency": test_rows["Amount"].to_numpy(),
        "benchmark_label": test_labels,
        "reconstruction_error": test_scores,
        "review_alert": test_alerts.astype(int),
    })
    predictions.to_csv(output / "test_predictions.csv", index=False)
    sample_dir = root / "data" / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    frame.iloc[test_idx[:25]][RAW_FEATURES].to_csv(
        sample_dir / "benchmark_scoring_sample.csv", index=False
    )
    top_alerts = predictions.sort_values("reconstruction_error", ascending=False).head(100)
    top_alerts.to_csv(output / "holdout_top_alerts.csv", index=False)

    alert_feature_mean = test_feature_errors[test_alerts].mean(axis=0) if test_alerts.any() else np.zeros(len(FEATURE_COLUMNS))
    feature_errors = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "mean_squared_reconstruction_error_on_test_alerts": alert_feature_mean,
    }).sort_values("mean_squared_reconstruction_error_on_test_alerts", ascending=False)
    feature_errors.to_csv(output / "feature_error_summary.csv", index=False)
    _save_plots(output, test_labels, test_scores, threshold)

    scaler_path = output / "scaler.joblib"
    import joblib
    joblib.dump(scaler, scaler_path)
    model.save(output / "model.keras")
    now = datetime.now(timezone.utc).isoformat()
    metadata = {
        "project_version": "1.0.0",
        "trained_at_utc": now,
        "dataset": {
            "name": "OpenML creditcard (dataset id 1597)",
            "source_url": "https://www.openml.org/d/1597",
            "raw_file": "data/raw/creditcard.csv",
            "raw_sha256": sha256_file(raw_path),
            "rows": int(len(frame)),
            "fraud_rows": int(labels.sum()),
            "genuine_rows": int((labels == 0).sum()),
            "time_start_seconds": float(frame["Time"].min()),
            "time_end_seconds": float(frame["Time"].max()),
            "original_feature_note": "V1-V28 are PCA components; only Time and Amount remain in original scale.",
        },
        "feature_columns": FEATURE_COLUMNS,
        "required_scoring_input_columns": ["Time", "Amount", *[f"V{i}" for i in range(1, 29)]],
        "model": {
            "type": "fully connected undercomplete autoencoder",
            "input_dimension": int(features.shape[1]),
            "hidden_dimension": int(config["hidden_dimension"]),
            "latent_dimension": int(config["latent_dimension"]),
            "loss": "mean squared error",
            "optimizer": "Adam",
            "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
        },
        "scaler": "RobustScaler (10th-90th percentile), fitted on chronological training genuine rows only",
        "threshold": {
            "value": float(threshold),
            "selection": "higher empirical quantile of validation genuine reconstruction scores only",
            "target_validation_false_positive_rate": float(config["target_validation_fpr"]),
            "achieved_validation_false_positive_rate": validation_normal_fpr,
        },
        "limitations": [
            "European credit-card benchmark proxy; not Nigerian mobile-money or POS transaction data.",
            "V1-V28 are anonymized PCA components, so named payment attributes and customer histories are unavailable.",
            "A review alert is an anomaly, not a finding of fraud.",
        ],
    }
    _write_json(output / "model_metadata.json", metadata)

    split_summary = {
        "train": {
            "rows_total": int(len(train_idx)),
            "rows_genuine_used_for_fit": int(len(train_normal_idx)),
            "rows_fraud_excluded_from_fit": int(np.sum(labels[train_idx] == 1)),
            "time_start": float(frame.iloc[train_idx[0]]["Time"]),
            "time_end": float(frame.iloc[train_idx[-1]]["Time"]),
        },
        "validation": {
            "rows_total": int(len(validation_idx)),
            "genuine_rows_used_for_threshold": int(len(validation_normal_idx)),
            "fraud_rows_not_used_for_threshold_or_early_stopping": int(np.sum(labels[validation_idx] == 1)),
            "time_start": float(frame.iloc[validation_idx[0]]["Time"]),
            "time_end": float(frame.iloc[validation_idx[-1]]["Time"]),
            "normal_only_threshold_fpr": validation_normal_fpr,
            "all_rows_metrics": validation_result,
        },
        "test": {
            "rows_total": int(len(test_idx)),
            "time_start": float(frame.iloc[test_idx[0]]["Time"]),
            "time_end": float(frame.iloc[test_idx[-1]]["Time"]),
            "metrics": test_result,
        },
    }
    results = {
        "title": "Autoencoder-Based Anomaly Detection for Mobile Money and POS Fraud",
        "benchmark_scope": "Real European credit-card data used as a transparent proxy only.",
        "training_protocol": "Normal-only autoencoder; known genuine training rows only; fraud labels are not fit targets.",
        "threshold_protocol": "Validation genuine rows only; target FPR selected before inspecting test metrics.",
        "split_method": "Contiguous chronological split after sorting by source Time; no random row split.",
        "threshold": float(threshold),
        "target_validation_fpr": float(config["target_validation_fpr"]),
        "achieved_validation_fpr": validation_normal_fpr,
        "split_summary": split_summary,
        "test_metrics": test_result,
        "reproducibility": {"seed": seed, "config": config},
        "trained_at_utc": now,
        "interpretation_warning": "Results do not establish performance on Nigerian mobile-money or POS transactions.",
    }
    _write_json(output / "training_results.json", results)

    dataset_summary = {
        "dataset": metadata["dataset"],
        "class_counts": {"genuine": int((labels == 0).sum()), "fraud": int((labels == 1).sum())},
        "fraud_prevalence": float(labels.mean()),
        "columns": [*FEATURE_COLUMNS, LABEL_COLUMN],
        "split_summary": split_summary,
    }
    _write_json(root / "data" / "processed" / "dataset_summary.json", dataset_summary)
    print(json.dumps({
        "best_epoch": metadata["model"]["best_epoch"],
        "threshold": threshold,
        "validation_normal_fpr": validation_normal_fpr,
        "test_metrics": test_result,
        "artifact_directory": str(output),
    }, indent=2))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    train_project(args.root, args.config)


if __name__ == "__main__":
    main()
