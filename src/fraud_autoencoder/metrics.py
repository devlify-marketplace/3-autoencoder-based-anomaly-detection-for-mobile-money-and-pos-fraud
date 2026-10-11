"""Threshold calibration and metrics for severely imbalanced fraud screening."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def calibrate_threshold(normal_validation_scores: np.ndarray, target_fpr: float = 0.005) -> float:
    """Set a score threshold using only known-normal validation scores."""
    scores = np.asarray(normal_validation_scores, dtype=float)
    if scores.ndim != 1 or scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("Threshold calibration requires a non-empty finite 1-D normal score array.")
    if not 0 < target_fpr < 1:
        raise ValueError("target_fpr must be between 0 and 1.")
    return float(np.quantile(scores, 1.0 - target_fpr, method="higher"))


def evaluate_scores(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores) or len(labels) == 0:
        raise ValueError("Labels and scores must be non-empty, aligned one-dimensional arrays.")
    if not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Expected finite scores and binary 0/1 labels.")
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    normal_count = int((labels == 0).sum())
    fraud_count = int((labels == 1).sum())
    return {
        "rows": int(len(labels)),
        "fraud_count": fraud_count,
        "genuine_count": normal_count,
        "fraud_prevalence": float(fraud_count / len(labels)),
        "threshold": float(threshold),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "average_precision": float(average_precision_score(labels, scores)) if fraud_count else None,
        "roc_auc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
        "false_positive_rate": float(fp / normal_count) if normal_count else 0.0,
        "specificity": float(tn / normal_count) if normal_count else 0.0,
        "accuracy": float((tn + tp) / len(labels)),
        "alert_rate": float((tp + fp) / len(labels)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
