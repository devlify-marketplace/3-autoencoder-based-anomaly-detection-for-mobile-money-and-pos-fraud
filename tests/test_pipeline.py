import numpy as np
import pandas as pd
import pytest

from fraud_autoencoder.data import chronological_splits
from fraud_autoencoder.features import FEATURE_COLUMNS, REQUIRED_INPUT_COLUMNS, engineer_features
from fraud_autoencoder.metrics import calibrate_threshold, evaluate_scores


def _raw_rows(n=12):
    data = {"Time": np.arange(n, dtype=float) * 60.0, "Amount": np.arange(n, dtype=float) + 1.0}
    data.update({f"V{i}": np.linspace(-1, 1, n) + i / 100 for i in range(1, 29)})
    return pd.DataFrame(data, columns=REQUIRED_INPUT_COLUMNS)


def test_feature_engineering_has_expected_shape_and_finite_values():
    features = engineer_features(_raw_rows(5))
    assert list(features.columns) == FEATURE_COLUMNS
    assert features.shape == (5, len(FEATURE_COLUMNS))
    assert np.isfinite(features.to_numpy()).all()
    assert features.dtypes.eq(np.float32).all()


def test_feature_engineering_requires_full_input_and_finite_values():
    with pytest.raises(ValueError, match="Missing required"):
        engineer_features(pd.DataFrame({"Time": [1], "Amount": [5]}))
    bad = _raw_rows(1)
    bad.loc[0, "V3"] = np.nan
    with pytest.raises(ValueError, match="finite numeric"):
        engineer_features(bad)


def test_chronological_splits_are_contiguous_and_non_overlapping():
    frame = pd.DataFrame({"Time": np.arange(100)})
    train, validation, test = chronological_splits(frame, 0.7, 0.15)
    assert (len(train), len(validation), len(test)) == (70, 15, 15)
    assert train[-1] + 1 == validation[0]
    assert validation[-1] + 1 == test[0]
    assert np.concatenate([train, validation, test]).tolist() == list(range(100))


def test_split_rejects_overlapping_or_invalid_fractions():
    frame = pd.DataFrame({"Time": np.arange(20)})
    with pytest.raises(ValueError):
        chronological_splits(frame, 0.8, 0.2)
    with pytest.raises(ValueError):
        chronological_splits(frame, 0, 0.1)


def test_threshold_calibration_uses_normal_scores_and_respects_target_tail():
    normal_validation_scores = np.arange(1000, dtype=float)
    threshold = calibrate_threshold(normal_validation_scores, target_fpr=0.01)
    assert threshold == 990.0
    assert np.mean(normal_validation_scores >= threshold) <= 0.01
    with pytest.raises(ValueError):
        calibrate_threshold(np.array([]), target_fpr=0.01)


def test_evaluation_reports_imbalance_aware_metrics_and_confusion_counts():
    labels = np.array([0, 0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.3, 0.9, 0.4])
    result = evaluate_scores(labels, scores, threshold=0.5)
    assert result["confusion_matrix"] == {"tn": 3, "fp": 1, "fn": 1, "tp": 1}
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["false_positive_rate"] == pytest.approx(0.25)
    assert 0 <= result["average_precision"] <= 1
    assert 0 <= result["roc_auc"] <= 1
