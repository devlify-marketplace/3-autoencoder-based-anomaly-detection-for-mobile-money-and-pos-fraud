"""Feature construction shared by offline training and online inference."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["Amount", *[f"V{i}" for i in range(1, 29)], "TimeOfDaySin", "TimeOfDayCos"]
REQUIRED_INPUT_COLUMNS = ["Time", "Amount", *[f"V{i}" for i in range(1, 29)]]


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build a fixed numeric feature matrix; Time becomes cyclic time-of-day features."""
    missing = [column for column in REQUIRED_INPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required transaction fields: {missing}")
    values = frame[REQUIRED_INPUT_COLUMNS].apply(pd.to_numeric, errors="raise")
    if values.isna().any().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Transaction features must be finite numeric values.")
    seconds = np.mod(values["Time"].to_numpy(dtype=np.float64), 86_400.0)
    angle = 2.0 * np.pi * seconds / 86_400.0
    output = values[["Amount", *[f"V{i}" for i in range(1, 29)]]].copy()
    output["TimeOfDaySin"] = np.sin(angle)
    output["TimeOfDayCos"] = np.cos(angle)
    return output[FEATURE_COLUMNS].astype(np.float32)
