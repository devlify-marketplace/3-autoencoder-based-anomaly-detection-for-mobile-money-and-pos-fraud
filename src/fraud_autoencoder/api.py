"""Local API and browser dashboard for the trained anomaly-screening prototype."""
from __future__ import annotations

import io
import json
import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .features import REQUIRED_INPUT_COLUMNS, engineer_features
from .model import reconstruction_errors

PROJECT_ROOT = Path(os.environ.get("FRAUD_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
ARTIFACTS = PROJECT_ROOT / "artifacts"
WEB_DIR = PROJECT_ROOT / "web"
MAX_CSV_BYTES = 50 * 1024 * 1024
MAX_CSV_ROWS = 50_000

app = FastAPI(
    title="Payment Fraud Anomaly Screening Prototype",
    description=(
        "Academic autoencoder demo trained on a real European credit-card benchmark. "
        "Its results do not establish performance on Nigerian mobile-money or POS transactions."
    ),
    version="1.0.0",
)
app.mount("/assets", StaticFiles(directory=ARTIFACTS, check_dir=False), name="assets")


@lru_cache(maxsize=1)
def _load_runtime():
    model_path = ARTIFACTS / "model.keras"
    scaler_path = ARTIFACTS / "scaler.joblib"
    metadata_path = ARTIFACTS / "model_metadata.json"
    if not all(path.exists() for path in [model_path, scaler_path, metadata_path]):
        return None
    import tensorflow as tf
    return (
        tf.keras.models.load_model(model_path, compile=False),
        joblib.load(scaler_path),
        json.loads(metadata_path.read_text(encoding="utf-8")),
    )


def _require_runtime():
    runtime = _load_runtime()
    if runtime is None:
        raise HTTPException(status_code=503, detail="Model artifacts not found. Run the training command first.")
    return runtime


def _score_frame(frame: pd.DataFrame, model, scaler, metadata) -> dict:
    engineered = engineer_features(frame)
    scaled = scaler.transform(engineered.to_numpy(dtype=np.float32)).astype(np.float32)
    scores = reconstruction_errors(model, scaled)
    threshold = float(metadata["threshold"]["value"])
    out = pd.DataFrame({
        "transaction_row": np.arange(len(frame), dtype=int),
        "time_seconds_from_file_origin": pd.to_numeric(frame["Time"], errors="raise").to_numpy(),
        "amount_source_currency": pd.to_numeric(frame["Amount"], errors="raise").to_numpy(),
        "anomaly_score": scores,
        "decision": np.where(scores >= threshold, "review", "no_alert"),
    })
    return {
        "rows_scored": int(len(out)),
        "threshold": threshold,
        "warning": "An anomaly score indicates unusualness, not confirmed fraud. Proxy-model results do not establish Nigerian mobile-money/POS performance.",
        "results": out.to_dict(orient="records"),
    }


@app.get("/", include_in_schema=False)
def dashboard():
    index = WEB_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Dashboard file not found.")
    return FileResponse(index)


@app.get("/api/health")
def health():
    runtime = _load_runtime()
    return {
        "status": "ok",
        "model_ready": runtime is not None,
        "project": "Topic 3 — normal-only transaction autoencoder",
        "benchmark": "OpenML European credit-card fraud dataset, id 1597",
        "scope_warning": "Not a Nigerian mobile-money or POS model validation.",
    }


@app.get("/api/metrics")
def metrics():
    path = ARTIFACTS / "training_results.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Training results not found. Train the model first.")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/sample")
def sample_csv():
    path = PROJECT_ROOT / "data" / "samples" / "benchmark_scoring_sample.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Train the model to generate the benchmark scoring sample.")
    return FileResponse(path, media_type="text/csv", filename=path.name)


@app.get("/api/examples")
def examples(limit: int = 12):
    if not 1 <= limit <= 100:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 100.")
    path = ARTIFACTS / "holdout_top_alerts.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Holdout examples not found. Train the model first.")
    frame = pd.read_csv(path).head(limit)
    frame = frame.drop(columns=["benchmark_label"], errors="ignore")
    return {"scope": "Offline chronological test-set examples; not operational transaction decisions.",
            "examples": frame.replace({np.nan: None}).to_dict(orient="records")}


@app.post("/api/score")
def score_one(payload: dict[str, float]):
    model, scaler, metadata = _require_runtime()
    frame = pd.DataFrame([payload])
    missing = [column for column in REQUIRED_INPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing required fields: {missing}")
    try:
        result = _score_frame(frame, model, scaler, metadata)
        return result["results"][0] | {"threshold": result["threshold"], "warning": result["warning"]}
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/score-csv")
async def score_csv(file: UploadFile = File(...)):
    model, scaler, metadata = _require_runtime()
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="Upload a CSV file with a .csv extension.")
    contents = await file.read(MAX_CSV_BYTES + 1)
    if len(contents) > MAX_CSV_BYTES:
        raise HTTPException(status_code=413, detail="CSV exceeds the 50 MiB upload limit.")
    try:
        frame = pd.read_csv(io.BytesIO(contents))
        if len(frame) == 0:
            raise ValueError("CSV has no transaction rows.")
        if len(frame) > MAX_CSV_ROWS:
            raise HTTPException(status_code=413, detail=f"CSV exceeds the {MAX_CSV_ROWS:,}-row scoring limit.")
        missing = [column for column in REQUIRED_INPUT_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")
        return _score_frame(frame, model, scaler, metadata)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not score CSV: {exc}") from exc
    finally:
        await file.close()
