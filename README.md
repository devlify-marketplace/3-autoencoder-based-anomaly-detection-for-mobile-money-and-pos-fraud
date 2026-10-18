# Autoencoder-Based Anomaly Detection for Mobile Money and POS Fraud

**Topic 3** from the attached final-year-project list. This is a complete, runnable academic prototype of a normal-only transaction autoencoder, chronological benchmark evaluation, local FastAPI service, and browser dashboard.

> **Critical scope note:** The trained/evaluated data are the **real European credit-card benchmark** distributed through OpenML (dataset 1597). This is an explicitly approved proxy because an open real Nigerian mobile-money/POS fraud dataset was not located. The results **do not establish performance on Nigerian mobile-money or POS transactions**. V1–V28 are anonymized PCA components, and customer, merchant, terminal, channel, and device context are unavailable. A deployment-specific, consented mobile-money/POS dataset is required before any operational claim.

## Devlify

An academic project in the [Devlify GitHub collection](https://github.com/devlify-marketplace).

[Visit Devlify](https://devlify.com.ng) · [Browse the marketplace](https://devlify.com.ng/marketplace)

## Included

- Public dataset downloader with an integrity manifest and source attribution.
- Deterministic chronological 70%/15%/15% split; no random shuffling across time.
- Robust scaling fit on genuine training rows only.
- Dense autoencoder trained only to reconstruct genuine training transactions; fraud labels are not model targets.
- Alert threshold set to the configured 0.5% empirical tail of **genuine validation** reconstruction errors, before test evaluation.
- Imbalance-aware, held-out evaluation (average precision, ROC AUC, precision, recall, F1, false-positive rate, confusion matrix).
- Saved model, scaler, metadata, test predictions, training history, feature reconstruction summary, and plots.
- FastAPI endpoints for saved benchmark metrics, one-row scoring, and CSV batch scoring.
- Responsive browser dashboard and automated tests.
- Research report, results, and data-provenance notes under `docs/`.

A review alert is **an anomaly requiring human review**, not a finding that the transaction is fraudulent. The prototype is read-only and does not block or reverse transactions.

## Requirements and setup

Use Python **3.10–3.12**. TensorFlow training is CPU-compatible; a GPU is not required. Downloading the source dataset needs internet access; the verified CSV is about 151 MB. Reserve additional disk space for the Python/TensorFlow environment and saved model outputs. Training reads the raw CSV directly and does not create a second full-size copy.

```bash
cd /workspace/topic-03-autoencoder-mobile-money-pos-fraud
python3 -m venv .venv
source .venv/bin/activate                 # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e '.[test]'
```

If editable installation is not desired, `pip install -r requirements.txt` is also provided.

## Download the benchmark

```bash
python scripts/download_dataset.py --root .
```

The script downloads the CSV mirror of OpenML dataset 1597 from the [Zenodo record](https://zenodo.org/records/7395559) into `data/raw/creditcard.csv`. It uses checked byte ranges and verifies the complete file size and the Zenodo record's published MD5 before saving `data/raw/manifest.json` with SHA-256 and source details. A partial download is rejected. It never substitutes generated transactions. Re-download only when needed with `--overwrite`.

## Train and evaluate

```bash
python -m fraud_autoencoder.train --root .
```

Training validates and reads the raw source CSV, writes a dataset/split summary under `data/processed/`, then produces the model and evaluation files under `artifacts/`. The defaults are in `configs/default.json` and are seeded. To use another config file:

```bash
python -m fraud_autoencoder.train --root . --config configs/default.json
```

The chronological splits are contiguous after sorting by source `Time`. The model and robust scaler are fit only on genuine transactions in the first 70%. Early stopping uses only genuine transactions in the following 15%; the score cutoff is calibrated only from their reconstruction errors. The final 15% is held out until the fixed threshold is evaluated. Its labels are used only for evaluation. Do not tune the threshold against the test results and continue to describe them as held-out performance.

## Run the dashboard and API

```bash
uvicorn fraud_autoencoder.api:app --app-dir src --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. API documentation is at <http://127.0.0.1:8000/docs>.

- `GET /api/health` — model readiness and explicit proxy warning.
- `GET /api/metrics` — dataset protocol, threshold, validation summary, and held-out test metrics.
- `GET /api/sample` — download a 25-row, labels-free example made from real chronological test transactions.
- `GET /api/examples?limit=12` — highest-scoring rows from the offline holdout (not operational decisions).
- `POST /api/score` — score one JSON transaction row.
- `POST /api/score-csv` — score up to 50,000 rows / 50 MiB from CSV; upload is held in memory and not saved.

A scoring CSV must have these 30 numeric fields: `Time`, `Amount`, and `V1` through `V28`. Extra columns, including `Class`, are ignored; labels never affect scoring. The `Time` field is seconds from the beginning of the benchmark's two-day period, not a Nigeria-local timestamp. The `Amount` values are in the source dataset's units, not Nigerian naira.

## Tests

```bash
pytest -q
```

Tests cover fixed feature engineering, finite-input validation, contiguous time splits, threshold calibration using normal-only validation scores, class-imbalance metrics, and API readiness behavior. Tests do not download data or start training.

## Project files

```text
configs/default.json
 data/raw/                 source CSV mirror + source/checksum manifest
 data/processed/           split/class summary
 data/samples/             small real benchmark input sample (Class excluded)
 src/fraud_autoencoder/    data, features, model, metrics, training, and API
 scripts/                  benchmark downloader
 artifacts/                trained model, scaler, metadata, metrics, plots, predictions
 docs/                     project report, data provenance, results, limitations
 tests/                    unit and API tests
 web/index.html             local dashboard
```

## Data source and citation

OpenML, **creditcard**, dataset ID 1597 (real European card transactions from September 2013; 284,807 rows and 492 frauds, 0.172% prevalence). The complete CSV mirror is [Zenodo record 7395559](https://doi.org/10.5281/zenodo.7395559), which states CC-BY-4.0 and cites Kaggle `mlg-ulb/creditcardfraud` as the file source. Data were collected and analysed through a collaboration between Worldline and the Machine Learning Group at Université Libre de Bruxelles (ULB). The only supplied non-PCA variables are `Time` and `Amount`; `V1`–`V28` are PCA components. Source details and citation are in [`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md).

## Responsible use and limitations

This is a coursework/research prototype. It has not been tested in Nigerian payments, against POS device behavior, on mobile-money account histories, or under live transaction throughput. Data are limited to two days, so temporal drift, longer account behavior, fraud adaptation, and local patterns are not assessed. Anomaly detection can flag unusual legitimate payments and miss fraud that resembles normal activity. Real thresholds require operator risk tolerance, review capacity, privacy/security safeguards, a local validation set, and independent monitoring. Never automatically reject a payment based on this prototype.
