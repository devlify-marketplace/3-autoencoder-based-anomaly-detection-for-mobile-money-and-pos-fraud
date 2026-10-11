# Experimental Results

> **Scope:** These metrics describe a held-out split of the real European credit-card benchmark (OpenML 1597). They do **not** establish detection performance on Nigerian mobile-money or POS transactions.

## Protocol

The 284,807 source records were ordered by elapsed transaction time and split into contiguous training (70%), validation (15%), and test (15%) portions. The normal-only autoencoder and RobustScaler were fitted using **198,980 genuine training rows**; 384 known training frauds were excluded from the reconstruction objective. Early stopping and threshold selection used genuine validation rows only. The fixed score threshold was selected as the higher 99.5th percentile of 42,665 genuine validation scores, targeting a 0.5% validation false-positive rate. The achieved validation rate was 0.5016% (214/42,665), the slight excess arising from the empirical quantile/tie convention. No test labels were used to fit the model or choose the threshold.

The final chronological test partition contains **42,722 transactions**, from source times 151,328 to 172,792 seconds. It contains 52 fraud labels (0.1217%) and 42,670 genuine labels. Metrics below were computed once with the validation-derived threshold, **0.472104 mean-squared reconstruction error**.

## Fixed-threshold test results

| Metric | Result |
| --- | ---: |
| Average precision | 0.4214 |
| ROC AUC | 0.9479 |
| Precision | 23.31% |
| Recall | 73.08% (38/52 frauds found) |
| F1 | 0.3535 |
| False-positive rate | 0.293% (125/42,670 genuine rows) |
| Specificity | 99.71% |
| Accuracy | 99.67% |
| Alert rate | 0.382% (163/42,722 rows) |

| Holdout outcome | Count |
| --- | ---: |
| True negatives | 42,545 |
| False positives | 125 |
| False negatives | 14 |
| True positives | 38 |

The high accuracy is expected in a dataset dominated by genuine transactions and should not be used alone to claim usefulness. Average precision is more informative for this imbalance. The alert queue contained 163 rows, of which 38 were fraud-labelled in the benchmark; the fixed cutoff also missed 14 of the 52 fraud-labelled rows. The 52 positive test examples are a small evaluation base, and precision/recall remain sample-dependent.

The achieved test false-positive rate was below the 0.5% validation target, but this does not guarantee a future rate: the two-day sample may not represent future drift, different cardholders, Nigerian payment products, POS terminals, or mobile wallets. V1–V28 are PCA components without business names. The benchmark contains no interpretable customer, merchant, terminal, mobile channel, account-history, or device signals.

## Artifacts

- `artifacts/training_results.json` — machine-readable split protocol, exact threshold, counts, metrics, and reproducibility config.
- `artifacts/model_metadata.json` — model/input details, source checksum reference, and threshold provenance.
- `artifacts/training_history.csv` — training and genuine-validation reconstruction loss (best epoch: 25).
- `artifacts/test_predictions.csv` and `artifacts/holdout_top_alerts.csv` — scores, flags, and offline benchmark labels for the test analysis; the API removes labels from its examples/scoring responses.
- `artifacts/confusion_matrix.png` and `artifacts/score_distribution.png` — diagnostic plots.
- `artifacts/feature_error_summary.csv` — mean reconstruction error by transformed input feature among test alerts.
- `data/samples/benchmark_scoring_sample.csv` — 25 real chronological holdout rows with `Class` omitted, for demonstrating CSV scoring.

## Interpretation

This is evidence that the implemented anomaly-ranking method can recover some fraud-labelled records in this specific benchmark while maintaining a low observed false-positive rate at the selected cutoff. It is **not** evidence of accuracy or readiness on Nigerian mobile-money/POS transactions. Local evaluation requires authorized, de-identified Nigerian transaction data, appropriate account/time-aware temporal partitions, operational alert-capacity decisions, and independent review.
