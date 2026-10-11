# Project Report

## Title

**Autoencoder-Based Anomaly Detection for Mobile Money and POS Fraud**

> **Data-scope qualification:** To make a reproducible build without misrepresenting generated data as real, this implementation uses the real OpenML European credit-card fraud dataset as a proxy. Its results do **not** establish detection performance on Nigerian mobile-money or POS transactions.

## Abstract

Digital payments create a need to identify unusual transactions while fraud labels are rare, delayed, or incomplete. This project designs, implements, and evaluates a normal-only dense autoencoder for transaction anomaly screening. The project uses the public OpenML credit-card benchmark (284,807 European transactions from September 2013, including 492 benchmark-labelled frauds) as an explicitly limited real-data proxy. The source features include 28 anonymized principal components, transaction amount, and elapsed time; they do not expose named transaction, merchant, customer, terminal, or device fields. Rows are kept in chronological order and divided into contiguous training, validation, and test periods. A robust scaler is fitted only on genuine training examples; the autoencoder is trained to reconstruct only those genuine examples. An alert cutoff is calibrated from genuine validation reconstruction errors at a 0.5% target false-positive rate. Fraud labels are not used as reconstruction targets or to choose the cutoff, and the final chronological test partition is used only for evaluation. The complete code, downloaded-source checksum, saved model and scaler, metrics, plots, API, browser dashboard, and tests are included. The measured benchmark outcomes are documented in `RESULTS.md`. Because the data are European credit-card transactions rather than Nigerian mobile-money/POS records, they support a reproducible anomaly-detection demonstration only; they cannot establish local deployment effectiveness.

## 1. Introduction and problem statement

Fraud detection is difficult because legitimate transaction volume greatly exceeds known fraud, fraudulent behaviors evolve, and labels can be scarce. An anomaly-detection model offers one research direction: learn a compact representation of normal transactions, then flag examples with unusually large reconstruction error for investigation. Such a score is a screening aid, not a verdict.

The motivating topic concerns mobile money and point-of-sale fraud in Nigeria. Real transaction logs from providers are sensitive and are rarely public. Public mobile-money benchmarks found during project scoping are simulated, whereas the available public real-labelled fraud benchmark found here is European credit-card data. To avoid calling simulation “real,” the project adopts the latter as an approved proxy and documents the domain mismatch.

## 2. Aim and objectives

**Aim:** Design and evaluate a normal-only autoencoder that ranks unusual payment transactions for possible manual fraud review, using a public real-world transaction-fraud benchmark as a transparent proxy.

**Objectives:**
1. Acquire the verified public OpenML credit-card benchmark and document its source and schema.
2. Validate, order, and split transactions chronologically to reduce temporal leakage.
3. Fit robust feature scaling and a dense undercomplete autoencoder using genuine training transactions only.
4. Calibrate an alert threshold from genuine validation reconstruction scores at a preselected false-positive target.
5. Evaluate the fixed threshold on a later holdout using precision, recall, F1, average precision, ROC AUC, false-positive rate, and confusion counts.
6. Deliver a reproducible training pipeline, saved artifacts, tests, local API, and browser dashboard.
7. State the evidence gap between this European credit-card proxy and Nigerian mobile-money/POS use.

## 3. Research questions

- Can a compact autoencoder trained only on genuine transactions assign higher reconstruction errors to some previously unseen fraudulent benchmark transactions?
- What precision/recall trade-off and false-positive rate result from a 0.5% genuine-validation score-tail threshold on the later chronological holdout?
- Which conclusions are limited by the proxy dataset's period, geography, schema, and anonymization?

These questions are evaluated on the downloaded benchmark only; they are not claims about local Nigerian payment rails.

## 4. Scope and assumptions

- **Benchmark:** OpenML dataset 1597, European cardholder transactions from September 2013, covering two days.
- **Labels:** `Class = 0` genuine, `Class = 1` fraud; 492 fraud examples among 284,807 records (0.172%, as described by OpenML).
- **Inputs:** `Time`, `Amount`, and `V1`–`V28`. The 28 V fields are anonymized PCA components. Time is converted to cyclic time-of-day sine and cosine features; amount and all PCA components remain numeric.
- **Excluded inputs:** The `Class` label is never an input feature. The dataset provides no interpretable account/customer, merchant, terminal, mobile channel, device, or Nigerian-specific field.
- **Intended output:** reconstruction error and a `review`/`no_alert` screening decision. It does not confirm a fraud event or execute payment actions.
- **Claim boundary:** results do not establish performance on Nigerian mobile-money or POS transactions; real domain-specific data are required for that evaluation.

## 5. Methodology

### 5.1 Data preparation and temporal split

The CSV mirror of OpenML dataset 1597 is downloaded from the linked Zenodo record in verified byte ranges. The project records the source URL, file size, SHA-256, and verified Zenodo MD5, checks required columns and numeric finiteness, then sorts rows stably by the source `Time` field. A contiguous 70%/15%/15% split is formed in that order. No random split is used. The raw source is preserved; no transactions are generated, interpolated, balanced, or duplicated.

### 5.2 Feature processing

The 28 PCA components and transaction `Amount` are retained. Source elapsed time (seconds since the first event) is converted to two cyclic features representing the time of day in a 24-hour period. A `RobustScaler` with 10th/90th percentile range is fitted solely on genuine examples in the earliest training partition. The fitted scaler is reused unchanged for validation, test, and API inputs.

### 5.3 Autoencoder

A fully connected undercomplete autoencoder maps each scaled input through a 32-unit ReLU layer and an 8-unit bottleneck, then mirrors the representation through a 32-unit ReLU decoder to a linear output. The model is optimized with Adam and mean squared reconstruction loss. The default seed is 42. Early stopping monitors reconstruction loss on genuine validation examples only and restores best weights. The model receives genuine training rows only; known fraud rows are excluded. The class label is not a model input or reconstruction target.

For input vector \(x_i\) and reconstruction \(\hat{x}_i\), the anomaly score is mean squared reconstruction error:

\[
s_i = \frac{1}{d}\sum_{j=1}^{d}(x_{ij}-\hat{x}_{ij})^2.
\]

### 5.4 Threshold selection

The initial alert threshold is the “higher” empirical quantile at 99.5% of validation reconstruction scores from genuine validation transactions. The cutoff is selected without using fraud-labelled validation scores. Therefore, an operator does not need a set of known fraud examples to set this prototype threshold, although a trusted sample of normal transactions is still required. The empirical achieved validation false-positive rate, including ties, is reported and may differ slightly from the 0.5% target.

### 5.5 Evaluation

After freezing the model, scaler, and threshold, the later 15% chronological partition is scored once. Held-out `Class` labels are used only to compute reported metrics. The report includes average precision (important under class imbalance), ROC AUC, precision, recall, F1, false-positive rate, specificity, alert rate, and confusion counts. These are benchmark-specific descriptive results, not estimates of live Nigerian payment performance. No threshold is tuned on the test partition.

## 6. System design

The project contains modules for dataset acquisition and validation, chronological partitioning, feature engineering, model construction, calibration and metrics, training, and inference. Training artifacts include a Keras model, joblib scaler, JSON metadata and results, CSV history/predictions, feature-error summary, confusion matrix, and score distribution. The local FastAPI application exposes health, saved metrics, benchmark examples, single-row scoring, and in-memory CSV batch scoring. The browser UI warns that alerts require review. The API does not store uploaded user CSVs and does not accept payment commands.

## 7. Evaluation results

On the final chronological 42,722-row holdout, the frozen 0.472104 threshold produced average precision 0.4214, ROC AUC 0.9479, 23.31% precision, 73.08% recall, 0.3535 F1, and a 0.293% false-positive rate. The holdout contains 52 benchmark fraud labels; the model flagged 38, missed 14, and generated 125 false alerts. The 0.5% target false-positive rate was selected using genuine validation scores only; its measured validation rate was 0.5016%. Exact splits, metrics, and the confusion matrix are in [`RESULTS.md`](RESULTS.md). These results describe only the two-day European credit-card proxy, are sensitive to the small number of test frauds, and do not establish Nigerian mobile-money/POS performance. High overall accuracy alone is not treated as evidence of utility because genuine transactions dominate the data.

## 8. Ethical, privacy, and responsible-use considerations

The released benchmark is already anonymized and its principal components cannot be directly translated into operational transaction rules. A real Nigerian deployment would require permission to use suitably de-identified data, minimization of personal information, access controls, retention limits, a lawful basis, documented review procedures, and oversight by the provider. Misclassifying a genuine payment can harm customers; missing fraud also has costs. A model alert should support a trained reviewer, not automatically block a customer. The European cardholder dataset and its two-day temporal window may not represent Nigerian customers, mobile network constraints, POS terminals, payment products, or current fraud patterns. The prototype is not production-ready and is not a compliance or legal determination.

## 9. Limitations and future work

1. **Domain mismatch:** Real European card transactions are a proxy, not mobile-money or Nigerian POS data. Obtain provider-approved local records and repeat chronological validation before any local claim.
2. **PCA anonymization:** The source hides original V1–V28 semantics; customer/account history and network/terminal behavior cannot be modeled.
3. **Short horizon:** Two days of data cannot assess seasonal behavior, long-term drift, or fraud adaptation.
4. **Threshold trade-off:** A 0.5% normal-validation FPR is a transparent starting value, not a Nigerian operator's optimal capacity or cost target.
5. **Normal-only assumption:** Training relies on class labels to exclude known fraud and uses a representative genuine baseline. Undetected fraud contamination or legitimate concept drift can affect scores.
6. **One-model baseline:** This deliverable focuses on the requested autoencoder and does not establish superiority over supervised models or other anomaly detectors.
7. **No live integration:** The app only scores input rows; it does not connect to a wallet, bank, POS terminal, or authorization system.

Future research should partner with a licensed Nigerian payment provider, form a privacy-preserving feature schema with interpretable behavioral inputs, create user/account/time-aware chronological partitions, set alert capacity and cost objectives with stakeholders, compare the autoencoder against simple operational baselines, validate under temporal drift, and conduct independent security and fairness review.

## 10. Key references

1. OpenML. [Creditcard dataset, ID 1597](https://www.openml.org/d/1597). Public record with source, original data description, feature definition, and author-provided citation.
2. Liu, L. (2022). [Credit Card Fraud Detection](https://doi.org/10.5281/zenodo.7395559). Zenodo record 7395559, CC-BY-4.0; mirror record states the file originates from Kaggle `mlg-ulb/creditcardfraud`.
3. Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015). *Calibrating Probability with Undersampling for Unbalanced Classification*. IEEE Symposium on Computational Intelligence and Data Mining. Citation as listed on OpenML.
4. Hajek, P. et al. (2022). [Fraud Detection in Mobile Payment Systems using an XGBoost-based approach: An empirical study](https://pmc.ncbi.nlm.nih.gov/articles/PMC9560719/). The paper uses PaySim simulated mobile transactions and discusses privacy-related barriers to public real mobile-payment data; its dataset is not used as training evidence here.
5. Azamuke, D. et al. (2025). [A labeled synthetic mobile money transaction dataset](https://www.sciencedirect.com/science/article/pii/S2352340925002665). This dataset article explicitly identifies its records as simulated; this project does not use it for empirical model evaluation.
6. Fraud Detection Handbook, Université Libre de Bruxelles. [Real-world data discussion](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/Baseline_RealWorldData.html), including a separate private Worldline dataset and the limits of public real transaction records.
