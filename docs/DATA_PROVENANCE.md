# Data Provenance

## Selected source and reason

The empirical benchmark is the public **OpenML `creditcard` dataset, ID 1597**: <https://www.openml.org/d/1597>. OpenML identifies it as European card transactions from September 2013, collected and analysed through a Worldline–Machine Learning Group at Université Libre de Bruxelles (ULB) collaboration. The record describes 284,807 transactions over two days, with 492 fraud transactions (0.172%). These are real credit-card transactions, not a synthetic mobile-money simulation.

This source was selected after no openly available real Nigerian mobile-money/POS fraud dataset suitable for reproducible public training was located. The user approved the real credit-card benchmark as a **clearly disclosed proxy**. Measured outcomes cannot be presented as Nigerian mobile-money/POS performance.

## Downloaded file and source chain

The downloader retrieves a full CSV mirror from [Zenodo record 7395559](https://zenodo.org/records/7395559), DOI [10.5281/zenodo.7395559](https://doi.org/10.5281/zenodo.7395559). The Zenodo record describes its `creditcard.csv` file as sourced from Kaggle's `mlg-ulb/creditcardfraud` dataset, reports the same 284,807/492 benchmark description, and states a **CC-BY-4.0** license for that record. OpenML dataset 1597 remains the referenced dataset definition and schema. The transfer is performed in byte ranges because a single-stream OpenML download returned incomplete data in this environment; the downloader rejects any chunk with the wrong `Content-Range` or byte count.

The source file is stored at `data/raw/creditcard.csv`. `data/raw/manifest.json` records the OpenML record, Zenodo mirror URL and DOI, stated license, exact file size, expected and verified Zenodo MD5, calculated SHA-256, and download time. Expected full size: **150,828,752 bytes**. Expected MD5 from the Zenodo record: **e90efcb83d69faf99fcab8b0255024de**. The computed SHA-256 is recorded in the local manifest after a complete download. A truncated OpenML `.arff` response from an initial attempt was not accepted for training and is excluded from the project.

The loader checks required columns, numeric types, finite values, binary class labels, and minimum row count. It sorts stably by source `Time`. It does not create synthetic transactions, interpolate, balance, or duplicate source rows. Dataset updates are explicit; the downloader verifies the existing file and does not overwrite unless `--overwrite` is specified.

## Source schema and transformation

- `Time`: seconds elapsed since the first transaction in the supplied data.
- `Amount`: transaction amount in the source benchmark's units; it is **not Nigerian naira**.
- `V1` through `V28`: principal-component transformed input variables. Original business feature names and meanings are not provided due to confidentiality.
- `Class`: benchmark response label, 0 genuine and 1 fraud.

Model input uses `Amount`, `V1`–`V28`, and sine/cosine transformations of time-of-day derived from `Time`. `Class` is never supplied to the model or scoring feature builder. In training/evaluation, labels are used to select genuine training and validation rows, to keep fraud examples out of threshold selection, and to score the later holdout. Fraud labels are not model targets; no fraud positives are needed to train or set the prototype alert cutoff, although genuine examples must be known.

## Limitations and permissible claims

The benchmark is from European cardholders, covers only two days, has a heavily anonymized PCA feature space, and does not expose local mobile-money/POS behaviors, customer identities, merchant categories, terminal identifiers, Nigerian currencies, or mobile-wallet-specific fields. It is suitable for reproducing an imbalanced transaction-anomaly experiment, not for claiming operational readiness or Nigerian-domain effectiveness. Obtain consented and lawfully processed Nigerian transaction data, document its source and sampling, and re-evaluate chronologically before making local claims.

## Citation and use

OpenML supplies the following citation for this dataset: Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi, “Calibrating Probability with Undersampling for Unbalanced Classification,” IEEE Symposium on Computational Intelligence and Data Mining (CIDM), 2015. Cite both this underlying benchmark paper and the Zenodo data record if redistributing the mirror. Check current source terms and attribution requirements before reuse.
