# Data Directory Guide

This directory contains raw and processed data assets for the **Leakage-Controlled and Calibration-Aware Pre-Shipment Delivery-Risk Prediction** research framework.

---

## 1. Dataset Origin & Acquisition

- **Source Dataset**: DataCo SMART Supply Chain for Big Data Analysis
- **Publisher**: DataCo Global / Mendeley Data & Kaggle
- **Primary Raw File**: `data/raw/Dataset_.csv` (180,519 historical records, ~85 MB)

### Automated or Manual Setup:
1. Download `DataCoSupplyChainDataset.csv` from Kaggle or Mendeley.
2. Place it in `data/raw/Dataset_.csv`.
3. Run `python scripts/run_full_pipeline.py` or execute Notebook `01_data_audit_and_scoping.ipynb`.

---

## 2. Pre-Shipment Scoping & Leakage Control

To guarantee zero post-dispatch target leakage, features are audited at the exact moment of order checkout:
- **Total Raw Records**: 180,519
- **Usable Pre-Shipment Samples**: 158,049 records (orders with valid scheduling & dispatch eligibility)
- **Pre-Shipment Features**: 39 strictly pre-shipment attributes (customer geography, order schedule, product category, payment mode, etc.)
- **Post-Shipment Leakage Exclusions**: Columns containing actual delivery timestamps, post-dispatch carrier status (`Delivery Status`, `Days for shipping (real)`), and late delivery flags are strictly quarantined.

---

## 3. Directory Layout

```text
data/
├── raw/
│   └── Dataset_.csv                  # Unprocessed raw DataCo records
├── processed/
│   ├── pre_shipment_dataset.csv      # Leakage-audited checkout-time dataset (158,049 rows)
│   ├── train_indices.json            # Train split index cache (70% holdout split)
│   └── test_indices.json             # Test split index cache (30% holdout split, N=47,415)
└── README.md                         # This data documentation file
```
