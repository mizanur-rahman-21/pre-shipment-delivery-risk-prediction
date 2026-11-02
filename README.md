# 📦 Leakage-Controlled Pre-Shipment Delivery-Risk Prediction with Probability Calibration, Temporal Generalization, Geographic Robustness, and Explainable Drift Analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Reproducibility: High](https://img.shields.io/badge/Reproducibility-100%25%20Verified-brightgreen.svg)](docs/reproducibility.md)

An academic machine-learning research framework for predicting order-level late-delivery risk at checkout (before warehouse fulfillment begins). The codebase implements strict prediction-time information governance, 5-fold cross-validated probability calibration, chronological out-of-time evaluation, geographic out-of-domain robustness testing, and TreeSHAP-based temporal attribution drift analysis.

---

## 📑 Table of Contents
1. [Research Overview](#1-research-overview)
2. [Research Questions](#2-research-questions)
3. [Research Contributions](#3-research-contributions)
4. [Methodological Framework](#4-methodological-framework)
5. [Dataset & Data Quality Audit](#5-dataset--data-quality-audit)
6. [Prediction-Time Feature Governance](#6-prediction-time-feature-governance)
7. [Machine-Learning Models](#7-machine-learning-models)
8. [Experimental Design](#8-experimental-design)
9. [Evaluation Metrics](#9-evaluation-metrics)
10. [Probability Calibration & Reliability Diagnostics](#10-probability-calibration--reliability-diagnostics)
11. [Temporal Generalization](#11-temporal-generalization)
12. [Geographic Out-of-Domain Generalization](#12-geographic-out-of-domain-generalization)
13. [Explainability & Temporal Attribution Drift](#13-explainability--temporal-attribution-drift)
14. [Operational Risk Scoring & Cost Simulation](#14-operational-risk-scoring--cost-simulation)
15. [Repository Structure](#15-repository-structure)
16. [Reproducibility & Execution](#16-reproducibility--execution)
17. [Experiment Traceability](#17-experiment-traceability)
18. [License & Citation](#18-license--citation)

---

## 1. Research Overview

### Operational Context
In global e-commerce and retail supply chains, unpredicted delivery delays damage customer lifetime value, trigger financial Service Level Agreement (SLA) penalties, and inflate safety stock buffers. Traditional logistics monitoring relies on **reactive post-dispatch tracking**, where order status is evaluated *after* a package has already been shipped. At that stage, operational interventions (e.g., rerouting or carrier escalation) are either cost-prohibitive or physically impossible.

### Predictive Solution & Methodological Gap
This study formulates a **leakage-controlled, pre-shipment predictive framework** to evaluate **Late Delivery Risk** at order checkout. Existing machine-learning literature in supply chain delay prediction often suffers from key methodological flaws:
- **Predictive Data Leakage**: Using post-checkout variables (e.g., actual fulfillment duration or dispatch timestamps) that are unobservable when an order is placed.
- **Unrealistic Random Train-Test Splits**: Evaluating models on random holdout splits that mask performance degradation under chronological temporal shifts and regional market changes.
- **Uncalibrated Output Probabilities**: Relying on raw classifier probabilities that are overconfident, rendering them unsuitable for cost-sensitive operational decision-making.

### Primary Research Objective
To establish a scientifically rigorous, fully reproducible methodology for pre-shipment delivery risk prediction that combines strict prediction-time feature scoping, leak-free probability calibration, temporal generalization testing, geographic out-of-domain evaluation, and explainable feature-attribution drift.

---

## 2. Research Questions

The empirical framework addresses six core research questions:

* **RQ1 (Pre-Shipment Predictability)**: *Can late-delivery risk be reliably predicted at order placement using exclusively pre-shipment information observable before warehouse dispatch begins?*
* **RQ2 (Temporal Generalization)**: *How does predictive accuracy degrade when a model trained on historical order data (2015–2016) is deployed on unseen future operating periods (2017)?*
* **RQ3 (Probability Reliability)**: *Does leak-free probability calibration (Platt Scaling and Isotonic Regression) improve predicted probability reliability across holdout, temporal, and out-of-domain test sets?*
* **RQ4 (Geographic Robustness)**: *How robustly does a pre-shipment model generalize to geographically unseen trade markets (e.g., Pacific Asia) under Out-of-Domain (OOD) evaluation?*
* **RQ5 (Attribution Drift)**: *Do feature attributions shift significantly between historical training windows and future operating periods, and how can TreeSHAP identify operational drift?*
* **RQ6 (Order-Value-Aware Decision Economics)**: *Can calibrated delivery-risk probabilities be translated into an interpretable, order-value-aware intervention policy under explicitly stated economic loss and efficacy assumptions?*

---

## 3. Research Contributions

1. **Strict Pre-Shipment Feature Governance**: Uniformly defines the prediction point at **Order Checkout** and enforces 5-fold stratified out-of-fold target encoding to guarantee zero test leakage.
2. **Leak-Free Probability Calibration Protocol**: Fits Platt Scaling and Isotonic Regression strictly within training cross-validation folds, reporting Expected Calibration Error (ECE), Maximum Calibration Error (MCE), 95% Bootstrap CIs, and Brier Score Decomposition.
3. **Connected Temporal & Geographic Out-of-Domain Validation**: Demonstrates calibration transferability across chronological time shifts (2015–2016 Train $\rightarrow$ 2017 Future Test) and Leave-One-Market-Out (LOMO) cross-validation across 5 global trade zones.
4. **Explainable Attribution Drift Analysis**: Uses TreeSHAP explainer values to quantify feature importance shifts between historical and future deployment periods without making invalid causal claims.
5. **Order-Value-Based Economic Decision Framework**: Connects calibrated risk probabilities to an order-value-dependent loss exposure model ($L_i = \alpha V_i$, baseline $\alpha = 0.05$), evaluates order-specific break-even thresholds, and conducts systematic sensitivity analysis across loss rates ($\alpha \in \{0.03, 0.05, 0.10\}$), intervention costs, and efficacy scenarios.

---

## 4. Methodological Framework

```text
Raw Ingested Dataset (DataCo 180,519 Records)
                       │
                       ▼
          [Data Quality & Scoping Audit]
       • Filter 22,469 audited exact duplicates
       • Exclude 22,470 missing target records
       • Final clean matrix: 158,049 records × 39 features
                       │
                       ▼
       [Prediction-Time Feature Scoping]
       • Enforce Order Checkout prediction point
       • Exclude post-fulfillment leakage features
       • 5-Fold Stratified Out-of-Fold Target Encoding
                       │
                       ▼
          [Classifier Benchmarking (E1)]
       • Train 7 core machine-learning models
       • Peak Holdout Performance: XGBoost (90.11% Accuracy)
       • Paired McNemar Test (χ² = 2457.39, p < 0.001 with Edwards' correction)
                       │
                       ▼
    [Leak-Free Probability Calibration (E3)]
       • Fit Platt Scaling & Isotonic Regression on X_train CV
       • Compute ECE, MCE, Brier Decomposition & 95% Bootstrap CIs
                       │
                       ▼
   ┌───────────────────┴───────────────────┐
   ▼                                       ▼
[Temporal Generalization (E2)]    [Geographic Generalization (E7)]
• Train: 2015–2016 (110,634)      • Train: Europe + LATAM (89,172)
• Test: 2017 Future (47,415)      • Test: Pacific Asia OOD (35,481)
• Calibrated ECE: 10.45% → 1.25%  • Calibrated ECE: 11.65% → 7.20%
   │                                       │
   └───────────────────┬───────────────────┘
                       ▼
      [TreeSHAP Attribution Drift Analysis]
       • Quantify top feature attribution shifts Δ (2015-16 vs 2017)
                       │
                       ▼
       [Order-Value-Based Economic Decision Layer]
         • Heterogeneous loss exposure L_i = α·V_i (benchmark α = 0.05)
         • Order-value-dependent break-even threshold p_BE = (1 - η) + C_I / (α·V_i)
         • Sensitivity analysis across α ∈ {0.03, 0.05, 0.10}, C_I, and η
```

---

## 5. Dataset & Data Quality Audit

### Dataset Description
The study analyzes the **DataCo SMART Supply Chain Dataset** (Constante et al., 2019, Mendeley Data, DOI: `10.17632/8gx2fvg2k6.1`), capturing global commercial transactions across 5 trade markets (Europe, LATAM, Pacific Asia, USCA, Africa).

### Data Audit & Disjointness Reconciliation
An exhaustive topological audit reveals that duplicate records are a strict subset of unlabelled missing-target records ($\mathcal{S}_{\text{dup}} \subset \mathcal{S}_{\text{missing}}$). All 22,469 duplicates reside within the 22,470 unlabelled rows (which contain exactly 1 unique record). Removing unlabelled rows simultaneously resolves all duplicate artifacts, yielding an exact reconciliation: $180,519 - 22,470 = \mathbf{158,049}$.

| Processing Stage | Record Count | Feature Count | Target Balance | Operational Filter & Audit Protocol |
| :--- | :---: | :---: | :---: | :--- |
| **Raw Ingested Dataset** | 180,519 | 53 | 54.82% Late | Ingested baseline load from raw repository (`10.17632/8gx2fvg2k6.1`). |
| **Exact Duplicate Check** | 22,469 | 53 | — | Audited duplicate rows; all 22,469 reside within unlabelled partition. |
| **Missing Target Removal** | 22,470 | 53 | — | Filtered unlabelled order records (removes all 22,469 duplicates + 1 unique row). |
| **Final Pre-Shipment Matrix** | **158,049** | **39** | **54.82% Late** | Usable clean pre-shipment feature matrix. |

*Note: Data files are stored locally in `data/raw/` and processed in `data/processed/`. External users can access the public DataCo repository [here](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data).*

---

## 6. Prediction-Time Feature Governance

### Single Uniform Prediction Point Definition
$$\text{Prediction Point: Immediately after order placement and before warehouse fulfillment/dispatch begins (Order Checkout)}$$

All predictive features must be strictly observable at order checkout. Post-checkout variables are excluded from model training to prevent target leakage.

### Representative Pre-Shipment Feature Roster

| Feature Symbol | Category | Availability Point | Decision | Operational Rationale | Encoding Protocol |
| :--- | :--- | :---: | :---: | :--- | :--- |
| `Days for shipment (scheduled)` | Order SLA | Checkout | **RETAINED** | Promised delivery window SLA target. | Raw numerical. |
| `Shipping Mode` | Carrier Tier | Checkout | **RETAINED** | Customer-selected fulfillment mode. | One-Hot / Frequency. |
| `Order Region` | Location | Checkout | **RETAINED** | Destination delivery region. | 5-Fold Stratified Target Encoding. |
| `mode_hour_combo_te` | Interaction | Checkout | **RETAINED** | Carrier cutoff pickup time interaction. | 5-Fold Stratified Out-of-Fold TE. |
| `Days for shipping (real)` | Post-Dispatch | Post-Delivery | 🚫 **REMOVED** | **DATA LEAKAGE**: Recorded after delivery. | Excluded. |
| `shipping date (DateOrders)` | Post-Dispatch | Post-Dispatch | 🚫 **REMOVED** | **DATA LEAKAGE**: Timestamp post-dispatch. | Excluded. |
| `Delivery Status` | Outcome | Post-Delivery | 🚫 **REMOVED** | **DATA LEAKAGE**: Direct target label. | Excluded. |
| `Order Status` | Outcome | Post-Checkout | 🚫 **REMOVED** | **DATA LEAKAGE**: Post-checkout status. | Excluded. |

*For the complete 53-feature audit table, see [docs/feature_availability.md](docs/feature_availability.md).*

---

## 7. Machine-Learning Models

The benchmark evaluates 7 standard classification algorithms:

1. **XGBoost Classifier (Tuned)**: Primary gradient-boosted tree model (`n_estimators=900`, `max_depth=16`, `learning_rate=0.025`, `subsample=0.88`, `colsample_bytree=0.88`).
2. **Random Forest Classifier**: Ensemble decision tree baseline (`n_estimators=500`, `max_depth=40`).
3. **Decision Tree Classifier**: Interpretable decision tree model (`max_depth=25`).
4. **Linear Discriminant Analysis (LDA)**: Parametric linear discriminant separator.
5. **Support Vector Machine (LinearSVC)**: Linear support vector classifier (`C=5.0`).
6. **Logistic Regression**: Standard linear logit model (`C=10.0`, `solver='lbfgs'`).
7. **Gaussian Naive Bayes**: Probabilistic baseline (`var_smoothing=1e-2`).

Implementation details are located in `src/models.py`.

---

## 8. Experimental Design

| Experiment ID | Scientific Purpose | Training Split | Test Split | Primary Metrics | Primary Output CSV File |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **E1** | Baseline Benchmark | 70% Holdout (110,634) | 30% Holdout (47,415) | Accuracy, ROC-AUC, PR-AUC, F1 | `results/metrics/E1_baseline_benchmark.csv` |
| **E2** | Temporal Generalization | Historical (2015–2016) | Future Unseen (2017) | Accuracy, Brier, ECE, MCE | `results/metrics/E2_temporal_robustness.csv` |
| **E3** | Leak-Free Calibration | $X_{\text{train}}$ 5-Fold CV | 30% Holdout (47,415) | Brier, ECE, MCE, Reliability | `results/metrics/E3_calibration_metrics.csv` |
| **E4** | Risk Stratification | Holdout Test Set | Operational Tiers | Risk Category Proportions | `results/metrics/E4_risk_score_distribution.csv` |
| **E5** | Feature Ablation Study | Feature Subsets A, B, C, D | 30% Holdout (47,415) | Accuracy, F1, ROC-AUC, PR-AUC | `results/metrics/E5_ablation_study.csv` |
| **E7** | Geographic Generalization | Europe + LATAM | Pacific Asia OOD | Accuracy, ROC-AUC, Brier, ECE | `results/metrics/E7_geographic_robustness.csv` |

### Baseline Model Benchmark Results (Experiment E1 — 47,415 Holdout Samples)

| Rank | Model Name | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | Training Time |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 1 | **XGBoost Classifier (Tuned)** | **90.11%** | **93.70%** | **88.02%** | **90.77%** | **96.13%** | **96.88%** | **57.45s** |
| 🥈 2 | **Random Forest Classifier** | **79.01%** | **88.58%** | **71.20%** | **78.95%** | **91.06%** | **92.91%** | **19.31s** |
| 🥉 3 | **Decision Tree Classifier** | **78.09%** | **81.54%** | **78.03%** | **79.75%** | **82.03%** | **87.12%** | **4.03s** |
| 4 | Linear Discriminant Analysis (LDA) | 72.23% | 84.27% | 61.19% | 70.90% | 77.32% | 83.28% | 0.43s |
| 5 | Support Vector Machine (LinearSVC) | 72.13% | 83.57% | 61.71% | 71.00% | 77.34% | 83.30% | 5.74s |
| 6 | Logistic Regression | 71.69% | 81.63% | 62.95% | 71.09% | 77.40% | 83.35% | 1.20s |
| 7 | Gaussian Naive Bayes | 69.45% | 79.92% | 59.74% | 68.37% | 72.46% | 79.27% | 0.11s |

*Full machine-readable benchmark records are stored in `results/tables/model_comparison_benchmark.csv` and `results/metrics/E1_baseline_benchmark.csv`.*

---

## 9. Evaluation Metrics

### Discrimination Metrics
- **Accuracy**: Overall proportion of correctly classified orders.
- **Precision**: Proportion of predicted late orders that were actually late ($\frac{TP}{TP + FP}$).
- **Recall (Sensitivity)**: Proportion of actual late orders correctly identified ($\frac{TP}{TP + FN}$).
- **F1 Score**: Harmonic mean of Precision and Recall.
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve.
- **PR-AUC**: Area under the Precision-Recall curve (critical for imbalanced decision problems).

### Calibration & Reliability Metrics
- **Brier Score**: Mean squared error between predicted probability $p_i$ and actual outcome $y_i \in \{0,1\}$. Higher reliability corresponds to lower Brier values ($0.0 = \text{perfect}$).
- **Expected Calibration Error (ECE)**: Weighted average absolute difference between predicted confidence and empirical accuracy across $K=10$ equal-width probability bins.
- **Maximum Calibration Error (MCE)**: Maximum absolute calibration error observed across any single bin.
- **Brier Decomposition**: $BS = \text{Reliability} - \text{Resolution} + \text{Uncertainty}$.

### Statistical Significance Test
- **McNemar's Paired Test**: Evaluates paired classification errors between XGBoost and Random Forest ($\chi^2$ statistic with 95% Bootstrap CIs).

---

## 10. Probability Calibration & Reliability Diagnostics

Classification models often output uncalibrated probabilities that misrepresent actual empirical likelihoods. To prevent data leakage, Platt Scaling (Sigmoid) and Isotonic Regression are fitted **strictly within 5-fold cross-validation folds on $X_{\text{train}}$**.

### Holdout Calibration Diagnostics (Experiment E3 — 47,415 Samples)

| Calibration Method | Brier Score | Brier 95% CI | ECE | ECE 95% CI | MCE | Reliability | Fit Protocol |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Raw Uncalibrated Model** | **0.0781** | [0.0767, 0.0796] | 5.24% | [4.98%, 5.51%] | 18.52% | 0.0056 | $X_{\text{train}}$ only (No Test Leakage) |
| **Platt Scaling (Sigmoid 5-fold CV)** | 0.0872 | [0.0856, 0.0890] | 3.45% | [3.21%, 3.74%] | 11.42% | 0.0023 | $X_{\text{train}}$ 5-fold CV |
| **Isotonic Regression (5-fold CV)** | 0.0868 | [0.0853, 0.0885] | **3.50%** | [3.24%, 3.77%] | **9.32%** | **0.0020** | $X_{\text{train}}$ 5-fold CV |

> **Methodological & Boundary Shift Note**: On random holdout, Isotonic calibration bounds ECE to **3.50%** and MCE to **9.32%**. While monotonic calibration preserves rank discrimination, applying a fixed operational cut-off ($\theta = 0.50$) to calibrated probabilities effectively shifts the decision boundary in raw probability space: $s^*(x) = m^{-1}(0.50) \approx 0.58$. Furthermore, because `CalibratedClassifierCV` averages five fold-level isotonic models, the resulting ensemble smoothing alters hard-classification metrics (increasing precision to 83.37% on the 2017 temporal horizon). Across both future temporal periods and out-of-domain markets, calibration consistently improves **both ECE and Brier score**.

---

## 11. Temporal Generalization

Random holdout validation assumes identical data distributions between training and deployment. In real-world supply chains, operating conditions shift chronologically. We train on **Historical 2015–2016 orders (110,634 samples)** and test on **Future 2017 orders (47,415 samples)**.

### Chronological Out-of-Time Performance (Experiment E2)

| Split Window | Sample Size | Model Configuration | Accuracy | F1 Score | ROC-AUC | PR-AUC | Brier Score | ECE | MCE |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Historical Train (2015–2016)** | **110,634** | 5-Fold Cross-Validation (OOF) | **92.44%** | **93.01%** | **97.42%** | **97.92%** | **0.0632** | **5.73%** | **21.62%** |
| **Future Test (2017)** | **47,415** | Raw Uncalibrated XGBoost | **69.58%** | **69.93%** | **76.10%** | **82.83%** | **0.2031** | **10.45%** | **22.07%** |
| **Future Test (2017)** | **47,415** | **Calibrated XGBoost (Isotonic)** | **71.01%** | **69.01%** | **76.16%** | **82.91%** | **0.1859** | **1.25%** | **5.35%** |

> **Key Finding**: Uncalibrated probabilities suffer severe calibration drift on future data ($ECE = 10.45\%$). Pre-fitting Isotonic calibration on historical CV folds successfully recovers probability reliability on future unseen data ($ECE = 1.25\%$, Brier score drops from $0.2031$ to $0.1859$).

---

## 12. Geographic Out-of-Domain Generalization

To test geographic robustness, models are trained on **Europe + LATAM (89,172 samples)** and evaluated on **Pacific Asia (35,481 samples out-of-domain)**.

### Geographic Out-of-Domain Performance (Experiment E7)

| Market Group | Sample Size | Evaluation Mode | Accuracy | F1 Score | ROC-AUC | PR-AUC | Brier Score | ECE | MCE |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Training Markets (Europe + LATAM)** | **89,172** | 5-Fold Cross-Validation (OOF) | **91.95%** | **92.50%** | **97.14%** | **97.68%** | **0.0651** | **4.73%** | **19.21%** |
| **Out-of-Domain Test (Pacific Asia)** | **35,481** | Raw Uncalibrated OOD Test | **70.22%** | **70.72%** | **77.26%** | **83.88%** | **0.2017** | **11.65%** | **20.98%** |
| **Out-of-Domain Test (Pacific Asia)** | **35,481** | **Calibrated XGBoost (Isotonic)** | **70.19%** | **70.99%** | **77.36%** | **83.97%** | **0.1902** | **7.20%** | **17.14%** |

> **Geographic Transfer Finding**: Isotonic calibration reduces Out-of-Domain ECE from **11.65% down to 7.20%** and Brier score from **0.2017 down to 0.1902** on Pacific Asia orders.

---

## 13. Explainability & Temporal Attribution Drift

TreeSHAP (SHapley Additive exPlanations) is used to interpret global model behavior and quantify attribution shifts between operating periods. SHAP values represent **associative model attributions**, not causal relationships.

### Top Feature Attribution Drift (2015–2016 Historical vs 2017 Future)

| Feature Symbol | Historical SHAP (2015–2016) | Future SHAP (2017) | Attribution Shift ($\Delta$) | Observed Relationship Shift |
| :--- | :---: | :---: | :---: | :--- |
| `mode_hour_combo_te` | **1.7608** | **1.6242** | **0.1366** | Attribution shift consistent with carrier cutoff schedule variation. |
| `Latitude` | 0.2024 | 0.0886 | **0.1138** | Attribution shift consistent with geographic node expansion. |
| `Order City` | 0.1508 | 0.0662 | **0.0846** | Attribution shift consistent with urban delivery density shifts. |
| `Order State` | 0.1243 | 0.0513 | **0.0730** | Attribution shift consistent with regional routing policy changes. |

Visual plots are exported to `figures/explainability/` (`fig_14_shap_summary_beeswarm.png`, `fig_15_shap_feature_importance.png`, `fig_16_shap_dependence_plots.png`).

---

## 14. Operational Risk Scoring & Cost Simulation

Calibrated probabilities are mapped into predefined operational risk tiers:
- **Low Risk ($p < 0.20$)**: Standard fulfillment processing.
- **Medium Risk ($0.20 \le p < 0.50$)**: Automated carrier tracking alert.
- **High Risk ($0.50 \le p < 0.80$)**: Priority warehouse dispatch allocation.
- **Critical Risk ($p \ge 0.80$)**: Expedited carrier rerouting intervention.

### Order-Value-Based Economic Decision Framework & Threshold Economics
- **Realized Historical Counterfactual Cost Formulation**:
  $$C_k = I_k \bigl[ C_I + Y_k (1 - \eta) L_k \bigr] + (1 - I_k) Y_k L_k$$
  where $L_k = \alpha V_{\text{ord}, k}$ (`Order Item Total` order sum). If an order is delivered on-time ($Y_k = 0$), intervention incurs only the expediting fee $C_I = \$15.00$ with zero residual late loss.
- **Observation-Specific Bayes Optimal Decision Rule**:
  $$\mathbb{E}[C \mid I_k = 1] < \mathbb{E}[C \mid I_k = 0] \iff \boxed{p_{\text{ord}, k} > \frac{C_I}{\eta \alpha V_{\text{ord}, k}} \equiv p_{\text{BE}, k}}$$
  $$I_k = \mathbf{1}\left[ p_{\text{ord}, k} > p_{\text{BE}, k} \right]$$
- **Operational Granularity & Performance**:
  - **Order-Level Operational Unit (`Order Id`, $N = 27{,}409$ unique orders)**: When intervention is evaluated per unique order ($C_I = \$15.00$ once per multi-item order), the Bayes decision rule intervenes on 3,840 orders (14.01%), yielding a **12.80% reduction in simulated economic cost (+\$33,085.28 savings)** over the \$258,522.01 baseline.
  - **Line-Item Level Execution ($N = 47{,}415$ items)**: Yields **+0.90% simulated cost reduction (+\$2,318.60)** across 1,073 intervened line items.
  - **Evaluation of Static Uniform Thresholds**: Under the specified economic assumptions, the evaluated uniform fixed-threshold policies ($\tau \in [0.05, 0.95]$) were economically inferior to the observation-specific value-aware decision rule, producing negative simulated returns (from $-69.26\%$ to $-0.89\%$) due to forced interventions on low-value orders.
- **Sensitivity Matrix**:
  - Across $\alpha \in \{0.03, 0.05, 0.10\}$: Simulated cost reductions are **+3.55%**, **+12.80%**, and **+35.46%**.
  - Across $C_I \in \{\$10, \$15, \$20\}$: Simulated cost reductions are **+26.06%**, **+12.80%**, and **+7.07%**.
  - Complete tables are saved in `results/tables/decision_cost_simulation.csv` and `results/tables/sensitivity_*.csv`.

---

## 15. Repository Structure

```text
pre-shipment-delivery-risk-prediction/
├── README.md                           # Master academic repository overview
├── RESEARCH_GUIDE.md                   # Comprehensive technical reference guide
├── LICENSE                             # MIT Open-Source License
├── CITATION.cff                        # Citation metadata file
├── requirements.txt                    # Pinned Python package dependencies
├── environment.yml                     # Conda environment configuration
├── pyproject.toml                      # PEP 517 build configuration
├── main.py                             # Master execution pipeline entrypoint
├── configs/                            # Centralized YAML configurations
│   ├── experiment.yaml                 # Experiment parameters & random seeds
│   ├── model.yaml                      # Model hyperparameter settings
│   └── paths.yaml                      # Data & output paths
├── data/                               # Data storage tiers
│   ├── README.md                       # Dataset documentation & acquisition guide
│   ├── raw/                            # Dataset_.csv (unmodified raw data)
│   └── processed/                      # Pre-shipment scoped clean dataset
├── src/                                # Modular Python package architecture
│   ├── calibration.py                  # Leak-free probability calibration & reliability
│   ├── data_loader.py                  # Ingestion & raw data cleaning
│   ├── data_preprocessing.py           # Preprocessing pipeline
│   ├── decision_simulation.py          # Cost-sensitive Bayes decision rules & simulation
│   ├── evaluation.py                   # Benchmark metrics & McNemar paired test
│   ├── experiments.py                  # Core experiment runners
│   ├── explainability.py               # TreeSHAP attribution & temporal drift
│   ├── feature_audit.py                # Pre-shipment scoping & feature audit
│   ├── feature_engineering.py          # Target encoding & transformations
│   ├── models.py                       # 7 baseline classifiers & tuning
│   ├── robustness.py                   # Temporal & geographic OOD robustness
│   └── visualization.py                # Publication figure generators
├── notebooks/                          # 8 Self-contained research notebooks
│   ├── 01_data_audit_and_scoping.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_leakage_free_preprocessing.ipynb
│   ├── 04_temporal_and_ablation_experiments.ipynb
│   ├── 05_model_calibration_and_risk_scoring.ipynb
│   ├── 06_shap_explainability_analysis.ipynb
│   ├── 07_robustness_and_geographic_generalization.ipynb
│   └── 08_order_value_economic_decision_simulation.ipynb
├── scripts/                            # Standalone execution entrypoints
│   ├── run_full_pipeline.py            # Master pipeline execution script
│   ├── compute_final_corrected_economics.py  # Order-level economics & sensitivity
│   ├── run_audit_and_reliability.py    # Order-level calibration & audit
│   └── generate_figures.py             # Publication figure organization script
├── results/                            # Experimental metrics & tables
│   ├── metrics/                        # Experiment metric CSV files (E1-E7)
│   └── tables/                         # Master summary tables & cost simulation
├── figures/                            # Publication visual figures (300 DPI)
│   ├── exploratory/                    # Figures 01-07 (Target & EDA distributions)
│   ├── model_performance/              # Figures 08-11 (Benchmark, ROC/PR curves)
│   ├── calibration/                    # Figures 12-13, 25 (Reliability diagrams)
│   ├── temporal/                       # Figures 17 & 20 (Temporal split & SHAP drift)
│   ├── geographic/                     # Figure 18 (Geographic OOD performance)
│   ├── explainability/                 # Figures 14-16 & 19 (SHAP & Risk tiers)
│   └── decision_simulation/            # Figures 21-25 (Cost curves, sensitivity, value dist)
├── docs/                               # Research documentation
│   ├── feature_availability.md         # Exhaustive feature availability audit
│   ├── experiment_traceability.md      # Paper Section → Code → Result CSV → Figure map
│   └── reproducibility.md              # Reproduction guide & environment setup
└── reports/                            # Publication manuscripts
    └── final_research_paper.md         # Full academic research paper
```

---

## 16. Reproducibility & Execution

### Environment Setup
```bash
# Clone repository
git clone https://github.com/mizanur-rahman-21/pre-shipment-delivery-risk-prediction.git
cd pre-shipment-delivery-risk-prediction

# Install dependencies via Pip
pip install -r requirements.txt
```

### Reproduce Full Research Pipeline
To run all experiments, generate CSV metric tables, and export all visual figures:
```bash
python main.py
```

*For detailed reproduction instructions and random seed settings, refer to [docs/reproducibility.md](docs/reproducibility.md).*

---

## 17. Experiment Traceability

Every table and figure in the paper maps directly to an experimental execution script and output file.

| Paper Section | Experiment ID | Source Script / Module | Primary Result CSV File | Key Figure(s) |
| :--- | :---: | :--- | :--- | :--- |
| **Section 3: Benchmark** | **E1** | `src/evaluation.py` | `results/metrics/E1_baseline_benchmark.csv` | `figures/model_performance/08_model_performance_comparison.png` |
| **Section 4: Calibration** | **E3** | `src/calibration.py` | `results/metrics/E3_calibration_metrics.csv` | `figures/calibration/12_reliability_diagram.png` |
| **Section 5: Temporal** | **E2** | `src/robustness.py` | `results/metrics/E2_temporal_robustness.csv` | `figures/temporal/17_temporal_performance_comparison.png` |
| **Section 6: Geographic** | **E7** | `src/robustness.py` | `results/metrics/E7_geographic_robustness.csv` | `figures/geographic/18_geographic_generalization_comparison.png` |
| **Section 7: TreeSHAP** | **Priority 3** | `src/explainability.py` | `results/metrics/shap_temporal_drift_analysis.csv` | `figures/explainability/14_shap_summary_beeswarm.png` |
| **Section 8: Decision Economics**| **Decision** | `src/decision_simulation.py`<br>`scripts/compute_final_corrected_economics.py` | `results/tables/decision_cost_simulation.csv`<br>`results/tables/sensitivity_*.csv` | `figures/decision_simulation/21_decision_cost_curve.png`<br>`figures/decision_simulation/25_order_level_reliability_diagram.png` |

*For complete mapping details, see [docs/experiment_traceability.md](docs/experiment_traceability.md).*

---

## 18. License & Citation

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Citation Format
```bibtex
@article{rahman2026deliveryrisk,
  title={Leakage-Controlled Pre-Shipment Delivery-Risk Prediction with Probability Calibration, Temporal Generalization, Geographic Robustness, and Explainable Drift Analysis},
  author={Rahman, Mizanur and Supply Chain Analytics Research Group},
  journal={Journal of Supply Chain Analytics},
  year={2026}
}
```
