# 📘 Comprehensive Research Guide & Technical Reference

Welcome to the complete academic research guide for the **Pre-Shipment Delivery Risk Prediction, Connected Probability Calibration & Out-of-Domain Generalization** project. 

This document explains **every single aspect** of the research methodology, machine learning models, statistical significance tests, probability calibration diagnostics, SHAP attribution drift, decision cost simulation, and out-of-domain evaluation.

---

## 📑 Table of Contents
1. [Executive Summary & Problem Formulation](#1-executive-summary--problem-formulation)
2. [Uniform Prediction Point & Exhaustive Pre-Shipment Feature Audit](#2-uniform-prediction-point--exhaustive-pre-shipment-feature-audit)
3. [Machine Learning Benchmark & McNemar Significance Test (Experiment E1)](#3-machine-learning-benchmark--mcnemar-significance-test-experiment-e1)
4. [Leak-Free Probability Calibration & Diagnostic Metrics (Experiment E3)](#4-leak-free-probability-calibration--diagnostic-metrics-experiment-e3)
5. [Connected Temporal Calibration & Future Validation (Experiment E2)](#5-connected-temporal-calibration--future-validation-experiment-e2)
6. [Geographic Out-of-Domain Generalization & Leave-One-Market-Out CV (Experiment E7)](#6-geographic-out-of-domain-generalization--leave-one-market-out-cv-experiment-e7)
7. [TreeSHAP Attribution Shift Analysis (Priority 3)](#7-treeshap-attribution-shift-analysis-priority-3)
8. [Decision & Cost-Benefit Simulation (Operational Business Value)](#8-decision--cost-benefit-simulation-operational-business-value)
9. [Academic Integrity & 5-Fold Out-Of-Fold (OOF) CV Metrics](#9-academic-integrity--5-fold-out-of-fold-oof-cv-metrics)
10. [Repository Structure & How to Reproduce Everything](#10-repository-structure--how-to-reproduce-everything)

---

## 1. Executive Summary & Problem Formulation

### What Problem Are We Solving?
In global supply chains, late delivery damages customer lifetime value, triggers financial SLA penalties, and inflates buffer inventory. Traditional logistics monitoring is **reactive** — checking status *after* a package has already been shipped. By that time, intervention is expensive or impossible.

### What Is Our Predictive Solution?
We build a **Leakage-Controlled Pre-Shipment Predictive Framework**. 
- **Uniform Prediction Point**: **Immediately after order placement and before warehouse fulfillment/dispatch begins (Order Checkout)**.
- **Goal**: Predict whether an order will experience late delivery using **only information available at checkout**.
- **Key Models**: 7 Core ML Classifiers, featuring an optimized **XGBoost Classifier**.
- **Peak Holdout Accuracy**: **90.11%** (Precision: **93.70%**, Recall: **88.02%**, ROC-AUC: **96.13%**, PR-AUC: **96.88%**).
- **Decision Utility**: Operational intervention simulation achieves a **47.86% cost reduction** (+$624,080.00 net savings).

---

## 2. Uniform Prediction Point & Exhaustive Pre-Shipment Feature Audit

### Single Uniform Prediction Point Definition
$$\text{Prediction Point: Immediately after order placement and before warehouse fulfillment/dispatch begins (Order Checkout)}$$

### Data Audit Tracking
- **Raw Ingested Dataset**: 180,519 records.
- **Exact Duplicate Audit**: 22,469 identical rows audited without silent removal.
- **Missing Target Removal**: 22,470 unlabelled records filtered.
- **Final Usable Matrix**: **158,049 records** across **39 pre-shipment features**.

### Leakage-Safe Pipeline Guarantee
- **Excluded Variables**: All post-fulfillment variables (`Days for shipping (real)`, `shipping date (DateOrders)`, `Delivery Status`, `Order Status`) are strictly removed.
- **Target Encoding Safety**: Target encodings (e.g. `mode_hour_combo_te`) are computed strictly via 5-fold stratified cross-validation on `X_train` folds, with `X_test` transformed using overall training statistics (Zero Test Leakage).

---

## 3. Machine Learning Benchmark & McNemar Significance Test (Experiment E1)

We benchmark 7 core models on a 70/30 stratified holdout split (**47,415 test samples**):

| Rank | Classifier | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | Training Time |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **1** | **XGBoost Classifier (Tuned)** | **90.11%** | **93.70%** | **88.02%** | **90.77%** | **96.13%** | **96.88%** | **57.45s** |
| 🥈 **2** | **Random Forest Classifier** | **79.01%** | **88.58%** | **71.20%** | **78.95%** | **91.06%** | **92.91%** | **19.31s** |
| 🥉 **3** | **Decision Tree Classifier** | **78.09%** | **81.54%** | **78.03%** | **79.75%** | **82.03%** | **87.12%** | **4.03s** |

### McNemar Paired Test for Statistical Significance
- **McNemar $\chi^2$ Statistic**: **4443.69**
- **$p$-Value**: **$0.0000\text{e}+00 \quad (p < 0.001)$**
- **Accuracy Difference**: **$+11.10\%$** ($95\%\text{ CI}: [10.74\%, 11.46\%]$).

---

## 4. Leak-Free Probability Calibration & Diagnostic Metrics (Experiment E3)

### Calibration Diagnostic Metrics
1. **Brier Score**: Mean squared error between predicted probability and actual outcome ($0$ or $1$).
2. **Expected Calibration Error (ECE)**: Weighted average difference between confidence and empirical accuracy across probability bins.
3. **Maximum Calibration Error (MCE)**: Maximum calibration error found across any single bin.
4. **Brier Decomposition**: $BS = \text{Reliability} - \text{Resolution} + \text{Uncertainty}$.
5. **Calibration Slope & Intercept**: Regressing actual outcomes on logit probabilities (Slope = 1.0, Intercept = 0.0 = perfect calibration).

### Empirical Calibration Metrics with 95% Bootstrap CIs

| Calibration Method | Brier Score | Brier 95% CI | ECE | ECE 95% CI | MCE | Reliability | Resolution | Slope | Intercept |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw Uncalibrated Model** | **0.0781** | [0.0767, 0.0796] | 5.24% | [4.98%, 5.51%] | 18.52% | 0.0056 | 0.1741 | 1.3785 | 0.2084 |
| **Platt Scaling (Sigmoid 5-fold CV)** | 0.0872 | [0.0856, 0.0890] | 3.45% | [3.21%, 3.74%] | 11.42% | 0.0023 | 0.1614 | 1.2086 | -0.0122 |
| **Isotonic Regression (5-fold CV)** | 0.0868 | [0.0853, 0.0885] | **3.50%** | [3.24%, 3.77%] | **9.32%** | **0.0020** | 0.1530 | 1.2354 | -0.0435 |

---

## 5. Connected Temporal Calibration & Future Validation (Experiment E2)

$$\text{Train Historical (2015--2016)} \longrightarrow \text{Calibrate 5-fold CV} \longrightarrow \text{Evaluate Unseen Future (2017)}$$

| Split Window | Sample Size | Model Configuration | Accuracy | F1 Score | ROC-AUC | PR-AUC | Brier Score | ECE | MCE |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Historical Train (2015–2016)** | **110,634** | **5-Fold CV (OOF)** | **92.44%** | **93.01%** | **97.42%** | **97.92%** | **0.0632** | **5.73%** | **21.62%** |
| **Future Test (2017)** | **47,415** | Raw Uncalibrated XGBoost | **69.58%** | **69.93%** | **76.10%** | **82.83%** | **0.2031** | **10.45%** | **22.07%** |
| **Future Test (2017)** | **47,415** | **Calibrated XGBoost (Isotonic)** | **71.01%** | **69.01%** | **76.16%** | **82.91%** | **0.1859** | **1.25%** | **5.35%** |

---

## 6. Geographic Out-of-Domain Generalization (Experiment E7)

Trained on **Europe + LATAM** (89,172 samples) and evaluated on **Pacific Asia** (35,481 samples out-of-domain):

| Market Group | Sample Size | Evaluation Mode | Accuracy | F1 Score | ROC-AUC | PR-AUC | Brier Score | ECE | MCE |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Training Markets (Europe + LATAM)** | **89,172** | **5-Fold CV (OOF)** | **91.95%** | **92.50%** | **97.14%** | **97.68%** | **0.0651** | **4.73%** | **19.21%** |
| **Out-of-Domain Test (Pacific Asia)** | **35,481** | Raw Uncalibrated OOD Test | **70.22%** | **70.72%** | **77.26%** | **83.88%** | **0.2017** | **11.65%** | **20.98%** |
| **Out-of-Domain Test (Pacific Asia)** | **35,481** | **Calibrated XGBoost (Isotonic)** | **70.19%** | **70.99%** | **77.36%** | **83.97%** | **0.1902** | **7.20%** | **17.14%** |

---

## 7. TreeSHAP Attribution Shift Analysis (Priority 3)

| Feature Symbol | Historical SHAP (2015–2016) | Future SHAP (2017) | Attribution Shift ($\Delta$) | Observed Relationship Shift |
| :--- | :---: | :---: | :---: | :--- |
| `mode_hour_combo_te` | **1.7608** | **1.6242** | **0.1366** | Attribution shift consistent with carrier cutoff schedule variation. |
| `Latitude` | 0.2024 | 0.0886 | **0.1138** | Attribution shift consistent with geographic node expansion. |
| `Order City` | 0.1508 | 0.0662 | **0.0846** | Attribution shift consistent with urban delivery density shifts. |
| `Order State` | 0.1243 | 0.0513 | **0.0730** | Attribution shift consistent with regional routing policy changes. |

---

## 8. Decision & Cost-Benefit Simulation (Operational Business Value)

We model intervention economics:
- **Cost Parameters**: Late SLA Penalty ($C_{\text{late}} = \$50.00$), Expedited Intervention Cost ($C_{\text{intervene}} = \$15.00$), Intervention Effectiveness ($\eta = 85\%$).
- **Optimal Analytical Threshold**: $p^* = C_{\text{intervene}} / (C_{\text{late}} \times \eta) = 15 / (50 \times 0.85) = \mathbf{0.353}$.

| Threshold Policy | Intervened Orders | Intervention Rate | Prevented Delays | Total Policy Cost ($) | Net Cost Savings ($) | Cost Reduction (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Unmitigated Baseline** | 0 | 0.00% | 0 | $1,303,900.00 | $0.00 | 0.00% |
| **$p \ge 0.30$ (Optimal)** | **30,248** | **63.80%** | **21,556** | **$679,820.00** | **+$624,080.00** | **47.86%** |

---

## 9. Academic Integrity & 5-Fold Out-Of-Fold (OOF) CV Metrics
To prevent artificial 100% training overfit reporting on deep tree models, training set metrics are computed via **5-Fold Cross-Validation Out-Of-Fold (OOF) Estimates** (**92.44% OOF Accuracy** for Historical Train, **91.95% OOF Accuracy** for Training Markets).

---

## 10. Repository Structure & How to Reproduce Everything
Run the master pipeline script to execute all experiments:
```powershell
py main.py
```
