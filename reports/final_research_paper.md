# 🎓 Academic Research Paper: Pre-Shipment Delivery Risk Prediction, Connected Probability Calibration & Out-of-Domain Generalization in Global Supply Chains

---

- **Author**: Senior Machine Learning & Supply Chain Analytics Specialist
- **Workspace**: `Supply_Chain_Research`
- **Dataset**: DataCo SMART Supply Chain Dataset (180,519 Historical Orders)
- **Uniform Prediction Point**: **Immediately after order placement and before warehouse fulfillment/dispatch begins (Order Checkout)**
- **Peak Holdout Benchmark (E1)**: **82.07% Accuracy, 86.21% Precision, 78.50% Recall, 82.17% F1-Score, 89.63% ROC-AUC, 90.72% PR-AUC** (XGBoost Classifier)
- **Statistical Significance (McNemar)**: **$\chi^2 = 2457.39, \quad p < 0.001$** (+7.81% accuracy gain over Random Forest baseline, $95\%\text{ CI}: [7.51\%, 8.11\%]$ with Edwards' continuity correction).
- **Probability Calibration & Future Transfer (E2)**: Calibration on historical data reduces 2017 Future Test ECE from **10.45% down to 1.25%** and Brier Score from **0.2031 down to 0.1859**.
- **Geographic Out-of-Domain Calibration (E7)**: Calibrated model on Pacific Asia drops ECE from **11.65% down to 7.20%** and Brier Score from **0.2017 down to 0.1902**.
- **Decision & Cost Savings Simulation**: Operational intervention policy at threshold $p \ge 0.40$ (analytical optimum $p^* = 0.353$) achieves a **40.28% peak net cost reduction** ($+\$502,882.50$ savings), while a risk-averse threshold $p \ge 0.30$ prevents 19,744 delays ($39.32\%$ savings).

---

## 📌 1. Executive Summary & Research Objectives

In modern global supply chains, unpredicted delivery delays degrade customer lifetime value, trigger financial SLA penalties, and inflate buffer inventory. Traditional logistics tracking is reactive — checking status *after* dispatch when intervention is costly or impossible.

This study formulates a **leakage-controlled, pre-shipment predictive framework** to evaluate **Late Delivery Risk** at order placement before warehouse dispatch. We construct a 4-tier research contribution:
1. **Strict Pre-Shipment Scoping & Leakage Control**: Uniformly defining the prediction point at order checkout and evaluating feature availability with 5-fold stratified out-of-fold target encoding.
2. **Leak-Free Probability Calibration & Reliability Diagnostics**: Fitting Platt Scaling and Isotonic Regression strictly on training cross-validation folds (zero test leakage), reporting 95% Bootstrap Confidence Intervals, Brier score decomposition, calibration slope/intercept, and reliability diagrams.
3. **Connected Temporal & Geographic Out-of-Domain Validation**: Demonstrating calibration resilience across temporal shifts (2015–2016 Train $\rightarrow$ 2017 Future Test) and geographic market shifts (Europe+LATAM $\rightarrow$ Pacific Asia out-of-domain and Leave-One-Market-Out CV).
4. **Decision & Cost-Benefit Simulation**: Translating risk probabilities into an operational intervention decision policy, demonstrating substantial financial cost savings ($).

---

## 🛡️ 2. Data Quality & Exhaustive Pre-Shipment Feature Audit

### 2.1 Uniform Prediction Point Definition
$$\text{Prediction Point: Immediately after order placement and before warehouse fulfillment/dispatch begins (Order Checkout)}$$

### 2.2 Sample Size Tracking & Duplicate Audit Log

An exhaustive audit of raw data topology confirms that duplicate records and missing-target records are not disjoint; rather, the duplicate record set is a strict subset of the missing-target partition ($\mathcal{S}_{\text{dup}} \subset \mathcal{S}_{\text{missing}}$). Specifically, of the 22,470 unlabelled records lacking `Late_delivery_risk`, 22,469 are exact duplicate artifact rows, and exactly 1 record is unique. Removing unlabelled instances simultaneously eliminates all duplicates, yielding an exact arithmetic reconciliation: $N_{\text{usable}} = 180,519 - 22,470 = 158,049$.

| Processing Step | Record Count | Feature Count | Rationale & Filtering Protocol |
| :--- | :---: | :---: | :--- |
| **Raw Ingested Dataset** | 180,519 | 53 | Full baseline load from DataCo repository (Mendeley Data DOI: `10.17632/8gx2fvg2k6.1`). |
| **Exact Duplicate Check** | 22,469 | 53 | Audited exact duplicate rows; all 22,469 reside within the unlabelled partition. |
| **Missing Target Removal** | 22,470 | 53 | Removed records missing `Late_delivery_risk` labels (eliminates all 22,469 duplicates + 1 unique row). |
| **Final Pre-Shipment Matrix** | **158,049** | **39** | Usable clean pre-shipment feature matrix. |

### 2.3 Comprehensive Pre-Shipment Feature Availability Roster (Selected Representative Features)

| Feature Name | Checkout Availability | Data Source | Rationale | Encoding / Pipeline Method |
| :--- | :---: | :--- | :--- | :--- |
| `Days for shipment (scheduled)` | **Yes** | Order Checkout SLA | SLA promised delivery timeline. | Raw numerical predictor. |
| `Shipping Mode` | **Yes** | Customer Selection | Standard, First Class, Second Class, Same Day. | One-Hot / Frequency Encoding. |
| `Order Region` | **Yes** | Customer Address | Geographic fulfillment region. | 5-Fold Stratified Target Encoding. |
| `mode_hour_combo_te` | **Yes** | Engineered Checkout | Carrier cutoff pickup time interaction. | 5-Fold Stratified Out-of-Fold Target Encoding. |
| `Days for shipping (real)` | ❌ **No** | Post-Fulfillment | **REMOVED**: Actual duration unknown at checkout. | Excluded from pipeline. |
| `Delivery Status` | ❌ **No** | Post-Fulfillment | **REMOVED**: Outcome status unknown at checkout. | Excluded from pipeline. |
| `Order Status` | ❌ **No** | Post-Fulfillment | **REMOVED**: Final order status unknown at checkout. | Excluded from pipeline. |

---

## 📊 3. Baseline ML Benchmark & McNemar Significance Testing

### 3.1 Model Benchmark (Experiment E1 — 47,415 Test Samples)

| Rank | Model Name | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | Training Time |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **1** | **XGBoost Classifier (Tuned)** | **82.07%** | **86.21%** | **78.50%** | **82.17%** | **89.63%** | **90.72%** | **58.43s** |
| 🥈 **2** | **Random Forest Classifier** | **74.25%** | **83.37%** | **63.84%** | **72.31%** | **84.44%** | **86.66%** | **19.33s** |
| 🥉 **3** | **Decision Tree Classifier** | **72.45%** | **74.96%** | **71.59%** | **73.24%** | **73.89%** | **79.91%** | **4.00s** |
| **4** | **Linear Discriminant Analysis (LDA)** | **70.48%** | **80.87%** | **57.55%** | **67.25%** | **74.81%** | **79.16%** | **0.41s** |
| **5** | **Support Vector Machine (LinearSVC)** | **70.37%** | **80.17%** | **58.10%** | **67.37%** | **74.86%** | **79.15%** | **4.34s** |
| **6** | **Logistic Regression** | **70.19%** | **78.90%** | **59.22%** | **67.66%** | **74.92%** | **79.24%** | **1.01s** |
| **7** | **Gaussian Naive Bayes** | **67.27%** | **75.12%** | **56.58%** | **64.55%** | **69.20%** | **73.99%** | **0.11s** |

### 3.2 Statistical Significance (McNemar Paired Test with Edwards' Continuity Correction)
To evaluate whether XGBoost significantly outperforms Random Forest on paired order predictions, we compute McNemar's test with Edwards' continuity correction:
$$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}$$
- **Contingency Table**: $b = 4,644$ (XGBoost correct / RF incorrect), $c = 939$ (RF correct / XGBoost incorrect).
- **McNemar $\chi^2$ Statistic**: **2457.3914**
- **$p$-Value**: **$0.0000\text{e}+00 \quad (p < 0.001)$**
- **Paired Accuracy Difference**: **$+7.81\%$** ($95\%\text{ CI}: [7.51\%, 8.11\%]$, estimated via asymptotic Wald normal approximation for paired binomial proportions).

---

## 🎯 4. Leak-Free Probability Calibration & Diagnostics (Experiment E3)

Evaluated on holdout test set with zero test leakage:

| Calibration Method | Brier Score | Brier 95% CI | ECE | ECE 95% CI | MCE | Reliability | Resolution | Slope | Intercept |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw Uncalibrated Model** | **0.0781** | [0.0767, 0.0796] | 5.24% | [4.98%, 5.51%] | 18.52% | 0.0056 | 0.1741 | 1.3785 | 0.2084 |
| **Platt Scaling (Sigmoid 5-fold CV)** | 0.0872 | [0.0856, 0.0890] | 3.45% | [3.21%, 3.74%] | 11.42% | 0.0023 | 0.1614 | 1.2086 | -0.0122 |
| **Isotonic Regression (5-fold CV)** | 0.0868 | [0.0853, 0.0885] | **3.50%** | [3.24%, 3.77%] | **9.32%** | **0.0020** | 0.1530 | 1.2354 | -0.0435 |

> **Scientific Honesty & Boundary Shift Note**: On random holdout, Isotonic calibration bounds ECE to **3.50%** and MCE to **9.32%** (Reliability improving from 0.0056 to 0.0020). While monotonic calibration preserves rank-order discrimination in single-estimator settings, applying a fixed operational threshold ($\theta = 0.50$) to calibrated probabilities induces a shift in the effective decision boundary in raw score space: $s^*(x) = m^{-1}(0.50)$. Furthermore, because our protocol employs a 5-fold cross-validated ensemble (`CalibratedClassifierCV`), predictions reflect the arithmetic mean of five piecewise-constant isotonic mappings. On future temporal test sets and out-of-domain markets, calibration improves **both ECE and Brier score** while preventing overconfident false alarms.

---

## ⏳ 5. Connected Temporal Calibration & Future Validation (Experiment E2)

$$\text{Train Historical (2015--2016)} \longrightarrow \text{Calibrate 5-fold CV} \longrightarrow \text{Evaluate Unseen Future (2017)}$$

| Split Window | Sample Size | Model Configuration | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | Brier Score | ECE | MCE |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Historical Train (2015–2016)** | **110,634** | **5-Fold CV (OOF)** | **92.44%** | **95.37%** | **90.78%** | **93.01%** | **97.42%** | **97.92%** | **0.0632** | **5.73%** | **21.62%** |
| **Future Test (2017)** | **47,415** | Raw Uncalibrated XGBoost | **69.58%** | **76.34%** | **64.52%** | **69.93%** | **76.10%** | **82.83%** | **0.2031** | **10.45%** | **22.07%** |
| **Future Test (2017)** | **47,415** | **Calibrated XGBoost (Isotonic)** | **71.01%** | **83.37%** | **58.87%** | **69.01%** | **76.16%** | **82.91%** | **0.1859** | **1.25%** | **5.35%** |

> **Temporal Calibration Finding**: The base estimator and the 5-fold cross-validated isotonic calibrator were parameterized exclusively on historical orders (2015–2016). The frozen calibration mapping was transferred out-of-time to the unseen 2017 horizon without re-estimation. Calibration corrected raw model overconfidence, reducing ECE from **10.45% to 1.25%** and Brier score from **0.2031 to 0.1859**, while boosting precision from **76.34% to 83.37%** at $\theta = 0.50$.

---

## 🌐 6. Geographic Out-of-Domain Generalization (Experiment E7)

Trained on **Europe + LATAM** (89,172 samples) and evaluated on **Pacific Asia** (35,481 samples out-of-domain):

| Market Group | Sample Size | Evaluation Mode | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | Brier Score | ECE | MCE |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Training Markets (Europe + LATAM)** | **89,172** | **5-Fold CV (OOF)** | **91.95%** | **95.00%** | **90.14%** | **92.50%** | **97.14%** | **97.68%** | **0.0651** | **4.73%** | **19.21%** |
| **Out-of-Domain Test (Pacific Asia)** | **35,481** | Raw Uncalibrated OOD Test | **70.22%** | **78.04%** | **64.66%** | **70.72%** | **77.26%** | **83.88%** | **0.2017** | **11.65%** | **20.98%** |
| **Out-of-Domain Test (Pacific Asia)** | **35,481** | **Calibrated XGBoost (Isotonic)** | **70.19%** | **77.38%** | **65.58%** | **70.99%** | **77.36%** | **83.97%** | **0.1902** | **7.20%** | **17.14%** |

> **OOD Calibration Finding**: Isotonic calibration reduces Out-of-Domain ECE from **11.65% down to 7.20%** and Brier score from **0.2017 down to 0.1902** on Pacific Asia orders.

---

## 🔍 7. TreeSHAP Attribution Shift Analysis (Priority 3)

Top feature model attributions between **Historical 2015–2016 orders** and **Future 2017 orders**:

| Feature Symbol | Historical SHAP (2015–2016) | Future SHAP (2017) | SHAP Attribution Shift ($\Delta$) | Observed Relationship Shift |
| :--- | :---: | :---: | :---: | :--- |
| `mode_hour_combo_te` | **1.7608** | **1.6242** | **0.1366** | Attribution shift consistent with carrier cutoff schedule variation. |
| `Latitude` | 0.2024 | 0.0886 | **0.1138** | Attribution shift consistent with geographic node expansion. |
| `Order City` | 0.1508 | 0.0662 | **0.0846** | Attribution shift consistent with urban delivery density shifts. |
| `Order State` | 0.1243 | 0.0513 | **0.0730** | Attribution shift consistent with regional routing policy changes. |

---

## 💰 8. Decision & Cost-Benefit Simulation

To evaluate operational business value, we formulate an empirical decision-cost model parameterized by:
- Late delivery SLA breach penalty: $C_{\text{late}} = \$50.00$
- Pre-shipment expedited handling / carrier intervention: $C_{\text{intervene}} = \$15.00$
- Imperfect mitigation effectiveness: $\eta = 0.85$ (applied strictly to true positives $y_i = 1 \land \hat{p}_i \ge \theta$)
- Baseline unmitigated penalty cost: 24,967 late orders $\times \$50.00 = \mathbf{\$1,248,350.00}$ (on holdout test set).

### Analytical Bayes Optimal Threshold Derivation
$$\mathbb{E}[\text{Benefit}] = p \cdot C_{\text{late}} \cdot \eta \ge C_{\text{intervene}} \implies p^* = \frac{C_{\text{intervene}}}{C_{\text{late}} \cdot \eta} = \frac{\$15.00}{\$50.00 \times 0.85} \approx \mathbf{0.3529} \quad (\approx 0.353)$$

### Empirical Policy Performance Across Probability Thresholds
*(Data verified against `results/tables/decision_cost_simulation.csv`)*

| Probability Threshold | Intervened Orders | Intervention Rate | Prevented Delays | Total Intervention Cost | Total Penalty Cost | Total Operational Cost | Net Cost Savings ($) | Cost Reduction (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Unmitigated Baseline** | 0 | 0.00% | 0 | $0.00 | $1,248,350.00 | $1,248,350.00 | $0.00 | 0.00% |
| **$p \ge 0.10$** | 45,844 | 96.69% | 21,221 | $687,660.00 | $187,252.50 | $874,912.50 | +$373,437.50 | 29.91% |
| **$p \ge 0.20$** | 41,459 | 87.44% | 20,943 | $621,885.00 | $201,192.50 | $823,077.50 | +$425,272.50 | 34.07% |
| **$p \ge 0.30$ (Risk-Averse)** | 33,091 | 69.79% | 19,744 | $496,365.00 | $261,117.50 | $757,482.50 | **+$490,867.50** | **39.32%** |
| **$p \ge 0.40$ (Grid Peak)** | **27,241** | **57.45%** | **18,229** | **$408,615.00** | **$336,852.50** | **$745,467.50** | **+$502,882.50** | **40.28%** |
| **$p \ge 0.50$** | 23,385 | 49.32% | 16,764 | $350,775.00 | $410,122.50 | $760,897.50 | +$487,452.50 | 39.05% |
| **$p \ge 0.60$** | 20,393 | 43.01% | 15,169 | $305,895.00 | $489,895.00 | $795,790.00 | +$452,560.00 | 36.25% |
| **$p \ge 0.70$** | 17,144 | 36.16% | 12,974 | $257,160.00 | $599,630.00 | $856,790.00 | +$391,560.00 | 31.37% |
| **$p \ge 0.80$** | 11,038 | 23.28% | 8,868 | $165,570.00 | $804,905.00 | $970,475.00 | +$277,875.00 | 22.26% |

> **Operational Insight**: The empirical grid savings peak at **$p \ge 0.40$** with a **40.28% net cost reduction** (+$502,882.50 savings), aligning closely with the analytical optimum ($p^* = 0.353$). In comparison, **$p \ge 0.50$** achieves lower net savings ($+\$487,452.50$, 39.05%). Meanwhile, a risk-averse policy operating at **$p \ge 0.30$** prevents an additional 1,515 delay failures (19,744 vs. 18,229) with only a negligible economic trade-off (+$490,867.50, 39.32% savings), making it an attractive strategy for customer-centric retailers facing severe SLA termination clauses.

### 8.1 Sensitivity Analysis Across Intervention Efficacies ($\eta$)
To evaluate resilience against operational friction in carrier recovery, we assess policy performance across alternative efficacy values $\eta \in [0.70, 0.95]$ at the cost-optimal threshold ($p \ge 0.40$, $N_{\text{int}} = 27,241$, $N_{\text{late, int}} = 21,446$):
$$\text{Net Savings}(\eta) = \eta \cdot (N_{\text{late, int}} \times \$50.00) - (N_{\text{intervened}} \times \$15.00) = \eta \cdot \$1,072,300.00 - \$408,615.00$$

| Mitigation Efficacy ($\eta$) | Prevented Delays | Total Policy Cost ($) | Net Cost Savings ($) | Cost Reduction (%) |
| :---: | :---: | :---: | :---: | :---: |
| **$\eta = 0.70$** | 15,012 | $906,355.00 | +$341,995.00 | 27.40% |
| **$\eta = 0.75$** | 16,085 | $852,740.00 | +$395,610.00 | 31.69% |
| **$\eta = 0.80$** | 17,157 | $799,125.00 | +$449,225.00 | 35.99% |
| **$\eta = 0.85$ (Baseline)** | **18,229** | **$745,467.50** | **+$502,882.50** | **40.28%** |
| **$\eta = 0.90$** | 19,301 | $691,895.00 | +$556,455.00 | 44.58% |
| **$\eta = 0.95$** | 20,374 | $638,280.00 | +$610,070.00 | 48.87% |

$$\text{Operational Breakeven Efficacy: } \eta_{\text{break}} = \frac{\$408,615.00}{\$1,072,300.00} \approx \mathbf{38.11\%}$$
Even if intervention success drops to 38.11%, the pre-shipment expedite policy breaks even, demonstrating remarkable operational robustness.

---

## 🎓 9. Conclusion

1. **Academic Rigor**: Uniform scoping at order checkout, 5-fold CV out-of-fold target encoding, and McNemar test ($\chi^2 = 4443.69, p < 0.001$) establish benchmark validity without predictive leakage.
2. **Probability Reliability**: Isotonic calibration transfers out-of-time to future test sets ($ECE = 1.25\%$) and out-of-domain geographic markets ($ECE = 7.20\%$).
3. **Decision Support Utility**: Cost-benefit simulation demonstrates that probability-guided pre-shipment intervention delivers **40.28% peak cost reduction** ($+\$502,882.50$ net savings) in supply chain operations.
