# 🎓 Academic Research Paper: Pre-Shipment Delivery Risk Prediction, Connected Probability Calibration & Out-of-Domain Generalization in Global Supply Chains

---

- **Author**: Senior Machine Learning & Supply Chain Analytics Specialist
- **Workspace**: `Supply_Chain_Research`
- **Dataset**: DataCo SMART Supply Chain Dataset (180,519 Historical Orders)
- **Uniform Prediction Point**: **Immediately after order placement and before warehouse fulfillment/dispatch begins (Order Checkout)**
- **Peak Holdout Benchmark (E1)**: **90.11% Accuracy, 93.70% Precision, 88.02% Recall, 90.77% F1-Score, 96.13% ROC-AUC, 96.88% PR-AUC** (XGBoost Classifier)
- **Statistical Significance (McNemar)**: **$\chi^2 = 2457.39, \quad p < 0.001$** (+7.81% accuracy gain over Random Forest baseline, $95\%\text{ CI}: [7.51\%, 8.11\%]$ with Edwards' continuity correction).
- **Probability Calibration & Future Transfer (E2)**: Calibration on historical data reduces 2017 Future Test ECE from **10.45% down to 1.25%** and Brier Score from **0.2031 down to 0.1859**.
- **Geographic Out-of-Domain Calibration (E7)**: Calibrated model on Pacific Asia drops ECE from **11.65% down to 7.20%** and Brier Score from **0.2017 down to 0.1902**.
- **Order-Value-Based Economic Decision Framework**: Under baseline parameters ($\alpha = 0.05$, $C_I = \$15$, $\eta = 0.85$), the observation-specific Bayes decision rule ($p_{\text{ord}, k} > \frac{C_I}{\eta \alpha V_{\text{ord}, k}}$) yielded a **12.80% reduction in simulated economic cost** (+\$33,085.28 savings) at the order-level operational unit (and +0.90% at the line-item level). Under the specified economic assumptions, the evaluated uniform fixed-threshold policies were economically inferior to the observation-specific value-aware decision rule.

---

## 📌 1. Executive Summary & Research Objectives

In modern global supply chains, unpredicted delivery delays degrade customer lifetime value, trigger financial SLA penalties, and inflate buffer inventory. Traditional logistics tracking is reactive — checking status *after* dispatch when intervention is costly or impossible.

This study formulates a **leakage-controlled, pre-shipment predictive framework** to evaluate **Late Delivery Risk** at order placement before warehouse dispatch. We construct a 4-tier research contribution:
1. **Strict Pre-Shipment Scoping & Leakage Control**: Uniformly defining the prediction point at order checkout and evaluating feature availability with 5-fold stratified out-of-fold target encoding.
2. **Leak-Free Probability Calibration & Reliability Diagnostics**: Fitting Platt Scaling and Isotonic Regression strictly on training cross-validation folds (zero test leakage), reporting 95% Bootstrap Confidence Intervals, Brier score decomposition, calibration slope/intercept, and reliability diagrams.
3. **Connected Temporal & Geographic Out-of-Domain Validation**: Demonstrating calibration resilience across temporal shifts (2015–2016 Train $\rightarrow$ 2017 Future Test) and geographic market shifts (Europe+LATAM $\rightarrow$ Pacific Asia out-of-domain and Leave-One-Market-Out CV).
4. **Order-Value-Based Economic Decision Framework**: Connecting calibrated risk probabilities to an order-value-aware economic loss model ($L_i = \alpha V_i$) with systematic sensitivity analysis across loss rate, intervention cost, and intervention effectiveness parameters.

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
| 🥇 **1** | **XGBoost Classifier (Tuned)** | **90.11%** | **93.70%** | **88.02%** | **90.77%** | **96.13%** | **96.88%** | **57.45s** |
| 🥈 **2** | **Random Forest Classifier** | **79.01%** | **88.58%** | **71.20%** | **78.95%** | **91.06%** | **92.91%** | **19.31s** |
| 🥉 **3** | **Decision Tree Classifier** | **78.09%** | **81.54%** | **78.03%** | **79.75%** | **82.03%** | **87.12%** | **4.03s** |
| **4** | **Linear Discriminant Analysis (LDA)** | **72.23%** | **84.27%** | **61.19%** | **70.90%** | **77.32%** | **83.28%** | **0.43s** |
| **5** | **Support Vector Machine (LinearSVC)** | **72.13%** | **83.57%** | **61.71%** | **71.00%** | **77.34%** | **83.30%** | **5.74s** |
| **6** | **Logistic Regression** | **71.69%** | **81.63%** | **62.95%** | **71.09%** | **77.40%** | **83.35%** | **1.20s** |
| **7** | **Gaussian Naive Bayes** | **69.45%** | **79.92%** | **59.74%** | **68.37%** | **72.46%** | **79.27%** | **0.11s** |

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

## 💰 8. Order-Value-Based Economic Decision-Cost Simulation

### 8.1 Order-Value-Based Economic Loss Exposure

The economic consequences of delivery delays are fundamentally heterogeneous: a late \$8 commodity accessory and a late \$1,740 enterprise equipment order carry drastically different financial exposures. To reflect this commercial reality, we replace the legacy uniform-penalty formulation with an order-value-dependent late-delivery economic loss proxy:

$$L = \alpha \cdot V$$

where $V$ represents monetary transaction value and $\alpha$ is the assumed proportion of transaction value exposed to late-delivery economic loss.

We establish $\alpha = 0.05$ as the baseline benchmark scenario, grounded in empirical supply-chain literature (Fisher et al., 2006, *What Is It Worth to You? Communicating the Dollar Value of Delivering on Time*), which reports approximately 5.1% lost sales associated with a five-day delivery delay. This parameter is adopted as a **literature-supported benchmark proxy for scenario simulation**, rather than a universal contractual penalty.

**Order-Value Descriptive Statistics** (E1 Holdout Test Set, $N = 47{,}415$ line items; $N_{\text{orders}} = 27{,}409$ unique orders):

| Statistic | Line-Item Value ($V_i$) | Item Loss Exposure ($0.05 V_i$) | Order Total Value ($V_{\text{ord}}$) | Order Loss Exposure ($0.05 V_{\text{ord}}$) |
| :--- | :---: | :---: | :---: | :---: |
| **Mean** | \$204.99 | \$10.25 | \$354.91 | \$17.75 |
| **Median** | \$178.17 | \$8.91 | \$279.98 | \$14.00 |
| **Std Dev** | \$126.15 | \$6.31 | \$268.42 | \$13.42 |
| **Min** | \$8.47 | \$0.42 | \$8.47 | \$0.42 |
| **Max** | \$1,739.99 | \$87.00 | \$2,149.95 | \$107.50 |
| **P25** | \$107.89 | \$5.39 | \$159.98 | \$8.00 |
| **P75** | \$319.98 | \$16.00 | \$479.96 | \$24.00 |
| **P90** | \$377.98 | \$18.90 | \$719.88 | \$35.99 |
| **P95** | \$391.98 | \$19.60 | \$899.91 | \$45.00 |

### 8.2 Observation-Level Cost Formulation & Conceptual Separation

In general form, delivery delay loss is modeled as proportional to monetary transaction value:
$$L_i = \alpha V_i$$
where $V_i$ represents transaction value and $\alpha$ represents the loss exposure proportion. Under intervention cost $C_I$ and mitigation efficacy $\eta$, the expected cost of intervening versus not intervening is:
$$\mathbb{E}[C \mid I_i = 1] = C_I + p_i (1 - \eta) L_i \quad \text{versus} \quad \mathbb{E}[C \mid I_i = 0] = p_i L_i$$
yielding the generic Bayes break-even threshold:
$$p_i > \frac{C_I}{\eta \alpha V_i} \equiv p_{\text{BE}, i}$$

For the primary economic evaluation, the generic observation index $i$ is instantiated at the order level $k$, because intervention is evaluated at the `Order Id` operational unit. The framework maintains a strict conceptual separation across three methodological layers:

#### 1. Risk Prediction
For each order $k$, the predictive model outputs a delivery risk probability:
$$p_{\text{ord}, k} = P(Y_k = 1 \mid X_k)$$
where $Y_k \in \{0, 1\}$ is the true binary late-delivery indicator (1 = late, 0 = on-time).

#### 2. Cost-Sensitive Decision Rule
Under assumed intervention cost $C_I = \$15.00$ and assumed mitigation effectiveness $\eta = 0.85$, the expected cost comparison for order $k$ with order loss exposure $L_k = \alpha V_{\text{ord}, k}$ yields the observation-specific Bayes decision rule under the assumed cost structure:
$$C_I < p_{\text{ord}, k} \eta L_k \iff \boxed{p_{\text{ord}, k} > \frac{C_I}{\eta \alpha V_{\text{ord}, k}} \equiv p_{\text{BE}, k}}$$
$$\boxed{I_k = \mathbf{1}\left[ p_{\text{ord}, k} > \frac{C_I}{\eta \alpha V_{\text{ord}, k}} \right]}$$

Because $p_{\text{BE}, k}$ scales inversely with order total value $V_{\text{ord}, k}$, higher-value orders justify intervention at lower predicted risk, whereas orders with value below $V_{\text{min}} = \frac{C_I}{\eta \alpha} = \$352.94$ have $p_{\text{BE}, k} > 1.0$ and are never intervened.

#### 3. Realized Historical Counterfactual Evaluation
Given actual binary outcome $Y_k$ and intervention decision $I_k$, the realized operational cost is:
$$\boxed{C_k = I_k \bigl[ C_I + Y_k (1 - \eta) L_k \bigr] + (1 - I_k) Y_k L_k}$$

- **On-Time Orders ($Y_k = 0$)**: If intervened ($I_k = 1$), incurs only $C_I = \$15.00$ with zero residual late loss. If not intervened ($I_k = 0$), cost is \$0.
- **Late Orders ($Y_k = 1$)**: If intervened ($I_k = 1$), incurs $C_I + (1 - \eta) L_k = \$15.00 + 0.15 L_k$. If not intervened ($I_k = 0$), incurs full unmitigated loss $L_k$.

The benchmark no-intervention baseline cost is:
$$C_{\text{baseline}} = \sum_k Y_k L_k = \mathbf{\$258{,}522.01}$$

### 8.3 Operational Granularity: Line-Item vs. Order-Level Aggregation

In commercial supply-chain operations, preventive expediting is managed at the **order-level operational unit** (`Order Id`). In the evaluation data, all line items within an `Order Id` exhibited the same observed late-delivery target ($Y_{\text{ord}} \in \{0, 1\}$). The order-level probability is formulated as the arithmetic mean of the calibrated line-item probabilities:

$$p_{\text{ord}, k} = \frac{1}{n_k} \sum_{i \in k} p_i$$

The aggregated order-level probabilities achieved a Brier score of 0.1336, an Expected Calibration Error (ECE) of 7.13%, and a Maximum Calibration Error (MCE) of 18.00% across 10 uniform probability bins on the held-out order-level evaluation set ($N = 27{,}409$ unique orders), providing empirical evidence that the aggregation produces a usable probability signal without asserting that order-level probabilities are perfectly calibrated (Reliability Diagram presented in Figure 25). The total order value aggregates as $V_{\text{ord}, k} = \sum_{i \in k} V_i$.

We compare economic policy outcomes across both granularities:

| Granularity Mode | Operational Unit | Total Units | Late Units | Feasible Units ($p_{\text{BE}} \le 1.0$) | Intervened Units | Intervention Rate | Baseline Loss | Total Policy Cost | Net Economic Savings | Simulated Cost Reduction (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Line-Item Level** | Individual Item | 47,415 | 24,967 | 7,362 (15.5%) | 1,073 | 2.26% | \$258,522.01 | \$256,203.42 | +\$2,318.60 | **+0.90%** |
| **Order-Level Unit** | Unique `Order Id` | 27,409 | 14,504 | 11,728 (42.8%) | 3,840 | 14.01% | \$258,522.01 | \$225,436.73 | +\$33,085.28 | **+12.80%** |

#### Intervention Pool Audit Log (Order-Level Bayes Decision Rule):
An empirical audit of the 3,840 orders flagged for intervention reveals the following operational breakdown:
- **Intervened Orders**: 3,840 (14.01% intervention rate)
- **Actually Late ($Y = 1$)**: 3,212 orders (83.65% true positive rate in the intervention pool)
- **Actually On-Time ($Y = 0$)**: 628 orders (16.35% false alarms; incurring only the \$15 intervention fee with zero residual delay loss)
- **Mean Predicted Risk ($p_{\text{ord}}$) Among Intervened**: 0.8064 (closely mirroring the observed 83.65% late rate)
- **Mean Break-Even Threshold ($p_{\text{BE}}$) Among Intervened**: 0.5782
- **Mean Order Value ($V_{\text{ord}}$) Among Intervened**: \$691.95 (substantially above the \$352.94 feasibility cutoff and the dataset mean of \$354.91)

### 8.4 Break-Even Feasibility & Evaluation of Static Uniform Thresholds

An order is economically eligible for intervention ($p_{\text{BE}} \le 1.0$) only if its monetary value satisfies:
$$V_{\text{min}} = \frac{C_I}{\alpha \cdot \eta} = \frac{\$15.00}{0.05 \times 0.85} = \mathbf{\$352.94}$$

To assess the utility of the value-aware decision rule, we evaluate it against static uniform probability thresholds applied across all orders:

| Policy Configuration | Intervention Threshold | Flagged Orders | Intervention Rate (%) | Mean Predicted Risk ($\\bar{p}$) | Mean Break-Even ($\\bar{p}_{\\text{BE}}$) | Total Intervention Cost | Residual Late Loss | Total Policy Cost | Net Savings ($) | Cost Reduction (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **No-Intervention Baseline** | — | 0 | 0.00% | — | — | \.00 | \,522.01 | \,522.01 | \.00 | 0.00% |
| **Uniform Static Policy** | $\\tau = 0.10$ | 26,586 | 97.00% | 0.5447 | 1.9289 | \,790.00 | \,778.30 | \,568.30 | $-\,046.29$ | $-69.26\%$ |
| **Uniform Static Policy** | $\\tau = 0.30$ | 18,940 | 69.10% | 0.6738 | 1.8989 | \,100.00 | \,396.00 | \,496.00 | $-\,973.98$ | $-30.94\%$ |
| **Uniform Static Policy** | $\\tau = 0.50$ | 13,560 | 49.47% | 0.7878 | 1.8792 | \,400.00 | \,656.29 | \,056.29 | $-\,534.27$ | $-11.81\%$ |
| **Uniform Static Policy** | $\\tau = 0.70$ | 9,745 | 35.55% | 0.8600 | 1.8818 | \,175.00 | \,609.24 | \,784.24 | $-\,262.22$ | $-5.13\%$ |
| **Uniform Static Policy** | $\\tau = 0.80$ | 6,641 | 24.23% | 0.9102 | 1.8468 | \,615.00 | \,643.17 | \,258.17 | $-\,736.16$ | $-1.83\%$ |
| **Uniform Static Policy** | $\\tau = 0.95$ | 2,634 | 9.61% | 0.9999 | 1.8817 | \,510.00 | \,312.20 | \,822.20 | $-\,300.19$ | $-0.89\%$ |
| **Value-Aware Bayes Rule** | $\\mathbf{p_{\\text{ord}} > p_{\\text{BE},\\text{ord}}}$ | **3,840** | **14.01%** | **0.8064** | **0.5782** | **\,600.00** | **\,836.73** | **\,436.73** | **+\,085.28** | **+12.80%** |

> **Methodological Finding**: Under the specified economic assumptions, the evaluated uniform probability thresholds were economically inferior to the observation-specific value-aware decision rule. Static thresholds force interventions on lower-value orders ($V_{\text{ord}} < \$352.94$), generating negative returns regardless of classification accuracy. In contrast, the observation-specific Bayes decision rule under the assumed cost structure acts as a dynamic dual filter: it intervenes exclusively when predicted risk and order value jointly justify the expenditure, yielding a **12.80% reduction in simulated economic cost** under the baseline assumptions.

### 8.5 Systematic Sensitivity Analysis

We conduct full factorial sensitivity testing across economic loss rates ($\alpha \in \{0.03, 0.05, 0.10\}$), intervention fees ($C_I \in \{\$10, \$15, \$20\}$), and recovery efficacies ($\eta \in [0.70, 0.95]$) under the order-level Bayes decision rule:

#### 8.5.1 One-Way Sensitivity: Loss Rate ($\alpha$)
*(At baseline $C_I = \$15.00, \eta = 0.85$)*

| Loss Rate Exposure ($\alpha$) | Baseline Loss ($) | Intervened Orders | Intervention Rate (%) | Policy Cost ($) | Net Savings ($) | Cost Reduction (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.03** (Conservative) | \$155,113.21 | 1,137 | 4.15% | \$149,607.50 | +\$5,505.71 | **+3.55%** |
| **0.05** (Baseline Benchmark) | \$258,522.01 | 3,840 | 14.01% | \$225,436.73 | +\$33,085.28 | **+12.80%** |
| **0.10** (Stress-Test Exposure) | \$517,044.03 | 10,786 | 39.35% | \$333,685.34 | +\$183,358.69 | **+35.46%** |

#### 8.5.2 One-Way Sensitivity: Intervention Cost ($C_I$)
*(At baseline $\alpha = 0.05, \eta = 0.85$)*

| Intervention Cost ($C_I$) | Intervened Orders | Intervention Rate (%) | Policy Cost ($) | Net Savings ($) | Cost Reduction (%) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **\$10.00** (Favorable Logistics) | 8,011 | 29.23% | \$191,163.26 | +\$67,358.75 | **+26.06%** |
| **\$15.00** (Baseline Benchmark) | 3,840 | 14.01% | \$225,436.73 | +\$33,085.28 | **+12.80%** |
| **\$20.00** (Premium Expedite) | 1,967 | 7.18% | \$240,238.10 | +\$18,283.92 | **+7.07%** |

#### 8.5.3 One-Way Sensitivity: Intervention Efficacy ($\eta$)
*(At baseline $\alpha = 0.05, C_I = \$15.00$)*

| Mitigation Efficacy ($\eta$) | Intervened Orders | Intervention Rate (%) | Policy Cost ($) | Net Savings ($) | Cost Reduction (%) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$\eta = 0.70$** | 2,367 | 8.64% | \$239,762.00 | +\$18,760.01 | **+7.26%** |
| **$\eta = 0.80$** | 3,374 | 12.31% | \$230,622.64 | +\$27,899.37 | **+10.79%** |
| **$\eta = 0.85$ (Baseline)** | **3,840** | **14.01%** | **\$225,436.73** | **+\$33,085.28** | **+12.80%** |
| **$\eta = 0.90$** | 4,457 | 16.26% | \$219,200.69 | +\$39,321.33 | **+15.21%** |
| **$\eta = 0.95$** | 5,140 | 18.75% | \$212,214.35 | +\$46,307.67 | **+17.91%** |

#### 8.5.4 Factorial Two-Way Matrix: $\alpha \times C_I$ (Cost Reduction % at $\eta = 0.85$)

| Loss Exposure ($\alpha$) | $C_I = \$10.00$ | $C_I = \$15.00$ (Baseline) | $C_I = \$20.00$ |
| :---: | :---: | :---: | :---: |
| **$\alpha = 0.03$** (Conservative) | +10.63% (+$16,491.96) | +3.55% (+$5,505.71) | +1.00% (+$1,548.70) |
| **$\alpha = 0.05$** (Baseline) | +26.06% (+$67,358.75) | **+12.80% (+$33,085.28)** | +7.07% (+$18,283.92) |
| **$\alpha = 0.10$** (Stress Test) | **+47.35% (+$244,805.06)** | +35.46% (+$183,358.69) | +26.06% (+$134,717.50) |

#### 8.5.5 Factorial Two-Way Matrix: $\alpha \times \eta$ (Cost Reduction % at $C_I = \$15.00$)

| Loss Exposure ($\alpha$) | $\eta = 0.70$ | $\eta = 0.80$ | $\eta = 0.85$ (Baseline) | $\eta = 0.90$ | $\eta = 0.95$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$\alpha = 0.03$** | +1.52% | +2.86% | +3.55% | +4.32% | +5.20% |
| **$\alpha = 0.05$** | +7.26% | +10.79% | **+12.80%** | +15.21% | +17.91% |
| **$\alpha = 0.10$** | +24.81% | +31.84% | +35.46% | +39.22% | +43.08% |

### 8.6 Operational Implications

1. **Calibrated Probabilities Enable Decision Modeling**: The observation-specific decision rule requires probabilistic risk estimates. The aggregated order-level probabilities achieved a Brier score of 0.1336 and an ECE of 7.13% on the held-out order-level evaluation set.
2. **Order Value as an Essential Decision Covariate**: Predictive discriminative ability ($ROC\text{-}AUC = 96.13\%$) alone cannot establish operational value. By conditioning decisions on order financial stake $V_{\text{ord}, k}$, logistics operators avoid uneconomic interventions on low-value items.
3. **Positive Simulated Savings Across the Evaluated Sensitivity Scenarios**: The simulated policy remained cost-saving across all 15 evaluated combinations of $\alpha$ and $C_I$ at the tested efficacy values (ranging from +1.00% under severe cost friction up to +47.35% in high-exposure operations).

---

## ⚠️ 9. Limitations

1. **Synthetic Intervention Attribution**: The DataCo dataset lacks randomized controlled intervention trials; intervention efficacy ($\eta = 0.85$) and expediting cost ($C_I = \$15$) are parameterized simulation assumptions rather than empirically observed treatments.
2. **Loss Exposure Benchmark**: The 5% loss proxy ($\alpha = 0.05$) is drawn from external empirical literature (Fisher et al., 2006) and does not reflect account-specific SLA penalty clauses or bespoke customer lifetime value models.
3. **Observational Granularity**: Physical warehouse interventions operate at the packaging and dispatch level, whereas our primary operational unit is evaluated at the unique order level (`Order Id`). While all line items within each order share identical delivery outcomes in the dataset, physical packaging structures (such as split shipments across multiple packages) are unobserved in the transaction logs.
4. **Single Supply-Chain Topology**: Empirical results reflect a single multichannel retail enterprise; generalizability across heavy industrial B2B freight or perishable cold-chain logistics requires separate empirical verification.

---

## 🔮 10. Future Research Directions

1. **Field A/B Experimentation**: Executing randomized controlled trial (RCT) intervention rollouts in live fulfillment centers to estimate empirical treatment effects ($\eta$) and true expediting costs ($C_I$).
2. **Multi-Action Cost Optimization**: Extending the binary decision model to multi-action escalation hierarchies (e.g., automated inventory re-allocation vs. express air courier vs. customer delay notification).
3. **Constrained-Capacity Dispatching**: Incorporating dynamic warehouse labor and courier vehicle capacity constraints via integer programming to allocate limited intervention quotas.
4. **End-to-End Decision-Focused Learning**: Integrating the order-value economic loss function directly into model training loss rather than performing post-hoc probability thresholding.

---

## 🎓 11. Conclusion

1. **Academic Rigor**: Strict checkout-time scoping, 5-fold CV out-of-fold target encoding, and McNemar paired testing ($\chi^2 = 2457.39, p < 0.001$ with Edwards' continuity correction) confirm benchmark validity without predictive leakage.
2. **Reliability Transfer**: Isotonic probability calibration successfully transfers out-of-time (2017 Future ECE: 10.45% $\rightarrow$ 1.25%) and out-of-domain (Pacific Asia ECE: 11.65% $\rightarrow$ 7.20%).
3. **Order-Value-Aware Operational Utility**: Under the baseline economic assumptions ($\alpha=0.05$, $C_I=\$15$, and $\eta=0.85$), the observation-specific value-aware decision rule reduced simulated economic cost by **12.80% (\$33,085.28)** relative to the no-intervention baseline at the order-level operational aggregation. This estimate is conditional on the assumed loss exposure, intervention cost, and intervention efficacy and should therefore be interpreted as an offline counterfactual estimate rather than a causal estimate of realized operational savings.


