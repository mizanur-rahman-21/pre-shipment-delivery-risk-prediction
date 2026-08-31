# 🔍 Academic Research Paper Experiment Traceability Matrix

This document maps every section of the research paper directly to its underlying experiment ID, source script, output CSV result file, and generated figure.

---

## 🗺️ Traceability Mapping Matrix

| Paper Section | Experiment ID | Source Script / Module | Primary Result CSV File | Key Figure(s) |
| :--- | :---: | :--- | :--- | :--- |
| **Section 2: Data Quality & Scoping Audit** | **Audit** | `src/feature_audit.py` | `results/metrics/dataset_audit_log.csv`<br>`results/metrics/feature_audit_table.csv` | `figures/exploratory/fig_01_target_distribution.png`<br>`figures/exploratory/fig_02_missingness_overview.png` |
| **Section 3.1: Baseline Benchmark** | **E1** | `src/evaluation.py` | `results/metrics/E1_baseline_benchmark.csv` | `figures/model_performance/fig_08_model_performance_comparison.png`<br>`figures/model_performance/fig_09_roc_curves.png`<br>`figures/model_performance/fig_10_precision_recall_curves.png` |
| **Section 3.2: McNemar Statistical Test** | **Significance** | `src/evaluation.py` | `results/statistical_tests/statistical_significance_mcnemar.csv` | `figures/model_performance/fig_11_confusion_matrix.png` |
| **Section 4: Leak-Free Calibration** | **E3** | `src/calibration.py` | `results/metrics/E3_calibration_metrics.csv` | `figures/calibration/fig_12_reliability_diagram.png`<br>`figures/calibration/fig_13_calibration_comparison.png` |
| **Section 5: Connected Temporal Validation** | **E2** | `src/robustness.py` | `results/metrics/E2_temporal_robustness.csv` | `figures/temporal/fig_17_temporal_performance_comparison.png` |
| **Section 6: TreeSHAP Attribution Drift** | **Priority 3** | `src/explainability.py`<br>`src/robustness.py` | `results/metrics/shap_temporal_drift_analysis.csv` | `figures/explainability/fig_14_shap_summary_beeswarm.png`<br>`figures/explainability/fig_15_shap_feature_importance.png`<br>`figures/temporal/fig_20_shap_temporal_drift.png` |
| **Section 7: Out-of-Domain Generalization** | **E7** | `src/robustness.py` | `results/metrics/E7_geographic_robustness.csv`<br>`results/metrics/leave_one_market_out_lomo_cv.csv` | `figures/geographic/fig_18_geographic_generalization_comparison.png` |
| **Section 8: Decision & Cost Simulation** | **Decision** | `src/decision_simulation.py` | `results/tables/decision_cost_simulation.csv` | `figures/explainability/fig_19_risk_category_distribution.png` |
| **Section 9: Feature Ablation Study** | **E5** | `src/feature_audit.py` | `results/metrics/E5_ablation_study.csv` | — |
| **Master Summary** | **Master** | `scripts/run_full_pipeline.py` | `results/tables/final_master_results.csv` | — |
