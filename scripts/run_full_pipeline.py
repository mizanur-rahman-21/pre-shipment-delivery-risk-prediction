"""
Master Research Execution Pipeline Script
=========================================
Runs the complete academic research workflow:
1. Pre-shipment scoping & data leakage audit
2. Model benchmark across 7 classifiers (E1)
3. McNemar paired statistical significance test
4. Leak-free probability calibration (E3)
5. Connected temporal validation & calibration transfer (E2)
6. TreeSHAP attribution temporal drift analysis
7. Decision & cost-benefit simulation
8. Geographic out-of-domain evaluation & LOMO CV (E7)
9. Feature setting ablation study (E5)
10. Publication figure generation (01-20)
"""

import sys
import os

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import audit_and_load_raw_data
from src.feature_audit import build_pre_shipment_dataset, run_feature_ablation_study
from src.feature_engineering import engineer_features, apply_bayesian_target_encoding, apply_quantile_transformation
from src.models import build_model_dictionary
from src.evaluation import evaluate_models_benchmark, run_mcnemar_statistical_test
from src.calibration import fit_and_evaluate_calibration_leakfree, assign_delivery_risk_scores
from src.decision_simulation import run_decision_cost_simulation
from src.explainability import run_treeshap_analysis
from src.robustness import run_temporal_calibrated_experiment, run_geographic_robustness_experiment
from src.visualization import generate_all_19_figures
from scripts.generate_figures import organize_figures
import pandas as pd

def main():
    print("================================================================================")
    print(" REFACTORED RESEARCH PIPELINE - LEAK-FREE CALIBRATION & TEMPORAL DRIFT")
    print("================================================================================")

    # 1. Load Data & Perform Leakage-Controlled Pre-Shipment Audit
    print("\n[POINT 1-3] Auditing Features, Scoping Prediction Point & Creating Pre-Shipment Dataset...")
    df_raw, df_clean, data_audit_log = audit_and_load_raw_data()
    df_pre, feature_audit_df = build_pre_shipment_dataset(df_clean)
    print(f"      [Usable Samples]: {len(df_pre):,} rows across {df_pre.shape[1]-1} pre-shipment predictors.")

    # 2. Engineer Features & Stratified Out-of-Fold Target Encoding
    X_raw, y = engineer_features(df_pre)
    split_idx = int(len(X_raw) * 0.70)
    X_train_raw, X_test_raw = X_raw.iloc[:split_idx].copy(), X_raw.iloc[split_idx:].copy()
    y_train, y_test = y.iloc[:split_idx].copy(), y.iloc[split_idx:].copy()

    X_train, X_test = apply_bayesian_target_encoding(X_train_raw, X_test_raw, y_train)
    X_train_qt, X_test_qt = apply_quantile_transformation(X_train, X_test)

    # Save processed datasets
    os.makedirs('data/processed', exist_ok=True)
    df_pre.to_csv('data/processed/pre_shipment_dataset.csv', index=False)

    # 3. Model Benchmark (E1)
    print(f"\n[POINT 6-7] Running Baseline Benchmark across Core 7 Classifiers ({len(y_test):,} test samples)...")
    models_dict = build_model_dictionary()
    e1_df, trained_models, test_preds, test_probs = evaluate_models_benchmark(
        models_dict, X_train, X_test, X_train_qt, X_test_qt, y_train, y_test
    )
    print("================================================================================")
    print(e1_df[['Model Name', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC', 'PR-AUC', 'Training Time']].to_string())
    print("================================================================================")

    # 4. McNemar Statistical Significance Test
    print("\n[POINT 13] Running McNemar's Paired Test for Statistical Significance (XGBoost vs Random Forest)...")
    mcnemar_df = run_mcnemar_statistical_test(y_test, test_preds['XGBoost Classifier'], test_preds['Random Forest Classifier'])
    print(mcnemar_df[['Comparison', 'McNemar Chi2 Statistic', 'P-Value', 'Statistical Significance', 'Accuracy Difference (b-c)/n']].to_string())

    # 5. Leak-Free Probability Calibration (E3)
    print("\n[PRIORITY 1] Running Leak-Free Probability Calibration (Fitted on X_train only)...")
    xgb_base = models_dict['XGBoost Classifier'][0]
    calibration_metrics_df, prob_dict = fit_and_evaluate_calibration_leakfree(xgb_base, X_train, y_train, X_test, y_test)
    print(calibration_metrics_df[['Calibration Method', 'Brier Score', 'Brier 95% CI', 'ECE', 'ECE 95% CI', 'MCE', 'Reliability', 'Fit Protocol']].to_string())

    # 6. Connected Temporal Calibration & SHAP Temporal Drift (E2)
    print("\n[PRIORITY 2 & 3] Running Connected Temporal Calibration & SHAP Temporal Drift Analysis...")
    temporal_df, drift_df = run_temporal_calibrated_experiment(df_raw)
    print("================================================================================")
    print("      Temporal Validation & Calibration Performance (2015-2016 Train / 2017 Test):")
    print(temporal_df.to_string())
    print("================================================================================")
    print("      Top SHAP Feature Importance Temporal Drift (2015-2016 vs 2017):")
    print(drift_df.to_string())

    # 7. Operational Risk Stratification & Cost Simulation
    print("\n[POINT 12] Generating Operational Delivery Risk Scores & Cost Simulation...")
    cal_probs = prob_dict['Isotonic Regression'][0]
    risk_df = assign_delivery_risk_scores(cal_probs)
    risk_summary = risk_df['Risk Category'].value_counts(normalize=True).map(lambda x: f"{x*100:.2f}%")
    print("      Operational Risk Category Distribution:")
    print(risk_summary)

    sim_df, _ = run_decision_cost_simulation(y_test, cal_probs)
    os.makedirs('results/tables', exist_ok=True)
    sim_df.to_csv('results/tables/decision_cost_simulation.csv', index=False)

    # 8. TreeSHAP Explainability
    print("\n[POINT 11] Running Complete TreeSHAP Explainability Analysis...")
    run_treeshap_analysis(trained_models['XGBoost Classifier'], X_test, y_test)

    # 9. Geographic Out-of-Domain Market Generalization (E7)
    print("\n[POINT 10] Running Geographic Out-of-Domain Market Generalization (Pacific Asia Test)...")
    geo_df = run_geographic_robustness_experiment(df_raw)
    print(geo_df.to_string())

    # 10. Feature Setting Ablation Study (E5)
    print("\n[POINT 14] Running Feature Setting Ablation Study...")
    ablation_df = run_feature_ablation_study(df_pre)
    print("================================================================================")
    print(ablation_df[['Feature Setting', 'Num Features', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC', 'PR-AUC']].to_string())
    print("================================================================================")

    # 11. Master Results Consolidation & Figures Export
    print("\n[POINT 15] Building Final Master Results Table (final_master_results.csv)...")
    master_results_df = pd.concat([
        e1_df.assign(Experiment='E1: Baseline Benchmark'),
        temporal_df.assign(Experiment='E2: Temporal Validation'),
        calibration_metrics_df.assign(Experiment='E3: Leak-Free Calibration'),
        ablation_df.assign(Experiment='E5: Ablation Study'),
        geo_df.assign(Experiment='E7: Geographic Robustness')
    ], ignore_index=True)
    master_results_df.to_csv('results/tables/final_master_results.csv', index=False)

    print("\n[POINT 16] Exporting All 19 Publication Figures to visuals/ and figures/...")
    generate_all_19_figures(
        df_raw, df_pre, e1_df, temporal_df, prob_dict, risk_df, ablation_df,
        trained_models, test_preds, test_probs, y_test
    )
    organize_figures()

    print("\n[OK] Priorities 1, 2, and 3 Pipeline Execution Completed Successfully!")

if __name__ == '__main__':
    main()
