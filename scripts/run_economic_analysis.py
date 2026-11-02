"""
Standalone Order-Value-Based Economic Analysis Script
=====================================================
Executes the complete order-value-based economic decision-cost analysis:
1. Loads pre-shipment dataset and reconstructs E1 70/30 split
2. Reconstructs calibrated probabilities (Isotonic, 5-fold CV on train only)
3. Runs the order-value-based cost simulation
4. Performs systematic sensitivity analysis
5. Computes break-even analysis
6. Saves all result CSVs
7. Generates all publication-grade figures (300 DPI)
8. Prints complete audit summary
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV

from src.feature_engineering import engineer_features, apply_bayesian_target_encoding
from src.decision_simulation import (
    run_order_value_cost_simulation,
    run_sensitivity_analysis,
    compute_order_value_stats,
    compute_breakeven_probabilities
)

# Matplotlib configuration
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except Exception:
    sns.set_theme(style="whitegrid")
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13


def main():
    print("=" * 80)
    print(" ORDER-VALUE-BASED ECONOMIC DECISION-COST ANALYSIS")
    print("=" * 80)

    # ----------------------------------------------------------------
    # 1. Load data and reconstruct E1 split
    # ----------------------------------------------------------------
    print("\n[1/7] Loading pre-shipment dataset and reconstructing E1 split...")
    df = pd.read_csv('data/processed/pre_shipment_dataset.csv')
    print(f"      Total samples: {len(df):,}")

    # Preserve Order Item Total before feature engineering drops it
    order_item_total_all = df['Order Item Total'].values.copy()

    # Feature engineering
    X_raw, y = engineer_features(df)
    split_idx = int(len(X_raw) * 0.70)
    X_train_raw = X_raw.iloc[:split_idx].copy()
    X_test_raw = X_raw.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx].copy()
    y_test = y.iloc[split_idx:].copy()

    # Get order values for test set
    order_values_test = order_item_total_all[split_idx:]

    X_train, X_test = apply_bayesian_target_encoding(X_train_raw, X_test_raw, y_train)

    print(f"      Train: {len(y_train):,} | Test: {len(y_test):,}")
    print(f"      Test late orders: {int(y_test.sum()):,}")
    print(f"      Test Order Item Total -- Mean: ${np.mean(order_values_test):.2f}, "
          f"Median: ${np.median(order_values_test):.2f}")

    # ----------------------------------------------------------------
    # 2. Reconstruct calibrated probabilities
    # ----------------------------------------------------------------
    print("\n[2/7] Reconstructing Isotonic calibrated probabilities (5-fold CV on train only)...")
    xgb_base = XGBClassifier(
        n_estimators=900, max_depth=16, learning_rate=0.025,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, use_label_encoder=False, eval_metric='logloss',
        tree_method='hist'
    )
    iso_calibrator = CalibratedClassifierCV(estimator=xgb_base, method='isotonic', cv=5)
    iso_calibrator.fit(X_train, y_train)
    cal_probs = iso_calibrator.predict_proba(X_test)[:, 1]
    print(f"      Calibrated probabilities -- Mean: {cal_probs.mean():.4f}, "
          f"Min: {cal_probs.min():.4f}, Max: {cal_probs.max():.4f}")

    # ----------------------------------------------------------------
    # 3. Order-value descriptive statistics
    # ----------------------------------------------------------------
    print("\n[3/7] Computing order-value descriptive statistics...")
    stats_df = compute_order_value_stats(order_values_test, alpha=0.05)
    os.makedirs('results/tables', exist_ok=True)
    stats_df.to_csv('results/tables/order_value_descriptive_stats.csv', index=False)
    print(stats_df.to_string(index=False))

    # ----------------------------------------------------------------
    # 4. Baseline simulation (alpha=0.05, C_I=$15, eta=0.85)
    # ----------------------------------------------------------------
    print("\n[4/7] Running baseline order-value cost simulation (alpha=0.05, C_I=$15, eta=0.85)...")
    y_true_arr = np.asarray(y_test)
    sim_df, summary = run_order_value_cost_simulation(
        y_true_arr, cal_probs, order_values_test,
        alpha=0.05, cost_intervene=15.0, eta=0.85
    )
    sim_df.to_csv('results/tables/decision_cost_simulation.csv', index=False)
    print(f"\n      Baseline Economic Loss (no intervention): ${summary['baseline_cost']:,.2f}")
    print(f"      Cost-Minimizing Threshold (tau*):           {summary['optimal_threshold']}")
    print(f"      Optimal Policy Cost:                      ${summary['optimal_policy_cost']:,.2f}")
    print(f"      Net Savings:                              ${summary['optimal_net_savings']:,.2f}")
    print(f"      Cost Reduction:                           {summary['optimal_cost_reduction_pct']:.2f}%")
    print(f"\n      Full threshold grid:")
    print(sim_df.to_string(index=False))

    # ----------------------------------------------------------------
    # 5. Break-even analysis
    # ----------------------------------------------------------------
    print("\n[5/7] Computing order-value-dependent break-even probabilities...")
    be_stats = compute_breakeven_probabilities(order_values_test, alpha=0.05, cost_intervene=15.0, eta=0.85)
    be_df = pd.DataFrame([{
        'Metric': 'Break-Even Probability (p_BE)',
        'Mean': round(be_stats['p_be_mean'], 4),
        'Median': round(be_stats['p_be_median'], 4),
        'Min': round(be_stats['p_be_min'], 4),
        'Max': round(be_stats['p_be_max'], 4),
        'P25': round(be_stats['p_be_p25'], 4),
        'P75': round(be_stats['p_be_p75'], 4),
        'Infeasible (p_BE > 1.0)': f"{be_stats['n_infeasible']} ({be_stats['pct_infeasible']:.2f}%)",
        'N': be_stats['n_total']
    }])
    be_df.to_csv('results/tables/breakeven_analysis.csv', index=False)
    print(f"      Mean p_BE:     {be_stats['p_be_mean']:.4f}")
    print(f"      Median p_BE:   {be_stats['p_be_median']:.4f}")
    print(f"      Min p_BE:      {be_stats['p_be_min']:.4f}")
    print(f"      Max p_BE:      {be_stats['p_be_max']:.4f}")
    print(f"      Infeasible:    {be_stats['n_infeasible']} orders ({be_stats['pct_infeasible']:.2f}%)")

    # ----------------------------------------------------------------
    # 6. Sensitivity analysis
    # ----------------------------------------------------------------
    print("\n[6/7] Running systematic sensitivity analysis...")
    sens = run_sensitivity_analysis(y_true_arr, cal_probs, order_values_test)

    sens['sensitivity_alpha'].to_csv('results/tables/sensitivity_alpha.csv', index=False)
    sens['sensitivity_ci'].to_csv('results/tables/sensitivity_cost_intervene.csv', index=False)
    sens['sensitivity_eta'].to_csv('results/tables/sensitivity_eta.csv', index=False)
    sens['sensitivity_alpha_x_ci'].to_csv('results/tables/sensitivity_alpha_x_ci.csv', index=False)
    sens['sensitivity_alpha_x_eta'].to_csv('results/tables/sensitivity_alpha_x_eta.csv', index=False)

    print("\n      --- One-Way: Loss Rate (alpha) ---")
    print(sens['sensitivity_alpha'].to_string(index=False))
    print("\n      --- One-Way: Intervention Cost (C_I) ---")
    print(sens['sensitivity_ci'].to_string(index=False))
    print("\n      --- One-Way: Intervention Effectiveness (eta) ---")
    print(sens['sensitivity_eta'].to_string(index=False))
    print("\n      --- Two-Way: alpha × C_I ---")
    print(sens['sensitivity_alpha_x_ci'].to_string(index=False))
    print("\n      --- Two-Way: alpha × eta ---")
    print(sens['sensitivity_alpha_x_eta'].to_string(index=False))

    # ----------------------------------------------------------------
    # 7. Generate publication figures
    # ----------------------------------------------------------------
    print("\n[7/7] Generating publication-grade figures (300 DPI)...")

    for d in ['visuals', 'figures/decision_simulation']:
        os.makedirs(d, exist_ok=True)

    # --- Figure 21: Threshold vs Policy Cost / Net Savings ---
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(sim_df['Threshold'], sim_df['Total Policy Cost ($)'],
             'o-', color='#e74c3c', linewidth=2, markersize=6, label='Total Policy Cost ($)')
    ax1.plot(sim_df['Threshold'], sim_df['Net Savings ($)'],
             's-', color='#2ecc71', linewidth=2, markersize=6, label='Net Savings ($)')
    ax1.axvline(x=summary['optimal_threshold'], color='#3498db',
                linestyle='--', linewidth=1.5, alpha=0.8,
                label=f'Cost-Minimizing tau* = {summary["optimal_threshold"]}')
    ax1.axhline(y=summary['baseline_cost'], color='#95a5a6',
                linestyle=':', linewidth=1.2, alpha=0.7,
                label=f'No-Intervention Baseline (${summary["baseline_cost"]:,.0f})')
    ax1.set_xlabel('Intervention Probability Threshold (tau)')
    ax1.set_ylabel('Dollar Amount ($)')
    ax1.set_title('Order-Value-Based Decision-Cost Simulation\n'
                   f'(alpha = 0.05, C_I = $15, eta = 0.85)', fontweight='bold')
    ax1.legend(loc='center right', fontsize=9)

    ax2 = ax1.twinx()
    ax2.fill_between(sim_df['Threshold'], sim_df['Cost Reduction (%)'],
                      alpha=0.15, color='#9b59b6')
    ax2.plot(sim_df['Threshold'], sim_df['Cost Reduction (%)'],
             '^--', color='#9b59b6', linewidth=1.5, markersize=5, alpha=0.7,
             label='Cost Reduction (%)')
    ax2.set_ylabel('Cost Reduction (%)')
    ax2.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    for path in ['visuals/21_decision_cost_curve.png',
                 'figures/decision_simulation/21_decision_cost_curve.png']:
        plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print("      [OK] Figure 21: Decision Cost Curve")

    # --- Figure 22: Sensitivity to eta ---
    eta_df = sens['sensitivity_eta']
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(eta_df['Intervention Effectiveness (eta)'].astype(str),
                  eta_df['Cost Reduction (%)'],
                  color=['#3498db', '#2ecc71', '#e67e22', '#e74c3c', '#9b59b6'],
                  edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, eta_df['Cost Reduction (%)']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}%', ha='center', fontweight='bold', fontsize=10)
    ax.set_xlabel('Intervention Effectiveness (eta)')
    ax.set_ylabel('Cost Reduction (%)')
    ax.set_title('Sensitivity to Intervention Effectiveness\n'
                 '(alpha = 0.05, C_I = $15)', fontweight='bold')
    plt.tight_layout()
    for path in ['visuals/22_mitigation_sensitivity_curve.png',
                 'figures/decision_simulation/22_mitigation_sensitivity_curve.png']:
        plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print("      [OK] Figure 22: Sensitivity to eta")

    # --- Figure 23: Sensitivity to alpha ---
    alpha_df = sens['sensitivity_alpha']
    fig, ax1 = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(alpha_df))
    width = 0.35
    bars1 = ax1.bar(x_pos - width/2, alpha_df['Net Savings ($)'],
                    width, label='Net Savings ($)', color='#2ecc71', edgecolor='white')
    ax1.set_xlabel('Loss Rate (alpha)')
    ax1.set_ylabel('Net Savings ($)')
    ax1.set_title('Sensitivity to Economic Loss Rate\n'
                  '(C_I = $15, eta = 0.85)', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'alpha = {a}' for a in alpha_df['Loss Rate (alpha)']])

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x_pos + width/2, alpha_df['Cost Reduction (%)'],
                    width, label='Cost Reduction (%)', color='#3498db', edgecolor='white')
    ax2.set_ylabel('Cost Reduction (%)')

    for bar, val in zip(bars1, alpha_df['Net Savings ($)']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                 f'${val:,.0f}', ha='center', fontsize=8, fontweight='bold')
    for bar, val in zip(bars2, alpha_df['Cost Reduction (%)']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f'{val:.1f}%', ha='center', fontsize=8, fontweight='bold')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
    plt.tight_layout()
    for path in ['visuals/23_loss_rate_sensitivity.png',
                 'figures/decision_simulation/23_loss_rate_sensitivity.png']:
        plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print("      [OK] Figure 23: Sensitivity to alpha")

    # --- Figure 24: Order-Value Distribution ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.hist(order_values_test, bins=60, color='#3498db', edgecolor='white',
             alpha=0.85, density=False)
    ax1.axvline(np.mean(order_values_test), color='#e74c3c', linestyle='--',
                linewidth=1.5, label=f'Mean = ${np.mean(order_values_test):.2f}')
    ax1.axvline(np.median(order_values_test), color='#2ecc71', linestyle='--',
                linewidth=1.5, label=f'Median = ${np.median(order_values_test):.2f}')
    ax1.set_xlabel('Order Item Total ($)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Order Values (V_i)\nE1 Test Set (N=47,415)', fontweight='bold')
    ax1.legend(fontsize=9)

    L_test = 0.05 * order_values_test
    ax2.hist(L_test, bins=60, color='#e67e22', edgecolor='white', alpha=0.85, density=False)
    ax2.axvline(np.mean(L_test), color='#e74c3c', linestyle='--',
                linewidth=1.5, label=f'Mean = ${np.mean(L_test):.2f}')
    ax2.axvline(np.median(L_test), color='#2ecc71', linestyle='--',
                linewidth=1.5, label=f'Median = ${np.median(L_test):.2f}')
    ax2.set_xlabel('Loss Exposure (L_i = 0.05 × V_i) ($)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Late-Delivery Loss Exposure\n(alpha = 0.05)', fontweight='bold')
    ax2.legend(fontsize=9)

    plt.tight_layout()
    for path in ['visuals/24_order_value_distribution.png',
                 'figures/decision_simulation/24_order_value_distribution.png']:
        plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print("      [OK] Figure 24: Order-Value Distribution")

    # ----------------------------------------------------------------
    # Final Summary Report
    # ----------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" FINAL IMPLEMENTATION REPORT")
    print("=" * 80)
    print(f"""
1. Selected Order-Value Field:  Order Item Total
   Definition:                  Net line-item cash value (Price × Qty - Discount)
   Reason:                      Actual customer payment per line item, post-discount

2. Prediction/Cost Unit:        Order-item (line-item) level -- no aggregation
   Observations:                {len(y_test):,} test samples

3. Baseline Parameters:
   alpha  = 0.05     (literature-supported benchmark loss rate)
   C_I = $15.00   (scenario-based intervention cost)
   eta  = 0.85     (assumed intervention effectiveness)

4. Evaluation Set:              E1 Random Holdout (47,415 samples)
   Probabilities:               Isotonic Regression calibrated (5-fold CV, train only)

5. Optimized Threshold:         tau* = {summary['optimal_threshold']}

6. Baseline Economic Loss:      ${summary['baseline_cost']:,.2f}

7. Optimized Policy Cost:       ${summary['optimal_policy_cost']:,.2f}

8. Net Savings:                 ${summary['optimal_net_savings']:,.2f}

9. Cost Reduction:              {summary['optimal_cost_reduction_pct']:.2f}%

10. Break-Even Analysis:
    Mean p_BE:    {be_stats['p_be_mean']:.4f}
    Median p_BE:  {be_stats['p_be_median']:.4f}
    Infeasible:   {be_stats['n_infeasible']} orders ({be_stats['pct_infeasible']:.2f}%)

11. Result CSVs Saved:
    - results/tables/decision_cost_simulation.csv
    - results/tables/order_value_descriptive_stats.csv
    - results/tables/breakeven_analysis.csv
    - results/tables/sensitivity_alpha.csv
    - results/tables/sensitivity_cost_intervene.csv
    - results/tables/sensitivity_eta.csv
    - results/tables/sensitivity_alpha_x_ci.csv
    - results/tables/sensitivity_alpha_x_eta.csv

12. Figures Saved:
    - visuals/21_decision_cost_curve.png
    - visuals/22_mitigation_sensitivity_curve.png
    - visuals/23_loss_rate_sensitivity.png
    - visuals/24_order_value_distribution.png
    (+ mirrors in figures/decision_simulation/)
""")
    print("=" * 80)
    print("[OK] Order-Value-Based Economic Analysis Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
