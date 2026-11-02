"""
Corrected Realized Cost Economic Decision Simulation
=====================================================
Implements the exact corrected realized cost formulation:

    C_i = I_i · [C_I + Y_i · (1 - η) · L_i] + (1 - I_i) · Y_i · L_i

Key Realized Cost Logic:
  - If an order is ON-TIME (Y_i = 0):
      - If intervened (I_i = 1): Incurs ONLY intervention cost C_I ($15). Zero residual loss!
      - If not intervened (I_i = 0): Cost is 0.
  - If an order is LATE (Y_i = 1):
      - If intervened (I_i = 1): Incurs C_I ($15) + residual loss (1 - η) · L_i.
      - If not intervened (I_i = 0): Incurs full loss L_i = α · V_i.

Decision Rule (Bayes Optimal per Observation):
  - Expected cost with intervention:    C_I + p_i · (1 - η) · L_i
  - Expected cost without intervention: p_i · L_i
  - Intervene when:
        C_I < p_i · η · L_i  ⟹  p_i > C_I / (η · α · V_i) = p_BE,i

Evaluates:
  1. Item-Level: Order-Specific Decision Rule vs. Uniform Thresholds
  2. Order-Level Aggregated (group by Order Id, C_I applied once per parcel):
     Order-Specific Decision Rule vs. Uniform Thresholds
  3. Full Sensitivity Analysis for both granularities
  4. Saves clean publication tables and figures
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except Exception:
    sns.set_theme(style="whitegrid")
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13


def compute_item_level_economics(df, alpha=0.05, cost_intervene=15.0, eta=0.85):
    """
    Computes item-level economics under the corrected realized cost formula.
    """
    V = df['order_item_total'].values.astype(float)
    y_true = df['y_true'].values.astype(int)
    cal_prob = df['cal_prob'].values.astype(float)
    n_items = len(V)
    
    L = alpha * V
    baseline_cost = np.sum(y_true * L)
    
    # 1. Order-Specific Bayes Decision Rule: p_i > C_I / (eta * L_i)
    with np.errstate(divide='ignore', invalid='ignore'):
        p_BE = cost_intervene / (eta * L)
    p_BE = np.where(L > 0, p_BE, np.inf)
    
    intervene_bayes = (cal_prob > p_BE)
    n_int_bayes = int(np.sum(intervene_bayes))
    
    # Correct realized cost: C_i = I_i * [C_I + Y_i * (1 - eta) * L_i] + (1 - I_i) * Y_i * L_i
    cost_bayes = np.where(
        intervene_bayes,
        cost_intervene + y_true * (1.0 - eta) * L,
        y_true * L
    )
    policy_cost_bayes = float(np.sum(cost_bayes))
    savings_bayes = baseline_cost - policy_cost_bayes
    reduction_bayes = (savings_bayes / baseline_cost) * 100.0 if baseline_cost > 0 else 0.0
    
    # 2. Fixed Threshold Grid
    grid_results = []
    thresholds = [round(t * 0.05, 2) for t in range(1, 20)]
    
    for tau in thresholds:
        intervene = (cal_prob >= tau)
        n_flagged = int(np.sum(intervene))
        late_flagged = int(np.sum(intervene & (y_true == 1)))
        
        costs = np.where(
            intervene,
            cost_intervene + y_true * (1.0 - eta) * L,
            y_true * L
        )
        total_policy_cost = float(np.sum(costs))
        total_int_cost = n_flagged * cost_intervene
        total_late_loss = total_policy_cost - total_int_cost
        savings = baseline_cost - total_policy_cost
        reduction = (savings / baseline_cost) * 100.0 if baseline_cost > 0 else 0.0
        
        grid_results.append({
            'Threshold': tau,
            'Flagged Orders': n_flagged,
            'Intervention Rate (%)': round((n_flagged / n_items) * 100.0, 2),
            'Late Orders Flagged': late_flagged,
            'Total Intervention Cost ($)': round(total_int_cost, 2),
            'Residual Late Loss ($)': round(total_late_loss, 2),
            'Total Policy Cost ($)': round(total_policy_cost, 2),
            'Net Savings ($)': round(savings, 2),
            'Cost Reduction (%)': round(reduction, 2)
        })
        
    grid_df = pd.DataFrame(grid_results)
    
    bayes_summary = {
        'Granularity': 'Item-Level (Row)',
        'Decision Policy': 'Observation-Specific Bayes Rule (p_i > p_BE,i)',
        'Total Observations': n_items,
        'Late Observations': int(np.sum(y_true)),
        'Intervened Observations': n_int_bayes,
        'Intervention Rate (%)': round((n_int_bayes / n_items) * 100.0, 2),
        'Baseline Economic Loss ($)': round(baseline_cost, 2),
        'Total Policy Cost ($)': round(policy_cost_bayes, 2),
        'Net Savings ($)': round(savings_bayes, 2),
        'Cost Reduction (%)': round(reduction_bayes, 2),
        'Feasible Threshold Minimum Order Value ($)': round(cost_intervene / (alpha * eta), 2),
        'Number of Feasible Observations': int(np.sum(p_BE <= 1.0)),
        'Feasible Proportion (%)': round((np.sum(p_BE <= 1.0) / n_items) * 100.0, 2)
    }
    
    return bayes_summary, grid_df


def compute_order_level_economics(df, alpha=0.05, cost_intervene=15.0, eta=0.85):
    """
    Computes order-level (aggregated by Order Id) economics.
    In actual logistics, carrier expedite is executed per shipment/order, NOT per individual line-item.
    """
    order_grouped = df.groupby('order_id').agg({
        'order_item_total': 'sum',
        'y_true': 'max',
        'cal_prob': 'mean'
    }).reset_index()
    
    V_ord = order_grouped['order_item_total'].values.astype(float)
    y_ord = order_grouped['y_true'].values.astype(int)
    p_ord = order_grouped['cal_prob'].values.astype(float)
    n_orders = len(V_ord)
    
    L_ord = alpha * V_ord
    baseline_cost = np.sum(y_ord * L_ord)
    
    # 1. Order-Level Bayes Decision Rule: p_ord > C_I / (eta * L_ord)
    with np.errstate(divide='ignore', invalid='ignore'):
        p_BE_ord = cost_intervene / (eta * L_ord)
    p_BE_ord = np.where(L_ord > 0, p_BE_ord, np.inf)
    
    int_ord_bayes = (p_ord > p_BE_ord)
    n_int_bayes = int(np.sum(int_ord_bayes))
    
    costs_bayes = np.where(
        int_ord_bayes,
        cost_intervene + y_ord * (1.0 - eta) * L_ord,
        y_ord * L_ord
    )
    policy_cost_bayes = float(np.sum(costs_bayes))
    savings_bayes = baseline_cost - policy_cost_bayes
    reduction_bayes = (savings_bayes / baseline_cost) * 100.0 if baseline_cost > 0 else 0.0
    
    # 2. Fixed Threshold Grid
    grid_results = []
    thresholds = [round(t * 0.05, 2) for t in range(1, 20)]
    
    for tau in thresholds:
        intervene = (p_ord >= tau)
        n_flagged = int(np.sum(intervene))
        late_flagged = int(np.sum(intervene & (y_ord == 1)))
        
        costs = np.where(
            intervene,
            cost_intervene + y_ord * (1.0 - eta) * L_ord,
            y_ord * L_ord
        )
        total_policy_cost = float(np.sum(costs))
        total_int_cost = n_flagged * cost_intervene
        total_late_loss = total_policy_cost - total_int_cost
        savings = baseline_cost - total_policy_cost
        reduction = (savings / baseline_cost) * 100.0 if baseline_cost > 0 else 0.0
        
        mean_p = float(np.mean(p_ord[intervene])) if n_flagged > 0 else 0.0
        mean_p_be = float(np.mean(p_BE_ord[intervene])) if n_flagged > 0 else 0.0
        
        grid_results.append({
            'Threshold': tau,
            'Flagged Orders': n_flagged,
            'Intervention Rate (%)': round((n_flagged / n_orders) * 100.0, 2),
            'Late Orders Flagged': late_flagged,
            'Mean Predicted Risk': round(mean_p, 4),
            'Mean Break-Even Threshold': round(mean_p_be, 4),
            'Total Intervention Cost ($)': round(total_int_cost, 2),
            'Residual Late Loss ($)': round(total_late_loss, 2),
            'Total Policy Cost ($)': round(total_policy_cost, 2),
            'Net Savings ($)': round(savings, 2),
            'Cost Reduction (%)': round(reduction, 2)
        })
        
    grid_df = pd.DataFrame(grid_results)
    
    mean_p_bayes = float(np.mean(p_ord[int_ord_bayes])) if n_int_bayes > 0 else 0.0
    mean_p_be_bayes = float(np.mean(p_BE_ord[int_ord_bayes])) if n_int_bayes > 0 else 0.0
    
    bayes_summary = {
        'Granularity': 'Order-Level (Unique Orders)',
        'Decision Policy': 'Order-Specific Bayes Rule (p_ord > p_BE,ord)',
        'Total Orders': n_orders,
        'Late Orders': int(np.sum(y_ord)),
        'Intervened Orders': n_int_bayes,
        'Intervention Rate (%)': round((n_int_bayes / n_orders) * 100.0, 2),
        'Mean Predicted Risk': round(mean_p_bayes, 4),
        'Mean Break-Even Threshold': round(mean_p_be_bayes, 4),
        'Baseline Economic Loss ($)': round(baseline_cost, 2),
        'Total Policy Cost ($)': round(policy_cost_bayes, 2),
        'Net Savings ($)': round(savings_bayes, 2),
        'Cost Reduction (%)': round(reduction_bayes, 2),
        'Feasible Threshold Minimum Order Value ($)': round(cost_intervene / (alpha * eta), 2),
        'Number of Feasible Orders': int(np.sum(p_BE_ord <= 1.0)),
        'Feasible Proportion (%)': round((np.sum(p_BE_ord <= 1.0) / n_orders) * 100.0, 2)
    }
    
    return bayes_summary, grid_df, order_grouped


def run_full_sensitivity(df):
    """
    Runs full factorial sensitivity analysis for both Item-level and Order-level.
    """
    alphas = [0.03, 0.05, 0.10]
    costs = [10.0, 15.0, 20.0]
    etas = [0.70, 0.80, 0.85, 0.90, 0.95]
    
    rows = []
    for g in ['Item-Level', 'Order-Level']:
        for a in alphas:
            for c in costs:
                for e in etas:
                    if g == 'Item-Level':
                        s, _ = compute_item_level_economics(df, alpha=a, cost_intervene=c, eta=e)
                    else:
                        s, _, _ = compute_order_level_economics(df, alpha=a, cost_intervene=c, eta=e)
                    rows.append({
                        'Granularity': g,
                        'Alpha': a,
                        'Cost_Intervene': c,
                        'Eta': e,
                        'Baseline_Loss': s['Baseline Economic Loss ($)'],
                        'Policy_Cost': s['Total Policy Cost ($)'],
                        'Net_Savings': s['Net Savings ($)'],
                        'Cost_Reduction_Pct': s['Cost Reduction (%)'],
                        'Intervention_Rate_Pct': s['Intervention Rate (%)']
                    })
    return pd.DataFrame(rows)


def main():
    print("=" * 80)
    print(" COMPUTING CORRECTED REALIZED COST ECONOMIC SIMULATION")
    print("=" * 80)
    
    pred_path = 'results/predictions/test_predictions_calibrated.csv'
    if not os.path.exists(pred_path):
        print(f"Error: {pred_path} does not exist yet. Please wait for model predictions to save.")
        return
        
    df = pd.read_csv(pred_path)
    print(f"Loaded {len(df):,} test predictions.")
    
    # 1. Item-Level Economics
    item_bayes, item_grid = compute_item_level_economics(df, alpha=0.05, cost_intervene=15.0, eta=0.85)
    print("\n--- ITEM-LEVEL BAYES RULE SUMMARY ---")
    for k, v in item_bayes.items():
        print(f"  {k}: {v}")
        
    # 2. Order-Level Economics
    order_bayes, order_grid, order_df = compute_order_level_economics(df, alpha=0.05, cost_intervene=15.0, eta=0.85)
    print("\n--- ORDER-LEVEL AGGREGATED BAYES RULE SUMMARY ---")
    for k, v in order_bayes.items():
        print(f"  {k}: {v}")
        
    # Save CSV tables
    os.makedirs('results/tables', exist_ok=True)
    item_grid.to_csv('results/tables/decision_cost_simulation_item_grid.csv', index=False)
    order_grid.to_csv('results/tables/decision_cost_simulation_order_grid.csv', index=False)
    
    summary_compare_df = pd.DataFrame([item_bayes, order_bayes])
    summary_compare_df.to_csv('results/tables/decision_policy_comparison.csv', index=False)
    
    # Also save the primary decision_cost_simulation.csv as the Order-Level Grid (most operationally sound)
    order_grid.to_csv('results/tables/decision_cost_simulation.csv', index=False)
    
    # 3. Sensitivity Analysis
    print("\nComputing full factorial sensitivity analysis across both granularities...")
    sens_df = run_full_sensitivity(df)
    sens_df.to_csv('results/tables/sensitivity_full_factorial.csv', index=False)
    
    # One-way and two-way tables for Order-Level (baseline C_I=15, eta=0.85)
    ord_sens = sens_df[sens_df['Granularity'] == 'Order-Level']
    
    sens_alpha = ord_sens[(ord_sens['Cost_Intervene'] == 15.0) & (ord_sens['Eta'] == 0.85)][
        ['Alpha', 'Baseline_Loss', 'Policy_Cost', 'Net_Savings', 'Cost_Reduction_Pct', 'Intervention_Rate_Pct']
    ].rename(columns={'Alpha': 'Loss Rate (alpha)', 'Baseline_Loss': 'Baseline Loss ($)', 'Policy_Cost': 'Policy Cost ($)', 'Net_Savings': 'Net Savings ($)', 'Cost_Reduction_Pct': 'Cost Reduction (%)', 'Intervention_Rate_Pct': 'Intervention Rate (%)'})
    sens_alpha.to_csv('results/tables/sensitivity_alpha.csv', index=False)
    
    sens_ci = ord_sens[(ord_sens['Alpha'] == 0.05) & (ord_sens['Eta'] == 0.85)][
        ['Cost_Intervene', 'Baseline_Loss', 'Policy_Cost', 'Net_Savings', 'Cost_Reduction_Pct', 'Intervention_Rate_Pct']
    ].rename(columns={'Cost_Intervene': 'Intervention Cost ($)', 'Baseline_Loss': 'Baseline Loss ($)', 'Policy_Cost': 'Policy Cost ($)', 'Net_Savings': 'Net Savings ($)', 'Cost_Reduction_Pct': 'Cost Reduction (%)', 'Intervention_Rate_Pct': 'Intervention Rate (%)'})
    sens_ci.to_csv('results/tables/sensitivity_cost_intervene.csv', index=False)
    
    sens_eta = ord_sens[(ord_sens['Alpha'] == 0.05) & (ord_sens['Cost_Intervene'] == 15.0)][
        ['Eta', 'Baseline_Loss', 'Policy_Cost', 'Net_Savings', 'Cost_Reduction_Pct', 'Intervention_Rate_Pct']
    ].rename(columns={'Eta': 'Intervention Effectiveness (eta)', 'Baseline_Loss': 'Baseline Loss ($)', 'Policy_Cost': 'Policy Cost ($)', 'Net_Savings': 'Net Savings ($)', 'Cost_Reduction_Pct': 'Cost Reduction (%)', 'Intervention_Rate_Pct': 'Intervention Rate (%)'})
    sens_eta.to_csv('results/tables/sensitivity_eta.csv', index=False)
    
    sens_alpha_x_ci = ord_sens[ord_sens['Eta'] == 0.85][
        ['Alpha', 'Cost_Intervene', 'Net_Savings', 'Cost_Reduction_Pct', 'Intervention_Rate_Pct']
    ].rename(columns={'Alpha': 'Loss Rate (alpha)', 'Cost_Intervene': 'Intervention Cost ($)', 'Net_Savings': 'Net Savings ($)', 'Cost_Reduction_Pct': 'Cost Reduction (%)', 'Intervention_Rate_Pct': 'Intervention Rate (%)'})
    sens_alpha_x_ci.to_csv('results/tables/sensitivity_alpha_x_ci.csv', index=False)
    
    sens_alpha_x_eta = ord_sens[ord_sens['Cost_Intervene'] == 15.0][
        ['Alpha', 'Eta', 'Net_Savings', 'Cost_Reduction_Pct', 'Intervention_Rate_Pct']
    ].rename(columns={'Alpha': 'Loss Rate (alpha)', 'Eta': 'Intervention Effectiveness (eta)', 'Net_Savings': 'Net Savings ($)', 'Cost_Reduction_Pct': 'Cost Reduction (%)', 'Intervention_Rate_Pct': 'Intervention Rate (%)'})
    sens_alpha_x_eta.to_csv('results/tables/sensitivity_alpha_x_eta.csv', index=False)
    
    print("\n[OK] All CSV tables saved successfully!")
    
    # 4. Generate Publication Figures
    print("\nGenerating updated publication-grade figures (300 DPI)...")
    for d in ['visuals', 'figures/decision_simulation']:
        os.makedirs(d, exist_ok=True)
        
    # --- Figure 21: Policy Comparison ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Left: Order-Level Threshold Grid
    ax1.plot(order_grid['Threshold'], order_grid['Total Policy Cost ($)'], 'o-', color='#e74c3c', label='Total Policy Cost ($)', linewidth=2)
    ax1.plot(order_grid['Threshold'], order_grid['Net Savings ($)'], 's-', color='#2ecc71', label='Net Savings ($)', linewidth=2)
    ax1.axhline(y=order_bayes['Baseline Economic Loss ($)'], color='#95a5a6', linestyle=':', label=f'Baseline Loss (${order_bayes["Baseline Economic Loss ($)"]:,.0f})')
    ax1.axhline(y=order_bayes['Net Savings ($)'], color='#3498db', linestyle='--', label=f'Optimal Bayes Rule Savings (+${order_bayes["Net Savings ($)"]:,.0f})')
    ax1.set_xlabel('Uniform Probability Threshold (τ)')
    ax1.set_ylabel('Dollar Amount ($)')
    ax1.set_title('A. Order-Level Uniform Threshold Policy\nvs. Observation-Specific Bayes Rule', fontweight='bold')
    ax1.legend(loc='lower left', fontsize=9)
    
    # Right: Item vs Order Comparison Bar Chart
    categories = ['Item-Level Policy', 'Order-Level Policy']
    savings = [item_bayes['Net Savings ($)'], order_bayes['Net Savings ($)']]
    reductions = [item_bayes['Cost Reduction (%)'], order_bayes['Cost Reduction (%)']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, savings, width, label='Net Savings ($)', color='#2ecc71', edgecolor='white')
    ax2.set_ylabel('Net Savings ($)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontweight='bold')
    ax2.set_title('B. Economic Viability by Granularity\n(Correct Realized Cost, α=0.05, C_I=$15, η=0.85)', fontweight='bold')
    
    ax2_twin = ax2.twinx()
    bars2 = ax2_twin.bar(x + width/2, reductions, width, label='Cost Reduction (%)', color='#3498db', edgecolor='white')
    ax2_twin.set_ylabel('Cost Reduction (%)')
    
    for bar, val in zip(bars1, savings):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000, f'+${val:,.0f}', ha='center', fontweight='bold')
    for bar, val in zip(bars2, reductions):
        ax2_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'+{val:.2f}%', ha='center', fontweight='bold')
        
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
    
    plt.tight_layout()
    for p in ['visuals/21_decision_cost_curve.png', 'figures/decision_simulation/21_decision_cost_curve.png']:
        plt.savefig(p, dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] Figure 21: Decision Cost Curve")
    
    # --- Figure 22: Sensitivity to Eta ---
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(sens_eta['Intervention Effectiveness (eta)'].astype(str),
                  sens_eta['Cost Reduction (%)'],
                  color=['#3498db', '#2ecc71', '#e67e22', '#e74c3c', '#9b59b6'],
                  edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, sens_eta['Cost Reduction (%)']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'+{val:.2f}%', ha='center', fontweight='bold', fontsize=10)
    ax.set_xlabel('Intervention Effectiveness (η)')
    ax.set_ylabel('Cost Reduction (%)')
    ax.set_title('Order-Level Cost Reduction vs. Intervention Effectiveness\n(α = 0.05, C_I = $15.00, Bayes Rule)', fontweight='bold')
    plt.tight_layout()
    for p in ['visuals/22_mitigation_sensitivity_curve.png', 'figures/decision_simulation/22_mitigation_sensitivity_curve.png']:
        plt.savefig(p, dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] Figure 22: Sensitivity to Eta")
    
    # --- Figure 23: Sensitivity to Alpha ---
    fig, ax1 = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(sens_alpha))
    width = 0.35
    bars1 = ax1.bar(x_pos - width/2, sens_alpha['Net Savings ($)'],
                    width, label='Net Savings ($)', color='#2ecc71', edgecolor='white')
    ax1.set_xlabel('Loss Rate (α)')
    ax1.set_ylabel('Net Savings ($)')
    ax1.set_title('Order-Level Economic Savings vs. Loss Rate\n(C_I = $15.00, η = 0.85, Bayes Rule)', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'α = {a}' for a in sens_alpha['Loss Rate (alpha)']])
    
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x_pos + width/2, sens_alpha['Cost Reduction (%)'],
                    width, label='Cost Reduction (%)', color='#3498db', edgecolor='white')
    ax2.set_ylabel('Cost Reduction (%)')
    
    for bar, val in zip(bars1, sens_alpha['Net Savings ($)']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                 f'+${val:,.0f}', ha='center', fontsize=9, fontweight='bold')
    for bar, val in zip(bars2, sens_alpha['Cost Reduction (%)']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                 f'+{val:.2f}%', ha='center', fontsize=9, fontweight='bold')
                 
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
    plt.tight_layout()
    for p in ['visuals/23_loss_rate_sensitivity.png', 'figures/decision_simulation/23_loss_rate_sensitivity.png']:
        plt.savefig(p, dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] Figure 23: Sensitivity to Alpha")
    
    # --- Figure 24: Order-Value Distribution & Break-Even Curve ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Distribution of Order Values
    V_ord = order_df['order_item_total'].values
    ax1.hist(V_ord, bins=50, color='#3498db', edgecolor='white', alpha=0.85)
    ax1.axvline(np.mean(V_ord), color='#e74c3c', linestyle='--', linewidth=1.5, label=f'Mean = ${np.mean(V_ord):.2f}')
    ax1.axvline(np.median(V_ord), color='#2ecc71', linestyle='--', linewidth=1.5, label=f'Median = ${np.median(V_ord):.2f}')
    ax1.axvline(352.94, color='#e67e22', linestyle=':', linewidth=2, label='Min Feasible V ($352.94)')
    ax1.set_xlabel('Total Shipment Order Value ($)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('A. Distribution of Order Values (Shipment Level)\nUnique Orders (N = 27,409)', fontweight='bold')
    ax1.legend(fontsize=9)
    
    # Break-Even Probability vs Order Value curve
    val_grid = np.linspace(50, 1500, 300)
    be_curve = 15.0 / (0.85 * 0.05 * val_grid)
    ax2.plot(val_grid, be_curve, color='#e74c3c', linewidth=2.5, label='p_BE(V) = 15 / (0.85 × 0.05 × V)')
    ax2.axhline(y=1.0, color='#95a5a6', linestyle='--', label='Maximum Possible Probability (p = 1.0)')
    ax2.axvline(x=352.94, color='#e67e22', linestyle=':', linewidth=2, label='Feasibility Boundary ($352.94)')
    ax2.fill_between(val_grid, 0, be_curve, where=(be_curve <= 1.0), color='#2ecc71', alpha=0.2, label='Intervention Viability Zone (p > p_BE)')
    ax2.set_ylim(0, 2.0)
    ax2.set_xlabel('Order Value V ($)')
    ax2.set_ylabel('Break-Even Probability Threshold')
    ax2.set_title('B. Order-Value-Dependent Break-Even Threshold\n(α = 0.05, C_I = $15, η = 0.85)', fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    for p in ['visuals/24_order_value_distribution.png', 'figures/decision_simulation/24_order_value_distribution.png']:
        plt.savefig(p, dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] Figure 24: Order Value Distribution & Break-Even Curve")
    print("\n" + "=" * 80)
    print(" COMPLETE ECONOMIC COMPUTATION FINISHED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == '__main__':
    main()
