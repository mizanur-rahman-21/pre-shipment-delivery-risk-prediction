"""
Decision & Cost-Benefit Simulation Module
==========================================
Simulates real-world operational decision making and intervention financial economics.
Evaluates cost savings ($) under risk probability thresholds (Low, Medium, High, Critical)
and calculates expected cost reduction compared to unmitigated late delivery baseline.
"""

import numpy as np
import pandas as pd

def run_decision_cost_simulation(y_true, y_prob, cost_late=50.0, cost_intervene=15.0, intervention_eff=0.85):
    """
    Simulates operational intervention decision policy:
    - Base Unmitigated Cost: If an order is late and not intervened, penalty = cost_late ($50.00).
    - Policy Intervention: If predicted probability p >= threshold, trigger intervention costing cost_intervene ($15.00).
    - Intervention Success: Intervention reduces late probability by intervention_eff (85% of late orders prevented).

    Evaluates across risk thresholds p in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80].
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n_orders = len(y_true)
    n_late_base = np.sum(y_true)
    
    baseline_unmitigated_cost = n_late_base * cost_late
    
    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    optimal_analytical_threshold = cost_intervene / (cost_late * intervention_eff)  # e.g. 15 / (50 * 0.85) = 0.3529
    
    simulation_results = []
    
    for th in thresholds:
        intervene_mask = (y_prob >= th)
        n_intervened = np.sum(intervene_mask)
        
        # Intervened late orders
        late_intervened = y_true & intervene_mask
        # Intervened on-time orders (false alarm intervention)
        ontime_intervened = (~y_true) & intervene_mask
        
        # Un-intervened late orders
        late_unintervened = y_true & (~intervene_mask)
        
        # Prevented late deliveries (85% success rate on intervened late orders)
        prevented_late = np.sum(late_intervened) * intervention_eff
        remaining_late = np.sum(late_intervened) * (1.0 - intervention_eff) + np.sum(late_unintervened)
        
        intervention_cost_total = n_intervened * cost_intervene
        late_penalty_cost_total = remaining_late * cost_late
        total_policy_cost = intervention_cost_total + late_penalty_cost_total
        
        cost_savings = baseline_unmitigated_cost - total_policy_cost
        cost_reduction_pct = (cost_savings / baseline_unmitigated_cost) * 100
        
        simulation_results.append({
            'Probability Threshold': f"p >= {th:.2f}",
            'Intervened Orders': int(n_intervened),
            'Intervention Rate': f"{(n_intervened / n_orders) * 100:.2f}%",
            'Prevented Delays': int(prevented_late),
            'Total Intervention Cost ($)': f"${intervention_cost_total:,.2f}",
            'Total Penalty Cost ($)': f"${late_penalty_cost_total:,.2f}",
            'Total Operational Cost ($)': f"${total_policy_cost:,.2f}",
            'Net Cost Savings ($)': f"${cost_savings:,.2f}",
            'Cost Reduction (%)': f"{cost_reduction_pct:.2f}%"
        })
        
    sim_df = pd.DataFrame(simulation_results)
    
    summary_dict = {
        'n_orders': n_orders,
        'baseline_cost': baseline_unmitigated_cost,
        'optimal_threshold': optimal_analytical_threshold,
        'sim_df': sim_df
    }
    return sim_df, summary_dict
