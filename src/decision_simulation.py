"""
Order-Value-Based Economic Decision-Cost Simulation Module
==========================================================
Replaces the previous fixed-dollar penalty model with an order-value-dependent
economic loss framework:

    L_i = α * V_i

where V_i is the monetary value of observation i (Order Item Total) and α is the
assumed proportion of order value exposed to late-delivery loss.

The framework implements:
  - Per-observation heterogeneous loss exposure
  - Threshold-based intervention policy: I_i(τ) = 1 if p_i ≥ τ, else 0
  - Order-value-dependent analytical break-even probability
  - Systematic sensitivity analysis across α, C_I, and η
  - Two-dimensional factorial sensitivity grids

Reference: Fisher et al. (2006), "What Is It Worth to You? Communicating
the Dollar Value of Delivering on Time."
"""

import numpy as np
import pandas as pd


def compute_order_value_stats(order_values, alpha=0.05):
    """
    Computes descriptive statistics for order values and loss exposure.

    Parameters
    ----------
    order_values : array-like
        Monetary values (V_i) for each observation.
    alpha : float
        Late-delivery economic loss rate.

    Returns
    -------
    pd.DataFrame
        Descriptive statistics for V_i and L_i = α*V_i.
    """
    v = np.asarray(order_values, dtype=float)
    L = alpha * v

    stats = {
        'Statistic': ['N', 'Mean', 'Median', 'Std Dev', 'Min', 'Max',
                       'P25', 'P75', 'P90', 'P95'],
        'Order Item Total (V_i)': [
            len(v), np.mean(v), np.median(v), np.std(v, ddof=1),
            np.min(v), np.max(v),
            np.percentile(v, 25), np.percentile(v, 75),
            np.percentile(v, 90), np.percentile(v, 95)
        ],
        f'Loss Exposure (L_i = {alpha}*V_i)': [
            len(L), np.mean(L), np.median(L), np.std(L, ddof=1),
            np.min(L), np.max(L),
            np.percentile(L, 25), np.percentile(L, 75),
            np.percentile(L, 90), np.percentile(L, 95)
        ]
    }
    return pd.DataFrame(stats)


def compute_breakeven_probabilities(order_values, alpha=0.05, cost_intervene=15.0, eta=0.85):
    """
    Computes order-value-dependent break-even probabilities.

    The break-even condition is:
        C_I + (1 - η)*L_i < p_i*L_i
    ⟹   p_i > (1 - η) + C_I / (α*V_i)

    Parameters
    ----------
    order_values : array-like
        Monetary values (V_i) for each observation.
    alpha : float
        Late-delivery economic loss rate.
    cost_intervene : float
        Per-observation intervention cost.
    eta : float
        Intervention effectiveness (proportion of loss eliminated).

    Returns
    -------
    dict
        Break-even statistics including mean, median, min, max, and
        proportion of observations where break-even exceeds 1.0.
    """
    v = np.asarray(order_values, dtype=float)
    L = alpha * v
    # Avoid division by zero for observations with L_i = 0
    with np.errstate(divide='ignore', invalid='ignore'):
        p_be = (1.0 - eta) + cost_intervene / L
    p_be = np.where(L > 0, p_be, np.inf)

    return {
        'p_be_mean': np.mean(p_be[np.isfinite(p_be)]),
        'p_be_median': np.median(p_be[np.isfinite(p_be)]),
        'p_be_min': np.min(p_be[np.isfinite(p_be)]),
        'p_be_max': np.max(p_be[np.isfinite(p_be)]),
        'p_be_p25': np.percentile(p_be[np.isfinite(p_be)], 25),
        'p_be_p75': np.percentile(p_be[np.isfinite(p_be)], 75),
        'pct_infeasible': np.mean(p_be > 1.0) * 100,
        'n_infeasible': int(np.sum(p_be > 1.0)),
        'n_total': len(p_be),
        'p_be_values': p_be
    }


def run_order_value_cost_simulation(y_true, y_prob, order_values,
                                     alpha=0.05, cost_intervene=15.0, eta=0.85,
                                     thresholds=None):
    """
    Runs the order-value-based economic decision-cost simulation.

    For each observation i:
        L_i = α * V_i  (late-delivery economic loss exposure)

    Correct Realized Policy Cost:
        C_i = I_i * [ C_I + Y_i * (1 - η) * L_i ] + (1 - I_i) * Y_i * L_i

    where I_i(τ) = 1 if p_i ≥ τ, else 0.
    Note: If Y_i = 0 (on-time), intervention incurs only C_I with zero residual late loss.

    Parameters
    ----------
    y_true : array-like
        Binary ground truth labels (1 = late delivery).
    y_prob : array-like
        Calibrated predicted probabilities.
    order_values : array-like
        Monetary values (V_i) for each observation.
    alpha : float
        Late-delivery economic loss rate (default: 0.05).
    cost_intervene : float
        Per-observation intervention cost (default: $15.00).
    eta : float
        Intervention effectiveness (default: 0.85).
    thresholds : list or None
        Threshold grid. If None, uses {0.05, 0.10, ..., 0.95}.

    Returns
    -------
    sim_df : pd.DataFrame
        Simulation results for each threshold.
    summary_dict : dict
        Summary statistics including baseline cost, optimal threshold, etc.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    v = np.asarray(order_values, dtype=float)

    n_orders = len(y_true)
    L = alpha * v  # per-observation loss exposure

    # Baseline: no intervention — full loss on all late orders
    baseline_cost = np.sum(y_true * L)

    if thresholds is None:
        thresholds = [round(t * 0.05, 2) for t in range(1, 20)]  # 0.05 to 0.95

    simulation_results = []
    best_threshold = None
    best_policy_cost = np.inf

    for tau in thresholds:
        # Intervention mask
        intervene = (y_prob >= tau)
        n_flagged = int(np.sum(intervene))

        # Late orders that are flagged
        late_flagged = y_true & intervene
        n_late_flagged = int(np.sum(late_flagged))

        # Late orders NOT flagged (missed)
        late_missed = y_true & (~intervene)

        # Cost components:
        # 1. Intervention cost for ALL flagged orders (TP + FP)
        total_intervention_cost = n_flagged * cost_intervene

        # 2. Residual loss on flagged late orders: (1-η)*L_i for each
        residual_loss_flagged = np.sum(late_flagged * (1.0 - eta) * L)

        # 3. Full loss on missed late orders: L_i for each
        loss_missed = np.sum(late_missed * L)

        # Total policy cost
        total_policy_cost = total_intervention_cost + residual_loss_flagged + loss_missed

        # Net savings
        net_savings = baseline_cost - total_policy_cost
        cost_reduction_pct = (net_savings / baseline_cost) * 100 if baseline_cost > 0 else 0.0

        # Prevented loss
        prevented_loss = np.sum(late_flagged * eta * L)

        simulation_results.append({
            'Threshold': tau,
            'Flagged Orders': n_flagged,
            'Intervention Rate (%)': round((n_flagged / n_orders) * 100, 2),
            'Late Orders in Test': int(np.sum(y_true)),
            'Late Orders Flagged': n_late_flagged,
            'Total Intervention Cost ($)': round(total_intervention_cost, 2),
            'Residual Late Loss ($)': round(residual_loss_flagged + loss_missed, 2),
            'Total Policy Cost ($)': round(total_policy_cost, 2),
            'Net Savings ($)': round(net_savings, 2),
            'Cost Reduction (%)': round(cost_reduction_pct, 2),
            'Prevented Loss ($)': round(prevented_loss, 2)
        })

        if total_policy_cost < best_policy_cost:
            best_policy_cost = total_policy_cost
            best_threshold = tau

    sim_df = pd.DataFrame(simulation_results)

    summary_dict = {
        'n_orders': n_orders,
        'n_late': int(np.sum(y_true)),
        'baseline_cost': round(baseline_cost, 2),
        'optimal_threshold': best_threshold,
        'optimal_policy_cost': round(best_policy_cost, 2),
        'optimal_net_savings': round(baseline_cost - best_policy_cost, 2),
        'optimal_cost_reduction_pct': round(((baseline_cost - best_policy_cost) / baseline_cost) * 100, 2),
        'alpha': alpha,
        'cost_intervene': cost_intervene,
        'eta': eta
    }

    return sim_df, summary_dict


def run_sensitivity_analysis(y_true, y_prob, order_values,
                              alphas=(0.03, 0.05, 0.10),
                              costs_intervene=(10.0, 15.0, 20.0),
                              etas=(0.70, 0.80, 0.85, 0.90, 0.95),
                              thresholds=None):
    """
    Performs systematic one-way and two-way sensitivity analyses.

    Parameters
    ----------
    y_true, y_prob, order_values : array-like
        Same as run_order_value_cost_simulation.
    alphas : tuple of float
        Loss rate scenarios.
    costs_intervene : tuple of float
        Intervention cost scenarios.
    etas : tuple of float
        Intervention effectiveness scenarios.
    thresholds : list or None
        Threshold grid.

    Returns
    -------
    dict of pd.DataFrames:
        'sensitivity_alpha', 'sensitivity_ci', 'sensitivity_eta',
        'sensitivity_alpha_x_ci', 'sensitivity_alpha_x_eta'
    """
    results = {}

    # --- One-way: varying alpha (baseline C_I=15, eta=0.85) ---
    alpha_rows = []
    for a in alphas:
        _, summary = run_order_value_cost_simulation(
            y_true, y_prob, order_values,
            alpha=a, cost_intervene=15.0, eta=0.85, thresholds=thresholds
        )
        alpha_rows.append({
            'Loss Rate (alpha)': a,
            'Baseline Loss ($)': summary['baseline_cost'],
            'Optimal Threshold': summary['optimal_threshold'],
            'Policy Cost ($)': summary['optimal_policy_cost'],
            'Net Savings ($)': summary['optimal_net_savings'],
            'Cost Reduction (%)': summary['optimal_cost_reduction_pct'],
        })
    results['sensitivity_alpha'] = pd.DataFrame(alpha_rows)

    # --- One-way: varying C_I (baseline alpha=0.05, eta=0.85) ---
    ci_rows = []
    for ci in costs_intervene:
        _, summary = run_order_value_cost_simulation(
            y_true, y_prob, order_values,
            alpha=0.05, cost_intervene=ci, eta=0.85, thresholds=thresholds
        )
        ci_rows.append({
            'Intervention Cost ($)': ci,
            'Baseline Loss ($)': summary['baseline_cost'],
            'Optimal Threshold': summary['optimal_threshold'],
            'Policy Cost ($)': summary['optimal_policy_cost'],
            'Net Savings ($)': summary['optimal_net_savings'],
            'Cost Reduction (%)': summary['optimal_cost_reduction_pct'],
        })
    results['sensitivity_ci'] = pd.DataFrame(ci_rows)

    # --- One-way: varying eta (baseline alpha=0.05, C_I=15) ---
    eta_rows = []
    for e in etas:
        _, summary = run_order_value_cost_simulation(
            y_true, y_prob, order_values,
            alpha=0.05, cost_intervene=15.0, eta=e, thresholds=thresholds
        )
        eta_rows.append({
            'Intervention Effectiveness (eta)': e,
            'Baseline Loss ($)': summary['baseline_cost'],
            'Optimal Threshold': summary['optimal_threshold'],
            'Policy Cost ($)': summary['optimal_policy_cost'],
            'Net Savings ($)': summary['optimal_net_savings'],
            'Cost Reduction (%)': summary['optimal_cost_reduction_pct'],
        })
    results['sensitivity_eta'] = pd.DataFrame(eta_rows)

    # --- Two-way: alpha × C_I (baseline eta=0.85) ---
    axci_rows = []
    for a in alphas:
        for ci in costs_intervene:
            _, summary = run_order_value_cost_simulation(
                y_true, y_prob, order_values,
                alpha=a, cost_intervene=ci, eta=0.85, thresholds=thresholds
            )
            axci_rows.append({
                'Loss Rate (alpha)': a,
                'Intervention Cost ($)': ci,
                'Optimal Threshold': summary['optimal_threshold'],
                'Net Savings ($)': summary['optimal_net_savings'],
                'Cost Reduction (%)': summary['optimal_cost_reduction_pct'],
            })
    results['sensitivity_alpha_x_ci'] = pd.DataFrame(axci_rows)

    # --- Two-way: alpha × eta (baseline C_I=15) ---
    axeta_rows = []
    for a in alphas:
        for e in etas:
            _, summary = run_order_value_cost_simulation(
                y_true, y_prob, order_values,
                alpha=a, cost_intervene=15.0, eta=e, thresholds=thresholds
            )
            axeta_rows.append({
                'Loss Rate (alpha)': a,
                'Intervention Effectiveness (eta)': e,
                'Optimal Threshold': summary['optimal_threshold'],
                'Net Savings ($)': summary['optimal_net_savings'],
                'Cost Reduction (%)': summary['optimal_cost_reduction_pct'],
            })
    results['sensitivity_alpha_x_eta'] = pd.DataFrame(axeta_rows)

    return results
