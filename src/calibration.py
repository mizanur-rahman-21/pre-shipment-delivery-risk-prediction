"""
Probability Calibration & Reliability Diagnostics Module
==========================================================
Fits Platt Scaling (Sigmoid) and Isotonic Regression strictly on internal cross-validation of X_train.
Guarantees ZERO test set leakage.
Calculates Brier Score, ECE, MCE, Brier Score Decomposition (Reliability, Resolution, Uncertainty),
Calibration Slope/Intercept, and 95% Bootstrap Confidence Intervals.
"""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, precision_recall_curve, auc
from sklearn.linear_model import LogisticRegression
from scipy.special import logit

def calculate_pr_auc(y_true, y_prob):
    p, r, _ = precision_recall_curve(y_true, y_prob)
    return auc(r, p)

def calculate_ece_mce(y_true, y_prob, n_bins=10):
    """
    Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    mce = 0.0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            diff = np.abs(accuracy_in_bin - avg_confidence_in_bin)
            ece += prop_in_bin * diff
            mce = max(mce, diff)
            
    return ece, mce

def brier_score_decomposition(y_true, y_prob, n_bins=10):
    """
    Computes Brier Score Decomposition: BS = Reliability - Resolution + Uncertainty.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    base_rate = np.mean(y_true)
    uncertainty = base_rate * (1.0 - base_rate)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    reliability = 0.0
    resolution = 0.0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        nk = np.sum(in_bin)
        if nk > 0:
            pk = np.mean(y_prob[in_bin])
            ok = np.mean(y_true[in_bin])
            reliability += (nk / n) * ((pk - ok) ** 2)
            resolution += (nk / n) * ((ok - base_rate) ** 2)
            
    return reliability, resolution, uncertainty

def calculate_calibration_slope_intercept(y_true, y_prob):
    """
    Computes Calibration Slope and Intercept using logistic regression of y on logit(p).
    """
    y_true = np.asarray(y_true)
    y_prob = np.clip(np.asarray(y_prob), 1e-6, 1 - 1e-6)
    logit_p = logit(y_prob).reshape(-1, 1)
    
    lr = LogisticRegression(solver='lbfgs')
    lr.fit(logit_p, y_true)
    
    slope = lr.coef_[0][0]
    intercept = lr.intercept_[0]
    return slope, intercept

def bootstrap_calibration_ci(y_true, y_prob, n_bootstraps=500, random_state=42):
    """
    Computes 95% Bootstrap Confidence Intervals for ECE and Brier Score.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    
    brier_list = []
    ece_list = []
    
    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        sample_y = y_true[idx]
        sample_p = y_prob[idx]
        brier_list.append(brier_score_loss(sample_y, sample_p))
        ece, _ = calculate_ece_mce(sample_y, sample_p)
        ece_list.append(ece * 100)
        
    brier_ci = (np.percentile(brier_list, 2.5), np.percentile(brier_list, 97.5))
    ece_ci = (np.percentile(ece_list, 2.5), np.percentile(ece_list, 97.5))
    return brier_ci, ece_ci

def fit_and_evaluate_calibration_leakfree(model_cls, X_train, y_train, X_test, y_test):
    """
    Fits CalibratedClassifierCV using 5-fold CV on X_train only.
    Evaluates calibrated probabilities on holdout test set with full diagnostic metrics.
    """
    y_test_arr = np.asarray(y_test)
    
    # 1. Raw Uncalibrated Model
    model = model_cls
    model.fit(X_train, y_train)
    raw_probs = model.predict_proba(X_test)[:, 1]
    raw_brier = brier_score_loss(y_test_arr, raw_probs)
    raw_ece, raw_mce = calculate_ece_mce(y_test_arr, raw_probs)
    raw_rel, raw_res, raw_unc = brier_score_decomposition(y_test_arr, raw_probs)
    raw_slope, raw_intercept = calculate_calibration_slope_intercept(y_test_arr, raw_probs)
    raw_brier_ci, raw_ece_ci = bootstrap_calibration_ci(y_test_arr, raw_probs)

    # 2. Platt Scaling (Sigmoid) fitted on X_train via 5-fold CV
    platt_calibrator = CalibratedClassifierCV(estimator=model_cls, method='sigmoid', cv=5)
    platt_calibrator.fit(X_train, y_train)
    platt_probs = platt_calibrator.predict_proba(X_test)[:, 1]
    platt_brier = brier_score_loss(y_test_arr, platt_probs)
    platt_ece, platt_mce = calculate_ece_mce(y_test_arr, platt_probs)
    platt_rel, platt_res, platt_unc = brier_score_decomposition(y_test_arr, platt_probs)
    platt_slope, platt_intercept = calculate_calibration_slope_intercept(y_test_arr, platt_probs)
    platt_brier_ci, platt_ece_ci = bootstrap_calibration_ci(y_test_arr, platt_probs)

    # 3. Isotonic Regression fitted on X_train via 5-fold CV
    iso_calibrator = CalibratedClassifierCV(estimator=model_cls, method='isotonic', cv=5)
    iso_calibrator.fit(X_train, y_train)
    iso_probs = iso_calibrator.predict_proba(X_test)[:, 1]
    iso_brier = brier_score_loss(y_test_arr, iso_probs)
    iso_ece, iso_mce = calculate_ece_mce(y_test_arr, iso_probs)
    iso_rel, iso_res, iso_unc = brier_score_decomposition(y_test_arr, iso_probs)
    iso_slope, iso_intercept = calculate_calibration_slope_intercept(y_test_arr, iso_probs)
    iso_brier_ci, iso_ece_ci = bootstrap_calibration_ci(y_test_arr, iso_probs)

    calibration_metrics = pd.DataFrame([
        {
            'Calibration Method': 'Raw Uncalibrated Model',
            'Brier Score': round(raw_brier, 4),
            'Brier 95% CI': f"[{raw_brier_ci[0]:.4f}, {raw_brier_ci[1]:.4f}]",
            'ECE': f"{raw_ece*100:.2f}%",
            'ECE 95% CI': f"[{raw_ece_ci[0]:.2f}%, {raw_ece_ci[1]:.2f}%]",
            'MCE': f"{raw_mce*100:.2f}%",
            'Reliability': round(raw_rel, 4),
            'Resolution': round(raw_res, 4),
            'Slope': round(raw_slope, 4),
            'Intercept': round(raw_intercept, 4),
            'Fit Protocol': 'X_train only (No Test Leakage)'
        },
        {
            'Calibration Method': 'Platt Scaling (Sigmoid 5-fold CV)',
            'Brier Score': round(platt_brier, 4),
            'Brier 95% CI': f"[{platt_brier_ci[0]:.4f}, {platt_brier_ci[1]:.4f}]",
            'ECE': f"{platt_ece*100:.2f}%",
            'ECE 95% CI': f"[{platt_ece_ci[0]:.2f}%, {platt_ece_ci[1]:.2f}%]",
            'MCE': f"{platt_mce*100:.2f}%",
            'Reliability': round(platt_rel, 4),
            'Resolution': round(platt_res, 4),
            'Slope': round(platt_slope, 4),
            'Intercept': round(platt_intercept, 4),
            'Fit Protocol': 'X_train 5-fold CV'
        },
        {
            'Calibration Method': 'Isotonic Regression (5-fold CV)',
            'Brier Score': round(iso_brier, 4),
            'Brier 95% CI': f"[{iso_brier_ci[0]:.4f}, {iso_brier_ci[1]:.4f}]",
            'ECE': f"{iso_ece*100:.2f}%",
            'ECE 95% CI': f"[{iso_ece_ci[0]:.2f}%, {iso_ece_ci[1]:.2f}%]",
            'MCE': f"{iso_mce*100:.2f}%",
            'Reliability': round(iso_rel, 4),
            'Resolution': round(iso_res, 4),
            'Slope': round(iso_slope, 4),
            'Intercept': round(iso_intercept, 4),
            'Fit Protocol': 'X_train 5-fold CV'
        }
    ])

    prob_dict = {
        'Raw Model': (raw_probs, model),
        'Platt Scaling': (platt_probs, platt_calibrator),
        'Isotonic Regression': (iso_probs, iso_calibrator)
    }

    return calibration_metrics, prob_dict

def assign_delivery_risk_scores(y_probs):
    """
    Assigns operational risk tiers at checkout.
    """
    risk_tiers = []
    for p in y_probs:
        if p < 0.20:
            risk_tiers.append('Low Risk')
        elif p < 0.50:
            risk_tiers.append('Medium Risk')
        elif p < 0.80:
            risk_tiers.append('High Risk')
        else:
            risk_tiers.append('Critical Risk')
            
    risk_df = pd.DataFrame({
        'Calibrated Probability': y_probs,
        'Risk Category': risk_tiers
    })
    return risk_df
