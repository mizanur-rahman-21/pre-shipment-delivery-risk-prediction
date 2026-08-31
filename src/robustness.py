"""
Temporal Robustness & Geographic Out-of-Domain Generalization Module
====================================================================
Evaluates 5-Fold Out-of-Fold (OOF) metrics on historical training splits to eliminate artificial overfit.
Evaluates Raw vs Calibrated performance on future temporal test sets (2017) and out-of-domain markets (Pacific Asia).
Implements Leave-One-Market-Out (LOMO) cross-validation across all 5 global geographic markets.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, brier_score_loss
)
from xgboost import XGBClassifier
from src.data_preprocessing import audit_and_load_raw_data
from src.feature_audit import build_pre_shipment_dataset
from src.feature_engineering import engineer_features, apply_bayesian_target_encoding
from src.calibration import calculate_ece_mce

def calculate_pr_auc(y_true, y_prob):
    p, r, _ = precision_recall_curve(y_true, y_prob)
    return auc(r, p)

def evaluate_metrics_row(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred) * 100
    prec = precision_score(y_true, y_pred) * 100
    rec = recall_score(y_true, y_pred) * 100
    f1 = f1_score(y_true, y_pred) * 100
    roc_auc = roc_auc_score(y_true, y_prob) * 100
    pr_auc = calculate_pr_auc(y_true, y_prob) * 100
    brier = brier_score_loss(y_true, y_prob)
    ece, mce = calculate_ece_mce(y_true, y_prob)
    return {
        'Accuracy': f"{acc:.2f}%",
        'Precision': f"{prec:.2f}%",
        'Recall': f"{rec:.2f}%",
        'F1 Score': f"{f1:.2f}%",
        'ROC-AUC': f"{roc_auc:.2f}%",
        'PR-AUC': f"{pr_auc:.2f}%",
        'Brier Score': f"{brier:.4f}",
        'ECE': f"{ece*100:.2f}%",
        'MCE': f"{mce*100:.2f}%"
    }

def run_temporal_calibrated_experiment(df_raw, output_dir='results/metrics', visuals_dir='visuals'):
    """
    Evaluates 5-Fold OOF metrics on Historical Train (2015-2016)
    and Out-of-Time Test metrics on Future 2017 Test Set (Raw vs Calibrated).
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(visuals_dir, exist_ok=True)

    df_raw_loaded, df_clean, _ = audit_and_load_raw_data()
    df_pre, _ = build_pre_shipment_dataset(df_clean)

    # Sort chronologically by order date
    df_pre['order_date'] = pd.to_datetime(df_raw.loc[df_pre.index, 'order date (DateOrders)'], errors='coerce')
    df_sorted = df_pre.sort_values(by='order_date').reset_index(drop=True)
    df_sorted = df_sorted.drop(columns=['order_date'], errors='ignore')

    split_idx = int(len(df_sorted) * 0.70)
    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()

    X_tr_raw, y_tr = engineer_features(train_df)
    X_te_raw, y_te = engineer_features(test_df)

    common_cols = [c for c in X_tr_raw.columns if c in X_te_raw.columns]
    X_tr_raw = X_tr_raw[common_cols]
    X_te_raw = X_te_raw[common_cols]

    X_tr, X_te = apply_bayesian_target_encoding(X_tr_raw, X_te_raw, y_tr)

    base_xgb = XGBClassifier(n_estimators=900, max_depth=16, learning_rate=0.025, subsample=0.88, colsample_bytree=0.88, gamma=0.05, random_state=42, eval_metric='logloss')
    
    # 1. 5-Fold CV OOF on Historical Train
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = cross_val_predict(base_xgb, X_tr, y_tr, cv=cv, method='predict_proba')[:, 1]
    oof_preds = (oof_probs >= 0.5).astype(int)
    tr_oof_metrics = evaluate_metrics_row(y_tr, oof_preds, oof_probs)

    base_xgb.fit(X_tr, y_tr)

    # Fit Isotonic Calibrator strictly on Historical Train 5-fold CV
    iso_calibrator = CalibratedClassifierCV(estimator=XGBClassifier(n_estimators=900, max_depth=16, learning_rate=0.025, subsample=0.88, colsample_bytree=0.88, gamma=0.05, random_state=42, eval_metric='logloss'), method='isotonic', cv=5)
    iso_calibrator.fit(X_tr, y_tr)

    # 2. Future 2017 Test Raw Evaluation
    te_raw_probs = base_xgb.predict_proba(X_te)[:, 1]
    te_raw_pred = base_xgb.predict(X_te)
    te_raw_metrics = evaluate_metrics_row(y_te, te_raw_pred, te_raw_probs)

    # 3. Future 2017 Test Calibrated Evaluation
    te_cal_probs = iso_calibrator.predict_proba(X_te)[:, 1]
    te_cal_pred = (te_cal_probs >= 0.5).astype(int)
    te_cal_metrics = evaluate_metrics_row(y_te, te_cal_pred, te_cal_probs)

    temporal_results = pd.DataFrame([
        {
            'Split Window': 'Historical Train (2015-2016)',
            'Sample Size': len(train_df),
            'Model Configuration': 'Historical 5-Fold Cross-Validation (OOF)',
            **tr_oof_metrics
        },
        {
            'Split Window': 'Future Test (2017 Unseen)',
            'Sample Size': len(test_df),
            'Model Configuration': 'Raw Uncalibrated XGBoost',
            **te_raw_metrics
        },
        {
            'Split Window': 'Future Test (2017 Unseen)',
            'Sample Size': len(test_df),
            'Model Configuration': 'Calibrated XGBoost (Isotonic 5-fold CV)',
            **te_cal_metrics
        }
    ])

    temporal_results.to_csv(os.path.join(output_dir, 'E2_temporal_robustness.csv'), index=False)

    # SHAP Temporal Drift Attribution Analysis
    explainer = shap.TreeExplainer(base_xgb)
    sample_tr = X_tr.sample(min(800, len(X_tr)), random_state=42)
    sample_te = X_te.sample(min(800, len(X_te)), random_state=42)
    
    shap_tr = explainer(sample_tr).values
    shap_te = explainer(sample_te).values
    
    if len(shap_tr.shape) == 3:
        shap_tr = shap_tr[:, :, 1]
        shap_te = shap_te[:, :, 1]
        
    imp_tr = pd.Series(np.abs(shap_tr).mean(axis=0), index=X_tr.columns)
    imp_te = pd.Series(np.abs(shap_te).mean(axis=0), index=X_te.columns)
    
    drift_df = pd.DataFrame({'Historical Train (2015-2016)': imp_tr, 'Future Test (2017)': imp_te})
    drift_df['SHAP Attribution Shift'] = (drift_df['Future Test (2017)'] - drift_df['Historical Train (2015-2016)']).abs()
    top_drift = drift_df.sort_values(by='SHAP Attribution Shift', ascending=False).head(10)
    top_drift.to_csv(os.path.join(output_dir, 'shap_temporal_drift_analysis.csv'))

    plt.figure(figsize=(9, 5.5))
    top_drift[['Historical Train (2015-2016)', 'Future Test (2017)']].plot(kind='barh', figsize=(9, 5.5), color=['#3498db', '#e74c3c'])
    plt.title('Figure 20: TreeSHAP Feature Attribution Drift (Historical 2015-2016 vs Future 2017)', fontweight='bold')
    plt.xlabel('Mean Absolute SHAP Attribution')
    plt.tight_layout()
    plt.savefig(os.path.join(visuals_dir, '20_shap_temporal_drift.png'), dpi=300)
    plt.close()

    return temporal_results, top_drift

def run_geographic_robustness_experiment(df_raw, output_dir='results/metrics'):
    """
    Evaluates 5-Fold OOF metrics on Training Markets (Europe + LATAM)
    and Raw vs Calibrated Out-of-Domain performance on Pacific Asia Test Market.
    Includes Leave-One-Market-Out (LOMO) cross-validation across all 5 markets.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    df_raw_loaded, df_clean, _ = audit_and_load_raw_data()
    df_pre, _ = build_pre_shipment_dataset(df_clean)
    df_pre['Market'] = df_raw.loc[df_pre.index, 'Market']
    
    train_markets = ['Europe', 'LATAM']
    test_markets = ['Pacific Asia']
    
    train_sub = df_pre[df_pre['Market'].isin(train_markets)].copy()
    test_sub = df_pre[df_pre['Market'].isin(test_markets)].copy()
    
    X_tr_raw, y_tr = engineer_features(train_sub)
    X_te_raw, y_te = engineer_features(test_sub)
    
    common_cols = [c for c in X_tr_raw.columns if c in X_te_raw.columns]
    X_tr_raw = X_tr_raw[common_cols]
    X_te_raw = X_te_raw[common_cols]
    
    X_tr, X_te = apply_bayesian_target_encoding(X_tr_raw, X_te_raw, y_tr)
    
    xgb = XGBClassifier(n_estimators=900, max_depth=16, learning_rate=0.025, subsample=0.88, colsample_bytree=0.88, gamma=0.05, random_state=42, eval_metric='logloss')
    
    # 1. 5-Fold CV OOF on Training Markets (Europe + LATAM)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tr_oof_probs = cross_val_predict(xgb, X_tr, y_tr, cv=cv, method='predict_proba')[:, 1]
    tr_oof_preds = (tr_oof_probs >= 0.5).astype(int)
    tr_metrics = evaluate_metrics_row(y_tr, tr_oof_preds, tr_oof_probs)

    xgb.fit(X_tr, y_tr)

    # Fit Isotonic Calibrator strictly on Training Markets 5-fold CV
    iso_calibrator = CalibratedClassifierCV(estimator=XGBClassifier(n_estimators=900, max_depth=16, learning_rate=0.025, subsample=0.88, colsample_bytree=0.88, gamma=0.05, random_state=42, eval_metric='logloss'), method='isotonic', cv=5)
    iso_calibrator.fit(X_tr, y_tr)

    # 2. Out-of-Domain Raw Test Performance (Pacific Asia)
    te_raw_probs = xgb.predict_proba(X_te)[:, 1]
    te_raw_pred = xgb.predict(X_te)
    te_raw_metrics = evaluate_metrics_row(y_te, te_raw_pred, te_raw_probs)

    # 3. Out-of-Domain Calibrated Test Performance (Pacific Asia)
    te_cal_probs = iso_calibrator.predict_proba(X_te)[:, 1]
    te_cal_pred = (te_cal_probs >= 0.5).astype(int)
    te_cal_metrics = evaluate_metrics_row(y_te, te_cal_pred, te_cal_probs)

    geo_df = pd.DataFrame([
        {
            'Market Group': 'Training Markets (Europe + LATAM)',
            'Sample Size': len(train_sub),
            'Evaluation Mode': '5-Fold Cross-Validation OOF',
            **tr_metrics
        },
        {
            'Market Group': 'Out-of-Domain Test Market (Pacific Asia)',
            'Sample Size': len(test_sub),
            'Evaluation Mode': 'Raw Uncalibrated OOD Test',
            **te_raw_metrics
        },
        {
            'Market Group': 'Out-of-Domain Test Market (Pacific Asia)',
            'Sample Size': len(test_sub),
            'Evaluation Mode': 'Calibrated XGBoost (Isotonic 5-fold CV) OOD Test',
            **te_cal_metrics
        }
    ])
    
    geo_df.to_csv(os.path.join(output_dir, 'E7_geographic_robustness.csv'), index=False)

    # 4. Leave-One-Market-Out (LOMO) Cross-Validation across all 5 markets
    all_markets = df_pre['Market'].dropna().unique().tolist()
    lomo_results = []

    for target_market in all_markets:
        lomo_tr_sub = df_pre[df_pre['Market'] != target_market].copy()
        lomo_te_sub = df_pre[df_pre['Market'] == target_market].copy()
        
        if len(lomo_te_sub) < 100:
            continue

        X_lomo_tr_raw, y_lomo_tr = engineer_features(lomo_tr_sub)
        X_lomo_te_raw, y_lomo_te = engineer_features(lomo_te_sub)
        
        c_cols = [c for c in X_lomo_tr_raw.columns if c in X_lomo_te_raw.columns]
        X_lomo_tr_raw = X_lomo_tr_raw[c_cols]
        X_lomo_te_raw = X_lomo_te_raw[c_cols]
        
        X_lomo_tr, X_lomo_te = apply_bayesian_target_encoding(X_lomo_tr_raw, X_lomo_te_raw, y_lomo_tr)
        
        lomo_model = XGBClassifier(n_estimators=500, max_depth=12, learning_rate=0.03, subsample=0.85, colsample_bytree=0.85, random_state=42, eval_metric='logloss')
        lomo_model.fit(X_lomo_tr, y_lomo_tr)
        
        lomo_calibrator = CalibratedClassifierCV(estimator=XGBClassifier(n_estimators=500, max_depth=12, learning_rate=0.03, subsample=0.85, colsample_bytree=0.85, random_state=42, eval_metric='logloss'), method='isotonic', cv=3)
        lomo_calibrator.fit(X_lomo_tr, y_lomo_tr)
        
        lomo_probs = lomo_calibrator.predict_proba(X_lomo_te)[:, 1]
        lomo_preds = (lomo_probs >= 0.5).astype(int)
        
        m_eval = evaluate_metrics_row(y_lomo_te, lomo_preds, lomo_probs)
        lomo_results.append({
            'Left-Out Market': target_market,
            'Test Sample Size': len(lomo_te_sub),
            **m_eval
        })
        
    lomo_df = pd.DataFrame(lomo_results)
    lomo_df.to_csv(os.path.join(output_dir, 'leave_one_market_out_lomo_cv.csv'), index=False)

    return geo_df
