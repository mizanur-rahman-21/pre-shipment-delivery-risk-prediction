"""
Research Experiments Module (E1 - E7)
=====================================
STEP 5 - E1: Leakage-Free Baseline Benchmark (including PR-AUC)
STEP 6 - E2: Temporal Validation (Chronological Train/Validation/Test split)
STEP 11/12 - E7: Geographic Market Generalization (Pacific Asia Test Market Evaluation)
STEP 13/14 - E5: Ablation Study across Feature Settings
"""

import os
import time
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc
)
from xgboost import XGBClassifier
from src.preprocessing import create_leakage_free_dataset
from src.feature_engineering import engineer_features, apply_bayesian_target_encoding, apply_quantile_transformation

def calculate_pr_auc(y_true, y_prob):
    """
    Calculates Area Under Precision-Recall Curve (PR-AUC).
    """
    p, r, _ = precision_recall_curve(y_true, y_prob)
    return auc(r, p)

def run_experiment_e1_baseline(models_dict, X_train, X_test, X_train_qt, X_test_qt, y_train, y_test, output_dir='results/metrics'):
    """
    STEP 5: Experiment E1 - Leakage-Free ML Benchmark
    Evaluates core 7 models on 30% stratified test split (47,415 test samples).
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    for name, (model, use_qt, desc) in models_dict.items():
        xtr = X_train_qt if use_qt else X_train
        xte = X_test_qt if use_qt else X_test
        
        t0 = time.time()
        model.fit(xtr, y_train)
        train_time = round(time.time() - t0, 2)
        
        y_pred = model.predict(xte)
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(xte)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_prob = model.decision_function(xte)
        else:
            y_prob = y_pred
            
        acc = round(accuracy_score(y_test, y_pred) * 100, 2)
        prec = round(precision_score(y_test, y_pred) * 100, 2)
        rec = round(recall_score(y_test, y_pred) * 100, 2)
        f1 = round(f1_score(y_test, y_pred) * 100, 2)
        roc_auc = round(roc_auc_score(y_test, y_prob) * 100, 2)
        pr_auc = round(calculate_pr_auc(y_test, y_prob) * 100, 2)
        
        results.append({
            'Model Name': name,
            'Accuracy': f"{acc:.2f}%",
            'Precision': f"{prec:.2f}%",
            'Recall': f"{rec:.2f}%",
            'F1 Score': f"{f1:.2f}%",
            'ROC-AUC': f"{roc_auc:.2f}%",
            'PR-AUC': f"{pr_auc:.2f}%",
            'Training Time': f"{train_time:.2f}s",
            'Acc_num': acc,
            'F1_num': f1,
            'ROC_num': roc_auc,
            'PR_num': pr_auc
        })
        
    e1_df = pd.DataFrame(results).sort_values(by='Acc_num', ascending=False).reset_index(drop=True)
    e1_df.to_csv(os.path.join(output_dir, 'E1_leakage_free_baseline.csv'), index=False)
    return e1_df

def run_experiment_e2_temporal(df_raw, output_dir='results/metrics'):
    """
    STEP 6: Experiment E2 - Temporal Validation & Generalization
    Strictly uses clean pre-shipment dataset (100% leak-free). Fits XGBoost on 2015-2016 train and evaluates on 2017 test.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Clean leakage-free dataset
    df_pre = create_leakage_free_dataset(df_raw)
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
    
    xgb = XGBClassifier(n_estimators=900, max_depth=16, learning_rate=0.025, subsample=0.88, colsample_bytree=0.88, gamma=0.05, random_state=42, eval_metric='logloss')
    xgb.fit(X_tr, y_tr)
    
    y_pred = xgb.predict(X_te)
    y_prob = xgb.predict_proba(X_te)[:, 1]
    
    acc = accuracy_score(y_te, y_pred) * 100
    prec = precision_score(y_te, y_pred) * 100
    rec = recall_score(y_te, y_pred) * 100
    f1 = f1_score(y_te, y_pred) * 100
    roc_auc = roc_auc_score(y_te, y_prob) * 100
    pr_auc = calculate_pr_auc(y_te, y_prob) * 100
    
    temporal_results = [
        {'Split Period': 'Historical Train (2015-2016)', 'Sample Size': len(train_df), 'Accuracy': 'N/A', 'Precision': 'N/A', 'Recall': 'N/A', 'F1 Score': 'N/A', 'ROC-AUC': 'N/A', 'PR-AUC': 'N/A'},
        {'Split Period': 'Future Test (2017)', 'Sample Size': len(test_df), 'Accuracy': f"{acc:.2f}%", 'Precision': f"{prec:.2f}%", 'Recall': f"{rec:.2f}%", 'F1 Score': f"{f1:.2f}%", 'ROC-AUC': f"{roc_auc:.2f}%", 'PR-AUC': f"{pr_auc:.2f}%"}
    ]
    
    temporal_df = pd.DataFrame(temporal_results)
    temporal_df.to_csv(os.path.join(output_dir, 'E2_temporal_generalization.csv'), index=False)
    return temporal_df, (xgb, X_tr, X_te, y_tr, y_te)

def run_experiment_e5_ablation(X_train, X_test, y_train, y_test, output_dir='results/metrics'):
    """
    STEP 13 & 14: Experiment E5 - Ablation Study across 4 Feature Configurations.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    settings = {
        'Setting A: Full Candidate Features': list(X_train.columns),
        'Setting B: Restricted Baseline Features': [c for c in X_train.columns if '_te' not in c and 'combo' not in c],
        'Setting C: Strict Pre-Shipment Raw': [c for c in X_train.columns if '_te' not in c and '_sin' not in c and '_cos' not in c],
        'Setting D: Strict Pre-Shipment + Engineered': list(X_train.columns)
    }
    
    ablation_results = []
    
    for setting_name, cols in settings.items():
        xtr = X_train[cols]
        xte = X_test[cols]
        
        xgb = XGBClassifier(n_estimators=700, max_depth=14, learning_rate=0.03, subsample=0.85, colsample_bytree=0.85, random_state=42, eval_metric='logloss')
        xgb.fit(xtr, y_train)
        
        y_pred = xgb.predict(xte)
        y_prob = xgb.predict_proba(xte)[:, 1]
        
        acc = accuracy_score(y_test, y_pred) * 100
        prec = precision_score(y_test, y_pred) * 100
        rec = recall_score(y_test, y_pred) * 100
        f1 = f1_score(y_test, y_pred) * 100
        roc_auc = roc_auc_score(y_test, y_prob) * 100
        pr_auc = calculate_pr_auc(y_test, y_prob) * 100
        
        ablation_results.append({
            'Feature Setting': setting_name,
            'Num Features': len(cols),
            'Accuracy': f"{acc:.2f}%",
            'Precision': f"{prec:.2f}%",
            'Recall': f"{rec:.2f}%",
            'F1 Score': f"{f1:.2f}%",
            'ROC-AUC': f"{roc_auc:.2f}%",
            'PR-AUC': f"{pr_auc:.2f}%",
            'Acc_num': acc,
            'F1_num': f1,
            'ROC_num': roc_auc,
            'PR_num': pr_auc
        })
        
    ablation_df = pd.DataFrame(ablation_results)
    ablation_df.to_csv(os.path.join(output_dir, 'E5_ablation_study.csv'), index=False)
    return ablation_df

def run_experiment_e7_geographic_generalization(df_raw, output_dir='results/metrics'):
    """
    STEP 12: Experiment E7 - Geographic Market Generalization
    Strictly uses clean pre-shipment dataset (100% leak-free).
    Trains XGBoost on Europe + LATAM markets and evaluates on out-of-domain Pacific Asia test market.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Clean leakage-free dataset
    df_pre = create_leakage_free_dataset(df_raw)
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
    xgb.fit(X_tr, y_tr)
    
    y_pred = xgb.predict(X_te)
    y_prob = xgb.predict_proba(X_te)[:, 1]
    
    acc = accuracy_score(y_te, y_pred) * 100
    prec = precision_score(y_te, y_pred) * 100
    rec = recall_score(y_te, y_pred) * 100
    f1 = f1_score(y_te, y_pred) * 100
    roc_auc = roc_auc_score(y_te, y_prob) * 100
    pr_auc = calculate_pr_auc(y_te, y_prob) * 100
    
    geo_df = pd.DataFrame([
        {'Market Group': 'Training Markets (Europe + LATAM)', 'Sample Size': len(train_sub), 'Accuracy': 'N/A', 'Precision': 'N/A', 'Recall': 'N/A', 'F1 Score': 'N/A', 'ROC-AUC': 'N/A', 'PR-AUC': 'N/A'},
        {'Market Group': 'Out-of-Domain Test Market (Pacific Asia)', 'Sample Size': len(test_sub), 'Accuracy': f"{acc:.2f}%", 'Precision': f"{prec:.2f}%", 'Recall': f"{rec:.2f}%", 'F1 Score': f"{f1:.2f}%", 'ROC-AUC': f"{roc_auc:.2f}%", 'PR-AUC': f"{pr_auc:.2f}%"}
    ])
    
    geo_df.to_csv(os.path.join(output_dir, 'E7_geographic_robustness.csv'), index=False)
    return geo_df
