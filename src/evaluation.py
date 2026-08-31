"""
Evaluation & Statistical Significance Module (Point 6, 7, 13)
================================================================
Calculates Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Training Time.
Performs paired McNemar's test for statistical significance (XGBoost vs Random Forest).
"""

import os
import time
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix
)

def calculate_pr_auc(y_true, y_prob):
    """
    Area Under Precision-Recall Curve (PR-AUC).
    """
    p, r, _ = precision_recall_curve(y_true, y_prob)
    return auc(r, p)

def evaluate_models_benchmark(models_dict, X_train, X_test, X_train_qt, X_test_qt, y_train, y_test, output_dir='results/metrics'):
    """
    Evaluates 7 core models on holdout test set.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []
    trained_models = {}
    test_preds = {}
    test_probs = {}
    
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
            
        trained_models[name] = model
        test_preds[name] = y_pred
        test_probs[name] = y_prob
        
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
            'Precision_num': prec,
            'Recall_num': rec,
            'F1_num': f1,
            'ROC_num': roc_auc,
            'PR_num': pr_auc
        })
        
    e1_df = pd.DataFrame(results).sort_values(by='Acc_num', ascending=False).reset_index(drop=True)
    e1_df.to_csv(os.path.join(output_dir, 'E1_baseline_benchmark.csv'), index=False)
    return e1_df, trained_models, test_preds, test_probs

def run_mcnemar_statistical_test(y_true, y_pred_xgb, y_pred_rf, output_dir='results/metrics'):
    """
    Point 13: McNemar's paired test for statistical significance (XGBoost vs Random Forest).
    Contingency table:
    b: XGBoost correct, Random Forest incorrect
    c: XGBoost incorrect, Random Forest correct
    """
    os.makedirs(output_dir, exist_ok=True)
    
    y_t = np.array(y_true)
    correct_xgb = (y_pred_xgb == y_t)
    correct_rf = (y_pred_rf == y_t)
    
    b = np.sum(correct_xgb & ~correct_rf)
    c = np.sum(~correct_xgb & correct_rf)
    
    # McNemar's chi-squared statistic with continuity correction
    statistic = ((abs(b - c) - 1) ** 2) / (b + c)
    p_value = chi2.sf(statistic, 1)
    
    # Risk difference confidence interval (95%)
    n = len(y_t)
    diff = (b - c) / n
    se = np.sqrt(((b + c) - ((b - c) ** 2) / n) / (n ** 2))
    ci_lower = diff - 1.96 * se
    ci_upper = diff + 1.96 * se
    
    mcnemar_df = pd.DataFrame([{
        'Comparison': 'XGBoost Classifier vs Random Forest Classifier',
        'XGB Correct / RF Wrong (b)': int(b),
        'XGB Wrong / RF Correct (c)': int(c),
        'McNemar Chi2 Statistic': round(statistic, 4),
        'P-Value': f"{p_value:.4e}",
        'Statistical Significance': 'Significant (p < 0.001)' if p_value < 0.001 else 'Not Significant',
        'Accuracy Difference (b-c)/n': f"{diff*100:.2f}%",
        '95% Confidence Interval': f"[{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]"
    }])
    
    mcnemar_df.to_csv(os.path.join(output_dir, 'statistical_significance_mcnemar.csv'), index=False)
    return mcnemar_df
