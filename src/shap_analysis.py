"""
SHAP Explainability Module (Complete Implementation)
=====================================================
Calculates TreeSHAP values for XGBoost model and generates:
1. SHAP Summary / Beeswarm Plot
2. SHAP Feature Importance Bar Plot
3. SHAP Dependence Plots for Top 5 Features
4. Individual Order Waterfall Explanations (3 sample orders)
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shap

def run_complete_shap_analysis(model, X_train, X_test, output_dir='visuals/E6_shap'):
    """
    Computes TreeSHAP values and exports all required visual explanations.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("      [SHAP] Computing TreeSHAP explainer values...")
    explainer = shap.TreeExplainer(model)
    
    # Subsample for smooth TreeSHAP visualization
    sample_size = min(1000, len(X_test))
    X_sample = X_test.sample(sample_size, random_state=42)
    shap_explanation = explainer(X_sample)
    shap_values = shap_explanation.values
    
    # Handle multi-class vs binary output shapes
    if len(shap_values.shape) == 3:
        shap_values_class1 = shap_values[:, :, 1]
        explanation_class1 = shap_explanation[:, :, 1]
    else:
        shap_values_class1 = shap_values
        explanation_class1 = shap_explanation

    # 1. SHAP Summary / Beeswarm Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_class1, X_sample, show=False)
    plt.title('TreeSHAP Global Summary & Feature Contribution Density', fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_summary_beeswarm.png'), dpi=300)
    plt.close()

    # 2. SHAP Feature Importance Bar Plot
    plt.figure(figsize=(10, 5))
    shap.summary_plot(shap_values_class1, X_sample, plot_type='bar', show=False)
    plt.title('TreeSHAP Mean Absolute Feature Importance Weights', fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_bar_importance.png'), dpi=300)
    plt.close()

    # 3. Top 5 Feature Dependence Plots
    mean_abs_shap = np.abs(shap_values_class1).mean(axis=0)
    top_feature_indices = np.argsort(mean_abs_shap)[::-1][:5]
    top_feature_names = X_sample.columns[top_feature_indices]
    
    for i, col in enumerate(top_feature_names, 1):
        plt.figure(figsize=(8, 5))
        shap.dependence_plot(col, shap_values_class1, X_sample, show=False)
        plt.title(f'TreeSHAP Dependence Plot #{i}: {col}', fontweight='bold', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'shap_dependence_top{i}_{col}.png'), dpi=300)
        plt.close()

    # 4. Individual Order Explanations (Waterfall Plots for 3 Sample Orders)
    for sample_idx in range(3):
        plt.figure(figsize=(9, 5))
        try:
            shap.waterfall_plot(explanation_class1[sample_idx], show=False)
            plt.title(f'Individual Order #{sample_idx+1} Local Risk Attribution (Waterfall)', fontweight='bold', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'shap_waterfall_order_{sample_idx+1}.png'), dpi=300)
        except Exception as e:
            # Fallback local bar plot if waterfall fails on specific shap versions
            local_shap = pd.Series(shap_values_class1[sample_idx], index=X_sample.columns).abs().sort_values(ascending=False).head(8)
            local_shap.plot(kind='barh', color='#e74c3c')
            plt.title(f'Individual Order #{sample_idx+1} Feature Contributions', fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'shap_waterfall_order_{sample_idx+1}.png'), dpi=300)
        plt.close()

    print("      [SHAP] Complete SHAP visuals generated: Summary, Bar, 5 Dependence Plots, 3 Individual Order Waterfalls.")
    return top_feature_names.tolist()
