"""
Explainability Module (Point 11 & Point 19)
===========================================
Calculates TreeSHAP values for XGBoost model and exports:
1. SHAP Beeswarm Plot (visuals/14_shap_summary_beeswarm.png)
2. SHAP Feature Importance Bar (visuals/15_shap_feature_importance.png)
3. SHAP Dependence Plots (visuals/16_shap_dependence_plots.png)
4. 3 Individual Order Waterfall Explanations
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shap

def generate_shap_visuals(model, X_test, y_test=None, output_dir='visuals'):
    """
    Computes TreeSHAP attributions and saves publication-grade figures.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("      [SHAP] Computing TreeSHAP explainer values...")
    explainer = shap.TreeExplainer(model)
    
    sample_size = min(1000, len(X_test))
    X_sample = X_test.sample(sample_size, random_state=42)
    shap_explanation = explainer(X_sample)
    shap_values = shap_explanation.values
    
    if len(shap_values.shape) == 3:
        shap_values_class1 = shap_values[:, :, 1]
        explanation_class1 = shap_explanation[:, :, 1]
    else:
        shap_values_class1 = shap_values
        explanation_class1 = shap_explanation

    # 1. SHAP Beeswarm Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_class1, X_sample, show=False)
    plt.title('Figure 14: TreeSHAP Global Feature Contribution Density (Beeswarm)', fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '14_shap_summary_beeswarm.png'), dpi=300)
    plt.close()

    # 2. SHAP Feature Importance Bar Plot
    plt.figure(figsize=(10, 5))
    shap.summary_plot(shap_values_class1, X_sample, plot_type='bar', show=False)
    plt.title('Figure 15: TreeSHAP Mean Absolute Feature Importance Weights', fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '15_shap_feature_importance.png'), dpi=300)
    plt.close()

    # 3. Top Feature Dependence Plot Grid
    mean_abs_shap = np.abs(shap_values_class1).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:4]
    top_cols = X_sample.columns[top_indices]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    for i, col in enumerate(top_cols):
        plt.sca(axes[i])
        shap.dependence_plot(col, shap_values_class1, X_sample, ax=axes[i], show=False)
        axes[i].set_title(f'SHAP Dependence: {col}', fontweight='bold', fontsize=11)
        
    plt.suptitle('Figure 16: TreeSHAP Dependence Plots for Top Risk Features', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '16_shap_dependence_plots.png'), dpi=300)
    plt.close()

    # 4. Individual Order Waterfall Explanations (3 Samples)
    for idx in range(3):
        plt.figure(figsize=(9, 5))
        try:
            shap.waterfall_plot(explanation_class1[idx], show=False)
            plt.title(f'Individual Order #{idx+1} TreeSHAP Waterfall Explanation', fontweight='bold', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'shap_waterfall_order_{idx+1}.png'), dpi=300)
        except Exception:
            local_s = pd.Series(shap_values_class1[idx], index=X_sample.columns).abs().sort_values(ascending=False).head(8)
            local_s.plot(kind='barh', color='#e74c3c')
            plt.title(f'Individual Order #{idx+1} Local Risk Contributions', fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'shap_waterfall_order_{idx+1}.png'), dpi=300)
        plt.close()

    print("      [SHAP] All SHAP visuals exported cleanly to visuals/")

run_treeshap_analysis = generate_shap_visuals
