"""
Figure Organization Script
==========================
Organizes figures from visuals/ into structured publication directories inside figures/
"""

import os
import shutil

def organize_figures():
    source_dir = 'visuals'
    base_fig_dir = 'figures'
    
    subdirs = {
        'exploratory': [
            '01_target_distribution.png',
            '02_missing_values_audit.png',
            '03_numerical_distributions.png',
            '04_categorical_distributions.png',
            '05_correlation_heatmap.png',
            '06_shipping_mode_analysis.png',
            '07_geographic_late_rates.png'
        ],
        'model_performance': [
            '08_model_performance_comparison.png',
            '09_roc_curves.png',
            '10_precision_recall_curves.png',
            '11_confusion_matrix.png'
        ],
        'calibration': [
            '12_reliability_diagram.png',
            '13_calibration_comparison.png'
        ],
        'temporal': [
            '17_temporal_performance_comparison.png',
            '20_shap_temporal_drift.png'
        ],
        'geographic': [
            '18_geographic_generalization_comparison.png'
        ],
        'explainability': [
            '14_shap_summary_beeswarm.png',
            '15_shap_feature_importance.png',
            '16_shap_dependence_plots.png',
            '19_risk_category_distribution.png',
            'shap_waterfall_order_1.png',
            'shap_waterfall_order_2.png',
            'shap_waterfall_order_3.png'
        ]
    }
    
    for sub, files in subdirs.items():
        target_path = os.path.join(base_fig_dir, sub)
        os.makedirs(target_path, exist_ok=True)
        for f in files:
            src_file = os.path.join(source_dir, f)
            if os.path.exists(src_file):
                shutil.copy2(src_file, os.path.join(target_path, f))
                
    print("[OK] Organized all visual figures into figures/ subdirectories.")

if __name__ == '__main__':
    organize_figures()
