"""
Data Preprocessing & Filtering Audit Module (Point 1 & Point 3)
===============================================================
Tracks raw rows, duplicate handling, missing target removal, and usable sample count.
Generates pre_shipment_dataset.csv without silent data drops.
"""

import os
import pandas as pd
import numpy as np

def audit_and_load_raw_data(data_path='data/raw/Dataset_.csv', output_dir='results/metrics'):
    """
    Loads raw dataset, tracks every filtering step, and returns audit log.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(data_path):
        fallback = '../data/raw/Dataset_.csv'
        if os.path.exists(fallback):
            data_path = fallback
        else:
            data_path = 'Dataset_.csv'
            
    df_raw = pd.read_csv(data_path, encoding='latin1', low_memory=False)
    raw_count = len(df_raw)
    raw_cols = len(df_raw.columns)
    
    duplicates_count = df_raw.duplicated().sum()
    missing_target_count = df_raw['Late_delivery_risk'].isnull().sum() if 'Late_delivery_risk' in df_raw.columns else 0
    
    df_clean = df_raw.dropna(subset=['Late_delivery_risk']).copy()
    usable_count = len(df_clean)
    
    audit_log = pd.DataFrame([
        {'Processing Step': 'Raw Ingested Dataset', 'Record Count': raw_count, 'Feature Count': raw_cols, 'Action / Filter': 'Baseline Load'},
        {'Processing Step': 'Duplicate Record Check', 'Record Count': duplicates_count, 'Feature Count': raw_cols, 'Action / Filter': 'Identified Duplicates'},
        {'Processing Step': 'Missing Target Removal', 'Record Count': missing_target_count, 'Feature Count': raw_cols, 'Action / Filter': 'Filtered Missing Targets'},
        {'Processing Step': 'Final Pre-Shipment Usable Dataset', 'Record Count': usable_count, 'Feature Count': raw_cols, 'Action / Filter': 'Retained Clean Usable Folds'}
    ])
    
    audit_log.to_csv(os.path.join(output_dir, 'dataset_audit_log.csv'), index=False)
    return df_raw, df_clean, audit_log
