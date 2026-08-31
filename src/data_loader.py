"""
Data Loader Module
==================
Handles loading, basic inspection, and initial summary statistics generation.
"""

import os
import pandas as pd
import numpy as np

def load_raw_data(data_path='data/raw/Dataset_.csv'):
    """
    Loads raw DataCo dataset from research path or root fallback.
    """
    if not os.path.exists(data_path):
        fallback = '../data/raw/Dataset_.csv'
        if os.path.exists(fallback):
            data_path = fallback
        else:
            fallback2 = 'Dataset_.csv'
            if os.path.exists(fallback2):
                data_path = fallback2
            else:
                raise FileNotFoundError(f"Dataset not found at {data_path}")
                
    df = pd.read_csv(data_path, encoding='latin1', low_memory=False)
    return df

def generate_data_summary(df, output_dir='results/metrics'):
    """
    Generates dataset summary statistics and exports missing value audits.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    summary_data = {
        'Metric': [
            'Total Records (Rows)',
            'Total Features (Columns)',
            'Numerical Features',
            'Categorical Features',
            'Duplicate Records',
            'Missing Target Values',
            'Clean Target Records'
        ],
        'Value': [
            len(df),
            len(df.columns),
            len(df.select_dtypes(include=[np.number]).columns),
            len(df.select_dtypes(include=['object', 'category', 'string']).columns),
            df.duplicated().sum(),
            df['Late_delivery_risk'].isnull().sum() if 'Late_delivery_risk' in df.columns else 0,
            df['Late_delivery_risk'].dropna().count() if 'Late_delivery_risk' in df.columns else 0
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'dataset_summary.csv'), index=False)
    
    return summary_df
