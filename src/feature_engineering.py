"""
Feature Engineering & Encoding Module
======================================
Engineers pre-shipment domain features, cyclical time encodings, financial ratios,
and 5-fold stratified Bayesian target encodings.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, QuantileTransformer

def engineer_features(df_features):
    """
    Engineers pre-fulfillment features from clean dataset.
    Strictly removes all raw datetime objects to ensure XGBoost matrix compatibility.
    """
    df_feat = df_features.copy()
    
    # 1. Temporal & Cyclical Order Placement Features
    order_date = pd.to_datetime(df_feat['order date (DateOrders)'], errors='coerce')
    df_feat['order_hour'] = order_date.dt.hour.fillna(12).astype(int)
    df_feat['order_dayofweek'] = order_date.dt.dayofweek.fillna(0).astype(int)
    df_feat['order_month'] = order_date.dt.month.fillna(1).astype(int)
    df_feat['order_dayofyear'] = order_date.dt.dayofyear.fillna(1).astype(int)
    df_feat['order_quarter'] = order_date.dt.quarter.fillna(1).astype(int)

    # Cyclical Sine/Cosine Features
    df_feat['order_hour_sin'] = np.sin(2 * np.pi * df_feat['order_hour'] / 24.0)
    df_feat['order_hour_cos'] = np.cos(2 * np.pi * df_feat['order_hour'] / 24.0)
    df_feat['order_day_sin'] = np.sin(2 * np.pi * df_feat['order_dayofweek'] / 7.0)
    df_feat['order_day_cos'] = np.cos(2 * np.pi * df_feat['order_dayofweek'] / 7.0)
    df_feat['order_month_sin'] = np.sin(2 * np.pi * df_feat['order_month'] / 12.0)
    df_feat['order_month_cos'] = np.cos(2 * np.pi * df_feat['order_month'] / 12.0)

    df_feat['order_is_weekend'] = (df_feat['order_dayofweek'] >= 5).astype(int)
    df_feat['order_after_cutoff'] = (df_feat['order_hour'] >= 17).astype(int)
    df_feat['order_is_night'] = ((df_feat['order_hour'] < 6) | (df_feat['order_hour'] >= 20)).astype(int)
    df_feat['is_holiday_season'] = (df_feat['order_month'] >= 11).astype(int)

    # 2. SLA & Shipping Tier Dynamics & Combo Interactions
    mode_expected_days = {'Same Day': 0, 'First Class': 1, 'Second Class': 2, 'Standard Class': 4}
    df_feat['expected_mode_days'] = df_feat['Shipping Mode'].map(mode_expected_days).fillna(3)
    df_feat['sla_mode_diff'] = df_feat['Days for shipment (scheduled)'] - df_feat['expected_mode_days']
    df_feat['scheduled_days_sq'] = df_feat['Days for shipment (scheduled)'] ** 2
    df_feat['is_sla_tight'] = (df_feat['Days for shipment (scheduled)'] <= 1).astype(int)
    df_feat['is_zero_buffer_sla'] = (df_feat['Days for shipment (scheduled)'] == 0).astype(int)

    df_feat['mode_scheduled_combo'] = df_feat['Shipping Mode'].astype(str) + '_' + df_feat['Days for shipment (scheduled)'].astype(str)
    df_feat['mode_region_combo'] = df_feat['Shipping Mode'].astype(str) + '_' + df_feat['Order Region'].astype(str)
    df_feat['mode_hour_combo'] = df_feat['Shipping Mode'].astype(str) + '_' + df_feat['order_hour'].astype(str)
    df_feat['category_mode_combo'] = df_feat['Category Name'].astype(str) + '_' + df_feat['Shipping Mode'].astype(str)

    # 3. Geographic Handoff & Distance Features
    df_feat['is_cross_country'] = (df_feat['Customer Country'].astype(str) != df_feat['Order Country'].astype(str)).astype(int)
    df_feat['is_cross_state'] = (df_feat['Customer State'].astype(str) != df_feat['Order State'].astype(str)).astype(int)
    df_feat['is_cross_city'] = (df_feat['Customer City'].astype(str) != df_feat['Order City'].astype(str)).astype(int)

    # 4. Financial & Order Ratios
    df_feat['unit_price_calc'] = np.where(df_feat['Order Item Quantity'] > 0, df_feat['Order Item Total'] / df_feat['Order Item Quantity'], df_feat['Product Price'])
    df_feat['discount_ratio'] = np.where(df_feat['Sales'] > 0, df_feat['Order Item Discount'] / df_feat['Sales'], 0)
    df_feat['profit_margin'] = np.where(df_feat['Sales'] > 0, df_feat['Order Profit Per Order'] / df_feat['Sales'], 0)
    df_feat['benefit_sales_ratio'] = np.where(df_feat['Sales'] > 0, df_feat['Benefit per order'] / df_feat['Sales'], 0)
    df_feat['total_order_value'] = df_feat['Order Item Total'] * df_feat['Order Item Quantity']
    df_feat['price_qty_interaction'] = df_feat['Product Price'] * df_feat['Order Item Quantity']

    # Explicitly drop raw datetime and ID columns
    drop_ids = ['order date (DateOrders)', 'order_date', 'Order Id', 'Order Item Id', 'Customer Id', 'Order Customer Id']
    df_feat = df_feat.drop(columns=drop_ids, errors='ignore')

    # Drop any remaining datetime dtypes by string type check
    for col in list(df_feat.columns):
        if 'datetime' in str(df_feat[col].dtype):
            df_feat = df_feat.drop(columns=[col])

    # Impute missing numericals
    num_cols = df_feat.select_dtypes(include=[np.number]).columns.drop('late_delivery', errors='ignore')
    df_feat[num_cols] = df_feat[num_cols].fillna(df_feat[num_cols].median())

    # Label & Frequency Encoding
    cat_cols = df_feat.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    for col in cat_cols:
        freq = df_feat[col].value_counts(normalize=True)
        df_feat[f'{col}_freq'] = df_feat[col].map(freq)
        df_feat[col] = LabelEncoder().fit_transform(df_feat[col].astype(str))

    X = df_feat.drop(columns=['late_delivery'], errors='ignore')
    y = df_feat['late_delivery']
    return X, y

def apply_bayesian_target_encoding(X_train, X_test, y_train, target_enc_cols=None):
    """
    Computes 5-fold stratified Bayesian target encodings (m=10) strictly inside training folds.
    """
    X_tr = X_train.copy()
    X_te = X_test.copy()
    
    if target_enc_cols is None:
        target_enc_cols = ['mode_scheduled_combo', 'mode_region_combo', 'mode_hour_combo', 'category_mode_combo', 'Order Region', 'Category Name', 'Department Name', 'Product Name', 'Customer City', 'Customer State', 'Market', 'Type']
        
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    global_mean = y_train.mean()
    m_smooth = 10

    for col in target_enc_cols:
        if col in X_tr.columns:
            te_train = np.zeros(len(X_tr))
            for tr_idx, val_idx in kf.split(X_tr, y_train):
                tr_df = pd.DataFrame({'cat': X_tr.iloc[tr_idx][col].values, 'target': y_train.iloc[tr_idx].values})
                stats = tr_df.groupby('cat')['target'].agg(['count', 'mean'])
                smoothed = (stats['count'] * stats['mean'] + m_smooth * global_mean) / (stats['count'] + m_smooth)
                val_cats = X_tr.iloc[val_idx][col].values
                te_train[val_idx] = pd.Series(val_cats).map(smoothed).fillna(global_mean).values
            
            X_tr[f'{col}_te'] = te_train
            
            overall_stats = pd.DataFrame({'cat': X_tr[col].values, 'target': y_train.values}).groupby('cat')['target'].agg(['count', 'mean'])
            overall_smoothed = (overall_stats['count'] * overall_stats['mean'] + m_smooth * global_mean) / (overall_stats['count'] + m_smooth)
            X_te[f'{col}_te'] = X_te[col].map(overall_smoothed).fillna(global_mean).values

    return X_tr, X_te

def apply_quantile_transformation(X_train, X_test):
    """
    Applies Gaussian Quantile Transformation for linear and probabilistic models.
    """
    qt = QuantileTransformer(output_distribution='normal', random_state=42)
    X_tr_qt = qt.fit_transform(X_train)
    X_te_qt = qt.transform(X_test)
    return X_tr_qt, X_te_qt
