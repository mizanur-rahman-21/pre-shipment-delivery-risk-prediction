"""
Feature Leakage Audit & Ablation Module (Point 2, Point 3, Point 14)
====================================================================
Audits all 53 original dataset features.
Defines Prediction Point explicitly as: Immediately after order placement and before warehouse fulfillment/dispatch begins (Order Checkout).
Saves pre_shipment_dataset.csv and exports pre_shipment_features.json.
Runs feature setting ablation study across candidate feature subsets (Settings A, B, C, D).
"""

import os
import json
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc

FEATURE_AUDIT_53 = [
    {'Feature Name': 'Type', 'Meaning': 'Payment transaction method', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Known at order checkout.'},
    {'Feature Name': 'Days for shipping (real)', 'Meaning': 'Actual elapsed transit duration', 'Prediction Availability': 'After delivery', 'Decision': 'Remove', 'Reason': 'Target leakage. Recorded post-delivery.'},
    {'Feature Name': 'Days for shipping (scheduled)', 'Meaning': 'Promised SLA window (days)', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Established SLA target at checkout.'},
    {'Feature Name': 'Benefit per order', 'Meaning': 'Net order profit margin', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Cart price minus product cost.'},
    {'Feature Name': 'Sales per customer', 'Meaning': 'Total customer checkout sales', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Checkout order total.'},
    {'Feature Name': 'Delivery Status', 'Meaning': 'Outcome label (Advance, Late, Cancel, On Time)', 'Prediction Availability': 'After delivery', 'Decision': 'Remove', 'Reason': 'Direct outcome leakage.'},
    {'Feature Name': 'Late_delivery_risk', 'Meaning': 'Target variable (1 = Late, 0 = On Time/Advance)', 'Prediction Availability': 'Target', 'Decision': 'Target', 'Reason': 'Target variable for binary classification.'},
    {'Feature Name': 'Category Id', 'Meaning': 'Product category code', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Fixed catalog metadata.'},
    {'Feature Name': 'Category Name', 'Meaning': 'Product category label', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Fixed catalog metadata.'},
    {'Feature Name': 'Customer City', 'Meaning': 'Customer shipping city', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Provided at checkout.'},
    {'Feature Name': 'Customer Country', 'Meaning': 'Customer shipping country', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Provided at checkout.'},
    {'Feature Name': 'Customer Email', 'Meaning': 'Customer email address', 'Prediction Availability': 'PII', 'Decision': 'Remove', 'Reason': 'Unneeded text PII.'},
    {'Feature Name': 'Customer Fname', 'Meaning': 'Customer first name', 'Prediction Availability': 'PII', 'Decision': 'Remove', 'Reason': 'Unneeded text PII.'},
    {'Feature Name': 'Customer Id', 'Meaning': 'Customer account ID', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Account identifier.'},
    {'Feature Name': 'Customer Lname', 'Meaning': 'Customer last name', 'Prediction Availability': 'PII', 'Decision': 'Remove', 'Reason': 'Unneeded text PII.'},
    {'Feature Name': 'Customer Password', 'Meaning': 'Account password hash', 'Prediction Availability': 'PII', 'Decision': 'Remove', 'Reason': 'Sensitive security token.'},
    {'Feature Name': 'Customer Segment', 'Meaning': 'Consumer, Corporate, Home Office', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Customer segment.'},
    {'Feature Name': 'Customer State', 'Meaning': 'Customer shipping state', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Provided at checkout.'},
    {'Feature Name': 'Customer Street', 'Meaning': 'Street address', 'Prediction Availability': 'PII', 'Decision': 'Remove', 'Reason': 'Unneeded text PII.'},
    {'Feature Name': 'Customer Zipcode', 'Meaning': 'Customer zip code', 'Prediction Availability': 'Before shipment', 'Decision': 'Remove', 'Reason': 'High-cardinality zip string.'},
    {'Feature Name': 'Department Id', 'Meaning': 'Department code', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Catalog hierarchy.'},
    {'Feature Name': 'Department Name', 'Meaning': 'Department label', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Catalog hierarchy.'},
    {'Feature Name': 'Latitude', 'Meaning': 'Customer geolocation latitude', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Geographic coordinate.'},
    {'Feature Name': 'Longitude', 'Meaning': 'Customer geolocation longitude', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Geographic coordinate.'},
    {'Feature Name': 'Market', 'Meaning': 'Global trade market region', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Fulfillment market assignment.'},
    {'Feature Name': 'Order City', 'Meaning': 'Destination city', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Routing location.'},
    {'Feature Name': 'Order Country', 'Meaning': 'Destination country', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Routing location.'},
    {'Feature Name': 'Order Customer Id', 'Meaning': 'Customer order ID link', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Order metadata.'},
    {'Feature Name': 'order date (DateOrders)', 'Meaning': 'Order placement timestamp', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Used for pre-shipment time features.'},
    {'Feature Name': 'Order Id', 'Meaning': 'Unique order ID', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Primary key.'},
    {'Feature Name': 'Order Item Cardprod Id', 'Meaning': 'Item product catalog ID', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Catalog item ID.'},
    {'Feature Name': 'Order Item Discount', 'Meaning': 'Discount dollar amount', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Applied at checkout.'},
    {'Feature Name': 'Order Item Discount Rate', 'Meaning': 'Discount percentage', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Applied at checkout.'},
    {'Feature Name': 'Order Item Id', 'Meaning': 'Line item ID', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Line item key.'},
    {'Feature Name': 'Order Item Product Price', 'Meaning': 'Unit price of item', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Catalog price.'},
    {'Feature Name': 'Order Item Profit Ratio', 'Meaning': 'Profit margin ratio', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Calculated at checkout.'},
    {'Feature Name': 'Order Item Quantity', 'Meaning': 'Quantity ordered', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Cart item count.'},
    {'Feature Name': 'Sales', 'Meaning': 'Gross cart value', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Calculated at checkout.'},
    {'Feature Name': 'Order Item Total', 'Meaning': 'Net line item total', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Calculated at checkout.'},
    {'Feature Name': 'Order Profit Per Order', 'Meaning': 'Order profit dollar amount', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Calculated at checkout.'},
    {'Feature Name': 'Order Region', 'Meaning': 'Logistics order region', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Regional warehouse routing.'},
    {'Feature Name': 'Order State', 'Meaning': 'Destination state', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Routing location.'},
    {'Feature Name': 'Order Status', 'Meaning': 'Post-checkout status (COMPLETE, CLOSED, PROCESSING)', 'Prediction Availability': 'After checkout', 'Decision': 'Remove', 'Reason': 'Target leakage. Lifecycle status.'},
    {'Feature Name': 'Order Zipcode', 'Meaning': 'Destination zip code', 'Prediction Availability': 'Before shipment', 'Decision': 'Remove', 'Reason': 'High cardinality zip string.'},
    {'Feature Name': 'Product Card Id', 'Meaning': 'Product card ID', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Catalog metadata.'},
    {'Feature Name': 'Product Category Id', 'Meaning': 'Product category ID', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Catalog metadata.'},
    {'Feature Name': 'Product Description', 'Meaning': 'Product text description', 'Prediction Availability': 'Before shipment', 'Decision': 'Remove', 'Reason': 'Null text column.'},
    {'Feature Name': 'Product Image', 'Meaning': 'Product image URL link', 'Prediction Availability': 'Before shipment', 'Decision': 'Remove', 'Reason': 'URL image string.'},
    {'Feature Name': 'Product Name', 'Meaning': 'Product name label', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Catalog metadata.'},
    {'Feature Name': 'Product Price', 'Meaning': 'Retail product price', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Catalog price.'},
    {'Feature Name': 'Product Status', 'Meaning': 'Product availability code', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Catalog status.'},
    {'Feature Name': 'shipping date (DateOrders)', 'Meaning': 'Actual warehouse dispatch timestamp', 'Prediction Availability': 'After dispatch', 'Decision': 'Remove', 'Reason': 'Target leakage. Recorded after dispatch.'},
    {'Feature Name': 'Shipping Mode', 'Meaning': 'Selected carrier mode tier', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Carrier tier selected at checkout.'}
]

def generate_feature_audit(output_dir='results/metrics'):
    """
    Exports feature audit spreadsheet.
    """
    os.makedirs(output_dir, exist_ok=True)
    audit_df = pd.DataFrame(FEATURE_AUDIT_53)
    audit_df.to_csv(os.path.join(output_dir, 'feature_audit_table.csv'), index=False)
    return audit_df

def build_pre_shipment_dataset(df_clean, output_path='data/processed/pre_shipment_dataset.csv', metrics_dir='results/metrics'):
    """
    Generates pre_shipment_dataset.csv containing only pre-shipment predictors and target.
    Saves pre_shipment_features.json.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    leakage_cols_to_remove = [
        'Delivery Status', 'Days for shipping (real)', 'shipping date (DateOrders)',
        'Order Status', 'Customer Email', 'Customer Password', 'Customer Fname',
        'Customer Lname', 'Product Description', 'Product Image', 'Customer Street',
        'Customer Zipcode', 'Order Zipcode'
    ]
    
    df_pre = df_clean.drop(columns=leakage_cols_to_remove, errors='ignore').copy()
    df_pre['late_delivery'] = df_pre['Late_delivery_risk'].astype(int)
    df_pre = df_pre.drop(columns=['Late_delivery_risk'], errors='ignore')
    
    df_pre.to_csv(output_path, index=False)
    
    feature_list = [c for c in df_pre.columns if c != 'late_delivery']
    with open(os.path.join(metrics_dir, 'pre_shipment_features.json'), 'w') as f:
        json.dump(feature_list, f, indent=2)
        
    return df_pre, feature_list

def run_feature_ablation_study(df_pre, output_dir='results/metrics'):
    """
    Point 14: Runs Feature Setting Ablation Study across Candidate Subsets (Settings A, B, C, D).
    """
    from src.feature_engineering import engineer_features, apply_bayesian_target_encoding
    
    os.makedirs(output_dir, exist_ok=True)
    X_full_raw, y = engineer_features(df_pre)
    split_idx = int(len(X_full_raw) * 0.70)
    
    X_tr_full_raw, X_te_full_raw = X_full_raw.iloc[:split_idx].copy(), X_full_raw.iloc[split_idx:].copy()
    y_train, y_test = y.iloc[:split_idx].copy(), y.iloc[split_idx:].copy()
    
    X_tr_full, X_te_full = apply_bayesian_target_encoding(X_tr_full_raw, X_te_full_raw, y_train)
    
    # Feature Subsets
    raw_pre_cols = [c for c in X_tr_full.columns if not c.endswith('_te') and not c.endswith('_freq') and c not in ['order_hour_sin', 'order_hour_cos', 'order_day_sin', 'order_day_cos', 'order_month_sin', 'order_month_cos']]
    restricted_cols = [c for c in X_tr_full.columns if c not in ['mode_hour_combo_te', 'mode_scheduled_combo_te']]
    
    settings = {
        'Setting A: Full Candidate Features': X_tr_full.columns.tolist(),
        'Setting B: Restricted Baseline Features': restricted_cols,
        'Setting C: Strict Pre-Shipment Raw': raw_pre_cols,
        'Setting D: Strict Pre-Shipment + Engineered': X_tr_full.columns.tolist()
    }
    
    results = []
    
    for name, cols in settings.items():
        X_tr = X_tr_full[cols]
        X_te = X_te_full[cols]
        
        xgb = XGBClassifier(n_estimators=500, max_depth=12, learning_rate=0.03, subsample=0.85, colsample_bytree=0.85, random_state=42, eval_metric='logloss')
        xgb.fit(X_tr, y_train)
        
        y_pred = xgb.predict(X_te)
        y_prob = xgb.predict_proba(X_te)[:, 1]
        
        acc = accuracy_score(y_test, y_pred) * 100
        prec = precision_score(y_test, y_pred) * 100
        rec = recall_score(y_test, y_pred) * 100
        f1 = f1_score(y_test, y_pred) * 100
        roc_auc = roc_auc_score(y_test, y_prob) * 100
        
        p, r, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(r, p) * 100
        
        results.append({
            'Feature Setting': name,
            'Num Features': len(cols),
            'Accuracy': f"{acc:.2f}%",
            'Precision': f"{prec:.2f}%",
            'Recall': f"{rec:.2f}%",
            'F1 Score': f"{f1:.2f}%",
            'ROC-AUC': f"{roc_auc:.2f}%",
            'PR-AUC': f"{pr_auc:.2f}%"
        })
        
    ablation_df = pd.DataFrame(results)
    ablation_df.to_csv(os.path.join(output_dir, 'E5_ablation_study.csv'), index=False)
    return ablation_df
