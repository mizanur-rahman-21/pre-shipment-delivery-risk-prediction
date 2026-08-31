"""
Preprocessing & Complete Feature Audit Module
===============================================
Enforces Prediction Point: Immediately before shipment/dispatch.
Audits every feature and automatically creates pre_shipment_dataset.csv.
"""

import os
import pandas as pd
import numpy as np

# STEP 2 & STEP 3: Complete Feature Audit Table (Prediction Point: Immediately before shipment)
FEATURE_AUDIT_ROSTER = [
    {'Feature Name': 'Type', 'Meaning': 'Payment transaction method (DEBIT, CASH, PAYMENT, TRANSFER)', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Known at order checkout.'},
    {'Feature Name': 'Days for shipping (real)', 'Meaning': 'Actual elapsed shipping duration', 'Prediction Availability': 'After delivery', 'Decision': 'Remove', 'Reason': 'Target leakage. Measured after package reaches customer.'},
    {'Feature Name': 'Days for shipment (scheduled)', 'Meaning': 'Promised SLA fulfillment window (days)', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Fixed target delivery SLA established at checkout.'},
    {'Feature Name': 'Benefit per order', 'Meaning': 'Net order profit margin', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Calculated from cart price and product cost.'},
    {'Feature Name': 'Sales per customer', 'Meaning': 'Total customer checkout sales value', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Known at transaction payment.'},
    {'Feature Name': 'Delivery Status', 'Meaning': 'Final text status (Advance, Late, Cancel, On Time)', 'Prediction Availability': 'After delivery', 'Decision': 'Remove', 'Reason': 'Direct outcome leakage.'},
    {'Feature Name': 'Late_delivery_risk', 'Meaning': 'Target variable (1 = Late, 0 = On Time/Advance)', 'Prediction Availability': 'Target', 'Decision': 'Target', 'Reason': 'Target variable for binary classification.'},
    {'Feature Name': 'Category Id', 'Meaning': 'Product category identifier', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Fixed catalog metadata.'},
    {'Feature Name': 'Category Name', 'Meaning': 'Product category label', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Fixed catalog metadata.'},
    {'Feature Name': 'Customer City', 'Meaning': 'Customer shipping city', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Provided at checkout.'},
    {'Feature Name': 'Customer Country', 'Meaning': 'Customer shipping country', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Provided at checkout.'},
    {'Feature Name': 'Customer Segment', 'Meaning': 'Consumer, Corporate, Home Office', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Customer account segment.'},
    {'Feature Name': 'Customer State', 'Meaning': 'Customer shipping state', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Provided at checkout.'},
    {'Feature Name': 'Department Id', 'Meaning': 'Internal department code', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Catalog organization.'},
    {'Feature Name': 'Department Name', 'Meaning': 'Department name', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Catalog organization.'},
    {'Feature Name': 'Market', 'Meaning': 'Global trade market region (Europe, LATAM, Pacific Asia, etc.)', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Fulfillment market assignment.'},
    {'Feature Name': 'Order City', 'Meaning': 'Destination city', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Order routing location.'},
    {'Feature Name': 'Order Country', 'Meaning': 'Destination country', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Order routing location.'},
    {'Feature Name': 'Order Region', 'Meaning': 'Logistics order region', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Regional warehouse routing.'},
    {'Feature Name': 'Order State', 'Meaning': 'Destination state', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Order routing location.'},
    {'Feature Name': 'order date (DateOrders)', 'Meaning': 'Timestamp of order checkout placement', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Used to extract pre-fulfillment temporal features.'},
    {'Feature Name': 'Order Item Discount', 'Meaning': 'Discount dollar amount', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Applied at checkout.'},
    {'Feature Name': 'Order Item Discount Rate', 'Meaning': 'Discount percentage', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Applied at checkout.'},
    {'Feature Name': 'Order Item Product Price', 'Meaning': 'Unit price of item', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Catalog price.'},
    {'Feature Name': 'Order Item Profit Ratio', 'Meaning': 'Profit margin ratio', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Calculated at checkout.'},
    {'Feature Name': 'Order Item Quantity', 'Meaning': 'Quantity ordered', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Cart item count.'},
    {'Feature Name': 'Sales', 'Meaning': 'Gross cart value', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Calculated at checkout.'},
    {'Feature Name': 'Order Item Total', 'Meaning': 'Net line item total', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Calculated at checkout.'},
    {'Feature Name': 'Order Profit Per Order', 'Meaning': 'Order profit', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Calculated at checkout.'},
    {'Feature Name': 'Order Status', 'Meaning': 'Post-checkout status (COMPLETE, CLOSED, PROCESSING)', 'Prediction Availability': 'After checkout', 'Decision': 'Remove', 'Reason': 'Target leakage. Status changes during fulfillment.'},
    {'Feature Name': 'Product Price', 'Meaning': 'Retail product price', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Catalog price.'},
    {'Feature Name': 'shipping date (DateOrders)', 'Meaning': 'Actual warehouse dispatch timestamp', 'Prediction Availability': 'After dispatch', 'Decision': 'Remove', 'Reason': 'Target leakage. Recorded after warehouse dispatches package.'},
    {'Feature Name': 'Shipping Mode', 'Meaning': 'Selected shipping mode tier (Standard, First, Second, Same Day)', 'Prediction Availability': 'Before shipment', 'Decision': 'Keep', 'Reason': 'Carrier tier selected at checkout.'}
]

def generate_feature_audit_table(output_dir='results/metrics'):
    """
    Exports the comprehensive feature audit spreadsheet/table.
    """
    os.makedirs(output_dir, exist_ok=True)
    audit_df = pd.DataFrame(FEATURE_AUDIT_ROSTER)
    audit_df.to_csv(os.path.join(output_dir, 'feature_audit_table.csv'), index=False)
    return audit_df

def create_leakage_free_dataset(df_raw, output_path='data/processed/pre_shipment_dataset.csv'):
    """
    STEP 4: Creates and saves data/processed/pre_shipment_dataset.csv automatically.
    Enforces Prediction Point: Immediately before shipment/dispatch.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Generate audit table
    generate_feature_audit_table(output_dir=os.path.dirname(output_path).replace('data/processed', 'results/metrics'))
    
    # Filter target
    df_clean = df_raw.dropna(subset=['Late_delivery_risk']).copy()
    df_clean['late_delivery'] = df_clean['Late_delivery_risk'].astype(int)
    
    # Strict list of features to remove due to post-fulfillment leakage or unneeded text PII
    leakage_remove_cols = [
        'Delivery Status', 'Late_delivery_risk', 'Days for shipping (real)',
        'shipping date (DateOrders)', 'Order Status', 'Customer Email',
        'Customer Password', 'Customer Fname', 'Customer Lname',
        'Product Description', 'Product Image', 'Customer Street',
        'Customer Zipcode', 'Order Zipcode'
    ]
    
    df_pre_shipment = df_clean.drop(columns=leakage_remove_cols, errors='ignore').copy()
    
    # Save pre-shipment dataset
    df_pre_shipment.to_csv(output_path, index=False)
    print(f"      [STEP 4] Leakage-Free Dataset Saved: {output_path} ({len(df_pre_shipment):,} rows, {len(df_pre_shipment.columns)} columns)")
    
    return df_pre_shipment
