import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from sklearn.metrics import brier_score_loss
from src.calibration import calculate_ece_mce

# Load test predictions
df = pd.read_csv('results/predictions/test_predictions_calibrated.csv')
print(f'Total line items: {len(df):,}')
print(f"Unique orders: {df['order_id'].nunique():,}")

# Group by order_id
grouped = df.groupby('order_id').agg(
    n_items=('order_item_total', 'count'),
    v_sum=('order_item_total', 'sum'),
    y_min=('y_true', 'min'),
    y_max=('y_true', 'max'),
    p_mean=('cal_prob', 'mean'),
    p_min=('cal_prob', 'min'),
    p_max=('cal_prob', 'max'),
    p_std=('cal_prob', 'std')
).reset_index()

print('\nCheck target consistency within orders:')
inconsistent_targets = (grouped['y_min'] != grouped['y_max']).sum()
print(f'Orders with inconsistent targets: {inconsistent_targets} (0 = perfectly consistent)')

print('\nCheck probability variation within multi-item orders:')
multi = grouped[grouped['n_items'] > 1]
print(f"Multi-item orders: {len(multi):,} ({len(multi)/len(grouped)*100:.1f}%)")
spread = multi['p_max'] - multi['p_min']
print(f"Mean probability spread (max - min) in multi-item orders: {spread.mean():.4f}")
print(f"Median probability spread (max - min): {spread.median():.4f}")
print(f"75th percentile probability spread: {spread.quantile(0.75):.4f}")
print(f"90th percentile probability spread: {spread.quantile(0.90):.4f}")

# Order-level calibration metrics
y_ord = grouped['y_max'].values
p_ord_mean = grouped['p_mean'].values
p_ord_max = grouped['p_max'].values

brier_mean = brier_score_loss(y_ord, p_ord_mean)
ece_mean, mce_mean = calculate_ece_mce(y_ord, p_ord_mean, n_bins=10)

brier_max = brier_score_loss(y_ord, p_ord_max)
ece_max, mce_max = calculate_ece_mce(y_ord, p_ord_max, n_bins=10)

print('\n=== ORDER-LEVEL CALIBRATION VALIDATION (N = 27,409 unique orders) ===')
print('Aggregation Method: p_ord = MEAN(p_i)')
print(f"  Brier Score: {brier_mean:.4f}")
print(f"  ECE:         {ece_mean*100:.2f}%")
print(f"  MCE:         {mce_mean*100:.2f}%")

print('\nAggregation Method: p_ord = MAX(p_i)')
print(f"  Brier Score: {brier_max:.4f}")
print(f"  ECE:         {ece_max*100:.2f}%")
print(f"  MCE:         {mce_max*100:.2f}%")

# Line-item calibration for comparison
brier_line = brier_score_loss(df['y_true'].values, df['cal_prob'].values)
ece_line, mce_line = calculate_ece_mce(df['y_true'].values, df['cal_prob'].values, n_bins=10)
print('\nLine-Item Level Calibration (for comparison, N = 47,415 items):')
print(f"  Brier Score: {brier_line:.4f}")
print(f"  ECE:         {ece_line*100:.2f}%")
print(f"  MCE:         {mce_line*100:.2f}%")

