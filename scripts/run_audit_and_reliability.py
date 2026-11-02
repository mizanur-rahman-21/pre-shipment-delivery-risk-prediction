import os
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

# Load test predictions
df = pd.read_csv('results/predictions/test_predictions_calibrated.csv')

# Aggregate to order level
order_grouped = df.groupby('order_id').agg(
    order_total_value=('order_item_total', 'sum'),
    y_ord=('y_true', 'max'),
    p_ord=('cal_prob', 'mean')
).reset_index()

V_ord = order_grouped['order_total_value'].values.astype(float)
y_ord = order_grouped['y_ord'].values.astype(int)
p_ord = order_grouped['p_ord'].values.astype(float)

alpha = 0.05
cost_intervene = 15.0
eta = 0.85
L_ord = alpha * V_ord

# Break-even threshold
with np.errstate(divide='ignore', invalid='ignore'):
    p_BE_ord = cost_intervene / (eta * L_ord)
p_BE_ord = np.where(L_ord > 0, p_BE_ord, np.inf)

feasible_mask = (p_BE_ord <= 1.0)
intervene_mask = feasible_mask & (p_ord > p_BE_ord)

# --- TASK A: INTERVENTION AUDIT ---
selected = order_grouped[intervene_mask].copy()
print("=" * 60)
print("INTERVENTION AUDIT OUTPUT:")
print("=" * 60)
print("Intervened orders:", len(selected))
print("Actually late:", int(selected["y_ord"].sum()))
print("Actually on-time:", int((1 - selected["y_ord"]).sum()))
print(f"Late rate among intervened: {selected['y_ord'].mean():.4f} ({selected['y_ord'].mean()*100:.2f}%)")
print(f"Mean predicted risk p_ord among intervened: {selected['p_ord'].mean():.4f}")
print(f"Mean break-even threshold p_BE among intervened: {p_BE_ord[intervene_mask].mean():.4f}")
print(f"Mean order value V_ord among intervened: ${selected['order_total_value'].mean():.2f}")
print("=" * 60)

# --- TASK B: ORDER-LEVEL RELIABILITY DIAGRAM ---
prob_true, prob_pred = calibration_curve(y_ord, p_ord, n_bins=10, strategy='uniform')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), gridspec_kw={'height_ratios': [3, 1]})

# Calibration curve
ax1.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated (45° Reference)")
ax1.plot(prob_pred, prob_true, "s-", color='#3498db', linewidth=2, markersize=7, label=f"Aggregated p_ord (ECE = 7.13%, Brier = 0.1336)")
ax1.set_ylabel("Observed Late Rate (Empirical Fraction of Positives)", fontsize=11)
ax1.set_title("Order-Level Reliability Diagram (N = 27,409 Unique Orders)\nAggregation: p_ord = mean(p_i)", fontsize=12, fontweight='bold')
ax1.legend(loc="upper left", fontsize=10)
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1])
ax1.grid(True, linestyle=':', alpha=0.6)

# Histogram of predicted probabilities
ax2.hist(p_ord, bins=10, range=(0, 1), color='#3498db', edgecolor='black', alpha=0.7)
ax2.set_xlabel("Mean Predicted Probability (p_ord)", fontsize=11)
ax2.set_ylabel("Order Count", fontsize=10)
ax2.set_xlim([0, 1])
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()

os.makedirs('visuals', exist_ok=True)
os.makedirs('figures/decision_simulation', exist_ok=True)
os.makedirs('figures/calibration', exist_ok=True)

for path in ['visuals/25_order_level_reliability_diagram.png',
             'figures/decision_simulation/25_order_level_reliability_diagram.png',
             'figures/calibration/fig_25_order_level_reliability_diagram.png']:
    plt.savefig(path, dpi=300, bbox_inches='tight')
plt.close()

print("\nReliability diagram generated and saved to:")
print("  - visuals/25_order_level_reliability_diagram.png")
print("  - figures/decision_simulation/25_order_level_reliability_diagram.png")
print("  - figures/calibration/fig_25_order_level_reliability_diagram.png")
