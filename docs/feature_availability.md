# 📋 Pre-Shipment Feature Availability & Leakage Scoping Rationale

## 1. Uniform Prediction Point Definition
$$\text{Prediction Point: Immediately after order placement and before warehouse fulfillment/dispatch begins (Order Checkout)}$$

All predictive models must rely exclusively on information observable at checkout. Any variable generated during or after warehouse processing, carrier dispatch, or final transit is strictly removed to prevent **predictive target leakage**.

---

## 2. Feature Classification & Scoping Matrix

| Feature Name | Category | Available at Checkout | Inclusion Status | Methodological Justification | Pipeline Processing Method |
| :--- | :--- | :---: | :---: | :--- | :--- |
| `Days for shipment (scheduled)` | Order SLA | **Yes** | **RETAINED** | Promised delivery SLA timeline. | Raw numerical predictor. |
| `Shipping Mode` | Transport Tier | **Yes** | **RETAINED** | Customer selected delivery tier. | Frequency / One-Hot Encoding. |
| `Order Region` | Location | **Yes** | **RETAINED** | Destination fulfillment region. | 5-Fold Stratified Target Encoding. |
| `Order City` | Location | **Yes** | **RETAINED** | Destination delivery city. | Frequency / Target Encoding. |
| `Order State` | Location | **Yes** | **RETAINED** | Destination delivery state. | Frequency / Target Encoding. |
| `Order Country` | Location | **Yes** | **RETAINED** | Destination country. | Frequency Encoding. |
| `Customer Country` | Location | **Yes** | **RETAINED** | Origin billing country. | Frequency Encoding. |
| `Market` | Trade Zone | **Yes** | **RETAINED** | Global economic trade zone. | Category / Target Encoding. |
| `Category Name` | Merchandise | **Yes** | **RETAINED** | Product merchandise category. | Frequency / Target Encoding. |
| `Department Name` | Department | **Yes** | **RETAINED** | Operational department tier. | Frequency / Target Encoding. |
| `Order Item Quantity` | Basket | **Yes** | **RETAINED** | Order item count. | Raw numerical predictor. |
| `Order Item Total` | Financial | **Yes** | **RETAINED** | Total basket monetary value. | Raw numerical predictor. |
| `Order Item Discount` | Financial | **Yes** | **RETAINED** | Applied promotional discount. | Numerical / Discount Ratio. |
| `Product Price` | Financial | **Yes** | **RETAINED** | Item unit list price. | Raw numerical predictor. |
| `Sales` | Financial | **Yes** | **RETAINED** | Total gross order sales. | Raw numerical predictor. |
| `Order Profit Per Order` | Financial | **Yes** | **RETAINED** | Estimated net profit per order. | Profit Margin Ratio. |
| `order_hour` | Temporal | **Yes** | **RETAINED** | Order placement hour (0-23). | Sine/Cosine Cyclical Encoding. |
| `order_dayofweek` | Temporal | **Yes** | **RETAINED** | Order day of week (0-6). | Sine/Cosine Cyclical Encoding. |
| `order_month` | Temporal | **Yes** | **RETAINED** | Order placement month (1-12). | Sine/Cosine Cyclical Encoding. |
| `order_dayofyear` | Temporal | **Yes** | **RETAINED** | Order placement day of year (1-365).| Numerical / Cyclical. |
| `mode_hour_combo_te` | Interaction | **Yes** | **RETAINED** | Carrier cutoff pickup interaction. | 5-Fold Stratified Out-of-Fold Target Encoding. |
| `mode_scheduled_combo_te` | Interaction | **Yes** | **RETAINED** | SLA tier interaction. | 5-Fold Stratified Out-of-Fold Target Encoding. |
| `Days for shipping (real)` | Post-Dispatch | ❌ **No** | 🚫 **EXCLUDED** | **DATA LEAKAGE**: Actual duration unknown at checkout. | Filtered & Excluded. |
| `shipping date (DateOrders)` | Post-Dispatch | ❌ **No** | 🚫 **EXCLUDED** | **DATA LEAKAGE**: Actual dispatch timestamp unknown at checkout. | Filtered & Excluded. |
| `Delivery Status` | Outcome | ❌ **No** | 🚫 **EXCLUDED** | **DATA LEAKAGE**: Delivery status unknown at checkout. | Filtered & Excluded. |
| `Order Status` | Outcome | ❌ **No** | 🚫 **EXCLUDED** | **DATA LEAKAGE**: Final order status unknown at checkout. | Filtered & Excluded. |
