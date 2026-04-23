# =============================================================================
# Vendor Performance Analysis
# =============================================================================
# Goal: Analyze vendor performance based on sales data to identify:
#   - Top-performing vendors and brands
#   - Pricing and promotional opportunities
#   - Inventory turnover and capital lock-up
#   - Statistical differences in profit margins
# =============================================================================

import os
import warnings
import logging
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from scipy.stats import ttest_ind
import scipy.stats as stats
from sqlalchemy import create_engine

warnings.filterwarnings('ignore')


# =============================================================================
# 1. DATABASE CONNECTION & DATA LOADING
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'inventory.db')

conn = sqlite3.connect(DB_PATH)

# Load vendor sales summary from database
df = pd.read_sql_query("SELECT * FROM vendor_sales_summary", conn)
print(df.head())

# Export to CSV (optional backup)
df.to_csv(os.path.join(BASE_DIR, 'vendor_sales_summary.csv'), index=False)


# =============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# =============================================================================
# Previously, we examined various tables in the database to identify key
# variables and relationships. Here we analyze the summary table to understand
# distributions, anomalies, and data quality.

# --- 2a. Summary Statistics ---
summary_stats = df.describe().T
print(summary_stats)

# Mode for each column
mode_values = df.mode().iloc[0]
print("\nMode Values:\n", mode_values)

# --- 2b. Distribution Plots (All Data) ---
numerical_cols = df.select_dtypes(include=np.number).columns

plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols):
    plt.subplot(4, 4, i + 1)
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(col)
plt.tight_layout()
plt.show()

# --- 2c. Outlier Detection with Boxplots ---
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols):
    plt.subplot(4, 4, i + 1)
    sns.boxplot(y=df[col])
    plt.title(col)
plt.tight_layout()
plt.show()

# INSIGHTS:
# - GrossProfit min = -52,002.78 → some products selling at a loss
# - ProfitMargin min = -inf → revenue is zero or below cost in some cases
# - TotalSalesQuantity min = 0 → some products purchased but never sold
# - FreightCost ranges from 0.09 to 257,032.07 → logistics inefficiencies
# - StockTurnover 0 to 274.5 → some products sell extremely fast


# =============================================================================
# 3. DATA FILTERING (Remove Inconsistencies)
# =============================================================================

df = pd.read_sql_query("""
    SELECT * 
    FROM vendor_sales_summary
    WHERE GrossProfit > 0
      AND ProfitMargin > 0
      AND TotalSalesQuantity > 0
""", conn)

print("Filtered data shape:", df.shape)

# --- 3a. Distribution Plots (Filtered Data) ---
numerical_cols = df.select_dtypes(include=np.number).columns

plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols):
    plt.subplot(4, 4, i + 1)
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(col)
plt.tight_layout()
plt.show()

# --- 3b. Count Plots for Categorical Columns ---
categorical_cols = ["VendorName", "Description"]

plt.figure(figsize=(12, 5))
for i, col in enumerate(categorical_cols):
    plt.subplot(1, 2, i + 1)
    sns.countplot(y=df[col], order=df[col].value_counts().index[:10])
    plt.title(f"Count Plot of {col}")
plt.tight_layout()
plt.show()

# --- 3c. Correlation Heatmap ---
plt.figure(figsize=(12, 8))
correlation_matrix = df[numerical_cols].corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# CORRELATION INSIGHTS:
# - PurchasePrice weakly correlated with TotalSalesDollars (-0.012) & GrossProfit (-0.016)
# - Strong correlation between TotalPurchaseQuantity & TotalSalesQuantity (0.999)
# - Negative correlation between ProfitMargin & TotalSalesPrice (-0.179)
# - StockTurnover weakly negatively correlated with GrossProfit & ProfitMargin


# =============================================================================
# 4. HELPER FUNCTION
# =============================================================================

def format_dollars(value):
    """Format large numbers into readable K/M format."""
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    elif value >= 1_000:
        return f"{value / 1_000:.2f}K"
    else:
        return str(value)


# =============================================================================
# 5. ANALYSIS 1 — Brands Needing Promotional or Pricing Adjustments
# =============================================================================
# Identify brands with low sales performance but high profit margins

brand_performance = df.groupby('Description').agg(
    TotalSalesDollars=('TotalSalesDollars', 'sum'),
    ProfitMargin=('ProfitMargin', 'mean')
).reset_index()

brand_performance.sort_values('ProfitMargin', inplace=True)

# Thresholds: bottom 15% sales, top 15% margin
low_sales_threshold  = brand_performance['TotalSalesDollars'].quantile(0.15)
high_margin_threshold = brand_performance['ProfitMargin'].quantile(0.85)

target_brands = brand_performance[
    (brand_performance['TotalSalesDollars'] <= low_sales_threshold) &
    (brand_performance['ProfitMargin'] >= high_margin_threshold)
]
print("Brands with Low Sales but High Profit Margins:")
print(target_brands.sort_values('TotalSalesDollars'))

# Filter for better visualization
brand_performance_viz = brand_performance[brand_performance['TotalSalesDollars'] < 10000]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=brand_performance_viz, x='TotalSalesDollars', y='ProfitMargin',
                color="blue", label="All Brands", alpha=0.2)
sns.scatterplot(data=target_brands, x='TotalSalesDollars', y='ProfitMargin',
                color="red", label="Target Brands")
plt.axhline(high_margin_threshold, linestyle='--', color='black', label="High Margin Threshold")
plt.axvline(low_sales_threshold,   linestyle='--', color='black', label="Low Sales Threshold")
plt.xlabel("Total Sales ($)")
plt.ylabel("Profit Margin (%)")
plt.title("Brands for Promotional or Pricing Adjustments")
plt.legend()
plt.grid(True)
plt.show()


# =============================================================================
# 6. ANALYSIS 2 — Top Vendors & Brands by Sales Performance
# =============================================================================

top_vendors = df.groupby("VendorName")["TotalSalesDollars"].sum().nlargest(10)
top_brands  = df.groupby("Description")["TotalSalesDollars"].sum().nlargest(10)

print(top_vendors.apply(format_dollars))

plt.figure(figsize=(15, 5))

# Top Vendors
plt.subplot(1, 2, 1)
ax1 = sns.barplot(y=top_vendors.index, x=top_vendors.values, palette="Blues_r")
plt.title("Top 10 Vendors by Sales")
for bar in ax1.patches:
    ax1.text(bar.get_width() + (bar.get_width() * 0.02),
             bar.get_y() + bar.get_height() / 2,
             format_dollars(bar.get_width()),
             ha='left', va='center', fontsize=10, color='black')

# Top Brands
plt.subplot(1, 2, 2)
ax2 = sns.barplot(y=top_brands.index.astype(str), x=top_brands.values, palette="Reds_r")
plt.title("Top 10 Brands by Sales")
for bar in ax2.patches:
    ax2.text(bar.get_width() + (bar.get_width() * 0.02),
             bar.get_y() + bar.get_height() / 2,
             format_dollars(bar.get_width()),
             ha='left', va='center', fontsize=10, color='black')

plt.tight_layout()
plt.show()


# =============================================================================
# 7. ANALYSIS 3 — Vendor Contribution to Total Procurement (Pareto Chart)
# =============================================================================

vendor_performance = df.groupby("VendorName").agg(
    TotalPurchaseDollars=('TotalPurchaseDollars', 'sum'),
    GrossProfit=('GrossProfit', 'sum'),
    TotalSalesDollars=('TotalSalesDollars', 'sum')
).reset_index()

vendor_performance["Purchase_Contribution%"] = (
    vendor_performance["TotalPurchaseDollars"] / vendor_performance["TotalPurchaseDollars"].sum()
) * 100

vendor_performance = round(vendor_performance.sort_values("TotalPurchaseDollars", ascending=False), 2)
top_vendors_perf = vendor_performance.head(10).copy()

# Format for display
top_vendors_perf['TotalSalesDollars']    = top_vendors_perf['TotalSalesDollars'].apply(format_dollars)
top_vendors_perf['TotalPurchaseDollars'] = top_vendors_perf['TotalPurchaseDollars'].apply(format_dollars)
top_vendors_perf['GrossProfit']          = top_vendors_perf['GrossProfit'].apply(format_dollars)
print(top_vendors_perf)

# Pareto Chart
top_vendors_perf['Cumulative_Contribution%'] = top_vendors_perf['Purchase_Contribution%'].cumsum()

fig, ax1 = plt.subplots(figsize=(10, 6))
sns.barplot(x=top_vendors_perf['VendorName'], y=top_vendors_perf['Purchase_Contribution%'],
            palette="mako", ax=ax1)

for i, value in enumerate(top_vendors_perf['Purchase_Contribution%']):
    ax1.text(i, value - 1, f"{value}%", ha='center', fontsize=10, color='white')

ax2 = ax1.twinx()
ax2.plot(top_vendors_perf['VendorName'], top_vendors_perf['Cumulative_Contribution%'],
         color='red', marker='o', linestyle='dashed', label='Cumulative %')

ax1.set_xticklabels(top_vendors_perf['VendorName'], rotation=90)
ax1.set_ylabel('Purchase Contribution %', color='blue')
ax2.set_ylabel('Cumulative Contribution %', color='red')
ax1.set_xlabel('Vendors')
ax1.set_title('Pareto Chart: Vendor Contribution to Total Purchases')
ax2.axhline(y=100, color='gray', linestyle='dashed', alpha=0.7)
ax2.legend(loc='upper right')
plt.show()

# Donut Chart — Vendor Procurement Share
print(f"Top 10 vendors contribute: {round(top_vendors_perf['Purchase_Contribution%'].sum(), 2)}%")

vendors_list = list(top_vendors_perf['VendorName'].values)
contributions = list(top_vendors_perf['Purchase_Contribution%'].values)
total_contribution = sum(contributions)
vendors_list.append("Other Vendors")
contributions.append(100 - total_contribution)

fig, ax = plt.subplots(figsize=(8, 8))
wedges, texts, autotexts = ax.pie(
    contributions, labels=vendors_list, autopct='%1.1f%%',
    startangle=140, pctdistance=0.85, colors=plt.cm.Paired.colors
)
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig.gca().add_artist(centre_circle)
plt.text(0, 0, f"Top 10 Total:\n{total_contribution:.2f}%",
         fontsize=14, fontweight='bold', ha='center', va='center')
plt.title("Top 10 Vendor's Purchase Contribution (%)")
plt.show()

# NOTE: Remaining vendors contribute ~34.31% → under-utilized or less competitive.
# High vendor dependency → consider identifying new suppliers to reduce risk.


# =============================================================================
# 8. ANALYSIS 4 — Bulk Purchasing Impact on Unit Price
# =============================================================================

df["UnitPurchasePrice"] = df["TotalPurchaseDollars"] / df["TotalPurchaseQuantity"]
df["OrderSize"] = pd.qcut(df["TotalPurchaseQuantity"], q=3, labels=["Small", "Medium", "Large"])

bulk_analysis = df.groupby("OrderSize")["UnitPurchasePrice"].mean().reset_index()
print(bulk_analysis)

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="OrderSize", y="UnitPurchasePrice", palette="Set2")
plt.title("Impact of Bulk Purchasing on Unit Price")
plt.xlabel("Order Size")
plt.ylabel("Average Unit Purchase Price")
plt.show()

# INSIGHTS:
# - Large orders: lowest unit price (~$10.78/unit)
# - ~72% cost reduction from Small to Large orders
# - Bulk pricing strategies successfully encourage larger purchase volumes


# =============================================================================
# 9. ANALYSIS 5 — Low Inventory Turnover Vendors (Slow-Moving Stock)
# =============================================================================

low_turnover_vendors = (
    df[df["StockTurnover"] < 1]
    .groupby("VendorName")["StockTurnover"]
    .mean()
    .reset_index()
    .sort_values("StockTurnover", ascending=True)
)
print(low_turnover_vendors.head(10))

# NOTE: Slow-moving inventory increases holding costs (warehouse, insurance, depreciation)


# =============================================================================
# 10. ANALYSIS 6 — Capital Locked in Unsold Inventory per Vendor
# =============================================================================

df["UnsoldInventoryValue"] = (
    (df["TotalPurchaseQuantity"] - df["TotalSalesQuantity"]) * df["PurchasePrice"]
)
print("Total Unsold Capital:", format_dollars(df["UnsoldInventoryValue"].sum()))

inventory_locked = (
    df.groupby("VendorName")["UnsoldInventoryValue"]
    .sum()
    .reset_index()
    .sort_values("UnsoldInventoryValue", ascending=False)
)
inventory_locked["UnsoldInventoryValue"] = inventory_locked["UnsoldInventoryValue"].apply(format_dollars)
print(inventory_locked.head(10))


# =============================================================================
# 11. ANALYSIS 7 — Confidence Intervals: Top vs Low-Performing Vendors
# =============================================================================

top_threshold = df["TotalSalesDollars"].quantile(0.75)
low_threshold  = df["TotalSalesDollars"].quantile(0.25)

top_vendor_margins = df[df["TotalSalesDollars"] >= top_threshold]["ProfitMargin"].dropna()
low_vendor_margins  = df[df["TotalSalesDollars"] <= low_threshold]["ProfitMargin"].dropna()

def confidence_interval(data, confidence=0.95):
    """Compute mean and confidence interval bounds for a data series."""
    mean_val  = np.mean(data)
    std_err   = np.std(data, ddof=1) / np.sqrt(len(data))
    t_critical = stats.t.ppf((1 + confidence) / 2, df=len(data) - 1)
    margin    = t_critical * std_err
    return mean_val, mean_val - margin, mean_val + margin

top_mean, top_lower, top_upper = confidence_interval(top_vendor_margins)
low_mean, low_lower, low_upper = confidence_interval(low_vendor_margins)

print(f"Top Vendors 95% CI: ({top_lower:.2f}, {top_upper:.2f}), Mean: {top_mean:.2f}")
print(f"Low Vendors 95% CI: ({low_lower:.2f}, {low_upper:.2f}), Mean: {low_mean:.2f}")

plt.figure(figsize=(12, 6))

sns.histplot(top_vendor_margins, kde=True, color="blue", bins=30, alpha=0.5, label="Top Vendors")
plt.axvline(top_lower, color="blue", linestyle="--", label=f"Top Lower: {top_lower:.2f}")
plt.axvline(top_upper, color="blue", linestyle="--", label=f"Top Upper: {top_upper:.2f}")
plt.axvline(top_mean,  color="blue", linestyle="-",  label=f"Top Mean: {top_mean:.2f}")

sns.histplot(low_vendor_margins, kde=True, color="red", bins=30, alpha=0.5, label="Low Vendors")
plt.axvline(low_lower, color="red", linestyle="--", label=f"Low Lower: {low_lower:.2f}")
plt.axvline(low_upper, color="red", linestyle="--", label=f"Low Upper: {low_upper:.2f}")
plt.axvline(low_mean,  color="red", linestyle="-",  label=f"Low Mean: {low_mean:.2f}")

plt.title("Confidence Interval Comparison: Top vs. Low Vendors (Profit Margin)")
plt.xlabel("Profit Margin (%)")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.show()

# INSIGHTS:
# - Low-performing vendors CI (40.48%–42.62%) > top-performing (30.74%–31.61%)
# - Vendors with lower sales tend to maintain higher profit margins (premium pricing)
# - High-performing vendors → explore price adjustments & cost optimization
# - Low-performing vendors → need better marketing or competitive pricing


# =============================================================================
# 12. ANALYSIS 8 — Hypothesis Test: Profit Margin Difference (T-Test)
# =============================================================================
# H₀: No significant difference in mean profit margins between top & low vendors
# H₁: Significant difference exists

t_stat, p_value = ttest_ind(top_vendor_margins, low_vendor_margins, equal_var=False)

print(f"T-Statistic: {t_stat:.4f}, P-Value: {p_value:.4f}")
if p_value < 0.05:
    print("Reject H₀: Significant difference in profit margins between top and low-performing vendors.")
else:
    print("Fail to Reject H₀: No significant difference in profit margins.")

# CONCLUSION:
# - Very small p-value → difference is statistically AND practically meaningful
# - The two vendor groups operate very differently in terms of profitability


# =============================================================================
# 13. CLOSE DATABASE CONNECTION
# =============================================================================

conn.close()
print("Database connection closed.")