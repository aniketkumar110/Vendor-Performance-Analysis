# =============================================================================
# MAIN PIPELINE — Vendor Performance Analysis
# =============================================================================
# Just run this file to execute the full pipeline:
#   Step 1: Ingest CSV data into the database
#   Step 2: Run Exploratory Data Analysis
#   Step 3: Run Vendor Performance Analysis
# =============================================================================

import os
import sys

# ── Make sure all scripts are found ──────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ── Create required folders if they don't exist ───────────────────────────────
os.makedirs(os.path.join(BASE_DIR, 'data'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'logs'), exist_ok=True)


# =============================================================================
# STEP 1 — INGEST CSV FILES INTO DATABASE
# =============================================================================

def run_ingestion():
    print("\n" + "="*60)
    print("  STEP 1: Ingesting CSV files into database...")
    print("="*60)

    import time
    import logging
    import pandas as pd
    from sqlalchemy import create_engine

    db_path = os.path.join(BASE_DIR, 'inventory.db')
    data_dir = os.path.join(BASE_DIR, 'data')

    logging.basicConfig(
        filename=os.path.join(BASE_DIR, 'logs', 'ingestion_db.log'),
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="a"
    )

    # Check if data folder has CSV files
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not csv_files:
        print("\n❌ ERROR: No CSV files found in the 'data/' folder!")
        print(f"   Please download 'vendor_sales_summary.csv' from GitHub and place it in:")
        print(f"   {data_dir}")
        print("\n   Download link:")
        print("   https://github.com/Hrishit31/Vendor-Performance-Analysis-/blob/main/vendor_sales_summary.csv\n")
        sys.exit(1)

    engine = create_engine(f'sqlite:///{db_path}')

    start = time.time()
    for file in csv_files:
        file_path = os.path.join(data_dir, file)
        table_name = file[:-4]  # strip .csv
        df = pd.read_csv(file_path)
        df.to_sql(table_name, con=engine, if_exists='replace', index=False)
        logging.info(f'Ingested {file} → table: {table_name}')
        print(f"   ✅ Ingested: {file} → table '{table_name}' ({len(df)} rows)")

    elapsed = round((time.time() - start) / 60, 2)
    logging.info(f'Ingestion Complete. Time taken: {elapsed} minutes')
    print(f"\n   ✅ Ingestion complete in {elapsed} minutes")


# =============================================================================
# STEP 2 — EXPLORATORY DATA ANALYSIS
# =============================================================================

def run_eda():
    print("\n" + "="*60)
    print("  STEP 2: Running Exploratory Data Analysis...")
    print("="*60)

    import sqlite3
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings('ignore')

    conn = sqlite3.connect(os.path.join(BASE_DIR, 'inventory.db'))

    # ── Load data ─────────────────────────────────────────────────────────────
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
    print("\n   Tables in database:", list(tables['name']))

    for table in tables['name']:
        count = pd.read_sql_query(f"SELECT COUNT(*) as count FROM {table}", conn)['count'].values[0]
        print(f"   → {table}: {count} records")

    # ── Vendor Sales Summary ──────────────────────────────────────────────────
    vendor_sales_summary = pd.read_sql_query("SELECT * FROM vendor_sales_summary", conn)

    print("\n   Data types:\n", vendor_sales_summary.dtypes)
    print("\n   Null values:\n", vendor_sales_summary.isnull().sum())

    # ── Data Cleaning ─────────────────────────────────────────────────────────
    vendor_sales_summary['GrossProfit']          = vendor_sales_summary['TotalSalesDollars'] - vendor_sales_summary['TotalPurchaseDollars']
    vendor_sales_summary['ProfitMargin']         = (vendor_sales_summary['GrossProfit'] / vendor_sales_summary['TotalSalesDollars']) * 100
    vendor_sales_summary['StockTurnover']        = vendor_sales_summary['TotalSalesQuantity'] / vendor_sales_summary['TotalPurchaseQuantity']
    vendor_sales_summary['SalesToPurchaseRatio'] = vendor_sales_summary['TotalSalesDollars'] / vendor_sales_summary['TotalPurchaseDollars']

    if 'Volume' in vendor_sales_summary.columns:
        vendor_sales_summary['Volume'] = pd.to_numeric(vendor_sales_summary['Volume'], errors='coerce')

    vendor_sales_summary.fillna(0, inplace=True)

    if 'VendorName' in vendor_sales_summary.columns:
        vendor_sales_summary['VendorName'] = vendor_sales_summary['VendorName'].str.strip()
    if 'Description' in vendor_sales_summary.columns:
        vendor_sales_summary['Description'] = vendor_sales_summary['Description'].str.strip()

    # Save cleaned summary back to DB
    vendor_sales_summary.to_sql('vendor_sales_summary', conn, if_exists='replace', index=False)
    print("\n   ✅ EDA complete — cleaned data saved back to database")

    conn.close()


# =============================================================================
# STEP 3 — VENDOR PERFORMANCE ANALYSIS
# =============================================================================

def run_analysis():
    print("\n" + "="*60)
    print("  STEP 3: Running Vendor Performance Analysis...")
    print("="*60)

    import sqlite3
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    from scipy.stats import ttest_ind
    import scipy.stats as stats
    warnings.filterwarnings('ignore')

    conn = sqlite3.connect(os.path.join(BASE_DIR, 'inventory.db'))

    # ── Load filtered data ────────────────────────────────────────────────────
    df = pd.read_sql_query("""
        SELECT * FROM vendor_sales_summary
        WHERE GrossProfit > 0
          AND ProfitMargin > 0
          AND TotalSalesQuantity > 0
    """, conn)

    df.to_csv(os.path.join(BASE_DIR, 'vendor_sales_summary.csv'), index=False)
    print(f"\n   Loaded {len(df)} rows for analysis")

    numerical_cols = df.select_dtypes(include=np.number).columns

    # ── Helper ────────────────────────────────────────────────────────────────
    def format_dollars(value):
        if value >= 1_000_000:
            return f"{value / 1_000_000:.2f}M"
        elif value >= 1_000:
            return f"{value / 1_000:.2f}K"
        return str(value)

    # ── Plot 1: Distribution ──────────────────────────────────────────────────
    print("\n   → Generating distribution plots...")
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numerical_cols):
        plt.subplot(4, 4, i + 1)
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(col)
    plt.suptitle("Distribution of Numerical Columns", y=1.01)
    plt.tight_layout()
    plt.show()

    # ── Plot 2: Correlation Heatmap ───────────────────────────────────────────
    print("   → Generating correlation heatmap...")
    plt.figure(figsize=(12, 8))
    sns.heatmap(df[numerical_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    # ── Plot 3: Top Vendors & Brands ──────────────────────────────────────────
    print("   → Generating top vendors & brands chart...")
    top_vendors = df.groupby("VendorName")["TotalSalesDollars"].sum().nlargest(10)
    top_brands  = df.groupby("Description")["TotalSalesDollars"].sum().nlargest(10)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    ax1 = sns.barplot(y=top_vendors.index, x=top_vendors.values, palette="Blues_r")
    plt.title("Top 10 Vendors by Sales")
    for bar in ax1.patches:
        ax1.text(bar.get_width() * 1.02, bar.get_y() + bar.get_height() / 2,
                 format_dollars(bar.get_width()), ha='left', va='center', fontsize=9)

    plt.subplot(1, 2, 2)
    ax2 = sns.barplot(y=top_brands.index.astype(str), x=top_brands.values, palette="Reds_r")
    plt.title("Top 10 Brands by Sales")
    for bar in ax2.patches:
        ax2.text(bar.get_width() * 1.02, bar.get_y() + bar.get_height() / 2,
                 format_dollars(bar.get_width()), ha='left', va='center', fontsize=9)
    plt.tight_layout()
    plt.show()

    # ── Plot 4: Bulk Purchasing Impact ────────────────────────────────────────
    print("   → Analyzing bulk purchasing impact...")
    df["UnitPurchasePrice"] = df["TotalPurchaseDollars"] / df["TotalPurchaseQuantity"]
    df["OrderSize"] = pd.qcut(df["TotalPurchaseQuantity"], q=3, labels=["Small", "Medium", "Large"])

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="OrderSize", y="UnitPurchasePrice", palette="Set2")
    plt.title("Impact of Bulk Purchasing on Unit Price")
    plt.xlabel("Order Size")
    plt.ylabel("Avg Unit Purchase Price ($)")
    plt.tight_layout()
    plt.show()

    # ── Plot 5: Brands Needing Promotion ─────────────────────────────────────
    print("   → Identifying brands needing promotion...")
    brand_perf = df.groupby('Description').agg(
        TotalSalesDollars=('TotalSalesDollars', 'sum'),
        ProfitMargin=('ProfitMargin', 'mean')
    ).reset_index()

    low_sales  = brand_perf['TotalSalesDollars'].quantile(0.15)
    high_margin = brand_perf['ProfitMargin'].quantile(0.85)
    target_brands = brand_perf[
        (brand_perf['TotalSalesDollars'] <= low_sales) &
        (brand_perf['ProfitMargin'] >= high_margin)
    ]

    brand_viz = brand_perf[brand_perf['TotalSalesDollars'] < 10000]
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=brand_viz, x='TotalSalesDollars', y='ProfitMargin',
                    color="blue", label="All Brands", alpha=0.2)
    sns.scatterplot(data=target_brands, x='TotalSalesDollars', y='ProfitMargin',
                    color="red", label="Target Brands")
    plt.axhline(high_margin, linestyle='--', color='black', label="High Margin Threshold")
    plt.axvline(low_sales,   linestyle='--', color='black', label="Low Sales Threshold")
    plt.title("Brands for Promotional or Pricing Adjustments")
    plt.xlabel("Total Sales ($)")
    plt.ylabel("Profit Margin (%)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ── Plot 6: Confidence Intervals ──────────────────────────────────────────
    print("   → Computing confidence intervals...")
    top_margins = df[df["TotalSalesDollars"] >= df["TotalSalesDollars"].quantile(0.75)]["ProfitMargin"].dropna()
    low_margins  = df[df["TotalSalesDollars"] <= df["TotalSalesDollars"].quantile(0.25)]["ProfitMargin"].dropna()

    def confidence_interval(data, confidence=0.95):
        mean_val   = np.mean(data)
        std_err    = np.std(data, ddof=1) / np.sqrt(len(data))
        t_critical = stats.t.ppf((1 + confidence) / 2, df=len(data) - 1)
        margin     = t_critical * std_err
        return mean_val, mean_val - margin, mean_val + margin

    top_mean, top_lower, top_upper = confidence_interval(top_margins)
    low_mean, low_lower, low_upper = confidence_interval(low_margins)

    print(f"   Top Vendors 95% CI: ({top_lower:.2f}, {top_upper:.2f}), Mean: {top_mean:.2f}")
    print(f"   Low Vendors 95% CI: ({low_lower:.2f}, {low_upper:.2f}), Mean: {low_mean:.2f}")

    plt.figure(figsize=(12, 6))
    sns.histplot(top_margins, kde=True, color="blue", bins=30, alpha=0.5, label="Top Vendors")
    plt.axvline(top_lower, color="blue", linestyle="--", label=f"Top Lower: {top_lower:.2f}")
    plt.axvline(top_upper, color="blue", linestyle="--", label=f"Top Upper: {top_upper:.2f}")
    plt.axvline(top_mean,  color="blue", linestyle="-",  label=f"Top Mean:  {top_mean:.2f}")
    sns.histplot(low_margins, kde=True, color="red", bins=30, alpha=0.5, label="Low Vendors")
    plt.axvline(low_lower, color="red", linestyle="--", label=f"Low Lower: {low_lower:.2f}")
    plt.axvline(low_upper, color="red", linestyle="--", label=f"Low Upper: {low_upper:.2f}")
    plt.axvline(low_mean,  color="red", linestyle="-",  label=f"Low Mean:  {low_mean:.2f}")
    plt.title("Confidence Interval: Top vs. Low Vendors (Profit Margin)")
    plt.xlabel("Profit Margin (%)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ── Hypothesis Test ───────────────────────────────────────────────────────
    print("\n   → Running T-Test (Profit Margin: Top vs Low vendors)...")
    t_stat, p_value = ttest_ind(top_margins, low_margins, equal_var=False)
    print(f"   T-Statistic: {t_stat:.4f}, P-Value: {p_value:.4f}")
    if p_value < 0.05:
        print("   ✅ Reject H₀: Significant difference in profit margins.")
    else:
        print("   ✅ Fail to Reject H₀: No significant difference.")

    conn.close()
    print("\n   ✅ Analysis complete!")


# =============================================================================
# RUN PIPELINE
# =============================================================================

if __name__ == '__main__':
    print("\n🚀 Starting Vendor Performance Analysis Pipeline...")

    run_ingestion()   # Step 1: Load CSVs → DB
    run_eda()         # Step 2: Clean & explore data
    run_analysis()    # Step 3: Full analysis & plots

    print("\n" + "="*60)
    print("  ✅ ALL STEPS COMPLETE!")
    print("="*60)