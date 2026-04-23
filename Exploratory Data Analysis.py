# =============================================================================
# Exploratory Data Analysis - Inventory Database
# =============================================================================
# Goal: Understand the dataset and create aggregated tables to support:
#   - Vendor selection for profitability
#   - Product Pricing Optimization
# =============================================================================

import sqlite3
import pandas as pd


# =============================================================================
# 1. DATABASE CONNECTION
# =============================================================================

conn = sqlite3.connect('inventory.db')
cursor = conn.cursor()


# =============================================================================
# 2. EXPLORE TABLES IN THE DATABASE
# =============================================================================

# Check all tables present in the database
tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
print(tables)

# Preview each table: record count and first 5 rows
for table in tables['name']:
    print('-' * 50, f'{table}', '-' * 50)
    count = pd.read_sql_query(f"SELECT COUNT(*) as count FROM {table}", conn)['count'].values[0]
    print('Count of records:', count)
    print(pd.read_sql_query(f"SELECT * FROM {table} LIMIT 5", conn))

# NOTE:
# - begin_inventory and end_inventory tables contain yearly snapshot data,
#   not useful for vendor behavior analysis → excluded.
# - We focus on: purchases, purchase_prices, vendor_invoice, sales


# =============================================================================
# 3. SINGLE VENDOR EXPLORATION (VendorNumber = 4466)
# =============================================================================

purchases = pd.read_sql_query("SELECT * FROM purchases WHERE VendorNumber = 4466", conn)
print(purchases)

purchase_prices = pd.read_sql_query("SELECT * FROM purchase_prices WHERE VendorNumber = 4466", conn)
print(purchase_prices)

vendor_invoice = pd.read_sql_query("SELECT * FROM vendor_invoice WHERE VendorNumber = 4466", conn)
print(vendor_invoice)

sales = pd.read_sql_query("SELECT * FROM sales WHERE VendorNo = 4466", conn)
print(sales)

# Quick aggregations for the sample vendor
print(purchases.groupby(["Brand", "PurchasePrice"])[["Quantity", "Dollars"]].sum())
print(purchases.groupby(["PONumber"])[["Quantity", "Dollars"]].sum())
print(sales.groupby("Brand")[['SalesQuantity', 'SalesDollars', 'SalesPrice', 'ExciseTax']].sum())

# TABLE INSIGHTS:
# - purchases       : actual purchase transactions (date, brand, dollars, quantity)
# - purchase_prices : product-wise actual vs purchase price (unique per vendor+brand)
# - vendor_invoice  : aggregated purchases per PO, includes freight costs
# - sales           : actual sales transactions (brand, quantity sold, revenue)


# =============================================================================
# 4. BUILD SUMMARY TABLES
# =============================================================================

# --- 4a. Freight Summary ---
freight_summary = pd.read_sql_query("""
    SELECT 
        VendorNumber, 
        SUM(Freight) AS FreightCost 
    FROM vendor_invoice 
    GROUP BY VendorNumber
""", conn)
print(freight_summary)

# --- 4b. Purchase Summary ---
purchase_summary = pd.read_sql_query("""
    SELECT 
        p.VendorNumber,
        p.VendorName,
        p.Brand,
        p.Description,
        p.PurchasePrice,
        pp.Price AS ActualPrice,
        pp.Volume,
        SUM(p.Quantity)  AS TotalPurchaseQuantity,
        SUM(p.Dollars)   AS TotalPurchaseDollars
    FROM purchases p
    JOIN purchase_prices pp ON p.Brand = pp.Brand
    WHERE p.PurchasePrice > 0
    GROUP BY p.VendorNumber, p.VendorName, p.Brand, p.Description, 
             p.PurchasePrice, pp.Price, pp.Volume
    ORDER BY TotalPurchaseDollars
""", conn)
print(purchase_summary)

# --- 4c. Sales Summary ---
sales_summary = pd.read_sql_query("""
    SELECT 
        VendorNo,
        Brand,
        SUM(SalesQuantity) AS TotalSalesQuantity,
        SUM(SalesDollars)  AS TotalSalesDollars,
        SUM(SalesPrice)    AS TotalSalesPrice,
        SUM(ExciseTax)     AS TotalExciseTax
    FROM sales
    GROUP BY VendorNo, Brand
""", conn)
print(sales_summary)

# --- 4d. Merged Vendor Sales Summary (Master Aggregated Table) ---
vendor_sales_summary = pd.read_sql_query("""
    WITH FreightSummary AS (
        SELECT 
            VendorNumber, 
            SUM(Freight) AS FreightCost 
        FROM vendor_invoice 
        GROUP BY VendorNumber
    ), 
    PurchaseSummary AS (
        SELECT 
            p.VendorNumber,
            p.VendorName,
            p.Brand,
            p.Description,
            p.PurchasePrice,
            pp.Price AS ActualPrice,
            pp.Volume,
            SUM(p.Quantity) AS TotalPurchaseQuantity,
            SUM(p.Dollars)  AS TotalPurchaseDollars
        FROM purchases p
        JOIN purchase_prices pp ON p.Brand = pp.Brand
        WHERE p.PurchasePrice > 0
        GROUP BY p.VendorNumber, p.VendorName, p.Brand, p.Description, 
                 p.PurchasePrice, pp.Price, pp.Volume
    ), 
    SalesSummary AS (
        SELECT 
            VendorNo,
            Brand,
            SUM(SalesQuantity) AS TotalSalesQuantity,
            SUM(SalesDollars)  AS TotalSalesDollars,
            SUM(SalesPrice)    AS TotalSalesPrice,
            SUM(ExciseTax)     AS TotalExciseTax
        FROM sales
        GROUP BY VendorNo, Brand
    ) 
    SELECT 
        ps.VendorNumber,
        ps.VendorName,
        ps.Brand,
        ps.Description,
        ps.PurchasePrice,
        ps.ActualPrice,
        ps.Volume,
        ps.TotalPurchaseQuantity,
        ps.TotalPurchaseDollars,
        ss.TotalSalesQuantity,
        ss.TotalSalesDollars,
        ss.TotalSalesPrice,
        ss.TotalExciseTax,
        fs.FreightCost
    FROM PurchaseSummary ps
    LEFT JOIN SalesSummary ss 
        ON ps.VendorNumber = ss.VendorNo AND ps.Brand = ss.Brand
    LEFT JOIN FreightSummary fs 
        ON ps.VendorNumber = fs.VendorNumber
    ORDER BY ps.TotalPurchaseDollars DESC
""", conn)

print(vendor_sales_summary.columns)


# =============================================================================
# 5. DATA CLEANING
# =============================================================================

print(vendor_sales_summary.dtypes)
print(vendor_sales_summary.isnull().sum())
print(vendor_sales_summary['Volume'].unique())
print(vendor_sales_summary['VendorName'].unique())

# Issues identified:
# - Volume is numeric but stored as object dtype
# - Some products have no sales → NaN values
# - Whitespace in categorical columns

# --- 5a. Add Derived Metric Columns ---
vendor_sales_summary['GrossProfit'] = (
    vendor_sales_summary['TotalSalesDollars'] - vendor_sales_summary['TotalPurchaseDollars']
)
vendor_sales_summary['ProfitMargin'] = (
    vendor_sales_summary['GrossProfit'] / vendor_sales_summary['TotalSalesDollars']
) * 100
vendor_sales_summary['StockTurnover'] = (
    vendor_sales_summary['TotalSalesQuantity'] / vendor_sales_summary['TotalPurchaseQuantity']
)
vendor_sales_summary['SalesToPurchaseRatio'] = (
    vendor_sales_summary['TotalSalesDollars'] / vendor_sales_summary['TotalPurchaseDollars']
)

# --- 5b. Fix Data Types & Missing Values ---
vendor_sales_summary['Volume'] = vendor_sales_summary['Volume'].astype('float')
vendor_sales_summary.fillna(0, inplace=True)

# --- 5c. Strip Whitespace from Categorical Columns ---
vendor_sales_summary['VendorName'] = vendor_sales_summary['VendorName'].str.strip()
vendor_sales_summary['Description'] = vendor_sales_summary['Description'].str.strip()


# =============================================================================
# 6. SAVE CLEANED DATA TO DATABASE
# =============================================================================

# Create the table schema (run only once — skipped if table already exists)
try:
    cursor.execute("""
        CREATE TABLE vendor_sales_summary (
            VendorNumber          INT,
            VendorName            VARCHAR(100),
            Brand                 INT,
            Description           VARCHAR(100),
            PurchasePrice         DECIMAL(10,2),
            ActualPrice           DECIMAL(10,2),
            Volume                INT,
            TotalPurchaseQuantity INT,
            TotalPurchaseDollars  DECIMAL(15,2),
            TotalSalesQuantity    INT,
            TotalSalesDollars     DECIMAL(15,2),
            TotalSalesPrice       DECIMAL(15,2),
            TotalExciseTax        DECIMAL(15,2),
            FreightCost           DECIMAL(15,2),
            GrossProfit           DECIMAL(15,2),
            ProfitMargin          DECIMAL(15,2),
            StockTurnover         DECIMAL(15,2),
            SalesToPurchaseRatio  DECIMAL(15,2),
            PRIMARY KEY (VendorNumber, Brand)
        );
    """)
    print("Table created successfully.")
except Exception as e:
    print(f"Table creation skipped: {e}")

# Write cleaned DataFrame to the database (replace if exists)
vendor_sales_summary.to_sql('vendor_sales_summary', conn, if_exists='replace', index=False)

# Verify the saved data
result = pd.read_sql_query("SELECT * FROM vendor_sales_summary", conn)
print(result)


# =============================================================================
# 7. CLOSE CONNECTION
# =============================================================================

conn.close()
print("Database connection closed.")