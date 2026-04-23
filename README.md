# Vendor Performance Analysis using Python and Power BI

## 📌 Overview
An end-to-end vendor performance analytics pipeline built with Python and Power BI,ingests raw sales data into a SQLite database, performs exploratory data analysis,uncovers pricing and procurement insights through statistical testing, and delivers,an interactive dashboard — all triggered by a single command.

## 📂 Project Structure
```
├── data/
│   └── vendor_sales_summary.csv       # Source data
├── logs/
│   └── ingestion_db.log               # Auto-generated pipeline logs
├── main.py                            # Run this to execute full pipeline
├── ingestion_db.py                    # Ingests CSV data into SQLite database
├── EDA_inventory_analysis.py          # Exploratory Data Analysis script
├── vendor_performance_analysis.py     # Vendor performance analysis & charts
├── get_vendor_summary.py              # Generates vendor summary report
├── inventory.db                       # Auto-generated SQLite database
├── vendor_performance.pbix            # Power BI dashboard
├── Vendor Performance Report.pdf      # Final insights report
└── README.md
```

## 🛠️ Tech Stack
- **Languages**: Python (Pandas, Matplotlib, Seaborn)
- **Tools**: Jupyter Notebook, Power BI, Excel
- **Other**: PDF Reporting

## 💡 Key Features
- Automated ingestion and cleaning of vendor sales data
- Detailed Exploratory Data Analysis (EDA) to identify trends and patterns
- Python scripts for automated vendor summary reports
- Interactive and visually rich dashboard built using Power BI
- Finalized insights compiled in a professional PDF report

## 🚀 How to Run
1. Clone the repository
2. Set up the data folder
3. Run the full pipeline(python main.py)
   This will:
   - ✅ Ingest CSV data into the SQLite database
   - ✅ Clean and explore the data (EDA)
   - ✅ Run all vendor performance analyses and generate charts
4. View the Power BI Dashboard
   - Open "vendor_performance.pbix" in Power BI Desktop
5. Review the final report
   - Open "Vendor Performance Report.pdf" for key insights and recommendations

## 📈 Outputs
- Vendor performance summary CSV
- Power BI Dashboard
- Final insights PDF report