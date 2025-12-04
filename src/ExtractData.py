import yfinance as yf
import pandas as pd
import os

# Define paths
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "Data")

# Create Data directory if it doesn't exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# ALL companies from the 3 groups (30 companies total)
all_tickers = {
    # First group (original)
    "Microsoft (MSFT)": "MSFT",
    "Apple (AAPL)": "AAPL",
    "Alphabet/Google (GOOGL)": "GOOGL",
    "Amazon (AMZN)": "AMZN",
    "Meta Platforms (META)": "META",
    "Nvidia (NVDA)": "NVDA",
    "Tesla (TSLA)": "TSLA",
    "Taiwan Semiconductor (TSM)": "TSM",
    "ASML (ASML)": "ASML",
    "Samsung (005930.KS)": "005930.KS",
    
    # Second group
    "Tencent (0700.HK)": "0700.HK",
    "Sony (SONY)": "SONY",
    "Netflix (NFLX)": "NFLX",
    "IBM (IBM)": "IBM",
    "Accenture (ACN)": "ACN",
    "Salesforce (CRM)": "CRM",
    "Palantir (PLTR)": "PLTR",
    "Adobe (ADBE)": "ADBE",
    "Intel (INTC)": "INTC",
    "Cisco (CSCO)": "CSCO",
    
    # Third group
    "Oracle (ORCL)": "ORCL",
    "ServiceNow (NOW)": "NOW",
    "Broadcom (AVGO)": "AVGO",
    "SAP (SAP)": "SAP",
    "Infosys (INFY)": "INFY",
    "Spotify (SPOT)": "SPOT",
    "Workday (WDAY)": "WDAY",
    "Fortinet (FTNT)": "FTNT",
    "Cloudflare (NET)": "NET",
    "Snowflake (SNOW)": "SNOW"
}

# Create a list to store all data
all_data = []
errors = []

print("Downloading stock data for 30 companies...")

for company_name, ticker in all_tickers.items():
    try:
        stock = yf.download(ticker, start="2018-01-01", end="2024-12-31", progress=False)
        monthly_data = stock['Close'].resample('ME').last().dropna()
        
        if not monthly_data.empty:
            print(f"✓ {company_name}")
            
            for date in monthly_data.index:
                price_value = float(monthly_data.loc[date])
                
                all_data.append({
                    "Company Name": company_name,
                    "Date": date.strftime('%Y-%m-%d'),
                    "Close Price": round(price_value, 2)
                })
        else:
            errors.append(f"{company_name}: No data available")
            
    except Exception as e:
        errors.append(f"{company_name}: {str(e)}")

if all_data:
    df = pd.DataFrame(all_data)
    
    # Define file path in Data directory
    filename = "data_of_30_companies.csv"
    file_path = os.path.join(DATA_DIR, filename)
    
    # Save the file to Data directory
    df.to_csv(file_path, index=False)
    
    print(f"\n✓ Download completed successfully!")
    print(f"  File saved: {file_path}")
    print(f"  Companies downloaded: {len(all_tickers) - len(errors)}/{len(all_tickers)}")
    
    if errors:
        print(f"\n✗ Errors ({len(errors)}):")
        for error in errors:
            print(f"  {error}")
else:
    print("✗ No data could be downloaded for any company")