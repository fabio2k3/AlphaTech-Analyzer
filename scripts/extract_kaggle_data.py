import os
import pandas as pd
import numpy as np
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

# =========================
# CONFIGURACIÓN
# =========================

DATASET = "borismarjanovic/price-volume-data-for-all-us-stocks-etfs"
RAW_DIR = "../data/raw"
OUTPUT_FILE = "../data/processed/tech_companies_2018_2024.csv"

START_DATE = "2018-01-01"
END_DATE = "2024-12-31"

TECH_COMPANIES = {
    "MSFT": "Microsoft",
    "AAPL": "Apple",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
    "META": "Meta",
    "NVDA": "Nvidia",
    "TSLA": "Tesla",
    "TSM": "TSMC",
    "ASML": "ASML",
    "005930.KS": "Samsung",
    "0700.HK": "Tencent",
    "SONY": "Sony",
    "NFLX": "Netflix",
    "IBM": "IBM",
    "ACN": "Accenture",
    "CRM": "Salesforce",
    "PLTR": "Palantir",
    "ADBE": "Adobe",
    "INTC": "Intel",
    "CSCO": "Cisco",
    "ORCL": "Oracle",
    "NOW": "ServiceNow",
    "AVGO": "Broadcom",
    "SAP": "SAP",
    "INFY": "Infosys",
    "SPOT": "Spotify",
    "WDAY": "Workday",
    "FTNT": "Fortinet",
    "NET": "Cloudflare",
    "SNOW": "Snowflake"
}

# =========================
# DESCARGA DATASET KAGGLE
# =========================

def download_dataset():
    api = KaggleApi()
    api.authenticate()

    os.makedirs(RAW_DIR, exist_ok=True)
    api.dataset_download_files(DATASET, path=RAW_DIR, unzip=True)

# =========================
# CARGA Y LIMPIEZA
# =========================

def load_and_filter_data():
    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv")]

    df_list = []
    for file in files:
        df = pd.read_csv(os.path.join(RAW_DIR, file))
        df_list.append(df)

    df = pd.concat(df_list, ignore_index=True)

    # Normalizar nombres de columnas
    df.columns = [c.lower().strip() for c in df.columns]

    # Ajusta estos nombres según el dataset real
    df = df.rename(columns={
        "date": "Date",
        "adj close": "AdjClose",
        "adjusted close": "AdjClose",
        "volume": "Volume",
        "ticker": "Ticker"
    })

    df["Date"] = pd.to_datetime(df["Date"])

    # Filtrar período
    df = df[(df["Date"] >= START_DATE) & (df["Date"] <= END_DATE)]

    # Filtrar empresas
    df = df[df["Ticker"].isin(TECH_COMPANIES.keys())]

    # Agregar nombre de empresa
    df["Company"] = df["Ticker"].map(TECH_COMPANIES)

    # Seleccionar columnas finales
    df = df[["Company", "Ticker", "Date", "AdjClose", "Volume"]]

    # Ordenar
    df = df.sort_values(["Company", "Date"])

    return df

# =========================
# CÁLCULO DE RETORNOS
# =========================

def compute_returns(df):
    df["Return"] = (
        df.groupby("Company")["AdjClose"]
        .apply(lambda x: np.log(x / x.shift(1)))
    )
    return df

# =========================
# PIPELINE PRINCIPAL
# =========================

def main():
    print("📥 Descargando dataset desde Kaggle...")
    download_dataset()

    print("🧹 Cargando y filtrando datos...")
    df = load_and_filter_data()

    print("📈 Calculando retornos...")
    df = compute_returns(df)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"✅ Dataset final guardado en: {OUTPUT_FILE}")
    print(f"📊 Total observaciones: {len(df)}")

if __name__ == "__main__":
    main()
