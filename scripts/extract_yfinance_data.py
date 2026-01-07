# scripts/extract_yfinance_data.py
import os
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# -------------------------
# CONFIG
# -------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")         # opcional, no usado por yfinance
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

START_DATE = "2018-01-01"
END_DATE = "2024-12-31"

# Market ETF / índice para beta (puedes cambiar a "^IXIC" o "SPY" si prefieres)
MARKET_TICKER = "QQQ"

OUTPUT_PANEL = os.path.join(PROCESSED_DIR, "tech30_panel_monthly_2018_2024.csv")
OUTPUT_AGG   = os.path.join(PROCESSED_DIR, "tech30_aggregated_stats_2018_2024.csv")

# Lista de empresas (ticker -> company display name)
TECH_COMPANIES = {
    "MSFT": "Microsoft",
    "AAPL": "Apple",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
    "META": "Meta Platforms",
    "NVDA": "Nvidia",
    "TSLA": "Tesla",
    "TSM": "Taiwan Semiconductor",
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

# -------------------------
# HELPERS
# -------------------------
def ensure_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

def download_monthly_for_ticker(ticker):
    """
    Descarga datos diarios y devuelve DataFrame mensual con columnas:
    ['AdjClose','Volume'] index = MonthEnd dates
    """
    try:
        # auto_adjust=False para mantener 'Adj Close' en la respuesta
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, auto_adjust=False)
        if df.empty:
            print(f"  ⚠️ {ticker}: DataFrame vacío")
            return None

        # Asegurarse de que existen las columnas necesarias
        # yf.download devuelve un DataFrame con columnas MultiIndex
        # Necesitamos aplanar las columnas
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1) if df.columns.nlevels > 1 else df.columns
        
        # Verificar columnas disponibles
        available_cols = df.columns.tolist()
        print(f"  {ticker}: Columnas disponibles: {available_cols}")

        # Buscar la columna de precio ajustado
        adj_close_col = None
        for col in ['Adj Close', 'AdjClose', 'Adj_Close', 'Adjclose']:
            if col in df.columns:
                adj_close_col = col
                break
        
        if not adj_close_col:
            print(f"  ⚠️ {ticker}: No existe columna de precio ajustado, usando 'Close'")
            if 'Close' in df.columns:
                df["Adj Close"] = df["Close"]
                adj_close_col = "Adj Close"
            else:
                print(f"  ❌ {ticker}: No hay columna 'Close' disponible")
                return None

        # Resample mensual: precio = último valor del mes, volume = suma del mes
        # Usar 'ME' en lugar de 'M' (deprecated) y asegurarse de que son Series
        monthly_price = df[adj_close_col].resample("ME").last()
        monthly_volume = df["Volume"].resample("ME").sum()

        # Si son DataFrames, convertirlos a Series
        if isinstance(monthly_price, pd.DataFrame):
            monthly_price = monthly_price.iloc[:, 0]
        if isinstance(monthly_volume, pd.DataFrame):
            monthly_volume = monthly_volume.iloc[:, 0]

        # Crear DataFrame
        monthly = pd.DataFrame({
            "AdjClose": monthly_price.values if hasattr(monthly_price, 'values') else monthly_price,
            "Volume": monthly_volume.values if hasattr(monthly_volume, 'values') else monthly_volume
        }, index=monthly_price.index)
        
        # eliminar meses sin datos (NaN)
        monthly = monthly.dropna(subset=["AdjClose"])
        
        if monthly.empty:
            print(f"  ⚠️ {ticker}: Datos mensuales vacíos después de dropna")
            return None
            
        return monthly
        
    except Exception as e:
        print(f"  ❌ Error procesando {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# -------------------------
# PIPELINE
# -------------------------
def main():
    ensure_dirs()
    errors = []
    panel_rows = []

    print("Descargando serie del índice de mercado para calcular Beta:", MARKET_TICKER)
    market_monthly = download_monthly_for_ticker(MARKET_TICKER)
    if market_monthly is None or market_monthly.empty:
        print(f"ERROR: No se pudo descargar datos del índice {MARKET_TICKER}. Beta no estará disponible.")
        market_returns = None
    else:
        market_returns = np.log(market_monthly["AdjClose"] / market_monthly["AdjClose"].shift(1))
        market_returns.name = "MarketReturn"

    print("Iniciando descarga de cada empresa (esto puede tardar unos minutos)...")
    for ticker, company in TECH_COMPANIES.items():
        try:
            monthly = download_monthly_for_ticker(ticker)
            if monthly is None or monthly.empty:
                errors.append(f"{company} ({ticker}): sin datos")
                print(f"✗ {company} ({ticker}) - sin datos")
                continue

            # calcular retornos log
            monthly = monthly.copy()
            monthly["Return"] = np.log(monthly["AdjClose"] / monthly["AdjClose"].shift(1))

            # añadir filas al panel
            for date, row in monthly.dropna(subset=["Return"]).iterrows():
                panel_rows.append({
                    "Company": company,
                    "Ticker": ticker,
                    "Date": date.strftime("%Y-%m-%d"),
                    "AdjClose": float(row["AdjClose"]),
                    "Volume": float(row["Volume"]),
                    "Return": float(row["Return"])
                })

            print(f"✓ {company} ({ticker}) - {len(monthly)} meses (retornos calculados: {monthly['Return'].dropna().shape[0]})")

        except Exception as e:
            errors.append(f"{company} ({ticker}): {str(e)}")
            print(f"✗ {company} ({ticker}) - ERROR: {e}")

    # construir panel DataFrame
    panel_df = pd.DataFrame(panel_rows)
    if panel_df.empty:
        print("No se descargó ningún dato válido. Revisa conectividad o tickers.")
        return

    # guardar panel
    panel_df.to_csv(OUTPUT_PANEL, index=False)
    print(f"\n✅ Panel mensual guardado en: {OUTPUT_PANEL}")
    print(f"Filas en panel: {len(panel_df)}")

    # -------------------------
    # Calcular estadísticas agregadas por empresa (para clustering & regresión)
    # -------------------------
    agg_list = []
    grouped = panel_df.groupby("Company")
    for company, g in grouped:
        # asegurarse orden temporal
        g = g.sort_values("Date")
        returns = g["Return"].astype(float)

        mean_return = returns.mean()
        volatility = returns.std(ddof=1)  # desviación estándar muestral
        avg_volume = g["Volume"].mean()

        # calcular beta si hay market_returns
        beta = np.nan
        if market_returns is not None:
            # alinear series por fecha
            # market_returns index are month-ends; convertir a strings YYYY-MM-DD
            mret = market_returns.copy()
            mret.index = mret.index.strftime("%Y-%m-%d")
            # alinear con los returns del panel (g["Date"])
            merged = pd.DataFrame({
                "r_i": returns.values,
                "Date": g["Date"].values
            })
            merged.set_index("Date", inplace=True)
            # seleccionar solo fechas que existan en market returns
            common = merged.join(mret.rename("r_m"), how="inner")
            if not common["r_i"].empty and not common["r_m"].empty:
                cov = np.cov(common["r_i"], common["r_m"], ddof=1)[0,1]
                var_m = np.var(common["r_m"], ddof=1)
                if var_m != 0:
                    beta = cov / var_m

        agg_list.append({
            "Company": company,
            "Ticker": g["Ticker"].iloc[0],
            "MeanReturn": mean_return,
            "Volatility": volatility,
            "Beta": beta,
            "AvgVolume": avg_volume,
            "N_months": len(g)
        })

    agg_df = pd.DataFrame(agg_list)
    agg_df.to_csv(OUTPUT_AGG, index=False)
    print(f"✅ Estadísticas agregadas guardadas en: {OUTPUT_AGG}")
    print(agg_df[["Company","Ticker","MeanReturn","Volatility","Beta","AvgVolume","N_months"]].to_string(index=False))

    if errors:
        print("\n⚠️ Errores durante la descarga:")
        for e in errors:
            print("  -", e)

if __name__ == "__main__":
    start_time = datetime.now()
    print("INICIANDO pipeline yfinance:", start_time.strftime("%Y-%m-%d %H:%M:%S"))
    main()
    end_time = datetime.now()
    print("FIN (duración):", (end_time - start_time))
