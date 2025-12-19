import yfinance as yf
import os

# Lista de empresas (Tickers)
tickers = [
    "MSFT", "AAPL", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "TSM", "ASML", "005930.KS",
    "0700.HK", "SONY", "NFLX", "IBM", "ACN", "CRM", "PLTR", "ADBE", "INTC", "CSCO",
    "ORCL", "NOW", "AVGO", "SAP", "INFY", "SPOT", "WDAY", "FTNT", "NET", "SNOW"
]

def download_data():
    if not os.path.exists('data/raw'):
        os.makedirs('data/raw')
    
    for ticker in tickers:
        print(f"Descargando: {ticker}...")
        # Periodo 2018-2024, datos mensuales
        data = yf.download(ticker, start="2018-01-01", end="2024-12-31", interval="1mo")
        if not data.empty:
            data.to_csv(f"data/raw/{ticker}.csv")
        else:
            print(f"Error con {ticker}")

if __name__ == "__main__":
    download_data()

