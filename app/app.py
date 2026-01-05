# app/app.py
from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# ---------- CONFIG / PATHS ----------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = (BASE_DIR / '..' / 'data' / 'processed' / 'tech30_aggregated_stats_2018_2024.csv').resolve()

# ---------- Cargar datos (seguro) ----------
try:
    df = pd.read_csv(DATA_PATH)
    logging.info(f"Datos cargados desde {DATA_PATH} — filas: {len(df)}")
except FileNotFoundError:
    logging.exception(f"No se encontró el fichero de datos en {DATA_PATH}. Asegúrate de la ruta.")
    df = pd.DataFrame()  # dataframe vacío para evitar romper la app

# ---------- Diccionario empresas (igual que tenías) ----------
COMPANIES = {
    "MSFT": "Microsoft", "AAPL": "Apple", "GOOGL": "Alphabet",
    "AMZN": "Amazon", "META": "Meta Platforms", "NVDA": "Nvidia",
    "TSLA": "Tesla", "TSM": "Taiwan Semiconductor", "ASML": "ASML",
    "005930.KS": "Samsung", "0700.HK": "Tencent", "SONY": "Sony",
    "NFLX": "Netflix", "IBM": "IBM", "ACN": "Accenture",
    "CRM": "Salesforce", "PLTR": "Palantir", "ADBE": "Adobe",
    "INTC": "Intel", "CSCO": "Cisco", "ORCL": "Oracle",
    "NOW": "ServiceNow", "AVGO": "Broadcom", "SAP": "SAP",
    "INFY": "Infosys", "SPOT": "Spotify", "WDAY": "Workday",
    "FTNT": "Fortinet", "NET": "Cloudflare", "SNOW": "Snowflake"
}

# ---------- Modelo PyTorch (simple) ----------
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

def train_and_evaluate_model(X_train, y_train, X_val, y_val, epochs=500, lr=1e-2, device='cpu'):
    """
    Entrena un modelo lineal y devuelve el modelo entrenado y RMSE en validación.
    X_* : numpy arrays shape (n, p)
    y_* : numpy arrays shape (n,)
    """
    torch.manual_seed(42)
    input_dim = X_train.shape[1]
    model = LinearRegressionModel(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Convertir a tensores
    Xtr = torch.tensor(X_train, dtype=torch.float32).to(device)
    ytr = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32).to(device)
    Xv = torch.tensor(X_val, dtype=torch.float32).to(device)
    yv = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32).to(device)

    for epoch in range(epochs):
        model.train()
        pred = model(Xtr)
        loss = criterion(pred, ytr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(Xv).cpu().numpy().reshape(-1)
        val_rmse = float(np.sqrt(np.mean((val_pred - y_val) ** 2)))

    return model, val_rmse

def prepare_features_targets(df_ticker):
    """
    Construye matriz de features y vector target desde df filtrado por ticker.
    Espera columnas: Beta, Volatility, AvgVolume, MeanReturn
    Devuelve X (n, p), y (n,)
    """
    # Seleccionar columnas necesarias y limpiar NaNs
    cols = ['Beta', 'Volatility', 'AvgVolume', 'MeanReturn']
    subset = df_ticker[cols].dropna()
    if subset.empty:
        return None, None

    # Features: Beta, Volatility, log(AvgVolume + 1)
    X = np.vstack([
        subset['Beta'].values,
        subset['Volatility'].values,
        np.log1p(subset['AvgVolume'].values)  # estabiliza escala volumes
    ]).T

    y = subset['MeanReturn'].values  # asumimos en fracción (ej. 0.012 -> 1.2%)
    return X, y

def normalize_train_apply(X_train, X_apply):
    """
    Normaliza X_apply con media/desv de X_train. Devuelve X_train_scaled, X_apply_scaled, scaler (mean,std).
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0, ddof=0)
    std[std == 0] = 1.0
    X_train_s = (X_train - mean) / std
    X_apply_s = (X_apply - mean) / std
    return X_train_s, X_apply_s, mean, std

def train_predict_for_ticker(df_all, ticker):
    """
    Dado el dataframe completo y el ticker:
    - Si hay suficiente historial (>=3 rows), entrena y evalúa modelo.
    - Si no, devuelve la predicción histórica (última MeanReturn) como fallback.
    Devuelve: predicted_return (float, fracción), model_val_rmse (float or None), historical_baseline_rmse (float or None)
    """
    df_t = df_all[df_all['Ticker'] == ticker].copy()
    if df_t.empty:
        raise ValueError(f"No hay datos para el ticker {ticker}")

    # Ordenar por fecha si existe columna Date (intentar)
    if 'Date' in df_t.columns:
        try:
            df_t['Date_parsed'] = pd.to_datetime(df_t['Date'], errors='coerce')
            df_t = df_t.sort_values('Date_parsed')
        except Exception:
            pass

    X, y = prepare_features_targets(df_t)
    if X is None or len(X) < 3:
        # No hay historial suficiente para entrenar. Usar último MeanReturn como predicción.
        last_return = float(df_t['MeanReturn'].dropna().iloc[-1])
        return last_return, None, None

    # Split train/val: usar 80/20 temporal (últimas filas -> validation)
    n = len(y)
    split = max(1, int(n * 0.8))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Normalizar
    X_train_s, X_val_s, mean, std = normalize_train_apply(X_train, X_val)

    # Entrenar
    model, model_val_rmse = train_and_evaluate_model(X_train_s, y_train, X_val_s, y_val, epochs=500, lr=1e-2)

    # Baseline histórico: predecir la media de train para val
    hist_pred = np.full_like(y_val, y_train.mean())
    hist_rmse = float(np.sqrt(np.mean((hist_pred - y_val) ** 2)))

    # Predicción "próximo mes": usar la última fila de X (no la val input) para predecir
    # Tomamos la última fila del dataframe (última observación conocida) como features de predicción
    last_row = X[-1].reshape(1, -1)
    # Normalizar con train mean/std
    last_row_s = (last_row - mean) / std
    with torch.no_grad():
        model.eval()
        last_tensor = torch.tensor(last_row_s, dtype=torch.float32)
        predicted_return = float(model(last_tensor).cpu().numpy().reshape(-1)[0])

    return predicted_return, float(model_val_rmse), hist_rmse

# ---------- Probabilidades ----------
def calculate_probabilities(mean_return, volatility):
    """
    Devuelve (prob_success, prob_failure) en fracción 0..1 asumiento distribución normal del retorno.
    mean_return, volatility deben estar en la misma unidad (fracción).
    """
    try:
        from scipy import stats
    except Exception as e:
        # Si scipy no está presente, fallback simple (sitio robusto)
        logging.warning("scipy no disponible, usando fallback simple para probabilidades.")
        if volatility and volatility > 0:
            z = mean_return / volatility
            # aproximación sigmoide al CDF
            prob_success = 1.0 / (1.0 + np.exp(-z))
        else:
            prob_success = 0.5 if mean_return == 0 else (1.0 if mean_return > 0 else 0.0)
        return float(prob_success), float(1.0 - prob_success)

    z_score = mean_return / volatility if (volatility and volatility > 0) else 0.0
    prob_success = float(stats.norm.cdf(z_score))
    prob_failure = float(1.0 - prob_success)
    return prob_success, prob_failure

# ---------- RUTAS ----------
@app.route('/')
def index():
    return render_template('index.html', companies=COMPANIES)

@app.route('/api/company/<ticker>')
def get_company_data(ticker):
    try:
        if df.empty:
            return jsonify({'error': 'Dataset no cargado en el servidor.'}), 500

        df_t = df[df['Ticker'] == ticker]
        if df_t.empty:
            return jsonify({'error': f'No hay datos para el ticker {ticker}'}), 404

        # Predicción con modelo (si hay historial suficiente)
        predicted_return, model_val_rmse, hist_rmse = train_predict_for_ticker(df, ticker)

        # Estadísticas (tomamos la última fila conocida)
        row = df_t.iloc[-1]
        mean_return = float(row.get('MeanReturn', np.nan))
        volatility = float(row.get('Volatility', np.nan))
        beta = float(row.get('Beta', np.nan) if not pd.isna(row.get('Beta', np.nan)) else 0.0)
        avg_volume = float(row.get('AvgVolume', np.nan) if not pd.isna(row.get('AvgVolume', np.nan)) else 0.0)

        # Probabilidades (devueltas como fracción 0..1)
        prob_success, prob_failure = calculate_probabilities(mean_return, volatility)

        # Determinar mejor método si tenemos métricas
        if model_val_rmse is not None and hist_rmse is not None:
            best_method = "Modelo PyTorch" if model_val_rmse < hist_rmse else "Histórico"
        else:
            # fallback conservador
            best_method = "Histórico"

        response = {
            'company': COMPANIES.get(ticker, ticker),
            'ticker': ticker,
            'stats': {
                'meanReturn': mean_return,         # fracción
                'volatility': volatility,         # fracción
                'beta': beta,
                'avgVolume': avg_volume
            },
            'predictions': {
                'historical': mean_return,        # fallback: último meanReturn
                'model': predicted_return,        # fracción
                'bestMethod': best_method,
                'modelLoss': model_val_rmse if model_val_rmse is not None else None
            },
            'probabilities': {
                'success': prob_success,          # fracción 0..1
                'failure': prob_failure
            }
        }

        return jsonify(response)

    except Exception as e:
        logging.exception("Error en /api/company")
        return jsonify({'error': str(e)}), 500

@app.route('/api/calculate_investment', methods=['POST'])
def calculate_investment():
    try:
        payload = request.get_json(force=True)
        investment = float(payload.get('investment', 0))
        predicted_return = float(payload.get('predictedReturn', 0))  # se espera fracción (ej 0.02)

        if investment <= 0:
            return jsonify({'error': 'investment debe ser > 0'}), 400

        # calcular
        profit = investment * predicted_return
        final_amount = investment + profit

        return jsonify({
            'investment': float(investment),
            'predictedReturn': float(predicted_return),            # fracción
            'predictedReturnPercent': float(predicted_return * 100),
            'profit': float(profit),
            'finalAmount': float(final_amount),
            'percentageChange': float(predicted_return * 100)
        })

    except Exception as e:
        logging.exception("Error en /api/calculate_investment")
        return jsonify({'error': str(e)}), 400

@app.route('/api/companies')
def get_all_companies():
    try:
        if df.empty:
            return jsonify([])

        companies_list = []
        for _, row in df.iterrows():
            companies_list.append({
                'ticker': row['Ticker'],
                'name': COMPANIES.get(row['Ticker'], row['Ticker']),
                'meanReturn': float(row.get('MeanReturn', np.nan)),
                'volatility': float(row.get('Volatility', np.nan)),
                'beta': float(row.get('Beta', np.nan))
            })
        return jsonify(companies_list)
    except Exception:
        logging.exception("Error en /api/companies")
        return jsonify([]), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
