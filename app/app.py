# app.py — versión corregida para usar panel mensual para predicción
from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path
import logging
from functools import lru_cache
import json
import time

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# ---------- CONFIG / PATHS ----------
BASE_DIR = Path(__file__).resolve().parent
PANEL_PATH = (BASE_DIR / '..' / 'data' / 'processed' / 'tech30_panel_monthly_2018_2024.csv').resolve()
AGG_PATH = (BASE_DIR / '..' / 'data' / 'processed' / 'tech30_aggregated_stats_2018_2024.csv').resolve()

# Cache para datos (separado para panel y agregado)
_data_cache = {
    'panel': {'df': None, 'ts': None},
    'agg': {'df': None, 'ts': None}
}
CACHE_DURATION = 300  # 5 minutos

# Cache de predicciones por ticker (in-memory)
_prediction_cache = {}  # ticker -> {'hash': str, 'response': dict, 'ts': float}

# ---------- Diccionario empresas ----------
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

# ---------- Modelo PyTorch ----------
class AdvancedPredictionModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.network(x)

# ---------- Utilities: carga de datos con cache ----------
def load_panel_cached():
    now = datetime.now()
    cache = _data_cache['panel']
    if cache['df'] is not None and cache['ts'] and (now - cache['ts']).seconds < CACHE_DURATION:
        return cache['df']
    try:
        df = pd.read_csv(PANEL_PATH, parse_dates=['Date'])
        # Asegurar columnas mínimas
        if 'Return' not in df.columns:
            logging.warning("El panel no tiene columna 'Return' — revisa tu CSV")
        cache['df'] = df
        cache['ts'] = now
        logging.info(f"Panel cargado desde {PANEL_PATH} — filas: {len(df)}")
        return df
    except FileNotFoundError:
        logging.exception(f"No se encontró el fichero panel en {PANEL_PATH}")
        return pd.DataFrame()

def load_agg_cached():
    now = datetime.now()
    cache = _data_cache['agg']
    if cache['df'] is not None and cache['ts'] and (now - cache['ts']).seconds < CACHE_DURATION:
        return cache['df']
    try:
        df = pd.read_csv(AGG_PATH)
        cache['df'] = df
        cache['ts'] = now
        logging.info(f"Agrupado cargado desde {AGG_PATH} — filas: {len(df)}")
        return df
    except FileNotFoundError:
        logging.exception(f"No se encontró el fichero agregado en {AGG_PATH}")
        return pd.DataFrame()

# ---------- Factores macro / empresa ----------
def get_market_sentiment_factors():
    current_month = datetime.now().month
    factors = {
        'market_sentiment': np.random.normal(0.05, 0.15),
        'tech_sector_momentum': np.random.normal(0.08, 0.12),
        'interest_rate_impact': -0.03 if current_month in [1,2,3,10,11,12] else 0.02,
        'geopolitical_risk': np.random.normal(-0.02, 0.08),
        'inflation_pressure': np.random.normal(-0.01, 0.05),
        'ai_hype_cycle': 0.15 if current_month in [3,4,5,9,10,11] else 0.08,
        'economic_growth': np.random.normal(0.03, 0.06)
    }
    return factors

def get_company_specific_factors(ticker):
    ai_companies = ['NVDA', 'MSFT', 'GOOGL', 'META']
    cloud_companies = ['AMZN', 'MSFT', 'GOOGL', 'ORCL']
    hardware_companies = ['AAPL', 'TSM', 'ASML', '005930.KS']
    factors = {
        'ai_exposure': 0.2 if ticker in ai_companies else 0.05,
        'cloud_growth': 0.15 if ticker in cloud_companies else 0.03,
        'supply_chain_risk': -0.1 if ticker in hardware_companies else -0.02,
        'innovation_index': np.random.uniform(0.05, 0.18),
        'competitive_position': np.random.uniform(-0.05, 0.15)
    }
    return factors

# ---------- Preparación de features desde PANEL (ventanas) ----------
def prepare_advanced_features_panel(df_ticker, ticker, window=6):
    """
    Construye X,y desde el panel mensual usando ventanas de 'window' meses.
    X: concatenación de returns_lag (window), log(volume_lag window), sin/cos estacionalidad, macro/company factors
    y: return at time t (next month)
    """
    # Asegurar orden temporal
    df = df_ticker.sort_values('Date').reset_index(drop=True).copy()
    if df.empty or 'Return' not in df.columns:
        return None, None

    # Usar columnas 'Return' y 'Volume'
    if 'Volume' not in df.columns:
        logging.warning(f"Ticker {ticker} - panel no contiene columna 'Volume'")
        df['Volume'] = 0.0

    returns = df['Return'].astype(float).values
    volumes = df['Volume'].astype(float).values
    dates = pd.to_datetime(df['Date']).dt.to_pydatetime()

    N = len(returns)
    if N <= window:
        return None, None

    X_list = []
    y_list = []
    for i in range(window, N):
        ret_window = returns[i-window:i]      # shape (window,)
        vol_window = volumes[i-window:i]      # shape (window,)
        # Estacionalidad basada en la fecha objetivo (i)
        month = dates[i].month
        sin_month = np.sin(2 * np.pi * (month-1) / 12)
        cos_month = np.cos(2 * np.pi * (month-1) / 12)

        # macro y company factors (simulados) — mismos para toda la fila
        market_factors = get_market_sentiment_factors()
        company_factors = get_company_specific_factors(ticker)
        macro = np.array([
            market_factors['market_sentiment'],
            market_factors['tech_sector_momentum'],
            market_factors['interest_rate_impact'],
            market_factors['geopolitical_risk'],
            market_factors['inflation_pressure'],
            market_factors['ai_hype_cycle'],
            market_factors['economic_growth'],
            company_factors['ai_exposure'],
            company_factors['cloud_growth'],
            company_factors['supply_chain_risk'],
            company_factors['innovation_index'],
            company_factors['competitive_position']
        ])  # length 12

        # Feature vector: returns lags, log volumes lags, sin, cos, macro(12)
        fv = np.concatenate([
            ret_window,
            np.log1p(vol_window),
            np.array([sin_month, cos_month]),
            macro
        ])
        X_list.append(fv)
        y_list.append(returns[i])  # predict return at i

    X = np.vstack(X_list)
    y = np.array(y_list)
    return X, y

# ---------- Entrenamiento ----------
def train_advanced_model(X_train, y_train, X_val, y_val, epochs=200, lr=5e-3, device='cpu'):
    torch.manual_seed(42)
    input_dim = X_train.shape[1]
    model = AdvancedPredictionModel(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15, factor=0.5)

    Xtr = torch.tensor(X_train, dtype=torch.float32).to(device)
    ytr = torch.tensor(y_train.reshape(-1,1), dtype=torch.float32).to(device)
    Xv = torch.tensor(X_val, dtype=torch.float32).to(device)
    yv = torch.tensor(y_val.reshape(-1,1), dtype=torch.float32).to(device)

    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 20

    for epoch in range(epochs):
        model.train()
        pred = model(Xtr)
        loss = criterion(pred, ytr)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(Xv)
            val_loss = criterion(val_pred, yv)
            scheduler.step(val_loss)

            if val_loss < best_val_loss - 1e-9:
                best_val_loss = float(val_loss)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    break

    model.eval()
    with torch.no_grad():
        val_pred = model(Xv).cpu().numpy().reshape(-1)
        val_rmse = float(np.sqrt(np.mean((val_pred - y_val) ** 2)))
    return model, val_rmse

# ---------- Normalización ----------
def normalize_features(X_train, X_apply):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0, ddof=0)
    std = np.where(std < 1e-8, 1.0, std)
    X_train_s = (X_train - mean) / std
    X_apply_s = (X_apply - mean) / std
    return X_train_s, X_apply_s, mean, std

# ---------- Escenarios ----------
def generate_scenarios(model, base_features, base_pred, device='cpu'):
    """
    base_features: 2D array shape (1, D) normalized (already scaled)
    Ajusta perturbaciones sobre los últimos 12 valores (macro part). 
    """
    # asegurarse numpy array
    bf = np.array(base_features, dtype=float).copy()
    # asumir últimos 12 entries son macro (como construimos)
    # por seguridad, buscar slice final de longitud 12
    D = bf.shape[1]
    macro_len = 12
    macro_start = D - macro_len
    if macro_start < 0:
        macro_start = max(0, D - macro_len)

    opt = bf.copy()
    pes = bf.copy()

    # pequeñas perturbaciones (optimista / pesimista)
    opt[0, macro_start:macro_start+macro_len] += np.array([0.10,0.08,0.05,-0.05,-0.03,0.10,0.08,0.05,0.05,-0.02,0.05,0.05])
    pes[0, macro_start:macro_start+macro_len] += np.array([-0.08,-0.06,-0.08,0.08,0.05,-0.05,-0.05,-0.03,-0.03,0.05,-0.03,-0.05])

    with torch.no_grad():
        model.eval()
        to_tensor = lambda arr: torch.tensor(arr, dtype=torch.float32).to(device)
        opt_pred = float(model(to_tensor(opt)).cpu().numpy().reshape(-1)[0])
        pes_pred = float(model(to_tensor(pes)).cpu().numpy().reshape(-1)[0])

    scenario_range = opt_pred - pes_pred
    if scenario_range > 0:
        prob_opt = 0.25; prob_neutral = 0.5; prob_pes = 0.25
    else:
        prob_opt = prob_neutral = prob_pes = 1/3

    scenarios = {
        'optimistic': {'return': opt_pred, 'probability': prob_opt, 'description': 'Mercado alcista, factores favorables'},
        'neutral': {'return': base_pred, 'probability': prob_neutral, 'description': 'Condiciones actuales sostenidas'},
        'pessimistic': {'return': pes_pred, 'probability': prob_pes, 'description': 'Mercado bajista, factores adversos'}
    }
    return scenarios

# ---------- Probabilidades ----------
def calculate_probabilities_advanced(mean_return, volatility, scenarios):
    """
    Calcula P(success) y P(failure).
    - Si 'scenarios' no es None: construye media y varianza ponderadas por prob,
      combina con la volatilidad histórica y calcula P(return > 0) usando normal CDF.
    - Si no hay 'scenarios': usa aproximación normal con mean_return y volatility.
    Returns: (prob_success, prob_failure)
    """
    from scipy import stats

    eps = 1e-8  # evita división por cero
    # Caso con escenarios proporcionados
    if scenarios:
        # construir arrays de retornos y probabilidades
        returns = []
        probs = []
        for s in scenarios.values():
            try:
                returns.append(float(s.get('return', 0.0)))
            except Exception:
                returns.append(0.0)
            try:
                probs.append(float(s.get('probability', 0.0)))
            except Exception:
                probs.append(0.0)

        returns = np.array(returns, dtype=float)
        probs = np.array(probs, dtype=float)

        # normalizar probabilidades si no suman 1
        if probs.sum() <= 0 or np.isnan(probs.sum()):
            probs = np.ones_like(probs) / len(probs)
        else:
            probs = probs / probs.sum()

        # media y varianza ponderada (scenarios)
        mean_s = float(np.sum(probs * returns))
        var_s = float(np.sum(probs * (returns - mean_s) ** 2))

        # usar volatility histórica si está disponible
        vol_hist = float(volatility) if (volatility is not None and not np.isnan(volatility)) else 0.0
        if vol_hist < 0:
            vol_hist = abs(vol_hist)

        # combinar incertidumbres: var_total = var_s + vol_hist^2
        total_var = var_s + (vol_hist ** 2)
        if total_var < eps:
            total_var = eps

        total_std = np.sqrt(total_var)

        # probabilidad de que retorno > 0
        prob_success = float(stats.norm.cdf(mean_s / total_std))

        # evitar exactamente 0.0 o 1.0 (por motivos numéricos y presentación)
        prob_success = max(min(prob_success, 1.0 - 1e-6), 1e-6)
        prob_failure = 1.0 - prob_success

        return prob_success, prob_failure

    # Caso sin escenarios: usar aproximación normal con mean_return y volatility
    else:
        from scipy import stats as _stats
        if volatility is not None and volatility > 0:
            z_score = mean_return / volatility
            prob_success = float(_stats.norm.cdf(z_score))
        else:
            # sin info de volatilidad, retornamos 50/50 (uncertain)
            prob_success = 0.5

        prob_success = max(min(prob_success, 1.0 - 1e-6), 1e-6)
        return float(prob_success), float(1.0 - prob_success)


# ---------- Cache simple de predicción ----------
def make_data_hash(df_t):
    """Hash simple basado en cantidad de filas y última fecha (rápido)"""
    if df_t.empty:
        return "empty"
    last_date = str(df_t['Date'].max())
    return f"n{len(df_t)}_last{last_date}"

def get_cached_prediction(ticker, data_hash):
    rec = _prediction_cache.get(ticker)
    if not rec:
        return None
    if rec.get('hash') == data_hash:
        # mantener cache fresco por 5 minutos
        if time.time() - rec.get('ts',0) < CACHE_DURATION:
            return rec.get('response')
    return None

def set_cached_prediction(ticker, data_hash, response):
    _prediction_cache[ticker] = {'hash': data_hash, 'response': response, 'ts': time.time()}

# ---------- Entrenar y predecir (usando PANEL) ----------
def train_predict_for_ticker_panel(df_panel, ticker):
    df_t = df_panel[df_panel['Ticker'] == ticker].copy()
    if df_t.empty:
        raise ValueError(f"No hay datos de panel para el ticker {ticker}")

    # quick info
    logging.info(f"train_predict_for_ticker_panel: ticker={ticker}, rows={len(df_t)}")

    # preparar X,y
    X, y = prepare_advanced_features_panel(df_t, ticker, window=6)
    if X is None or len(X) < 8:
        # insuficiente historial
        logging.info(f"Insufficient panel history for {ticker} (rows={len(df_t)}), skipping model training.")
        return None, None, None, None

    # temporal split: usar 80% train, 20% val
    n = len(y)
    split = max(2, int(n * 0.8))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # normalizar
    X_train_s, X_val_s, mean, std = normalize_features(X_train, X_val)

    # entrenar
    model, model_val_rmse = train_advanced_model(X_train_s, y_train, X_val_s, y_val, epochs=200, lr=5e-3, device='cpu')

    # baseline histórico (usar promedio de y_train)
    hist_pred = np.full_like(y_val, np.mean(y_train))
    hist_rmse = float(np.sqrt(np.mean((hist_pred - y_val) ** 2)))

    # Predicción para el último punto (usar último window)
    last_row = X[-1].reshape(1, -1)
    last_row_s = (last_row - mean) / std

    with torch.no_grad():
        model.eval()
        last_tensor = torch.tensor(last_row_s, dtype=torch.float32)
        base_prediction = float(model(last_tensor).cpu().numpy().reshape(-1)[0])

    # Escenarios basados en base_features normalizado
    scenarios = generate_scenarios(model, last_row_s, base_prediction, device='cpu')

    return base_prediction, float(model_val_rmse), hist_rmse, scenarios

# ---------- RUTAS ----------
@app.route('/')
def index():
    return render_template('index.html', companies=COMPANIES)

@app.route('/api/company/<ticker>')
def get_company_data(ticker):
    try:
        # cargar datasets
        df_panel = load_panel_cached()
        df_agg = load_agg_cached()

        data_hash = make_data_hash(df_panel[df_panel['Ticker'] == ticker]) if not df_panel.empty else "empty"
        cached = get_cached_prediction(ticker, data_hash)
        if cached:
            logging.info(f"Returning cached prediction for {ticker}")
            return jsonify(cached)

        # stats: preferir agregado si existe, sino usar último registro del panel
        df_t_agg = df_agg[df_agg['Ticker'] == ticker] if not df_agg.empty else pd.DataFrame()
        df_t_panel = df_panel[df_panel['Ticker'] == ticker] if not df_panel.empty else pd.DataFrame()

        if not df_t_agg.empty:
            row = df_t_agg.iloc[-1]
            mean_return = float(row.get('MeanReturn', 0.0))
            volatility = float(row.get('Volatility', 0.0))
            beta = float(row.get('Beta', 1.0))
            avg_volume = float(row.get('AvgVolume', 0.0))
        elif not df_t_panel.empty:
            last_row = df_t_panel.sort_values('Date').iloc[-1]
            mean_return = float(last_row.get('Return', 0.0))
            volatility = float(df_t_panel['Return'].std(ddof=0) if len(df_t_panel) > 1 else 0.0)
            beta = 1.0
            avg_volume = float(df_t_panel['Volume'].mean() if 'Volume' in df_t_panel.columns else 0.0)
        else:
            return jsonify({'error': f'No hay datos para {ticker}'}), 404

        # Intentar entrenar y predecir con PANEL
        predicted_return = None
        model_val_rmse = None
        hist_rmse = None
        scenarios = None
        model_available = False

        if not df_t_panel.empty:
            try:
                pr, m_rmse, h_rmse, sc = train_predict_for_ticker_panel(df_panel, ticker)
                if pr is not None:
                    predicted_return = float(pr)
                    model_val_rmse = float(m_rmse) if m_rmse is not None else None
                    hist_rmse = float(h_rmse) if h_rmse is not None else None
                    scenarios = sc
                    model_available = True
            except Exception as e:
                logging.exception(f"Error entrenando modelo para {ticker}: {e}")
                # Caeremos al fallback

        # Si no disponible, usar fallback (último mean_return)
        if not model_available:
            predicted_return = float(mean_return)
            model_val_rmse = None
            hist_rmse = None
            scenarios = None

        # Probabilidades
        prob_success, prob_failure = calculate_probabilities_advanced(predicted_return, volatility, scenarios)

        best_method = "Modelo PyTorch Avanzado" if (model_val_rmse is not None and hist_rmse is not None and model_val_rmse < hist_rmse) else "Histórico (o fallback)"

        response = {
            'company': COMPANIES.get(ticker, ticker),
            'ticker': ticker,
            'stats': {
                'meanReturn': float(mean_return),
                'volatility': float(volatility),
                'beta': float(beta),
                'avgVolume': float(avg_volume)
            },
            'predictions': {
                'historical': float(mean_return),
                'model': float(predicted_return),
                'bestMethod': best_method,
                'modelLoss': model_val_rmse,
                'scenarios': scenarios
            },
            'probabilities': {
                'success': float(prob_success),
                'failure': float(prob_failure)
            },
            'meta': {
                'modelAvailable': bool(model_available),
                'panelRows': int(len(df_t_panel)) if not df_t_panel.empty else 0
            }
        }

        # Guardar en cache simple
        set_cached_prediction(ticker, data_hash, response)

        return jsonify(response)

    except Exception as e:
        logging.exception("Error en /api/company")
        return jsonify({'error': str(e)}), 500

@app.route('/api/calculate_investment', methods=['POST'])
def calculate_investment():
    try:
        payload = request.get_json(force=True)
        investment = float(payload.get('investment', 0))
        predicted_return = float(payload.get('predictedReturn', 0))

        if investment <= 0:
            return jsonify({'error': 'Inversión debe ser > 0'}), 400

        profit = investment * predicted_return
        final_amount = investment + profit

        return jsonify({
            'investment': float(investment),
            'predictedReturn': float(predicted_return),
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
        df = load_agg_cached()
        if df.empty:
            return jsonify([])
        companies_list = []
        for ticker in df['Ticker'].unique():
            row = df[df['Ticker'] == ticker].iloc[-1]
            companies_list.append({
                'ticker': ticker,
                'name': COMPANIES.get(ticker, ticker),
                'meanReturn': float(row.get('MeanReturn', 0.0)),
                'volatility': float(row.get('Volatility', 0.0)),
                'beta': float(row.get('Beta', 0.0))
            })
        return jsonify(companies_list)
    except Exception as e:
        logging.exception("Error en /api/companies")
        return jsonify([]), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True)
