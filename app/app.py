# app.py — versión corregida para usar panel mensual para predicción
# Comentarios en español: explicación línea a línea / función por función.
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

# Inicializa la aplicación Flask y el logging básico
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# ---------- CONFIG / PATHS ----------
# Directorio base y rutas hacia los CSV de panel y agregados
BASE_DIR = Path(__file__).resolve().parent
PANEL_PATH = (BASE_DIR / '..' / 'data' / 'processed' / 'tech30_panel_monthly_2018_2024.csv').resolve()
AGG_PATH = (BASE_DIR / '..' / 'data' / 'processed' / 'tech30_aggregated_stats_2018_2024.csv').resolve()

# Cache para datos en memoria (separado para panel y agregado)
# Cada entrada guarda el DataFrame y la marca temporal de carga 'ts'
_data_cache = {
    'panel': {'df': None, 'ts': None},
    'agg': {'df': None, 'ts': None}
}
# Duración de cache en segundos (5 minutos)
CACHE_DURATION = 300  # 5 minutos

# Cache de predicciones por ticker (in-memory)
# Formato: _prediction_cache[ticker] = {'hash': str, 'response': dict, 'ts': float}
_prediction_cache = {}  # ticker -> {'hash': str, 'response': dict, 'ts': float}

# ---------- Diccionario empresas ----------
# Mapa ticker -> nombre oficial (usado en respuestas / UI)
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
    # Modelo feed-forward simple para regresión de retornos
    def __init__(self, input_dim: int):
        super().__init__()
        # Arquitectura: Linear -> BatchNorm -> ReLU -> Dropout -> ... -> Linear(1)
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
        # Forward pass: devuelve un tensor (batch_size, 1)
        return self.network(x)

# ---------- Utilities: carga de datos con cache ----------
def load_panel_cached():
    """
    Carga el CSV del PANEL y lo cachea en _data_cache['panel'] durante CACHE_DURATION.
    Si la carga falla (FileNotFoundError) devuelve un DataFrame vacío.
    """
    now = datetime.now()
    cache = _data_cache['panel']
    # Si hay DF en cache y no expiró, devolverlo
    if cache['df'] is not None and cache['ts'] and (now - cache['ts']).seconds < CACHE_DURATION:
        return cache['df']
    try:
        # Leer CSV y parsear columna 'Date' como fecha
        df = pd.read_csv(PANEL_PATH, parse_dates=['Date'])
        # Validación mínima: columna 'Return' esperada
        if 'Return' not in df.columns:
            logging.warning("El panel no tiene columna 'Return' — revisa tu CSV")
        # Guardar en cache con timestamp
        cache['df'] = df
        cache['ts'] = now
        logging.info(f"Panel cargado desde {PANEL_PATH} — filas: {len(df)}")
        return df
    except FileNotFoundError:
        # Si no encuentra el fichero, loguea la excepción y devuelve DF vacío
        logging.exception(f"No se encontró el fichero panel en {PANEL_PATH}")
        return pd.DataFrame()

def load_agg_cached():
    """
    Carga el CSV de datos agregados y lo cachea en _data_cache['agg'].
    Devuelve DataFrame vacío si el fichero no existe.
    """
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
    """
    Devuelve un diccionario con factores macro simulados.
    Usa la fecha actual (mes) para ajustar algunos factores discretos.
    """
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
    """
    Devuelve factores por empresa (ai_exposure, cloud_growth, etc.) simulados
    en base a listas predefinidas de tipos de empresas.
    """
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
    Construye X, y desde el panel mensual usando ventanas de tamaño `window`.
    - X: concatenación de returns_lag (window), log(volumen_lag window), sin/cos mes, macro/company factors
    - y: retorno en el mes objetivo (returns[i])
    Devuelve (None, None) si no hay suficientes datos o falta 'Return'.
    """
    # Ordenar por fecha y resetear índice
    df = df_ticker.sort_values('Date').reset_index(drop=True).copy()
    if df.empty or 'Return' not in df.columns:
        return None, None

    # Si falta 'Volume', crear columna con ceros y loggear advertencia
    if 'Volume' not in df.columns:
        logging.warning(f"Ticker {ticker} - panel no contiene columna 'Volume'")
        df['Volume'] = 0.0

    # Extraer arrays numpy de retornos, volúmenes y fechas
    returns = df['Return'].astype(float).values
    volumes = df['Volume'].astype(float).values
    dates = pd.to_datetime(df['Date']).dt.to_pydatetime()

    N = len(returns)
    # Si no hay al menos window+1 observaciones, no se puede crear muestra
    if N <= window:
        return None, None

    X_list = []
    y_list = []
    # Para cada punto i (desde window hasta N-1) construimos una muestra
    for i in range(window, N):
        # Ventana de returns y volúmenes (lags)
        ret_window = returns[i-window:i]      # shape (window,)
        vol_window = volumes[i-window:i]      # shape (window,)

        # Estacionalidad: sin y cos sobre el mes objetivo (fecha i)
        month = dates[i].month
        sin_month = np.sin(2 * np.pi * (month-1) / 12)
        cos_month = np.cos(2 * np.pi * (month-1) / 12)

        # Obtener factores macro y por empresa (simulados)
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

        # Vector de features: returns lags, log1p(volume lags), sin_month, cos_month, macro(12)
        fv = np.concatenate([
            ret_window,
            np.log1p(vol_window),
            np.array([sin_month, cos_month]),
            macro
        ])
        X_list.append(fv)
        # Etiqueta: retorno en el mes objetivo (i)
        y_list.append(returns[i])

    # Apilar en matrices numpy
    X = np.vstack(X_list)
    y = np.array(y_list)
    return X, y

# ---------- Entrenamiento ----------
def train_advanced_model(X_train, y_train, X_val, y_val, epochs=200, lr=5e-3, device='cpu'):
    """
    Entrena el modelo PyTorch pasando X_train/y_train y validando en X_val/y_val.
    - Usa AdamW, MSELoss, ReduceLROnPlateau y early stopping por paciencia.
    - Devuelve (model_entrenado, rmse_validación).
    """
    # Fijar semilla para reproducibilidad parcial
    torch.manual_seed(42)
    input_dim = X_train.shape[1]
    model = AdvancedPredictionModel(input_dim).to(device)

    # Definir loss y optimizador
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15, factor=0.5)

    # Convertir numpy arrays a tensores en el device (cpu o cuda)
    Xtr = torch.tensor(X_train, dtype=torch.float32).to(device)
    ytr = torch.tensor(y_train.reshape(-1,1), dtype=torch.float32).to(device)
    Xv = torch.tensor(X_val, dtype=torch.float32).to(device)
    yv = torch.tensor(y_val.reshape(-1,1), dtype=torch.float32).to(device)

    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 20

    # Bucle de entrenamiento
    for epoch in range(epochs):
        model.train()
        pred = model(Xtr)
        loss = criterion(pred, ytr)
        optimizer.zero_grad()
        loss.backward()
        # Clipping de gradientes para estabilidad numérica
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Evaluar en validación y ajustar scheduler
        model.eval()
        with torch.no_grad():
            val_pred = model(Xv)
            val_loss = criterion(val_pred, yv)
            scheduler.step(val_loss)

            # Early stopping por paciencia
            if val_loss < best_val_loss - 1e-9:
                best_val_loss = float(val_loss)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    break

    # Calcular RMSE final sobre validación
    model.eval()
    with torch.no_grad():
        val_pred = model(Xv).cpu().numpy().reshape(-1)
        val_rmse = float(np.sqrt(np.mean((val_pred - y_val) ** 2)))
    return model, val_rmse

# ---------- Normalización ----------
def normalize_features(X_train, X_apply):
    """
    Normaliza X_apply usando media y desviación estándar calculadas sobre X_train.
    - Devuelve X_train_s, X_apply_s, mean, std
    - Evita dividir por 0 si std es muy pequeño (reemplaza por 1.0)
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0, ddof=0)
    std = np.where(std < 1e-8, 1.0, std)
    X_train_s = (X_train - mean) / std
    X_apply_s = (X_apply - mean) / std
    return X_train_s, X_apply_s, mean, std

# ---------- Escenarios ----------
def generate_scenarios(model, base_features, base_pred, device='cpu'):
    """
    Genera tres escenarios (optimista, neutral, pesimista) a partir de base_features.
    - Asume que las últimas 12 columnas de base_features corresponden a factores macro/company.
    - Crea perturbaciones predefinidas en ese bloque y solicita predicción al modelo.
    - Devuelve diccionario con 'optimistic', 'neutral', 'pessimistic' (cada uno con return y probability).
    """
    # Copiar base_features a numpy array para no mutar entrada
    bf = np.array(base_features, dtype=float).copy()
    # Determinar posición del bloque macro (últimas 12 dimensiones)
    D = bf.shape[1]
    macro_len = 12
    macro_start = D - macro_len
    if macro_start < 0:
        macro_start = max(0, D - macro_len)

    # Copias para perturbar
    opt = bf.copy()
    pes = bf.copy()

    # Perturbaciones fijas (optimista / pesimista)
    opt[0, macro_start:macro_start+macro_len] += np.array([0.10,0.08,0.05,-0.05,-0.03,0.10,0.08,0.05,0.05,-0.02,0.05,0.05])
    pes[0, macro_start:macro_start+macro_len] += np.array([-0.08,-0.06,-0.08,0.08,0.05,-0.05,-0.05,-0.03,-0.03,0.05,-0.03,-0.05])

    # Predecir con el modelo sin gradientes
    with torch.no_grad():
        model.eval()
        to_tensor = lambda arr: torch.tensor(arr, dtype=torch.float32).to(device)
        opt_pred = float(model(to_tensor(opt)).cpu().numpy().reshape(-1)[0])
        pes_pred = float(model(to_tensor(pes)).cpu().numpy().reshape(-1)[0])

    # Calcular probabilidades heurísticas basadas en rango entre opt y pes
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
    Calcula P(success) y P(failure) de que el retorno futuro sea > 0.
    - Si se pasan 'scenarios', construye media y varianza ponderada por probabilidades
      y combina con la volatilidad histórica para obtener desviación total.
    - Si no hay 'scenarios', usa aproximación normal con mean_return y volatility.
    - Devuelve (prob_success, prob_failure) en formato float.
    """
    from scipy import stats

    eps = 1e-8  # evita división por cero
    # Caso con escenarios proporcionados
    if scenarios:
        # Extraer retornos y probabilidades desde el diccionario de escenarios
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

        # Normalizar probabilidades si no suman 1 o son inválidas
        if probs.sum() <= 0 or np.isnan(probs.sum()):
            probs = np.ones_like(probs) / len(probs)
        else:
            probs = probs / probs.sum()

        # Media y varianza ponderada de escenarios
        mean_s = float(np.sum(probs * returns))
        var_s = float(np.sum(probs * (returns - mean_s) ** 2))

        # Volatilidad histórica (si está disponible)
        vol_hist = float(volatility) if (volatility is not None and not np.isnan(volatility)) else 0.0
        if vol_hist < 0:
            vol_hist = abs(vol_hist)

        # Var total = var escenarios + vol_hist^2
        total_var = var_s + (vol_hist ** 2)
        if total_var < eps:
            total_var = eps

        total_std = np.sqrt(total_var)

        # Probabilidad de que retorno > 0 según N(mean_s, total_std)
        prob_success = float(stats.norm.cdf(mean_s / total_std))

        # Evitar 0 o 1 exactos por presentación
        prob_success = max(min(prob_success, 1.0 - 1e-6), 1e-6)
        prob_failure = 1.0 - prob_success

        return prob_success, prob_failure

    # Caso sin escenarios: usar aproximación normal simple
    else:
        from scipy import stats as _stats
        if volatility is not None and volatility > 0:
            z_score = mean_return / volatility
            prob_success = float(_stats.norm.cdf(z_score))
        else:
            # Si no hay volatilidad conocida, devolver 0.5 (incertidumbre)
            prob_success = 0.5

        prob_success = max(min(prob_success, 1.0 - 1e-6), 1e-6)
        return float(prob_success), float(1.0 - prob_success)

# ---------- Cache simple de predicción ----------
def make_data_hash(df_t):
    """Hash simple basado en cantidad de filas y última fecha (rápido)."""
    if df_t.empty:
        return "empty"
    last_date = str(df_t['Date'].max())
    return f"n{len(df_t)}_last{last_date}"

def get_cached_prediction(ticker, data_hash):
    """
    Devuelve la predicción cacheada si existe y si el hash coincide y no expiró.
    - Comprueba timestamp contra CACHE_DURATION.
    """
    rec = _prediction_cache.get(ticker)
    if not rec:
        return None
    if rec.get('hash') == data_hash:
        # Mantener cache fresco por CACHE_DURATION segundos
        if time.time() - rec.get('ts',0) < CACHE_DURATION:
            return rec.get('response')
    return None

def set_cached_prediction(ticker, data_hash, response):
    """Guarda la predicción en cache (in-memory) con timestamp actual."""
    _prediction_cache[ticker] = {'hash': data_hash, 'response': response, 'ts': time.time()}

# ---------- Entrenar y predecir (usando PANEL) ----------
def train_predict_for_ticker_panel(df_panel, ticker):
    """
    Flujo completo para un ticker usando datos del panel:
    - Filtra el panel por ticker
    - Prepara X,y con prepare_advanced_features_panel
    - Split temporal train/val (80/20)
    - Normaliza features, entrena modelo y calcula RMSE de validación
    - Calcula baseline histórico (promedio) y su RMSE
    - Predice el último punto (base_prediction) y genera escenarios
    - Retorna (base_prediction, model_val_rmse, hist_rmse, scenarios)
    """
    df_t = df_panel[df_panel['Ticker'] == ticker].copy()
    if df_t.empty:
        raise ValueError(f"No hay datos de panel para el ticker {ticker}")

    # Info rápida en logs
    logging.info(f"train_predict_for_ticker_panel: ticker={ticker}, rows={len(df_t)}")

    # Preparar X,y
    X, y = prepare_advanced_features_panel(df_t, ticker, window=6)
    if X is None or len(X) < 8:
        # Insuficiente historia para entrenar de forma fiable
        logging.info(f"Insufficient panel history for {ticker} (rows={len(df_t)}), skipping model training.")
        return None, None, None, None

    # Split temporal: primeros 80% para train, últimos 20% para validación
    n = len(y)
    split = max(2, int(n * 0.8))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Normalizar features
    X_train_s, X_val_s, mean, std = normalize_features(X_train, X_val)

    # Entrenar modelo y obtener RMSE en validación
    model, model_val_rmse = train_advanced_model(X_train_s, y_train, X_val_s, y_val, epochs=200, lr=5e-3, device='cpu')

    # Baseline histórico: predecir con el promedio de y_train sobre el conjunto de validación
    hist_pred = np.full_like(y_val, np.mean(y_train))
    hist_rmse = float(np.sqrt(np.mean((hist_pred - y_val) ** 2)))

    # Predicción para el último punto usando la última ventana X[-1]
    last_row = X[-1].reshape(1, -1)
    last_row_s = (last_row - mean) / std

    with torch.no_grad():
        model.eval()
        last_tensor = torch.tensor(last_row_s, dtype=torch.float32)
        base_prediction = float(model(last_tensor).cpu().numpy().reshape(-1)[0])

    # Generar escenarios a partir del feature vector normalizado del último punto
    scenarios = generate_scenarios(model, last_row_s, base_prediction, device='cpu')

    return base_prediction, float(model_val_rmse), hist_rmse, scenarios

# ---------- RUTAS ----------
@app.route('/')
def index():
    """
    Ruta raíz que renderiza index.html y pasa la lista de empresas ordenada por nombre.
    La plantilla espera la variable 'companies'.
    """
    companies_sorted = sorted(
        [{'ticker': t, 'name': n} for t, n in COMPANIES.items()],
        key=lambda x: x['name'].lower()
    )
    return render_template('index.html', companies=companies_sorted)


@app.route('/api/company/<ticker>')
def get_company_data(ticker):
    """
    Endpoint principal para obtener estadísticas, predicciones y probabilidades de un ticker.
    Flujo:
    - Cargar panel y agregado usando funciones cacheadas
    - Intentar devolver resultado desde cache de predicción rápida
    - Obtener estadísticas (preferir agregado si está disponible)
    - Intentar entrenar modelo usando panel (train_predict_for_ticker_panel)
    - Si el modelo no está disponible, usar fallback histórico (mean_return)
    - Calcular probabilidades y construir JSON de respuesta
    - Guardar respuesta en cache simple y devolverla
    """
    try:
        # Cargar datasets
        df_panel = load_panel_cached()
        df_agg = load_agg_cached()

        # Construir hash de datos para la parte del panel del ticker (para cache)
        data_hash = make_data_hash(df_panel[df_panel['Ticker'] == ticker]) if not df_panel.empty else "empty"
        cached = get_cached_prediction(ticker, data_hash)
        if cached:
            logging.info(f"Returning cached prediction for {ticker}")
            return jsonify(cached)

        # Preparar DataFrames filtrados (agregado y panel)
        df_t_agg = df_agg[df_agg['Ticker'] == ticker] if not df_agg.empty else pd.DataFrame()
        df_t_panel = df_panel[df_panel['Ticker'] == ticker] if not df_panel.empty else pd.DataFrame()

        # Obtener stats preferiendo el agregado
        if not df_t_agg.empty:
            row = df_t_agg.iloc[-1]
            mean_return = float(row.get('MeanReturn', 0.0))
            volatility = float(row.get('Volatility', 0.0))
            beta = float(row.get('Beta', 1.0))
            avg_volume = float(row.get('AvgVolume', 0.0))
        elif not df_t_panel.empty:
            # Si no hay agregado, usar último registro del panel
            last_row = df_t_panel.sort_values('Date').iloc[-1]
            mean_return = float(last_row.get('Return', 0.0))
            volatility = float(df_t_panel['Return'].std(ddof=0) if len(df_t_panel) > 1 else 0.0)
            beta = 1.0
            avg_volume = float(df_t_panel['Volume'].mean() if 'Volume' in df_t_panel.columns else 0.0)
        else:
            # No hay datos para el ticker
            return jsonify({'error': f'No hay datos para {ticker}'}), 404

        # Variables para almacenar resultados del modelo
        predicted_return = None
        model_val_rmse = None
        hist_rmse = None
        scenarios = None
        model_available = False

        # Intentar entrenar y predecir con el panel si hay datos del panel
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
                # Si hay error durante el entrenamiento/ejecución, se captura
                logging.exception(f"Error entrenando modelo para {ticker}: {e}")
                # Se caerá al fallback histórico más abajo

        # Si no se pudo usar el modelo, usar el estadístico histórico como predicho
        if not model_available:
            predicted_return = float(mean_return)
            model_val_rmse = None
            hist_rmse = None
            scenarios = None

        # Calcular probabilidades usando la función avanzada
        prob_success, prob_failure = calculate_probabilities_advanced(predicted_return, volatility, scenarios)

        # Seleccionar mejor método entre modelo y histórico según RMSE
        best_method = "Modelo PyTorch Avanzado" if (model_val_rmse is not None and hist_rmse is not None and model_val_rmse < hist_rmse) else "Histórico (o fallback)"

        # Construir respuesta JSON
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

        # Guardar en cache simple y devolver
        set_cached_prediction(ticker, data_hash, response)
        return jsonify(response)

    except Exception as e:
        # Captura de errores generales en el endpoint
        logging.exception("Error en /api/company")
        return jsonify({'error': str(e)}), 500

@app.route('/api/calculate_investment', methods=['POST'])
def calculate_investment():
    """
    Endpoint que calcula resultado monetario esperado para una inversión dada.
    Input JSON esperado: {'investment': float, 'predictedReturn': float}
    - predictedReturn se interpreta como retorno relativo (ej. 0.05 => 5%).
    Devuelve JSON con profit, finalAmount y porcentajes.
    """
    try:
        payload = request.get_json(force=True)
        investment = float(payload.get('investment', 0))
        predicted_return = float(payload.get('predictedReturn', 0))

        # Validación básica: inversión positiva
        if investment <= 0:
            return jsonify({'error': 'Inversión debe ser > 0'}), 400

        # Cálculos simples: profit y monto final
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
    """
    Endpoint que devuelve la lista de empresas presentes en el CSV agregado.
    Para cada ticker devuelve ticker, nombre, meanReturn, volatility y beta.
    """
    try:
        df = load_agg_cached()
        if df.empty:
            return jsonify([])
        companies_list = []
        # Iterar por tickers únicos en el CSV agregado
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

# ---------- Entrada principal ----------
if __name__ == '__main__':
    # Ejecuta la app Flask en modo debug en el puerto 5000 (uso local)
    app.run(debug=True, port=5000, threaded=True)
