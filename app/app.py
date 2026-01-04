from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import os

app = Flask(__name__)

# Cargar datos
DATA_PATH = os.path.join('..', 'data', 'processed', 'tech30_aggregated_stats_2018_2024.csv')
df = pd.read_csv(DATA_PATH)

# Diccionario de empresas
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

# Modelo de regresión lineal con PyTorch
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x)

def train_predict_model(company_data):
    """Entrena modelo y predice retorno próximo mes"""
    # Preparar features: Beta, Volatility, AvgVolume
    X = torch.tensor([[
        company_data['Beta'],
        company_data['Volatility'],
        company_data['AvgVolume'] / 1e9  # Normalizar
    ]], dtype=torch.float32)
    
    # Target: MeanReturn
    y = torch.tensor([[company_data['MeanReturn']]], dtype=torch.float32)
    
    # Crear y entrenar modelo
    model = LinearRegressionModel(3)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    
    # Entrenamiento simple (100 epochs)
    for epoch in range(100):
        prediction = model(X)
        loss = criterion(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Predicción para próximo mes
    with torch.no_grad():
        next_month_return = model(X).item()
    
    return next_month_return, loss.item()

def calculate_probabilities(mean_return, volatility):
    """Calcula probabilidades de éxito/fracaso usando distribución normal"""
    # Probabilidad de retorno positivo
    from scipy import stats
    z_score = mean_return / volatility if volatility > 0 else 0
    prob_success = stats.norm.cdf(z_score)
    prob_failure = 1 - prob_success
    
    return prob_success, prob_failure

@app.route('/')
def index():
    return render_template('index.html', companies=COMPANIES)

@app.route('/api/company/<ticker>')
def get_company_data(ticker):
    try:
        company_row = df[df['Ticker'] == ticker].iloc[0]
        
        # Predicción con modelo PyTorch
        predicted_return, model_loss = train_predict_model(company_row)
        
        # Probabilidades
        prob_success, prob_failure = calculate_probabilities(
            company_row['MeanReturn'],
            company_row['Volatility']
        )
        
        # Determinar método más preciso (comparar con histórico)
        historical_return = company_row['MeanReturn']
        error_model = abs(predicted_return - historical_return)
        
        # El histórico siempre es más confiable que un modelo simple
        best_method = "Histórico" if error_model > 0.01 else "Modelo PyTorch"
        
        data = {
            'company': COMPANIES.get(ticker, ticker),
            'ticker': ticker,
            'stats': {
                'meanReturn': float(company_row['MeanReturn']),
                'volatility': float(company_row['Volatility']),
                'beta': float(company_row['Beta']),
                'avgVolume': float(company_row['AvgVolume'])
            },
            'predictions': {
                'historical': float(historical_return),
                'model': float(predicted_return),
                'bestMethod': best_method,
                'modelLoss': float(model_loss)
            },
            'probabilities': {
                'success': float(prob_success * 100),
                'failure': float(prob_failure * 100)
            }
        }
        
        return jsonify(data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/api/calculate_investment', methods=['POST'])
def calculate_investment():
    try:
        data = request.json
        investment = float(data['investment'])
        predicted_return = float(data['predictedReturn'])
        
        # Calcular ganancia/pérdida
        profit = investment * predicted_return
        final_amount = investment + profit
        
        return jsonify({
            'investment': investment,
            'predictedReturn': predicted_return * 100,
            'profit': profit,
            'finalAmount': final_amount,
            'percentageChange': predicted_return * 100
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/companies')
def get_all_companies():
    companies_list = []
    for _, row in df.iterrows():
        companies_list.append({
            'ticker': row['Ticker'],
            'name': COMPANIES.get(row['Ticker'], row['Ticker']),
            'meanReturn': float(row['MeanReturn']),
            'volatility': float(row['Volatility']),
            'beta': float(row['Beta'])
        })
    return jsonify(companies_list)

if __name__ == '__main__':
    app.run(debug=True, port=5000)