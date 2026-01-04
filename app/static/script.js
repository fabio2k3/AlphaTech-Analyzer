let currentCompanyData = null;
let probabilityChart = null;
let riskReturnChart = null;

// Inicializar eventos
document.addEventListener('DOMContentLoaded', function() {
    const companySelect = document.getElementById('companySelect');
    const calculateBtn = document.getElementById('calculateBtn');
    
    companySelect.addEventListener('change', handleCompanyChange);
    calculateBtn.addEventListener('click', calculateInvestment);
});

async function handleCompanyChange(event) {
    const ticker = event.target.value;
    
    if (!ticker) {
        document.getElementById('resultsSection').style.display = 'none';
        return;
    }
    
    // Mostrar spinner
    document.getElementById('loadingSpinner').style.display = 'block';
    document.getElementById('resultsSection').style.display = 'none';
    
    try {
        const response = await fetch(`/api/company/${ticker}`);
        const data = await response.json();
        
        if (data.error) {
            alert('Error al cargar datos: ' + data.error);
            return;
        }
        
        currentCompanyData = data;
        displayCompanyData(data);
        
    } catch (error) {
        alert('Error de conexión: ' + error.message);
    } finally {
        document.getElementById('loadingSpinner').style.display = 'none';
    }
}

function displayCompanyData(data) {
    // Mostrar sección de resultados
    document.getElementById('resultsSection').style.display = 'block';
    
    // Header
    document.getElementById('companyName').textContent = data.company;
    document.getElementById('companyTicker').textContent = data.ticker;
    
    // Estadísticas
    document.getElementById('meanReturn').textContent = formatPercentage(data.stats.meanReturn);
    document.getElementById('volatility').textContent = formatPercentage(data.stats.volatility);
    document.getElementById('beta').textContent = data.stats.beta.toFixed(3);
    document.getElementById('avgVolume').textContent = formatVolume(data.stats.avgVolume);
    
    // Predicciones
    const historicalEl = document.getElementById('historicalPrediction');
    const modelEl = document.getElementById('modelPrediction');
    
    historicalEl.textContent = formatPercentage(data.predictions.historical);
    modelEl.textContent = formatPercentage(data.predictions.model);
    
    // Colorear según positivo/negativo
    historicalEl.className = 'prediction-value ' + (data.predictions.historical >= 0 ? 'positive' : 'negative');
    modelEl.className = 'prediction-value ' + (data.predictions.model >= 0 ? 'positive' : 'negative');
    
    document.getElementById('modelLoss').textContent = data.predictions.modelLoss.toFixed(6);
    document.getElementById('bestMethod').textContent = data.predictions.bestMethod;
    
    // Probabilidades
    document.getElementById('probSuccess').textContent = data.probabilities.success.toFixed(2) + '%';
    document.getElementById('probFailure').textContent = data.probabilities.failure.toFixed(2) + '%';
    
    // Gráficos
    createProbabilityChart(data.probabilities);
    createRiskReturnChart(data);
    
    // Ocultar resultados de inversión
    document.getElementById('investmentResults').style.display = 'none';
    document.getElementById('investmentAmount').value = '';
}

function createProbabilityChart(probabilities) {
    const ctx = document.getElementById('probabilityChart');
    
    if (probabilityChart) {
        probabilityChart.destroy();
    }
    
    probabilityChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Probabilidad de Éxito', 'Probabilidad de Pérdida'],
            datasets: [{
                data: [probabilities.success, probabilities.failure],
                backgroundColor: [
                    'rgba(16, 185, 129, 0.8)',
                    'rgba(239, 68, 68, 0.8)'
                ],
                borderColor: [
                    'rgba(16, 185, 129, 1)',
                    'rgba(239, 68, 68, 1)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        font: {
                            size: 14
                        },
                        padding: 20
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.label + ': ' + context.parsed.toFixed(2) + '%';
                        }
                    }
                }
            }
        }
    });
}

function createRiskReturnChart(data) {
    const ctx = document.getElementById('riskReturnChart');
    
    if (riskReturnChart) {
        riskReturnChart.destroy();
    }
    
    // Crear dataset con la empresa actual
    const chartData = {
        datasets: [{
            label: data.company,
            data: [{
                x: data.stats.volatility * 100,
                y: data.stats.meanReturn * 100
            }],
            backgroundColor: 'rgba(37, 99, 235, 0.8)',
            borderColor: 'rgba(37, 99, 235, 1)',
            pointRadius: 10,
            pointHoverRadius: 15
        }]
    };
    
    riskReturnChart = new Chart(ctx, {
        type: 'scatter',
        data: chartData,
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                title: {
                    display: true,
                    text: 'Posición de Riesgo vs Retorno',
                    font: {
                        size: 16
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return [
                                'Empresa: ' + context.dataset.label,
                                'Volatilidad: ' + context.parsed.x.toFixed(2) + '%',
                                'Retorno: ' + context.parsed.y.toFixed(2) + '%'
                            ];
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Volatilidad (Riesgo) %',
                        font: {
                            size: 14
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Retorno Promedio %',
                        font: {
                            size: 14
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                }
            }
        }
    });
}

async function calculateInvestment() {
    const investmentAmount = parseFloat(document.getElementById('investmentAmount').value);
    
    if (!investmentAmount || investmentAmount <= 0) {
        alert('Por favor ingresa un monto válido');
        return;
    }
    
    if (!currentCompanyData) {
        alert('Primero selecciona una empresa');
        return;
    }
    
    // Usar la mejor predicción (modelo o histórico)
    const predictedReturn = currentCompanyData.predictions.bestMethod === "Modelo PyTorch" 
        ? currentCompanyData.predictions.model 
        : currentCompanyData.predictions.historical;
    
    try {
        const response = await fetch('/api/calculate_investment', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                investment: investmentAmount,
                predictedReturn: predictedReturn
            })
        });
        
        const result = await response.json();
        
        if (result.error) {
            alert('Error al calcular: ' + result.error);
            return;
        }
        
        displayInvestmentResults(result);
        
    } catch (error) {
        alert('Error de conexión: ' + error.message);
    }
}

function displayInvestmentResults(result) {
    document.getElementById('investmentResults').style.display = 'block';
    
    document.getElementById('initialInvestment').textContent = formatCurrency(result.investment);
    document.getElementById('expectedReturn').textContent = result.predictedReturn.toFixed(2) + '%';
    
    const profitEl = document.getElementById('profitLoss');
    profitEl.textContent = formatCurrency(result.profit);
    profitEl.style.color = result.profit >= 0 ? '#10b981' : '#ef4444';
    
    document.getElementById('finalAmount').textContent = formatCurrency(result.finalAmount);
    
    // Scroll suave a resultados
    document.getElementById('investmentResults').scrollIntoView({ 
        behavior: 'smooth', 
        block: 'nearest' 
    });
}

// Funciones de formato
function formatPercentage(value) {
    return (value * 100).toFixed(2) + '%';
}

function formatVolume(value) {
    if (value >= 1e9) {
        return (value / 1e9).toFixed(2) + 'B';
    } else if (value >= 1e6) {
        return (value / 1e6).toFixed(2) + 'M';
    } else if (value >= 1e3) {
        return (value / 1e3).toFixed(2) + 'K';
    }
    return value.toFixed(0);
}

function formatCurrency(value) {
    return new Intl.NumberFormat('es-MX', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2
    }).format(value);
}