/* Script optimizado con cache, debounce y manejo de escenarios */

let currentCompanyData = null;
let probabilityChart = null;
let riskReturnChart = null;

// Control de peticiones y cache
let fetchAbortController = null;
let debounceTimer = null;
const DEBOUNCE_MS = 200;
const dataCache = new Map(); // Cache de respuestas

// Spinner con delay
let spinnerTimeout = null;
const SPINNER_DELAY = 160;

// Últimos valores para evitar updates innecesarios
let lastProbSuccess = null;
let lastProbFailure = null;
let lastRiskPoint = { x: null, y: null };

// Utilidades
const qs = id => document.getElementById(id);

function showError(message) {
    const el = qs('errorMessage');
    if (!el) return;
    el.textContent = message;
    el.style.display = 'block';
    clearTimeout(showError._t);
    showError._t = setTimeout(() => { el.style.display = 'none'; }, 5000);
}

// Inicialización
document.addEventListener('DOMContentLoaded', () => {
    const results = qs('resultsSection');
    if (results) {
        results.style.transition = 'opacity 220ms ease, transform 220ms ease';
        results.style.opacity = 0;
        results.style.transform = 'translateY(8px)';
        results.style.pointerEvents = 'none';
    }
    
    const spinner = qs('loadingSpinner');
    if (spinner) {
        spinner.style.transition = 'opacity 160ms linear';
        spinner.style.opacity = 0;
        spinner.style.pointerEvents = 'none';
    }

    const companySelect = qs('companySelect');
    const calculateBtn = qs('calculateBtn');
    
    if (companySelect) {
        companySelect.addEventListener('change', (e) => {
            if (debounceTimer) clearTimeout(debounceTimer);
            debounceTimer = setTimeout(() => handleCompanyChange(e), DEBOUNCE_MS);
        });
    }
    
    if (calculateBtn) calculateBtn.addEventListener('click', calculateInvestment);
    
    console.log('✅ App inicializada - modo optimizado');
});

// Spinner con delay (fix: gestionar display)
function showSpinnerWithDelay() {
    const spinner = qs('loadingSpinner');
    if (!spinner) return;
    if (spinnerTimeout) clearTimeout(spinnerTimeout);
    spinnerTimeout = setTimeout(() => {
        spinner.style.display = 'block';            // <-- mostrar
        // Forzar reflow opcional (mejora transiciones)
        void spinner.offsetWidth;
        spinner.style.opacity = 1;
        spinner.style.pointerEvents = 'auto';
        spinner.setAttribute('aria-hidden', 'false');
    }, SPINNER_DELAY);
}

function hideSpinnerImmediate() {
    const spinner = qs('loadingSpinner');
    if (!spinner) return;
    if (spinnerTimeout) {
        clearTimeout(spinnerTimeout);
        spinnerTimeout = null;
    }
    // desvanecer y luego ocultar para respetar la transición
    spinner.style.opacity = 0;
    spinner.style.pointerEvents = 'none';
    spinner.setAttribute('aria-hidden', 'true');
    setTimeout(() => { 
        // solo ocultar después de la transición
        spinner.style.display = 'none';
    }, 200);
}

// Mostrar/ocultar resultados (fix: gestionar display)
function showResultsSection() {
    const results = qs('resultsSection');
    if (!results) return;
    // Asegurar que esté renderizable primero
    results.style.display = 'block';
    // Forzar reflow para que la transición funcione correctamente
    void results.offsetWidth;
    const op = parseFloat(getComputedStyle(results).opacity || 0);
    if (op > 0.5) return;
    results.style.opacity = 1;
    results.style.transform = 'translateY(0)';
    results.style.pointerEvents = 'auto';
}

function hideResultsSection() {
    const results = qs('resultsSection');
    if (!results) return;
    results.style.opacity = 0;
    results.style.transform = 'translateY(8px)';
    results.style.pointerEvents = 'none';
    // ocultar del flow después de la transición
    setTimeout(() => { results.style.display = 'none'; }, 240);
}


// Fetch con cache y abort
async function handleCompanyChange(event) {
    const ticker = event.target.value;
    if (!ticker) {
        hideResultsSection();
        return;
    }

    // Verificar cache primero
    if (dataCache.has(ticker)) {
        console.log('📦 Usando datos cacheados para', ticker);
        const cachedData = dataCache.get(ticker);
        window.requestAnimationFrame(() => {
            displayCompanyData(cachedData);
            showResultsSection();
        });
        return;
    }

    // Cancelar fetch anterior
    if (fetchAbortController) {
        try { fetchAbortController.abort(); } catch(e) {}
    }
    fetchAbortController = new AbortController();
    const signal = fetchAbortController.signal;

    showSpinnerWithDelay();
    hideResultsSection();

    try {
        const res = await fetch(`/api/company/${encodeURIComponent(ticker)}`, { 
            cache: 'no-store', 
            signal 
        });
        
        if (!res.ok) {
            let txt;
            try { txt = await res.text(); } catch(e){ txt = res.statusText; }
            throw new Error(`HTTP ${res.status} - ${txt}`);
        }
        
        const data = await res.json();
        if (data.error) {
            showError('Error: ' + data.error);
            return;
        }
        
        currentCompanyData = data;
        dataCache.set(ticker, data); // Guardar en cache

        window.requestAnimationFrame(() => {
            displayCompanyData(data);
            showResultsSection();
        });

    } catch (err) {
        if (err.name === 'AbortError') return;
        console.error(err);
        showError('Error al cargar datos: ' + (err.message || err));
    } finally {
        hideSpinnerImmediate();
    }
}

// Display con escenarios
function displayCompanyData(data) {
    // Header
    qs('companyName').textContent = data.company || '';
    qs('companyTicker').textContent = data.ticker || '';

    // Stats
    qs('meanReturn').textContent = formatPercentageSafe(data.stats?.meanReturn);
    qs('volatility').textContent = formatPercentageSafe(data.stats?.volatility);
    qs('beta').textContent = (data.stats && !isNaN(data.stats.beta)) ? 
        Number(data.stats.beta).toFixed(3) : 'N/A';
    qs('avgVolume').textContent = formatVolume(data.stats?.avgVolume || 0);

    // Predictions
    const hist = data.predictions?.historical;
    const mod = data.predictions?.model;
    
    qs('historicalPrediction').textContent = formatPercentageSafe(hist);
    qs('modelPrediction').textContent = formatPercentageSafe(mod);

    toggleClass('historicalPrediction', 'positive', (hist !== undefined && hist >= 0));
    toggleClass('historicalPrediction', 'negative', (hist !== undefined && hist < 0));
    toggleClass('modelPrediction', 'positive', (mod !== undefined && mod >= 0));
    toggleClass('modelPrediction', 'negative', (mod !== undefined && mod < 0));

    const modelLoss = data.predictions?.modelLoss;
    qs('modelLoss').textContent = (modelLoss !== undefined && !isNaN(modelLoss)) ? 
        Number(modelLoss).toFixed(6) : 'N/A';
    qs('bestMethod').textContent = data.predictions?.bestMethod || '';

    // Renderizar escenarios
    displayScenarios(data.predictions?.scenarios);

    // Probabilidades
    let rawSuccess = Number(data.probabilities?.success);
    let rawFailure = Number(data.probabilities?.failure);
    if (isNaN(rawSuccess)) rawSuccess = 0;
    if (isNaN(rawFailure)) rawFailure = 0;

    let ps = (rawSuccess <= 1) ? rawSuccess * 100 : rawSuccess;
    let pf = (rawFailure <= 1) ? rawFailure * 100 : rawFailure;

    qs('probSuccess').textContent = ps.toFixed(2) + '%';
    qs('probFailure').textContent = pf.toFixed(2) + '%';

    updateProbabilityChartSmart(ps, pf);

    // Risk-return
    const vol = Number(data.stats?.volatility) || 0;
    const mean = Number(data.stats?.meanReturn) || 0;
    updateRiskReturnChartSmart({ 
        x: vol * 100, 
        y: mean * 100, 
        label: data.company || '' 
    });

    // Reset inversión
    qs('investmentResults').style.display = 'none';
    qs('investmentAmount').value = '';
}

// Renderizar tarjetas de escenarios
function displayScenarios(scenarios) {
    const container = qs('scenariosContainer');
    if (!container) return;
    
    if (!scenarios) {
        container.innerHTML = '<p style="text-align:center;color:#6b7280;">No hay análisis de escenarios disponible</p>';
        return;
    }

    const scenarioOrder = ['optimistic', 'neutral', 'pessimistic'];
    const scenarioIcons = {
        'optimistic': '📈',
        'neutral': '➡️',
        'pessimistic': '📉'
    };
    const scenarioTitles = {
        'optimistic': 'Escenario Optimista',
        'neutral': 'Escenario Neutral',
        'pessimistic': 'Escenario Pesimista'
    };

    let html = '';
    scenarioOrder.forEach(key => {
        const scenario = scenarios[key];
        if (!scenario) return;

        const returnVal = scenario.return || 0;
        const prob = (scenario.probability || 0) * 100;
        const desc = scenario.description || '';
        const returnClass = returnVal >= 0 ? 'positive' : 'negative';

        html += `
            <div class="scenario-card ${key}">
                <div class="scenario-header">
                    <span class="scenario-title">${scenarioIcons[key]} ${scenarioTitles[key]}</span>
                    <span class="scenario-prob">${prob.toFixed(0)}% prob.</span>
                </div>
                <div class="scenario-return ${returnClass}">
                    ${formatPercentageSafe(returnVal)}
                </div>
                <div class="scenario-desc">${desc}</div>
            </div>
        `;
    });

    container.innerHTML = html;
}

function toggleClass(id, cls, on) {
    const el = qs(id);
    if (!el) return;
    if (on) el.classList.add(cls); 
    else el.classList.remove(cls);
}

// Smart chart updates
function updateProbabilityChartSmart(successPercent, failurePercent) {
    const EPS = 0.05;
    if (lastProbSuccess !== null && 
        Math.abs(lastProbSuccess - successPercent) < EPS &&
        Math.abs(lastProbFailure - failurePercent) < EPS) {
        return;
    }
    lastProbSuccess = successPercent;
    lastProbFailure = failurePercent;

    const ctx = qs('probabilityChart');
    if (!ctx || typeof Chart === 'undefined') return;

    if (!probabilityChart) {
        probabilityChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Éxito', 'Pérdida'],
                datasets: [{
                    data: [successPercent, failurePercent],
                    backgroundColor: ['rgba(16,185,129,0.9)', 'rgba(239,68,68,0.9)'],
                    borderWidth: 2,
                    borderColor: '#fff'
                }]
            },
            options: {
                animation: { duration: 300, easing: 'easeOutCubic' },
                plugins: { 
                    legend: { position: 'bottom' }, 
                    tooltip: { 
                        callbacks: {
                            label: context => `${context.label}: ${context.parsed.toFixed(2)}%`
                        }
                    } 
                },
                responsive: true,
                maintainAspectRatio: true
            }
        });
    } else {
        probabilityChart.data.datasets[0].data = [successPercent, failurePercent];
        probabilityChart.update('none'); // Update sin animación
        setTimeout(() => probabilityChart.update(300), 10); // Luego animar
    }
}

function updateRiskReturnChartSmart(point) {
    const ctx = qs('riskReturnChart');
    if (!ctx || typeof Chart === 'undefined') return;

    const EPS = 0.01;
    if (lastRiskPoint.x !== null && 
        Math.abs(lastRiskPoint.x - point.x) < EPS && 
        Math.abs(lastRiskPoint.y - point.y) < EPS) {
        return;
    }
    lastRiskPoint = { x: point.x, y: point.y };

    if (!riskReturnChart) {
        riskReturnChart = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: point.label || '',
                    data: [{ x: point.x, y: point.y }],
                    backgroundColor: 'rgba(37,99,235,0.9)',
                    pointRadius: 10,
                    pointHoverRadius: 12
                }]
            },
            options: {
                animation: { duration: 300, easing: 'easeOutCubic' },
                plugins: { 
                    legend: { display: true },
                    tooltip: {
                        callbacks: {
                            label: context => {
                                return `${context.dataset.label}: Volatilidad ${context.parsed.x.toFixed(2)}%, Retorno ${context.parsed.y.toFixed(2)}%`;
                            }
                        }
                    }
                },
                scales: {
                    x: { 
                        title: { display: true, text: 'Volatilidad (%)' },
                        beginAtZero: true
                    },
                    y: { 
                        title: { display: true, text: 'Retorno Esperado (%)' }
                    }
                },
                responsive: true,
                maintainAspectRatio: true
            }
        });
    } else {
        riskReturnChart.data.datasets[0].data = [{ x: point.x, y: point.y }];
        if (point.label) riskReturnChart.data.datasets[0].label = point.label;
        riskReturnChart.update('none');
        setTimeout(() => riskReturnChart.update(300), 10);
    }
}

// Calculadora de inversión
async function calculateInvestment() {
    const el = qs('investmentAmount');
    const investmentAmount = parseFloat(el?.value);
    
    if (!investmentAmount || investmentAmount <= 0) {
        showError('Por favor ingresa un monto válido');
        return;
    }
    
    if (!currentCompanyData) {
        showError('Primero selecciona una empresa');
        return;
    }

    const predictedReturnFraction = (currentCompanyData.predictions?.bestMethod?.includes("PyTorch"))
        ? currentCompanyData.predictions.model
        : currentCompanyData.predictions.historical;

    try {
        const res = await fetch('/api/calculate_investment', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                investment: investmentAmount, 
                predictedReturn: predictedReturnFraction 
            })
        });
        
        if (!res.ok) {
            let t;
            try { t = await res.text(); } catch(e) { t = res.statusText; }
            throw new Error(`HTTP ${res.status} - ${t}`);
        }
        
        const result = await res.json();
        if (result.error) { 
            showError('Error: ' + result.error); 
            return; 
        }
        
        qs('initialInvestment').textContent = formatCurrency(result.investment);
        qs('expectedReturn').textContent = formatPercentageSafe(result.predictedReturn);
        qs('profitLoss').textContent = formatCurrency(result.profit);
        qs('finalAmount').textContent = formatCurrency(result.finalAmount);

        const resultsDiv = qs('investmentResults');
        resultsDiv.style.display = 'block';
        resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    } catch (err) {
        console.error(err);
        showError('Error de conexión: ' + (err.message || err));
    }
}

// Formatters
function formatPercentageSafe(v) {
    if (v === null || v === undefined || isNaN(v)) return 'N/A';
    const n = Number(v);
    const percent = (n <= 1) ? n * 100 : n;
    const sign = percent >= 0 ? '+' : '';
    return sign + percent.toFixed(2) + '%';
}

function formatVolume(value) {
    if (!value) return '0';
    if (value >= 1e9) return (value / 1e9).toFixed(2) + 'B';
    if (value >= 1e6) return (value / 1e6).toFixed(2) + 'M';
    if (value >= 1e3) return (value / 1e3).toFixed(2) + 'K';
    return Number(value).toFixed(0);
}

function formatCurrency(value) {
    if (value === null || value === undefined || isNaN(value)) return 'N/A';
    return new Intl.NumberFormat('es-MX', { 
        style: 'currency', 
        currency: 'USD', 
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }).format(value);
}