/* app/static/script.js (versión anti-flash: debounce, abort, spinner delay, transitions) */

let currentCompanyData = null;
let probabilityChart = null;
let riskReturnChart = null;

// Control de peticiones
let fetchAbortController = null;
let debounceTimer = null;
const DEBOUNCE_MS = 200;

// Spinner/min delay
let spinnerTimeout = null;
const SPINNER_DELAY = 160; // ms: si la petición es más rápida no mostramos spinner

// Últimos valores de gráfico (evitar updates innecesarios)
let lastProbSuccess = null;
let lastProbFailure = null;
let lastRiskPoint = { x: null, y: null };

// Utilidades
function qs(id) { return document.getElementById(id); }

function showError(message) {
    const el = qs('errorMessage');
    if (!el) return;
    el.textContent = message;
    el.style.display = 'block';
    clearTimeout(showError._t);
    showError._t = setTimeout(() => { el.style.display = 'none'; }, 5000);
}

// Inicialización: preparar transiciones suaves en resultados y spinner
document.addEventListener('DOMContentLoaded', () => {
    const results = qs('resultsSection');
    if (results) {
        results.style.transition = 'opacity 220ms ease, transform 220ms ease';
        // Inicialmente oculto con opacidad 0 y pointer-events none
        if (results.style.display === '' || results.style.display === 'block') {
            // leave as is; only set if not configured
        } else {
            results.style.opacity = 0;
            results.style.transform = 'translateY(8px)';
            results.style.pointerEvents = 'none';
            results.style.display = 'block'; // keep in flow to avoid re-layout on show
        }
    }
    const spinner = qs('loadingSpinner');
    if (spinner) {
        spinner.style.transition = 'opacity 160ms linear';
        spinner.style.opacity = 0;
        spinner.style.pointerEvents = 'none';
    }

    // Attach debounced handler
    const companySelect = qs('companySelect');
    const calculateBtn = qs('calculateBtn');
    if (companySelect) {
        companySelect.addEventListener('change', (e) => {
            if (debounceTimer) clearTimeout(debounceTimer);
            debounceTimer = setTimeout(() => {
                handleCompanyChange(e);
            }, DEBOUNCE_MS);
        });
    }
    if (calculateBtn) calculateBtn.addEventListener('click', calculateInvestment);

    console.log('script (anti-flash) cargado');
});

// Spinner helpers with delay
function showSpinnerWithDelay() {
    const spinner = qs('loadingSpinner');
    if (!spinner) return;
    // solo mostrar si no aparece en SPINNER_DELAY ms
    spinnerTimeout = setTimeout(() => {
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
    spinner.style.opacity = 0;
    spinner.style.pointerEvents = 'none';
    spinner.setAttribute('aria-hidden', 'true');
}

// Mostrar/ocultar resultados con transición suave
function showResultsSection() {
    const results = qs('resultsSection');
    if (!results) return;
    // If already visible (opacity > 0.5) don't touch
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
}

// Fetch + cancel previous via AbortController
async function handleCompanyChange(event) {
    const ticker = event.target.value;
    if (!ticker) {
        hideResultsSection();
        return;
    }

    // Cancel previous fetch if exists
    if (fetchAbortController) {
        try { fetchAbortController.abort(); } catch(e) {}
    }
    fetchAbortController = new AbortController();
    const signal = fetchAbortController.signal;

    // Show spinner after delay
    showSpinnerWithDelay();
    hideResultsSection(); // keep hidden until first render (but using opacity)

    try {
        const res = await fetch(`/api/company/${encodeURIComponent(ticker)}`, { cache: 'no-store', signal });
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

        // Batch DOM updates in next animation frame to reduce reflow flicker
        window.requestAnimationFrame(() => {
            displayCompanyData(data);
            showResultsSection();
        });

    } catch (err) {
        if (err.name === 'AbortError') {
            // request aborted intentionally; ignore
            return;
        }
        console.error(err);
        showError('Error al cargar datos: ' + (err.message || err));
    } finally {
        hideSpinnerImmediate();
    }
}

// Display minimal updates and avoid heavy layout churn
function displayCompanyData(data) {
    // Header
    qs('companyName').textContent = data.company || '';
    qs('companyTicker').textContent = data.ticker || '';

    // Stats (formatters are defensive)
    qs('meanReturn').textContent = formatPercentageSafe(data.stats?.meanReturn);
    qs('volatility').textContent = formatPercentageSafe(data.stats?.volatility);
    qs('beta').textContent = (data.stats && !isNaN(data.stats.beta)) ? Number(data.stats.beta).toFixed(3) : 'N/A';
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

    qs('modelLoss').textContent = (data.predictions?.modelLoss !== undefined && !isNaN(data.predictions.modelLoss)) ? Number(data.predictions.modelLoss).toFixed(6) : 'N/A';
    qs('bestMethod').textContent = data.predictions?.bestMethod || '';

    // Probabilities: accept both fraction (0..1) or percent (0..100)
    let rawSuccess = Number(data.probabilities?.success);
    let rawFailure = Number(data.probabilities?.failure);
    if (isNaN(rawSuccess)) rawSuccess = 0;
    if (isNaN(rawFailure)) rawFailure = 0;

    let ps = (rawSuccess <= 1) ? rawSuccess * 100 : rawSuccess;
    let pf = (rawFailure <= 1) ? rawFailure * 100 : rawFailure;

    // if both zero and predictions exist, try deriving sign-based prob (fallback)
    if (ps === 0 && pf === 0 && typeof hist === 'number' && typeof mod === 'number') {
        // simple heuristic: if mean positive, success >50
        ps = (hist + 1) * 50; // heuristic, rarely used
        pf = 100 - ps;
    }

    qs('probSuccess').textContent = ps.toFixed(2) + '%';
    qs('probFailure').textContent = pf.toFixed(2) + '%';

    // Update charts (only if meaningful change)
    updateProbabilityChartSmart(ps, pf);

    // Risk-return
    const vol = Number(data.stats?.volatility) || 0;
    const mean = Number(data.stats?.meanReturn) || 0;
    updateRiskReturnChartSmart({ x: vol * 100, y: mean * 100, label: data.company || '' });

    // Reset investment ui
    qs('investmentResults').style.display = 'none';
    qs('investmentAmount').value = '';
}

function toggleClass(id, cls, on) {
    const el = qs(id);
    if (!el) return;
    if (on) el.classList.add(cls); else el.classList.remove(cls);
}

// Smart update for probability chart (avoid update if differences tiny)
function updateProbabilityChartSmart(successPercent, failurePercent) {
    // Tolerance: only update if change > 0.05%
    const EPS = 0.05;
    if (lastProbSuccess !== null && Math.abs(lastProbSuccess - successPercent) < EPS &&
        lastProbFailure !== null && Math.abs(lastProbFailure - failurePercent) < EPS) {
        return; // negligible change
    }
    lastProbSuccess = successPercent;
    lastProbFailure = failurePercent;

    // create or update
    const ctx = qs('probabilityChart');
    if (!ctx) return;
    if (typeof Chart === 'undefined') return;

    if (!probabilityChart) {
        probabilityChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Éxito', 'Pérdida'],
                datasets: [{
                    data: [successPercent, failurePercent],
                    backgroundColor: ['rgba(16,185,129,0.9)', 'rgba(239,68,68,0.9)'],
                    borderWidth: 1
                }]
            },
            options: {
                animation: { duration: 300, easing: 'easeOutCubic' },
                plugins: { legend: { position: 'bottom' }, tooltip: { mode: 'index' } },
                responsive: true,
                maintainAspectRatio: false
            }
        });
    } else {
        probabilityChart.data.datasets[0].data = [successPercent, failurePercent];
        probabilityChart.update(300); // 300ms animated update
    }
}

// Smart update for risk-return chart
function updateRiskReturnChartSmart(point) {
    const ctx = qs('riskReturnChart');
    if (!ctx) return;
    if (typeof Chart === 'undefined') return;

    // tolerance check
    const EPS = 0.01;
    if (lastRiskPoint.x !== null && Math.abs(lastRiskPoint.x - point.x) < EPS && Math.abs(lastRiskPoint.y - point.y) < EPS) {
        // nothing meaningful changed
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
                    pointRadius: 9
                }]
            },
            options: {
                animation: { duration: 300, easing: 'easeOutCubic' },
                plugins: { legend: { display: true } },
                scales: {
                    x: { title: { display: true, text: 'Volatilidad %' } },
                    y: { title: { display: true, text: 'Retorno %' } }
                },
                responsive: true,
                maintainAspectRatio: false
            }
        });
    } else {
        riskReturnChart.data.datasets[0].data = [{ x: point.x, y: point.y }];
        if (point.label) riskReturnChart.data.datasets[0].label = point.label;
        riskReturnChart.update(300);
    }
}

// Investment calculation (unchanged core logic)
async function calculateInvestment() {
    const el = qs('investmentAmount');
    const investmentAmount = parseFloat(el && el.value);
    if (!investmentAmount || investmentAmount <= 0) {
        showError('Por favor ingresa un monto válido');
        return;
    }
    if (!currentCompanyData) {
        showError('Primero selecciona una empresa');
        return;
    }

    const predictedReturnFraction = (currentCompanyData.predictions?.bestMethod === "Modelo PyTorch")
        ? currentCompanyData.predictions.model
        : currentCompanyData.predictions.historical;

    try {
        const res = await fetch('/api/calculate_investment', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ investment: investmentAmount, predictedReturn: predictedReturnFraction })
        });
        if (!res.ok) {
            let t;
            try { t = await res.text(); } catch(e) { t = res.statusText; }
            throw new Error(`HTTP ${res.status} - ${t}`);
        }
        const result = await res.json();
        if (result.error) { showError('Error al calcular: ' + result.error); return; }
        // show results
        qs('initialInvestment').textContent = formatCurrency(result.investment);
        qs('expectedReturn').textContent = formatPercentageSafe(result.predictedReturn);
        qs('profitLoss').textContent = formatCurrency(result.profit);
        qs('finalAmount').textContent = formatCurrency(result.finalAmount);

        qs('investmentResults').style.display = 'block';
        qs('investmentResults').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    } catch (err) {
        console.error(err);
        showError('Error de conexión: ' + (err.message || err));
    }
}

// Formatters (defensive)
function formatPercentageSafe(v) {
    if (v === null || v === undefined || isNaN(v)) return 'N/A';
    const n = Number(v);
    return ((n <= 1) ? n * 100 : n).toFixed(2) + '%';
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
    return new Intl.NumberFormat('es-MX', { style: 'currency', currency: 'USD', minimumFractionDigits: 2 }).format(value);
}
