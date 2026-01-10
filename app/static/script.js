/* script.js — versión final corregida y robusta */

let currentCompanyData = null;
let probabilityChart = null;
let riskReturnChart = null;

// Control de peticiones y cache
let fetchAbortController = null;
let debounceTimer = null;
const DEBOUNCE_MS = 200;
const dataCache = new Map(); // Cache de respuestas (guardamos clones)

// Spinner con delay
let spinnerTimeout = null;
const SPINNER_DELAY = 160;

// Timers para evitar colisiones al mostrar/ocultar resultados
let resultsHideTimer = null;

// Últimos valores para evitar updates innecesarios
let lastProbSuccess = null;
let lastProbFailure = null;
let lastRiskPoint = { x: null, y: null };

// Utilidades
const qs = id => document.getElementById(id);

function safeNumber(v, fallback = 0) {
    const n = Number(v);
    return (Number.isFinite(n) ? n : fallback);
}

function clone(obj) {
    try { return JSON.parse(JSON.stringify(obj)); } catch (e) { return obj; }
}

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
        results.style.display = 'none';
    }
    
    const spinner = qs('loadingSpinner');
    if (spinner) {
        spinner.style.transition = 'opacity 160ms linear';
        spinner.style.opacity = 0;
        spinner.style.pointerEvents = 'none';
        spinner.style.display = 'none';
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
    
    console.log('✅ App inicializada - modo final robusto');
    console.log('Estado inicial results display:', qs('resultsSection')?.style.display);
});

// Spinner con delay (gestionar display) — cancela hides pendientes
function showSpinnerWithDelay() {
    const spinner = qs('loadingSpinner');
    if (!spinner) return;
    // cancelar hideResults pendientes (defensa)
    if (resultsHideTimer) { clearTimeout(resultsHideTimer); resultsHideTimer = null; }

    if (spinnerTimeout) clearTimeout(spinnerTimeout);
    spinnerTimeout = setTimeout(() => {
        spinner.style.display = 'block';
        void spinner.offsetWidth;
        spinner.style.opacity = 1;
        spinner.style.pointerEvents = 'auto';
        spinner.setAttribute('aria-hidden', 'false');
        console.info('[UI] spinner mostrado por delay');
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
    setTimeout(() => { 
        try { spinner.style.display = 'none'; } catch(e) {}
    }, 200);
    console.info('[UI] spinner ocultado inmediatamente');
}

// Mostrar/ocultar resultados (gestionar display y evitar colisiones)
function showResultsSection() {
    const results = qs('resultsSection');
    if (!results) return;
    // cancelar cualquier hide pendiente
    if (resultsHideTimer) { clearTimeout(resultsHideTimer); resultsHideTimer = null; }

    results.style.display = 'block';
    void results.offsetWidth;
    const op = parseFloat(getComputedStyle(results).opacity || 0);
    if (op > 0.5) {
        console.info('[UI] results ya visible');
        return;
    }
    results.style.opacity = 1;
    results.style.transform = 'translateY(0)';
    results.style.pointerEvents = 'auto';
    console.info('[UI] results mostrados');
}

function hideResultsSection(delay = 240) {
    const results = qs('resultsSection');
    if (!results) return;
    if (resultsHideTimer) clearTimeout(resultsHideTimer);
    results.style.opacity = 0;
    results.style.transform = 'translateY(8px)';
    results.style.pointerEvents = 'none';
    resultsHideTimer = setTimeout(() => {
        try { results.style.display = 'none'; } catch(e) {}
        resultsHideTimer = null;
        console.info('[UI] results ocultados (timeout terminado)');
    }, delay);
    console.info('[UI] iniciada transición de ocultado de results');
}

// Fetch con cache y abort (robusto)
async function handleCompanyChange(event) {
    const ticker = event.target.value;
    if (!ticker) {
        hideResultsSection();
        return;
    }

    console.info('[UI] handleCompanyChange inicio para', ticker);

    // Verificar cache primero
    if (dataCache.has(ticker)) {
        try {
            console.info('[UI] Usando datos cacheados para', ticker);
            const cachedData = clone(dataCache.get(ticker));
            // cancelar spinner pending si existiera (evita que aparezca después)
            if (spinnerTimeout) { clearTimeout(spinnerTimeout); spinnerTimeout = null; }
            hideSpinnerImmediate();

            window.requestAnimationFrame(() => {
                try {
                    displayCompanyData(cachedData);
                    showResultsSection();
                } catch (err) {
                    console.error('[UI] Error mostrando datos cacheados:', err);
                    showError('Error mostrando datos cacheados: ' + err.message);
                    hideResultsSection();
                }
            });
            return;
        } catch (err) {
            console.warn('[UI] Cache corrupta, forzando fetch', err);
        }
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
        const res = await fetch(`/api/company/${encodeURIComponent(ticker)}`, { cache: 'no-store', signal });
        if (!res.ok) {
            let txt;
            try { txt = await res.text(); } catch(e){ txt = res.statusText; }
            throw new Error(`HTTP ${res.status} - ${txt}`);
        }

        const data = await res.json();
        console.info('[API] respuesta /api/company/', ticker, data);

        if (data.error) {
            showError('Error: ' + data.error);
            hideSpinnerImmediate();
            return;
        }

        // Guardar en cache (clonamos)
        try { dataCache.set(ticker, clone(data)); } catch(e) { dataCache.set(ticker, data); }

        currentCompanyData = data;

        window.requestAnimationFrame(() => {
            try {
                displayCompanyData(data);
                showResultsSection();
            } catch (err) {
                console.error('[UI] Error en displayCompanyData:', err);
                showError('Error al procesar datos: ' + err.message);
                hideResultsSection();
            } finally {
                hideSpinnerImmediate();
            }
        });

    } catch (err) {
        if (err.name === 'AbortError') {
            console.info('[UI] Fetch abortado para', ticker);
            return;
        }
        console.error('[UI] Error al cargar datos:', err);
        showError('Error al cargar datos: ' + (err.message || err));
        hideSpinnerImmediate();
        hideResultsSection();
    }
}

// Display con escenarios (defensivo)
function displayCompanyData(data) {
    if (!data || typeof data !== 'object') {
        throw new Error('Datos inválidos');
    }

    try {
        // Header
        qs('companyName').textContent = data.company || '';
        qs('companyTicker').textContent = data.ticker || '';

        // Stats
        qs('meanReturn').textContent = formatPercentageSafe(data.stats?.meanReturn);
        qs('volatility').textContent = formatPercentageSafe(data.stats?.volatility);
        qs('beta').textContent = (data.stats && !isNaN(data.stats.beta)) ? Number(data.stats.beta).toFixed(3) : 'N/A';
        qs('avgVolume').textContent = formatVolume(data.stats?.avgVolume || 0);

        // Predictions
        const hist = (data.predictions && 'historical' in data.predictions) ? data.predictions.historical : undefined;
        const mod = (data.predictions && 'model' in data.predictions) ? data.predictions.model : undefined;

        qs('historicalPrediction').textContent = formatPercentageSafe(hist);
        qs('modelPrediction').textContent = formatPercentageSafe(mod);

        toggleClass('historicalPrediction', 'positive', (hist !== undefined && hist >= 0));
        toggleClass('historicalPrediction', 'negative', (hist !== undefined && hist < 0));
        toggleClass('modelPrediction', 'positive', (mod !== undefined && mod >= 0));
        toggleClass('modelPrediction', 'negative', (mod !== undefined && mod < 0));

        const modelLoss = data.predictions?.modelLoss;
        qs('modelLoss').textContent = (modelLoss !== undefined && !isNaN(modelLoss)) ? Number(modelLoss).toFixed(6) : 'N/A';
        qs('bestMethod').textContent = data.predictions?.bestMethod || '';

        // Renderizar escenarios (protegido)
        try { displayScenarios(data.predictions?.scenarios); } 
        catch (err) { 
            console.warn('[UI] Error renderizando escenarios:', err);
            const container = qs('scenariosContainer');
            if (container) container.innerHTML = '<p style="text-align:center;color:#6b7280;">No hay análisis de escenarios disponible</p>';
        }

        // Probabilidades
        let rawSuccess = Number(data.probabilities?.success);
        let rawFailure = Number(data.probabilities?.failure);
        if (isNaN(rawSuccess)) rawSuccess = 0;
        if (isNaN(rawFailure)) rawFailure = 0;

        let ps = (rawSuccess <= 1) ? rawSuccess * 100 : rawSuccess;
        let pf = (rawFailure <= 1) ? rawFailure * 100 : rawFailure;

        // Guardar valores seguros
        ps = safeNumber(ps, 0);
        pf = safeNumber(pf, 0);

        // Si ambos son 0 => poner 50/50 para que el chart no falle
        if (ps === 0 && pf === 0) { ps = 50; pf = 50; }

        qs('probSuccess').textContent = ps.toFixed(2) + '%';
        qs('probFailure').textContent = pf.toFixed(2) + '%';

        try { updateProbabilityChartSmart(ps, pf); } catch (err) { console.error('[UI] Error updating probability chart:', err); }

        // Risk-return (usar stats agregadas; proteger NaN)
        const vol = safeNumber(data.stats?.volatility, 0);
        const mean = safeNumber(data.stats?.meanReturn, 0);
        try { updateRiskReturnChartSmart({ x: vol * 100, y: mean * 100, label: data.company || '' }); } 
        catch (err) { console.error('[UI] Error updating risk-return chart:', err); }

        // Reset inversión UI
        const invRes = qs('investmentResults');
        if (invRes) invRes.style.display = 'none';
        const invAmt = qs('investmentAmount');
        if (invAmt) invAmt.value = '';

    } catch (err) {
        console.error('[UI] Excepción en displayCompanyData general:', err);
        throw err;
    }
}

// Renderizar tarjetas de escenarios (defensivo)
function displayScenarios(scenarios) {
    const container = qs('scenariosContainer');
    if (!container) return;
    
    if (!scenarios || typeof scenarios !== 'object') {
        container.innerHTML = '<p style="text-align:center;color:#6b7280;">No hay análisis de escenarios disponible</p>';
        return;
    }

    const scenarioOrder = ['optimistic', 'neutral', 'pessimistic'];
    const scenarioIcons = { 'optimistic': '📈', 'neutral': '➡️', 'pessimistic': '📉' };
    const scenarioTitles = { 'optimistic': 'Escenario Optimista', 'neutral': 'Escenario Neutral', 'pessimistic': 'Escenario Pesimista' };

    let html = '';
    scenarioOrder.forEach(key => {
        const scenario = scenarios[key];
        if (!scenario) return;

        const returnVal = safeNumber(scenario.return, 0);
        const prob = safeNumber(scenario.probability, 0) * 100;
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

// Smart chart updates (con defensas contra NaN y recreación segura)
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

    // Asegurar números válidos
    successPercent = safeNumber(successPercent, 50);
    failurePercent = safeNumber(failurePercent, 50);

    try {
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
            // Actualizar datos de manera segura
            probabilityChart.data.datasets[0].data = [successPercent, failurePercent];
            probabilityChart.update('none');
            setTimeout(() => {
                try { probabilityChart.update(300); } catch(e) {
                    console.warn('[UI] Recreating probabilityChart due to update failure', e);
                    try { probabilityChart.destroy(); } catch(e2) {}
                    probabilityChart = null;
                    updateProbabilityChartSmart(successPercent, failurePercent);
                }
            }, 10);
        }
    } catch (err) {
        console.error('[UI] Exception in updateProbabilityChartSmart:', err);
        try { if (probabilityChart) probabilityChart.destroy(); } catch(e) {}
        probabilityChart = null;
    }
}

function updateRiskReturnChartSmart(point) {
    const ctx = qs('riskReturnChart');
    if (!ctx || typeof Chart === 'undefined') return;

    const EPS = 0.01;
    // validar números
    const x = safeNumber(point.x, 0);
    const y = safeNumber(point.y, 0);
    if (lastRiskPoint.x !== null && 
        Math.abs(lastRiskPoint.x - x) < EPS && 
        Math.abs(lastRiskPoint.y - y) < EPS) {
        return;
    }
    lastRiskPoint = { x, y };

    try {
        if (!riskReturnChart) {
            riskReturnChart = new Chart(ctx, {
                type: 'scatter',
                data: {
                    datasets: [{
                        label: point.label || '',
                        data: [{ x, y }],
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
            riskReturnChart.data.datasets[0].data = [{ x, y }];
            if (point.label) riskReturnChart.data.datasets[0].label = point.label;
            riskReturnChart.update('none');
            setTimeout(() => {
                try { riskReturnChart.update(300); } catch(e) {
                    console.warn('[UI] Recreating riskReturnChart due to update failure', e);
                    try { riskReturnChart.destroy(); } catch(e2) {}
                    riskReturnChart = null;
                    updateRiskReturnChartSmart(point);
                }
            }, 10);
        }
    } catch (err) {
        console.error('[UI] Exception in updateRiskReturnChartSmart:', err);
        try { if (riskReturnChart) riskReturnChart.destroy(); } catch(e) {}
        riskReturnChart = null;
    }
}

// Reemplazar la función calculateInvestment por esta versión
async function calculateInvestment() {
    const el = qs('investmentAmount');
    const investmentAmount = parseFloat(el?.value);

    if (!investmentAmount || investmentAmount <= 0) {
        showError('Por favor ingresa un monto válido');
        return;
    }

    if (!currentCompanyData || typeof currentCompanyData !== 'object') {
        showError('Primero selecciona una empresa');
        return;
    }

    // Obtener valores desde currentCompanyData (seguro)
    const stats = currentCompanyData.stats || {};
    const preds = currentCompanyData.predictions || {};
    const scenarios = preds.scenarios || null;
    const modelAvailable = currentCompanyData.meta ? !!currentCompanyData.meta.modelAvailable : true;

    // Mostrar resumen
    qs('initialInvestment').textContent = formatCurrency(investmentAmount);
    qs('panelRows').textContent = (currentCompanyData.meta && currentCompanyData.meta.panelRows !== undefined) ? String(currentCompanyData.meta.panelRows) : 'N/A';
    qs('modelAvailable').textContent = modelAvailable ? 'Sí' : 'No';

    // Recoger retornos (asegurarnos de que son fracciones)
    function toFraction(v) {
        if (v === null || v === undefined || isNaN(v)) return NaN;
        const n = Number(v);
        if (Math.abs(n) > 2) return n / 100.0;
        return n;
    }

    const histFrac = toFraction(preds.historical);
    const modelFrac = toFraction(preds.model);

    // Construir filas: Histórico, Modelo (si disponible), escenarios (si disponibles)
    const rows = [];

    rows.push({
        key: 'Histórico',
        frac: Number.isFinite(histFrac) ? histFrac : null,
        note: 'Basado en promedio histórico'
    });

    if (preds.model !== undefined && preds.model !== null && !isNaN(modelFrac)) {
        rows.push({
            key: 'Modelo PyTorch',
            frac: modelFrac,
            note: preds.modelLoss ? `RMSE: ${Number(preds.modelLoss).toFixed(6)}` : ''
        });
    } else {
        rows.push({
            key: 'Modelo PyTorch',
            frac: null,
            note: 'No disponible'
        });
    }

    if (scenarios && typeof scenarios === 'object') {
        const order = ['optimistic', 'neutral', 'pessimistic'];
        const titles = { optimistic: 'Escenario Optimista', neutral: 'Escenario Neutral', pessimistic: 'Escenario Pesimista' };
        order.forEach(k => {
            const s = scenarios[k];
            if (!s) return;
            const frac = toFraction(s.return);
            rows.push({
                key: titles[k],
                frac: Number.isFinite(frac) ? frac : null,
                note: `Prob ${safeNumber(s.probability, 0) * 100}%`
            });
        });
    }

    // Llenar tabla
    const tbody = qs('investmentTableBody');
    tbody.innerHTML = ''; // limpiar

    rows.forEach(r => {
        let frac = r.frac;
        let fracText = (frac === null || !Number.isFinite(frac)) ? 'N/A' : formatPercentageSafe(frac);
        let profit = (frac === null || !Number.isFinite(frac)) ? null : (investmentAmount * frac);
        let finalAmount = (profit === null) ? null : (investmentAmount + profit);

        const tr = document.createElement('tr');
        tr.style.borderBottom = '1px solid #efefef';

        const tdMethod = document.createElement('td');
        tdMethod.style.padding = '10px';
        tdMethod.style.verticalAlign = 'top';
        tdMethod.innerHTML = `<strong>${r.key}</strong><div style="color:var(--text-secondary); font-size:0.85em; margin-top:4px;">${r.note || ''}</div>`;

        const tdRet = document.createElement('td');
        tdRet.style.padding = '10px';
        tdRet.style.textAlign = 'right';
        tdRet.textContent = fracText;

        const tdProfit = document.createElement('td');
        tdProfit.style.padding = '10px';
        tdProfit.style.textAlign = 'right';
        tdProfit.textContent = (profit === null || !Number.isFinite(profit)) ? 'N/A' : formatCurrency(profit);

        const tdFinal = document.createElement('td');
        tdFinal.style.padding = '10px';
        tdFinal.style.textAlign = 'right';
        tdFinal.textContent = (finalAmount === null || !Number.isFinite(finalAmount)) ? 'N/A' : formatCurrency(finalAmount);

        tr.appendChild(tdMethod);
        tr.appendChild(tdRet);
        tr.appendChild(tdProfit);
        tr.appendChild(tdFinal);
        tbody.appendChild(tr);
    });

    // Mostrar el contenedor de resultados y hacer scroll a la tabla
    const resultsDiv = qs('investmentResults');
    if (resultsDiv) {
        resultsDiv.style.display = 'block';
        resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
}


// Formatters
function formatPercentageSafe(v) {
    if (v === null || v === undefined || isNaN(v)) return 'N/A';
    const n = Number(v);
    const percent = (Math.abs(n) <= 1 && n !== 0) ? n * 100 : n;
    const sign = percent >= 0 ? '+' : '';
    return sign + percent.toFixed(2) + '%';
}

function formatVolume(value) {
    if (value === null || value === undefined || isNaN(value) || value === 0) return '0';
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
