# Análisis Exploratorio de Datos - Extensiones Implementadas

## Análisis de Valores Faltantes y Detección de Outliers

### Descripción

Se ha implementado un módulo de análisis de valores faltantes (missing data) y detección de valores atípicos (outliers) en el notebook `eda.ipynb`, siguiendo las mejores prácticas establecidas en la literatura estadística y de análisis financiero cuantitativo.

### Componentes Implementados

#### 1. Análisis de Valores Faltantes

**Librerías utilizadas:**
- `missingno`: Visualización especializada de patrones de datos faltantes
- `pandas`: Análisis y manipulación de datos

**Análisis realizados:**
- Conteo de valores faltantes por columna
- Porcentaje de datos faltantes
- Visualización matricial de patrones (heatmap)
- Identificación de empresas con datos incompletos

**Fundamento teórico:**
- Clasificación de Little & Rubin (2002):
  - **MCAR** (Missing Completely At Random): Aleatorio puro
  - **MAR** (Missing At Random): Depende de variables observadas
  - **MNAR** (Missing Not At Random): Depende del valor faltante mismo

#### 2. Detección de Outliers

**Métodos implementados:**

##### Método 1: Rango Intercuartílico (IQR) - Tukey (1977)
```
Outlier si: x < Q1 - 1.5×IQR  o  x > Q3 + 1.5×IQR
donde IQR = Q3 - Q1
```

**Ventajas:**
- Robusto (no asume normalidad)
- No se ve afectado por los propios outliers
- Ideal para datos financieros

##### Método 2: Z-Score
```
z = (x - μ) / σ
Outlier si: |z| > 3
```

**Limitaciones:**
- Asume distribución normal
- Los outliers afectan μ y σ

### Visualizaciones Generadas

1. **missing_values_pattern.png**: 
   - Matriz de valores faltantes
   - Muestra patrones temporales y por variable

2. **outliers_analysis.png**:
   - Boxplot con outliers marcados
   - Histograma con límites IQR (líneas rojas)
   - Cuartiles Q1 y Q3 (líneas verdes)

### Resultados del Análisis

**Valores Faltantes:**
- ✓ No hay valores faltantes en columnas principales
- ⚠️ 4 empresas con menos observaciones (IPO posterior al 2018):
  - Palantir: 51 observaciones
  - Snowflake: 51 observaciones
  - Cloudflare: 63 observaciones
  - Spotify: 80 observaciones

**Outliers:**
- 84 outliers (3.50%) - Método IQR
- 33 outliers (1.37%) - Método Z-score
- Eventos identificados:
  - COVID-19 crash (marzo 2020)
  - Recuperación tecnológica (2020-2021)
  - Corrección del mercado tech (2022)

### Cómo Usar

1. Instalar dependencias:
```bash
pip install missingno numpy pandas matplotlib seaborn scipy
```

2. Ejecutar las celdas en orden:
   - Celda de imports
   - Celda de carga de datos
   - Celdas de análisis de mejora 1

3. Los gráficos se guardan automáticamente en:
   - `data/processed/missing_values_pattern.png`
   - `data/processed/outliers_analysis.png`

### Referencias

1. **Little, R.J.A. & Rubin, D.B. (2002)**. *Statistical Analysis with Missing Data*. 2nd Edition. Wiley.
   - Referencia principal sobre análisis de datos faltantes
   - Introduce clasificación MCAR/MAR/MNAR

2. **Tukey, J.W. (1977)**. *Exploratory Data Analysis*. Addison-Wesley.
   - Método IQR para detección de outliers
   - Fundamentos del análisis exploratorio moderno

3. **Barnett, V. & Lewis, T. (1994)**. *Outliers in Statistical Data*. 3rd Edition. Wiley.
   - Métodos de detección de outliers
   - Tests formales (Grubbs, Dixon)

### Mejores Prácticas Aplicadas

**Documentación**: Comentarios detallados en código siguiendo estándares de proyectos académicos

**Visualización**: Gráficos con anotaciones claras, leyendas informativas y guardado automático

**Validación**: Implementación de múltiples métodos (IQR y Z-score) para comparación robusta

**Interpretación financiera**: Análisis contextual de outliers como eventos de mercado significativos

**Reproducibilidad**: Código modular con funciones reutilizables y parámetros configurables

## Matriz de Correlación entre Activos

### Descripción

Implementación de análisis de correlación mediante coeficiente de Pearson para evaluar relaciones lineales entre retornos de activos. La matriz de correlación es fundamental para la construcción de portafolios eficientes y la gestión de riesgo según la teoría moderna de portafolios.

### Marco Teórico

**Coeficiente de correlación de Pearson:**

```
ρ_ij = Cov(R_i, R_j) / (σ_i · σ_j)
```

**Interpretación:**
- ρ = +1: Movimientos perfectamente sincronizados
- ρ = 0: Movimientos independientes
- ρ = -1: Movimientos perfectamente opuestos

**Importancia en portafolios (Markowitz, 1952):**

La varianza de un portafolio de dos activos:
```
σ_p² = w₁²σ₁² + w₂²σ₂² + 2w₁w₂σ₁σ₂ρ₁₂
```

El beneficio de diversificación aumenta cuando ρ₁₂ < 1. Activos con baja correlación proporcionan mayor reducción de riesgo.

### Componentes Implementados

#### 1. Reestructuración de Datos

- Transformación de panel largo a formato wide (fechas × empresas)
- Preparación de matriz de retornos para cálculo de correlaciones
- Verificación de dimensiones y completitud de datos

#### 2. Cálculo de Matriz de Correlación

- Correlación producto-momento de Pearson entre todos los pares de activos
- Matriz simétrica n×n donde n = número de empresas
- Estadísticas descriptivas de la distribución de correlaciones

#### 3. Visualización: Heatmap

- Mapa de calor con escala divergente (rojo-amarillo-verde)
- Anotaciones numéricas en cada celda (formato decimal)
- Codificación visual: rojo = baja correlación, verde = alta correlación
- Matriz completa de 30×30 empresas

#### 4. Identificación de Pares Extremos

**Pares con mayor correlación:**
- Identificación de activos con movimientos más sincronizados
- Interpretación: menor beneficio de diversificación

**Pares con menor correlación:**
- Identificación de activos con movimientos más independientes
- Interpretación: mayor potencial de diversificación

**Clasificación por intensidad:**
- Alta correlación: ρ ≥ 0.7
- Media correlación: 0.4 ≤ ρ < 0.7
- Baja correlación: ρ < 0.4

### Resultados del Análisis

**Estadísticas generales:**
- 30 empresas analizadas → 435 pares únicos
- Correlación promedio entre activos tecnológicos: ~0.50
- Mayoría de pares presenta correlación media-alta (típico del sector tech)

**Observaciones:**
- Empresas del mismo subsector tienden a mostrar mayores correlaciones
- Empresas con modelos de negocio diferenciados presentan menores correlaciones
- Útil para construcción de portafolios diversificados dentro del sector tecnológico

### Archivos Generados

- `correlation_heatmap.png`: Visualización completa de matriz de correlación (990 KB)

### Referencias

- Markowitz, H. (1952). "Portfolio Selection". *Journal of Finance*, 7(1), 77-91.
- Elton, E.J. & Gruber, M.J. (1997). *Modern Portfolio Theory and Investment Analysis*. Wiley.
- Campbell, J.Y., Lo, A.W., & MacKinlay, A.C. (1997). *The Econometrics of Financial Markets*. Princeton University Press.

---

## Análisis de Series Temporales Comparativas

### Descripción

Implementación de análisis comparativo de series temporales financieras mediante normalización a base 100, permitiendo evaluar el desempeño relativo de múltiples activos independientemente de sus niveles de precio absolutos.

### Marco Teórico

**Características de series financieras (Tsay, 2010):**

1. **No estacionariedad**: Presencia de tendencias estocásticas (raíces unitarias)
2. **Volatility clustering**: Agrupamiento temporal de periodos de alta volatilidad (Mandelbrot, 1963)
3. **Distribuciones leptocúrticas**: Colas más pesadas que la distribución normal
4. **Asimetría en shocks**: Movimientos negativos más abruptos que positivos (efecto apalancamiento)

**Normalización implementada:**

```
P_norm(t) = [P(t) / P(t₀)] × 100
```

Esta transformación representa el valor de una inversión de 100 unidades monetarias en el periodo inicial.

### Componentes Implementados

#### 1. Selección y Preparación de Datos

- Filtrado de empresas representativas (FAANG + alto crecimiento)
- Reestructuración de panel a formato wide (fechas × empresas)
- Normalización a base 100 para comparabilidad

#### 2. Visualización Comparativa

- Gráfico de evolución temporal de todas las series normalizadas
- Marcadores de eventos macroeconómicos significativos:
  - COVID-19 crash (marzo 2020)
  - Tech selloff (enero 2022)
- Línea de referencia en valor inicial (100)

#### 3. Métricas de Rendimiento

**Rendimiento Total:**
```
R_total = [(P_final / P_inicial) - 1] × 100
```

**CAGR (Compound Annual Growth Rate):**
```
CAGR = [(P_final / P_inicial)^(1/años) - 1] × 100
```

### Resultados del Análisis

**Periodo analizado:** Febrero 2018 - Diciembre 2024 (6.92 años)

**Rendimientos destacados:**
- Nvidia: 2,196% total (57.3% CAGR)
- Tesla: 1,725% total (52.2% CAGR)
- Apple: 499% total (29.6% CAGR)

**Observaciones:**
- Alta dispersión en rendimientos (σ = 822.5%)
- Impacto significativo de eventos macroeconómicos en trayectorias
- Volatilidad diferenciada entre empresas tecnológicas maduras y de alto crecimiento

### Archivos Generados

- `series_normalizadas.png`: Visualización de evolución comparativa

### Referencias

- Tsay, R.S. (2010). *Analysis of Financial Time Series*. 3rd Edition. Wiley.
- Mandelbrot, B. (1963). "The Variation of Certain Speculative Prices". *Journal of Business*, 36(4), 394-419.

---

## 4. Análisis Detallado de Distribución: Momentos Estadísticos

### Descripción

Se ha implementado un análisis exhaustivo de los momentos estadísticos de las distribuciones de retornos, permitiendo caracterizar completamente el comportamiento probabilístico de cada activo y evaluar desviaciones respecto a la distribución normal.

### Componentes Implementados

#### Análisis de Momentos por Empresa

**Estadísticos calculados:**

1. **Media (primer momento):** $\mu = E[X]$
   - Retorno esperado del activo
   - Medida de tendencia central

2. **Varianza (segundo momento):** $\sigma^2 = E[(X-\mu)^2]$
   - Volatilidad (riesgo total)
   - Medida de dispersión

3. **Asimetría (Skewness, tercer momento):**
   $$\gamma_1 = E\left[\left(\frac{X-\mu}{\sigma}\right)^3\right]$$
   - $\gamma_1 = 0$: Distribución simétrica
   - $\gamma_1 > 0$: Cola derecha más larga (sesgo positivo)
   - $\gamma_1 < 0$: Cola izquierda más larga (sesgo negativo)

4. **Curtosis (cuarto momento):**
   $$\gamma_2 = E\left[\left(\frac{X-\mu}{\sigma}\right)^4\right] - 3$$
   - $\gamma_2 = 0$: Curtosis normal (mesocúrtica)
   - $\gamma_2 > 0$: Colas pesadas (leptocúrtica)
   - $\gamma_2 < 0$: Colas livianas (platicúrtica)

#### Test de Jarque-Bera

**Hipótesis nula:** Los datos provienen de una distribución normal

**Estadístico:**
$$JB = \frac{n}{6}\left[S^2 + \frac{(K-3)^2}{4}\right]$$

donde $S$ es skewness, $K$ es curtosis, y $n$ es el tamaño muestral.

**Distribución bajo $H_0$:** $JB \sim \chi^2(2)$

**Criterio de decisión:**
- $p < 0.05$: Rechazar normalidad (significancia 5%)
- $p \geq 0.05$: No rechazar normalidad

### Resultados Obtenidos

#### Estadísticas Agregadas
- **Asimetría promedio:** -0.1525 (ligero sesgo negativo)
- **Curtosis promedio:** 1.0619 (colas pesadas)
- **Empresas con distribución normal:** 20 de 30 (66.7%)

#### Clasificación por Curtosis
- **Leptocúrticas** (K > 0): 22 empresas (73.3%)
  - Más eventos extremos de lo esperado bajo normalidad
  - Netflix presenta la mayor curtosis (K = 10.16)
- **Mesocúrticas** (|K| ≤ 0.5): 13 empresas (43.3%)
- **Platicúrticas** (K < -0.5): 2 empresas (6.7%)

#### Clasificación por Asimetría
- **Sesgo positivo** (S > 0.5): 2 empresas (6.7%)
  - Palantir (S = 1.30): Mayor probabilidad de retornos positivos extremos
- **Aproximadamente simétricas** (|S| ≤ 0.5): 22 empresas (73.3%)
- **Sesgo negativo** (S < -0.5): 6 empresas (20.0%)
  - Netflix (S = -2.19): Mayor probabilidad de pérdidas extremas

### Visualizaciones Generadas

#### Panel 1: Asimetría vs Curtosis
- **Tipo:** Gráfico de dispersión
- **Características:**
  - Eje X: Skewness (asimetría)
  - Eje Y: Excess Kurtosis (exceso de curtosis)
  - Color: Volatilidad (desviación estándar)
  - Líneas de referencia: Normal (S=0, K=0)
- **Interpretación:**
  - Distancia del origen indica magnitud de desviación de normalidad
  - Cuadrante superior derecho: Sesgo positivo + colas pesadas
  - Color más intenso: Mayor riesgo (volatilidad)

#### Panel 2: Distribución Empírica vs Normal Teórica
- **Tipo:** Histograma con curva superpuesta
- **Características:**
  - Barras azules: Distribución empírica (datos reales)
  - Curva roja: Distribución normal teórica
  - Estadísticos anotados: S, K, N
- **Interpretación:**
  - Barras más altas en el centro: Pico leptocúrtico
  - Barras en extremos: Colas más pesadas que la normal
  - Asimetría visible si distribución no simétrica

### Archivos Generados

- `distribution_analysis.png`: Visualización completa de momentos estadísticos

### Implicaciones Financieras

1. **Rechazo de normalidad (33.3% de empresas):**
   - Invalida supuestos de modelos basados en normalidad (CAPM básico)
   - Requiere uso de modelos alternativos (t-Student, distribuciones estables)

2. **Exceso de curtosis generalizado:**
   - Subestimación de VaR (Value at Risk) con modelos normales
   - Necesidad de ajustes en cálculo de primas de riesgo

3. **Asimetría negativa predominante:**
   - Mayor probabilidad de crashes que de alzas equivalentes
   - Justifica demanda de opciones de protección (puts)

### Referencias

- Jarque, C.M. & Bera, A.K. (1980). "Efficient tests for normality, homoscedasticity and serial independence of regression residuals". *Economics Letters*, 6(3), 255-259.
- Mandelbrot, B. (1963). "The variation of certain speculative prices". *Journal of Business*, 36(4), 394-419.
- Fama, E.F. (1965). "The behavior of stock-market prices". *Journal of Business*, 38(1), 34-105.

---

## 5. Análisis de Drawdown (Caídas Máximas)

### Descripción

Se ha implementado un análisis exhaustivo de drawdown para cuantificar las pérdidas máximas desde picos históricos, proporcionando una métrica fundamental de riesgo que complementa la volatilidad tradicional en la evaluación de activos financieros.

### Componentes Implementados

#### Definiciones Fundamentales

**Drawdown en el tiempo t:**
$$DD_t = \frac{P_t - \max_{s \leq t} P_s}{\max_{s \leq t} P_s}$$

donde $P_t$ es el precio actual y $\max_{s \leq t} P_s$ es el máximo histórico hasta ese momento.

**Maximum Drawdown (MDD):**
$$MDD = \min_{t} DD_t$$

Representa la peor caída histórica observada desde cualquier pico anterior.

#### Métricas Calculadas por Empresa

1. **Max_Drawdown (MDD):**
   - Peor caída porcentual desde un máximo histórico
   - Valor más negativo de toda la serie de drawdowns

2. **Avg_Drawdown:**
   - Promedio de todos los drawdowns observados
   - Indica la severidad típica de las caídas

3. **Time_Underwater:**
   - Número de meses cotizando bajo el máximo histórico
   - Refleja la persistencia de las pérdidas

4. **Pct_Underwater:**
   - Porcentaje del periodo total bajo agua
   - Indica la frecuencia relativa de estar en drawdown

### Resultados Obtenidos

#### Estadísticas Agregadas
- **MDD promedio:** -51.66%
- **MDD mediano:** -49.55%
- **Peor MDD:** -81.75% (Palantir)
- **Mejor MDD:** -24.50% (IBM)
- **Tiempo bajo agua promedio:** 59.8 meses (75.1% del periodo)

#### Clasificación por Severidad
- **Severo** (MDD ≤ -50%): 14 empresas (46.7%)
  - Palantir, Cloudflare, Meta Platforms, Spotify, Netflix
  - Requieren retornos superiores a 100% para recuperación
- **Moderado** (-50% < MDD ≤ -30%): 15 empresas (50.0%)
  - Mayoría del sector tecnológico
- **Leve** (MDD > -30%): 1 empresa (3.3%)
  - IBM: -24.50% (empresa más estable)

#### Casos Destacados

**Tesla:**
- MDD: -67.72% (Diciembre 2022)
- Tiempo bajo agua: 67 meses (80.7%)
- Retorno requerido: 209.8%

**Nvidia:**
- MDD: -62.82% (Septiembre 2022)
- Tiempo bajo agua: 54 meses (65.1%)
- Retorno requerido: 169.0%

**Apple:**
- MDD: -30.46% (Diciembre 2018)
- Tiempo bajo agua: 56 meses (67.5%)
- Retorno requerido: 43.8%

**Microsoft:**
- MDD: -30.53% (Octubre 2022)
- Tiempo bajo agua: 48 meses (57.8%)
- Retorno requerido: 43.9%

### Visualizaciones Generadas

#### Panel de Series Temporales de Drawdown
- **Tipo:** 4 subplots verticales (Tesla, Nvidia, Apple, Microsoft)
- **Características:**
  - Área roja sombreada: Periodos bajo máximo histórico
  - Profundidad del área: Magnitud del drawdown
  - Marcador triangular invertido (▼): Punto de MDD
  - Líneas de referencia: -20% (moderado), -50% (severo)
  - Anotaciones con fecha y porcentaje del MDD
- **Interpretación:**
  - Visualiza evolución temporal de pérdidas acumuladas
  - Identifica periodos de recuperación vs periodos de caída
  - Permite comparar patrones de drawdown entre activos

### Implicaciones para Gestión de Riesgo

1. **Asimetría de recuperación:**
   - Una caída de 50% requiere retorno de 100% para recuperar
   - Ejemplo: Palantir con MDD de -81.75% necesita +440% para volver al pico

2. **Persistencia de pérdidas:**
   - 75.1% del tiempo promedio bajo máximo histórico
   - Indica alta volatilidad y ciclos prolongados de recuperación

3. **Planificación de capital:**
   - El MDD histórico define el colchón mínimo de liquidez necesario
   - Inversores deben tolerar caídas superiores al 50% en sector tech

4. **Comparación riesgo-retorno:**
   - Empresas de alto crecimiento (Tesla, Nvidia) muestran MDD severos
   - Empresas estables (IBM, Microsoft) tienen drawdowns moderados

### Limitaciones del Análisis

1. **Dependencia temporal:** MDD varía según periodo analizado
2. **No captura frecuencia:** Múltiples caídas de 25% vs una de 50%
3. **No garantiza protección futura:** Historia no predice eventos extremos

### Archivos Generados

- `drawdown_analysis.png`: Visualización de series temporales de drawdown (348 KB)

### Referencias

- Magdon-Ismail, M. & Atiya, A. (2004). "Maximum Drawdown". *Risk Magazine*, 17(10), 99-102.
- Chekhlov, A., Uryasev, S. & Zabarankin, M. (2005). "Drawdown Measure in Portfolio Optimization". *International Journal of Theoretical and Applied Finance*, 8(1), 13-58.
- Calmar, T.W. (1991). Calmar Ratio: A smoother tool. *Futures*, 20(1), 40.

---

### Extensiones Futuras

Módulos adicionales a considerar para análisis completo:

- Descomposición temporal (tendencia, estacionalidad, ruido)
- Rolling statistics (ventanas móviles de media y volatilidad)
- Visualización multivariante (pair plots, scatter matrices)

---

**Autor:** AlphaTech-Analyzer Project  
**Fecha:** Enero 2026  
**Versión:** 1.2
