# Quant Strategy Backtester & Dashboard v0.2.5

Este proyecto es un laboratorio interactivo de análisis cuantitativo (Backtesting) construido para simular estrategias de trading sobre los activos más importantes de Wall Street con una profundidad de **30 años de historia**.

Combina un motor de cálculo de alta velocidad escrito en **Python** con un **Dashboard Interactivo en Vanilla JS/HTML**, operando bajo una arquitectura Cliente-Servidor local.

## ✨ Características Principales

* **Motor de Simulación a 30 Años:** Descarga y cachea la historia completa de precios mediante `yfinance` para **18 activos líderes** (Acciones, ETFs, Oro y Cripto).
* **26 Estrategias Algorítmicas Incorporadas:** Incluye 15 estrategias "Simples" (SS01 a SS15) con cruces de medias, momentum, estocásticos y "Crash Guards Macro", y 11 estrategias de "Inteligencia Artificial" (AIS01 a AIS11) impulsadas por Google TimesFM, IBM TSPulse, MiniRocket y ensambles Multi-IA.
* **Motor Multi-IA Combinatorio (Nuevas AIS10 & AIS11):** Integración simultánea y ponderada de 5 modelos de IA avanzados junto a 13 indicadores técnicos con normalización sigmoidal (0..100).
* **Stop Loss Dinámico y Filtro Estricto de Re-entrada (-15%):** Integra cálculos de Stop Loss fijos por porcentaje (ej. -10%, -15%) parametrizables individualmente por estrategia para proteger el capital. En las estrategias líderes, tras tocar el stop loss se exige que el precio supere la **SMA 20** antes de permitir un reingreso, evitando atrapar cuchillos cayendo.
* **Dashboard "Mejor por Activo":** Una nueva sección que evalúa automáticamente la estrategia con mejor rendimiento histórico para cada instrumento del portafolio.
* **Precios en Vivo e Interfaz Dinámica:** El Dashboard hace un polling asíncrono inyectando el último precio del mercado y notificando cambios con destellos visuales.
* **Filtros por Fechas y Cálculo Dinámico de Comisiones:** El servidor backend recalcula los 30 años de trades en milisegundos si ajustas las comisiones del broker o filtras por rangos de fechas personalizados.
* **Exportación a TradingView (Pine Script):** Cada estrategia te entrega su propio código generado dinámicamente en Pine Script v5 para configurar alertas en tiempo real (con inclusión de Stop Loss automático y filtros de recuperación SMA20).

## 🚀 Requisitos de Instalación

**Para usuarios de Windows con tarjeta gráfica NVIDIA (Recomendado):**
Hemos preparado un instalador automático que creará tu entorno virtual, instalará todas las dependencias necesarias y descargará automáticamente los modelos pre-entrenados de IA (TimesFM y TSPulse) que pesan varios Gigabytes. 
Solo debes dar doble clic en el archivo o ejecutar en la consola:
```bash
install_windows_nvidia.bat
```

*(Si eres un usuario avanzado o usas Linux/Mac, puedes revisar el código del instalador para replicar las dependencias o correr simplemente `uv sync`).*

Para aprovechar todas las estrategias de Inteligencia Artificial (AIS01 a AIS11) se requiere aceleración por hardware (NVIDIA GPU).

## 💻 ¿Cómo ejecutar el Dashboard?

1. Abre tu terminal en el directorio del proyecto.
2. **Actualizar la Inteligencia Artificial (Diario/Semanal):**
   Para pre-calcular las predicciones de los modelos locales (Google TimesFM, IBM TSPulse, MiniRocket, XGBoost Stack), ejecuta el archivo de actualización. Este proceso utiliza una **Caché Incremental Inteligente**, por lo que solo calculará los días nuevos que falten, demorando solo segundos si se corre a diario:
```bash
actualizar_ia.bat
```
3. Ejecuta el motor completo para cargar los datos y evaluar el rendimiento:
```bash
uv run python backtester.py
```
4. Ejecuta el servidor API local y levanta el Dashboard:
```bash
uv run python web/server.py
```
5. Abre tu navegador web y ve a la dirección: **http://localhost:8000**

## 📐 Arquitectura del Proyecto

* `web/server.py`: Servidor web ligero que atiende los archivos estáticos y expone la API de comisiones y cálculo por fechas.
* `backtester.py`: El corazón matemático del proyecto. Contiene las definiciones de las 26 estrategias, y realiza la simulación vectorizada.
* `web/app.js` / `web/index.html` / `web/styles.css`: Interfaz de usuario interactiva y responsiva.
* `models/`: Directorio dedicado a la carga, pre-entrenamiento e inferencia de modelos de Machine Learning (TimesFM, TSPulse, MiniRocket, XGBoost).
* `.data_cache/`: Carpeta (ignorada por Git) temporal para los CSV de Yahoo Finance.

## 📜 Historial de Versiones

### Versión 0.2.5 (Detalles respecto a la v0.2.0)
* **Expansión del Portafolio Algorítmico (de 20 a 26 estrategias)**:
  * Incorporación de las estrategias **SS12, SS13, SS14 y SS15**: Nuevas fórmulas multifactoriales combinando señales de volumen, momentum, estocásticos y modelos de IA, calibradas específicamente para entornos con 0% y 0.4% de comisión.
  * Incorporación de las estrategias super-ensamble **AIS10 y AIS11**: Motores Multi-IA combinatorios que evalúan miles de ponderaciones de forma vectorizada.
* **Motor Optimizado Multi-IA (`optimize_ais10.py`)**:
  * Integración en paralelo de 5 señales avanzadas de Machine Learning (`TimesFM`, `TSPulse`, `MiniRocket GPU`, `MiniRocket Bin`, `XGBoost Stack`) escaladas dinámicamente al rango 0..100 mediante funciones sigmoidales y probabilísticas.
  * Re-ponderación dinámica en tiempo real que gestiona fechas de inicio escalonadas (ej. 1996 vs 2005 vs 2012) sin distorsionar el historial pre-existente.
* **Estrategia Líder Absoluta (AIS10 - Puesto #1)**:
  * Optimizada para 0% comisiones, alcanza un **rendimiento promedio de 4,139.7%** batiendo a Buy & Hold en 15 de 18 activos. Fórmula: `AI_TIMESFM` (50%), `ROC_3_NORM` (25%), `AI_TSPULSE` (20%), `VOL_NORM` (5%).
* **Consistencia Máxima en Portafolio (AIS11 - Puesto #6)**:
  * Optimizada para comisiones reales del 0.4%, supera a Buy & Hold en **17 de 18 activos** (éxito en el 94.4% de las acciones y ETFs). Fórmula: `AI_TSPULSE` (35%), `ATR_NORM` (30%), `ROC_3_NORM` (25%), `ROC_NORM` (10%).
* **Seguro de Re-entrada Post Stop-Loss (SMA 20)**:
  * Todas las estrategias con Stop Loss estricto del -15% exigen ahora una doble condición de recuperación (`Close > SMA20` + señal activa de compra) para reingresar al mercado tras un corte de pérdidas, eliminando el riesgo de atrapar cuchillos cayendo durante crashes bursátiles.
* **Mejoras en el Dashboard y API**:
  * Capacidad de filtrado por rangos de fechas personalizados directamente desde la interfaz web.
  * Homologación y alineación de calendarios bursátiles con `ffill` entre los optimizadores y el motor principal.

## ⚠️ Descargo de Responsabilidad

Este código ha sido creado con propósitos estrictamente **educativos y de investigación matemática**. No constituye consejo financiero ni recomendación de inversión. Opere bajo su propio riesgo.
