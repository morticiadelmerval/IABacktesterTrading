# Quant Strategy Backtester & Dashboard

Este proyecto es un laboratorio interactivo de análisis cuantitativo (Backtesting) construido para simular estrategias de trading sobre los activos más importantes de Wall Street (SPY, QQQ, MSFT, GOOG, etc.) con una profundidad de **30 años de historia**.

Combina un motor de cálculo de alta velocidad escrito en **Python** con un **Dashboard Interactivo en Vanilla JS/HTML**, operando bajo una arquitectura Cliente-Servidor local.

## ✨ Características Principales

* **Motor de Simulación a 30 Años:** Descarga y cachea la historia completa de precios mediante `yfinance` para **18 activos líderes** (Acciones, ETFs, Oro y Cripto).
* **13 Estrategias Algorítmicas Incorporadas:** Incluye desde cruces de medias y "Crash Guards Macro" tradicionales, hasta un **Oráculo de Inteligencia Artificial** basado en el modelo *TimesFM* de Google (optimizaciones S11, S12 y S13).
* **Precios en Vivo e Interfaz Dinámica:** El Dashboard hace un polling asíncrono cada 10 segundos inyectando el último precio del mercado y notificando cambios con destellos visuales.
* **Cálculo Dinámico de Comisiones:** El Dashboard web permite a los usuarios ingresar la comisión real de su broker (ej. 0.4%). El servidor backend en Python recibe este dato vía API, recalcula los 30 años de trades, y re-ordena el ranking de estrategias al instante.
* **Visualización de Curvas de Capital (Equity Curves):** Gráficos comparativos frente al rendimiento de "Buy & Hold" (Comprar y Mantener) utilizando `Chart.js`.
* **Exportación a TradingView (Pine Script):** Cada estrategia te entrega su propio código generado dinámicamente en Pine Script v5 para configurar alertas en tiempo real.

## 🚀 Requisitos de Instalación

Para una instalación ligera (solo algoritmos tradicionales):
```bash
pip install yfinance pandas numpy
```

Para usar la versión con Inteligencia Artificial (NVIDIA RTX 3000/4000 requerida):
Ejecuta directamente el archivo **`install_windows_nvidia.bat`**. Este script te creará un entorno virtual e instalará PyTorch (CUDA 12.1), HuggingFace Transformers y TimesFM de forma automatizada.

## 💻 ¿Cómo ejecutar el Dashboard?

1. Abre tu terminal o consola de comandos en el directorio del proyecto.
2. (Opcional si usas IA) Pre-calcula las señales del oráculo local:
```bash
python precalculate_timesfm.py
```
3. Ejecuta el servidor API local y Dashboard:
```bash
python server.py
```
*(Opcional en Windows: haz doble clic en `run_dashboard.bat`)*
4. Abre tu navegador web y ve a la dirección: **http://localhost:8000**

## 📐 Arquitectura del Proyecto

* `server.py`: Servidor web ligero nativo que atiende los archivos estáticos y expone la API `/api/recalculate?commission=X`.
* `backtester.py`: El corazón matemático del proyecto. Contiene las definiciones de las estrategias (Señales de Entrada/Salida), descarga la data y realiza la simulación vectorizada de portafolios.
* `app.js` / `index.html` / `styles.css`: Interfaz de usuario interactiva y responsiva. Lee los resultados JSON que devuelve el servidor y pinta el dashboard.
* `.data_cache/`: Carpeta (ignorada por Git) donde el script guarda temporalmente los archivos CSV de Yahoo Finance para no sobrecargar la red y hacer los recálculos dinámicos en cuestión de microsegundos.

## ⚠️ Descargo de Responsabilidad

Este código ha sido creado con propósitos estrictamente **educativos y de investigación matemática**. Las estrategias de *Backtesting* asumen ejecuciones perfectas que no siempre se cumplen en los mercados reales debido a la liquidez, *slippage* y eventos de cisne negro. No constituye consejo financiero ni recomendación de inversión. Opere bajo su propio riesgo.
