# Quant Strategy Backtester & Dashboard v0.2

Este proyecto es un laboratorio interactivo de análisis cuantitativo (Backtesting) construido para simular estrategias de trading sobre los activos más importantes de Wall Street con una profundidad de **30 años de historia**.

Combina un motor de cálculo de alta velocidad escrito en **Python** con un **Dashboard Interactivo en Vanilla JS/HTML**, operando bajo una arquitectura Cliente-Servidor local.

## ✨ Características Principales (v0.2)

* **Motor de Simulación a 30 Años:** Descarga y cachea la historia completa de precios mediante `yfinance` para **18 activos líderes** (Acciones, ETFs, Oro y Cripto).
* **20 Estrategias Algorítmicas Incorporadas:** Incluye 11 estrategias "Simples" (SS) con cruces de medias, momentum y "Crash Guards Macro", y 9 estrategias de "Inteligencia Artificial" (AIS) impulsadas por Google TimesFM, IBM TSPulse y MiniRocket.
* **Stop Loss Dinámico y Estricto:** Integra cálculos de Stop Loss fijos por porcentaje (ej. -10%, -15%) parametrizables individualmente por estrategia para proteger el capital.
* **Dashboard "Mejor por Activo":** Una nueva sección que evalúa automáticamente la estrategia con mejor rendimiento histórico para cada instrumento del portafolio.
* **Precios en Vivo e Interfaz Dinámica:** El Dashboard hace un polling asíncrono inyectando el último precio del mercado y notificando cambios con destellos visuales.
* **Cálculo Dinámico de Comisiones:** El servidor backend recalcula los 30 años de trades en milisegundos si ajustas las comisiones del broker.
* **Exportación a TradingView (Pine Script):** Cada estrategia te entrega su propio código generado dinámicamente en Pine Script v5 para configurar alertas en tiempo real (con inclusión de Stop Loss automático).

## 🚀 Requisitos de Instalación

Para la instalación del backtester principal y soporte a IA:
```bash
uv sync  # (Recomendado) Usa uv para manejar las dependencias
# O con pip:
# pip install -r requirements.txt
```

Para usar las estrategias de Inteligencia Artificial (AIS01 a AIS09) se requiere aceleración por hardware (NVIDIA GPU).

## 💻 ¿Cómo ejecutar el Dashboard?

1. Abre tu terminal en el directorio del proyecto.
2. (Opcional) Pre-calcula las señales de los oráculos locales:
```bash
python models/precalculate_timesfm.py
python models/precalculate_tspulse_multi.py
```
3. Ejecuta el motor completo para generar el pipeline:
```bash
python backtester.py
```
4. Ejecuta el servidor API local y Dashboard:
```bash
python web/server.py
```
5. Abre tu navegador web y ve a la dirección: **http://localhost:8000**

## 📐 Arquitectura del Proyecto

* `web/server.py`: Servidor web ligero que atiende los archivos estáticos y expone la API de comisiones.
* `backtester.py`: El corazón matemático del proyecto. Contiene las definiciones de las 20 estrategias, y realiza la simulación vectorizada.
* `web/app.js` / `web/index.html` / `web/styles.css`: Interfaz de usuario interactiva y responsiva.
* `models/`: Directorio dedicado a la carga, pre-entrenamiento e inferencia de modelos de Machine Learning (TimesFM, TSPulse, MiniRocket, XGBoost).
* `.data_cache/`: Carpeta (ignorada por Git) temporal para los CSV de Yahoo Finance.

## ⚠️ Descargo de Responsabilidad

Este código ha sido creado con propósitos estrictamente **educativos y de investigación matemática**. No constituye consejo financiero ni recomendación de inversión. Opere bajo su propio riesgo.
