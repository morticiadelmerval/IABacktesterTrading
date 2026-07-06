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

## 🚀 Cómo Instalar y Ejecutar el Dashboard

Para empezar a utilizar el laboratorio cuantitativo en Windows (se recomienda tarjeta gráfica NVIDIA), sigue estos 3 sencillos pasos en orden:

### 1️⃣ Instalar dependencias y modelos
Haz doble clic o ejecuta en tu consola el instalador automático. Este script creará tu entorno virtual, instalará todas las librerías necesarias y descargará los pesos pre-entrenados de Inteligencia Artificial (Google TimesFM e IBM TSPulse):
```bash
install_windows_nvidia.bat
```

### 2️⃣ Actualizar las señales de IA (Diario/Semanal)
Antes de abrir el dashboard, ejecuta el actualizador para descargar las velas más recientes de la Bolsa y entrenar los modelos locales (MiniRocket y XGBoost). Gracias a su **Caché Incremental Inteligente**, en el uso diario este proceso solo toma unos segundos:
```bash
actualizar_ia.bat
```

### 3️⃣ Abrir el Dashboard Interactivo
Para iniciar el servidor local y abrir la interfaz gráfica automáticamente en tu navegador web, simplemente ejecuta:
```bash
run_dashboard.bat
```

> [!TIP]
> **Actualización desde la Web:** Una vez dentro del Dashboard, verás un **banner superior** con un botón interactivo que te permite actualizar los datos del mercado y re-calcular las señales de los modelos de IA directamente desde tu navegador con un solo clic.

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
* **Búsqueda Exhaustiva Total (3,060 Combinaciones y 13 Millones de Simulaciones)**:
  * Optimización simultánea y sistemática de las 4 estrategias supremas (**AIS10, AIS11, SS14 y SS15**) evaluando el 100% de los 3,060 grupos posibles de 4 indicadores (repartiendo 2,345 grupos con IA para AIS10/AIS11 y 715 grupos 100% técnicos para SS14/SS15).
  * Carga dinámica en tiempo real en `backtester.py` desde archivos JSON sincronizados, garantizando 100% de homologación entre el motor de búsqueda y el backtester.
* **Estrategias Líderes Absolutas (Resultados Finales de Optimización)**:
  * **AIS10 (IA - 0% comisiones - Puesto #1)**: Bate a Buy & Hold en **18 de 18 activos (100% de éxito en todo el portafolio)**, alcanzando un **rendimiento promedio de 4,438%**. Fórmula ganadora: `AI_MINIROCKET_BIN` (10%), `AI_TIMESFM` (55%), `KONCORDE_MD` (15%), `ROC_NORM` (20%). Entry: 60, Exit: 15.
  * **AIS11 (IA - 0.4% comisiones)**: Bate a Buy & Hold en **17 de 18 activos**, alcanzando un **rendimiento promedio de 3,842%**. Fórmula ganadora: `AI_MINIROCKET_GPU` (5%), `AI_TIMESFM` (20%), `AI_TSPULSE` (65%), `ROC_3_NORM` (10%). Entry: 55, Exit: 5.
  * **SS14 (Simple/Tradicional - 0% comisiones)**: Bate a Buy & Hold en **17 de 18 activos**, alcanzando un **rendimiento promedio de 4,349%**. Fórmula ganadora: `ROC_3_NORM` (40%), `ROC_NORM` (10%), `VOL_EXT_NORM` (5%), `ATR_NORM` (45%). Entry: 55, Exit: 45.
  * **SS15 (Simple/Tradicional - 0.4% comisiones)**: Bate a Buy & Hold en **17 de 18 activos**, alcanzando un **rendimiento promedio neto de 3,693%** con 302 operaciones totales (~16 trades por activo en 30 años). Fórmula ganadora: `ATR_NORM` (50%), `VOL_EXT_NORM` (50%). Entry: 55, Exit: 10.
* **Seguro de Re-entrada Post Stop-Loss (SMA 20)**:
  * Todas las estrategias con Stop Loss estricto del -15% exigen ahora una doble condición de recuperación (`Close > SMA20` + señal activa de compra) para reingresar al mercado tras un corte de pérdidas, eliminando el riesgo de atrapar cuchillos cayendo durante crashes bursátiles.
* **Mejoras en el Dashboard y API**:
  * Capacidad de filtrado por rangos de fechas personalizados directamente desde la interfaz web.
  * Homologación y alineación de calendarios bursátiles con `ffill` entre los optimizadores y el motor principal.

## 📄 Licencia

Este proyecto se distribuye bajo la **Licencia MIT (MIT License)**.
Esto significa que tienes **libertad total** para descargar, usar, modificar, hacer forks, integrar e incluso comercializar o ganar dinero con este software sin ninguna restricción ni necesidad de pagar regalías o pedir permiso, siempre y cuando mantengas el aviso de copyright original del archivo `LICENSE`.

## ⚠️ Descargo de Responsabilidad

Este código ha sido creado con propósitos estrictamente **educativos y de investigación matemática**. No constituye consejo financiero ni recomendación de inversión. Opere bajo su propio riesgo.
