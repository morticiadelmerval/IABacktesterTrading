# Plan de Implementación: Remediación y Optimización de TSPulse + MiniRocket

## Contexto

Este plan surge del cruce entre el documento [Análisis Profundo de Modelos de Trading IA.pdf](file:///e:/ProyectosIA/TradingViewPersonalsIndicators/models/Análisis%20Profundo%20de%20Modelos%20de%20Trading%20IA.pdf) y una auditoría exhaustiva de los 7 scripts de entrenamiento/inferencia + la lógica de negocio en [backtester.py](file:///e:/ProyectosIA/TradingViewPersonalsIndicators/backtester.py).

---

## Diagnóstico: Estado Actual vs. Estándar Institucional

### Hallazgos Críticos de la Auditoría

| Archivo | Data Leakage | Normalización | Walk-Forward | Severidad |
|---|---|---|---|---|
| [finetune_tspulse.py](file:///e:/ProyectosIA/TradingViewPersonalsIndicators/models/finetune_tspulse.py) | 🔴 Sin holdout | ❌ Ninguna | ❌ No | **CRÍTICA** |
| [finetune_tspulse_multi.py](file:///e:/ProyectosIA/TradingViewPersonalsIndicators/models/finetune_tspulse_multi.py) | 🔴 Sin holdout | 🟡 Parcial | ❌ No | **CRÍTICA** |
| [precalculate_tspulse.py](file:///e:/ProyectosIA/TradingViewPersonalsIndicators/models/precalculate_tspulse.py) | 🔴 Hereda leakage | ❌ Ninguna | ❌ No | **CRÍTICA** |
| [precalculate_tspulse_multi.py](file:///e:/ProyectosIA/TradingViewPersonalsIndicators/models/precalculate_tspulse_multi.py) | 🔴 Hereda leakage | 🟡 Parcial | ❌ No | **CRÍTICA** |
| [train_minirocket.py](file:///e:/ProyectosIA/TradingViewPersonalsIndicators/models/train_minirocket.py) | ✅ Walk-forward OOS | ✅ StandardScaler | ✅ Sí | BAJA |
| [train_minirocket_gpu.py](file:///e:/ProyectosIA/TradingViewPersonalsIndicators/models/train_minirocket_gpu.py) | ✅ Walk-forward OOS | 🟡 Implícita | ✅ Sí | BAJA |
| [precalculate_timesfm.py](file:///e:/ProyectosIA/TradingViewPersonalsIndicators/models/precalculate_timesfm.py) | ✅ No fine-tune | ✅ Z-score/ventana | N/A | MEDIA (bugs) |

> [!CAUTION]
> **El problema más grave**: TSPulse (S14, S15, S16) entrena con el 100% de los datos históricos sin ningún holdout ni walk-forward. Luego, el precalculador genera predicciones sobre **los mismos datos** que el modelo ya memorizó. Esto significa que los retornos de ~3679% de S15 podrían estar **severamente inflados** por data leakage. Hasta que no se corrija esto, no podemos confiar en que S15 realmente merezca el Puesto 3 del ranking.

> [!IMPORTANT]
> **MiniRocket (S17/S18)** ya implementa Walk-Forward Expanding Window correctamente. Sus resultados de backtest son confiables. Las mejoras propuestas para MiniRocket son de **optimización**, no de remediación.

---

## Propuesta de Cambios (6 Fases)

---

### FASE 1 — Remediación Crítica de TSPulse (Data Leakage)
**Prioridad:** 🔴 MÁXIMA — Sin esto, los resultados de S14/S15/S16 no son confiables.

#### [MODIFY] [finetune_tspulse.py](file:///e:/ProyectosIA/TradingViewPersonalsIndicators/models/finetune_tspulse.py)
**Problema actual:** Entrena con 100% de los datos. No hay train/test split.
**Cambio propuesto:** Implementar **Walk-Forward Expanding Window** idéntico al que ya usa MiniRocket:

```python
# ANTES (problemático):
dataset = TSPulseDataset(all_windows, all_targets)
# Entrena con TODO, sin separar

# DESPUÉS (correcto):
min_train_pct = 0.30  # Primer 30% = solo training
step_pct = 0.05       # Avanzar 5% por iteración

for train_end in walk_forward_splits(len(data), min_train_pct, step_pct):
    train_data = data[:train_end]
    # Fine-tune solo con datos del pasado
    # El modelo se guarda/actualiza en cada paso
```

**Detalle técnico:**
- En cada paso del walk-forward, el modelo se re-inicializa desde el checkpoint base de IBM y se fine-tunea solo con los datos `[0 : train_end]`.
- Se guardan múltiples checkpoints intermedios (uno por ventana) para que el precalculador pueda usar el modelo correcto para cada período temporal.
- Alternativa más eficiente: fine-tunear **incrementalmente** (sin re-inicializar) avanzando la ventana, aceptando un sesgo leve a cambio de velocidad.

#### [MODIFY] [finetune_tspulse_multi.py](file:///e:/ProyectosIA/TradingViewPersonalsIndicators/models/finetune_tspulse_multi.py)
**Mismos cambios que el univariado**, adaptados para 4 canales.

**Cambio adicional — Normalización por ventana:**
```python
# ANTES (problemático):
features[:, 1] = rsi / 100.0   # Escala fija
features[:, 2] = rel_vol        # Sin escalar (outliers posibles)
features[:, 3] = sma_dist        # Sin escalar

# DESPUÉS (correcto):
# Z-score rolling por cada ventana de contexto (512 días)
for window in windows:
    for channel in [1, 2, 3]:  # No el canal 0 (retornos ya son ~estacionarios)
        mean = window[:, channel].mean()
        std = window[:, channel].std() + 1e-8
        window[:, channel] = (window[:, channel] - mean) / std
```

---

### FASE 2 — Remediación de Precalculadores TSPulse
**Prioridad:** 🔴 MÁXIMA — Depende de Fase 1.

#### [MODIFY] [precalculate_tspulse.py](file:///e:/ProyectosIA/TradingViewPersonalsIndicators/models/precalculate_tspulse.py)
**Problema actual:** Usa un único modelo fine-tuneado para predecir sobre todos los datos históricos (incluyendo datos que el modelo ya vio en training).

**Cambio propuesto:** Implementar inferencia walk-forward sincronizada con los checkpoints de la Fase 1:

```python
# Para cada paso walk-forward:
#   1. Cargar el checkpoint que fue entrenado con datos [0:train_end]
#   2. Generar predicciones SOLO para el período [train_end:next_train_end]
#   3. Guardar esas predicciones como out-of-sample (OOS)
#
# El resultado final: tspulse_signals.json solo contiene predicciones OOS
```

**Alternativa simplificada (recomendada por eficiencia):**
En lugar de múltiples checkpoints, usar un **único fine-tune** con los primeros N% de datos y generar señales solo para el período posterior. Esto sacrifica algo de datos de señal pero es mucho más simple:

```python
TRAIN_CUTOFF = 0.60  # Fine-tune con el primer 60% de los datos
# Generar señales SOLO para el 40% restante
# Las fechas del primer 60% quedan sin señal (rellenadas con 0.0 en el backtester → neutras)
```

#### [MODIFY] [precalculate_tspulse_multi.py](file:///e:/ProyectosIA/TradingViewPersonalsIndicators/models/precalculate_tspulse_multi.py)
**Mismos cambios** que la versión univariada.

---

### FASE 3 — Corrección de Bugs en TimesFM
**Prioridad:** 🟡 MEDIA — Afecta la calidad de las señales de S11/S12/S13.

#### [MODIFY] [precalculate_timesfm.py](file:///e:/ProyectosIA/TradingViewPersonalsIndicators/models/precalculate_timesfm.py)
**Bug 1 — Desnormalización incorrecta:** Las variables `seq_mean` y `seq_std` se sobreescriben en el loop y al desnormalizar se usa la media/std de la **última** secuencia del batch en lugar de la correspondiente a cada predicción.

```python
# ANTES (buggy):
for j, pf in enumerate(point_forecasts):
    pred_price = (pf[-1] * seq_std) + seq_mean  # ← usa stats del último elemento

# DESPUÉS (correcto):
# Guardar mean/std de cada secuencia del batch
batch_means = []
batch_stds = []
for seq in batch_sequences:
    batch_means.append(seq.mean())
    batch_stds.append(seq.std() + 1e-8)

for j, pf in enumerate(point_forecasts):
    pred_price = (pf[-1] * batch_stds[j]) + batch_means[j]  # ← stats correctas
```

**Bug 2 — Índice de `current_price`:** El cálculo del índice actual es confuso y propenso a errores off-by-one. Refactorizar para almacenar el precio actual junto con la secuencia de input.

---

### FASE 4 — Optimización de MiniRocket (Ensamble con XGBoost)
**Prioridad:** 🟢 MEJORA — MiniRocket ya funciona bien, esto lo potencia.

El PDF recomienda enérgicamente combinar las 10,000 features de MiniRocket con un clasificador no-lineal (XGBoost o CatBoost) en lugar del RidgeClassifierCV lineal actual. Esto permitiría capturar interacciones no-lineales entre las features convolucionales.

#### [MODIFY] [train_minirocket.py](file:///e:/ProyectosIA/TradingViewPersonalsIndicators/models/train_minirocket.py)
**Cambio propuesto:** Reemplazar `RidgeClassifierCV` por `XGBClassifier` como opción configurable:

```python
# ANTES:
from sklearn.linear_model import RidgeClassifierCV
clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
clf.fit(X_train_scaled, y_train)

# DESPUÉS (configurable):
USE_XGBOOST = True

if USE_XGBOOST:
    from xgboost import XGBClassifier
    clf = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.3,  # Crítico con 10K features
        reg_lambda=10.0,       # Regularización L2 fuerte
        use_label_encoder=False,
        eval_metric='logloss',
        tree_method='gpu_hist',  # GPU acceleration
        random_state=42
    )
    clf.fit(X_train_scaled, y_train)
else:
    clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    clf.fit(X_train_scaled, y_train)
```

**Beneficio esperado:** XGBoost puede capturar patrones no-lineales en las features convolucionales de MiniRocket que el clasificador lineal Ridge no puede explotar. El PDF cita estudios donde este ensamble superó significativamente al Ridge en entornos de alta volatilidad.

**Riesgo:** Mayor complejidad = mayor riesgo de overfitting. La regularización agresiva (`reg_lambda=10`, `colsample_bytree=0.3`) mitiga esto.

#### [MODIFY] [train_minirocket_gpu.py](file:///e:/ProyectosIA/TradingViewPersonalsIndicators/models/train_minirocket_gpu.py)
**Cambio propuesto — Aumentar epochs y agregar early stopping:**

```python
# ANTES:
learn.fit_one_cycle(5, 1e-3)  # Solo 5 epochs, puede underfit

# DESPUÉS:
from fastai.callback.tracker import EarlyStoppingCallback
learn.fit_one_cycle(
    20,     # Más epochs
    1e-3,
    cbs=[EarlyStoppingCallback(monitor='valid_loss', patience=3)]
)
```

**Cambio adicional — Agregar features exógenas:**
Incorporar indicadores técnicos como canales adicionales en la ventana de entrada (similar a lo que hace TSPulse multi):

```python
# ANTES: Solo retornos (1 canal)
X[i] = returns[start:end].reshape(1, -1)

# DESPUÉS: Retornos + RSI + RelVol (3 canales)
X[i, 0, :] = returns[start:end]       # Canal 0: retornos
X[i, 1, :] = rsi_values[start:end]    # Canal 1: RSI normalizado
X[i, 2, :] = relvol_values[start:end] # Canal 2: Volumen relativo
```

---

### FASE 5 — Covariables Exógenas para TimesFM (XReg)
**Prioridad:** 🟢 MEJORA — El PDF enfatiza que TimesFM univariado es "operar ciegamente".

#### [MODIFY] [precalculate_timesfm.py](file:///e:/ProyectosIA/TradingViewPersonalsIndicators/models/precalculate_timesfm.py)
**Cambio propuesto:** Aprovechar el framework XReg de TimesFM 2.5 para inyectar covariables macro:

```python
# Covariables exógenas posibles:
# - SPY returns (proxy de mercado general)
# - VIX (si disponible, volatilidad implícita)
# - Yield del Tesoro a 10 años
# - Volumen relativo del activo

xreg_data = pd.DataFrame({
    'spy_returns': spy_df['Close'].pct_change(),
    'rel_vol': df['RelVol'],
    'rsi': df['RSI_14'] / 100.0
})

# TimesFM 2.5 soporta XReg nativamente:
forecasts = model.forecast(
    context=price_window,
    xreg=xreg_window,  # Covariables alineadas temporalmente
    freq="B"           # Business days
)
```

> [!WARNING]
> Esto requiere actualizar a TimesFM 2.5 (verificar si la versión instalada lo soporta). Si no lo soporta, esta fase se pospone.

---

### FASE 6 — Corrección del Backtester (Lógica de Negocio S18)
**Prioridad:** 🟡 MEDIA — Impacta la coherencia de S18.

#### [MODIFY] [backtester.py](file:///e:/ProyectosIA/TradingViewPersonalsIndicators/backtester.py)
**Problema actual:** El umbral de salida de S18 (`preds_aligned < -0.10`) es inalcanzable porque las probabilidades de MiniRocket GPU son siempre `[0.0, 1.0]`. Es decir, la IA **nunca** genera señal de venta por sí misma. S18 depende al 100% del filtro macro SPY para salir.

**Cambio propuesto:** Actualizar el comentario para reflejar la realidad y considerar un umbral alcanzable:

```python
# OPCIÓN A: Mantener como está (Hold Inteligente con Macro Safety Net)
# Dejar -0.10 (inalcanzable) = S18 solo sale por crash macro.
# Rendimiento actual: 3657% (Rank 4). Funciona, pero son solo 5 trades en 30 años.

# OPCIÓN B: Agregar un umbral de salida real basado en probabilidad
ai_wants_out = preds_aligned < 0.25  # Sale si la IA ve < 25% prob de subir
# Esto generaría más trades y daría más peso a las predicciones de la IA.
```

---

## Open Questions

> [!IMPORTANT]
> **Pregunta 1:** Para la Fase 1 (TSPulse Walk-Forward), ¿preferís la opción de **múltiples checkpoints** (más precisa pero más lenta, ~2-3 horas de entrenamiento por ticker) o la opción de **corte único al 60%** (más rápida, ~30 min total, pero genera señales solo para el 40% posterior de la historia)?

> [!IMPORTANT]
> **Pregunta 2:** Para la Fase 4 (MiniRocket + XGBoost), hay que instalar la librería `xgboost`. ¿Tenés alguna restricción de librerías o preferís que probemos primero con `LGBMClassifier` (LightGBM) que es más liviano?

> [!IMPORTANT]
> **Pregunta 3:** ¿Querés que ejecutemos las 6 fases en orden o preferís priorizar solo las fases críticas (1, 2, 3) primero, correr el backtest para ver el impacto real de corregir el data leakage, y luego decidir si las fases de mejora (4, 5, 6) valen la pena?

---

## Verification Plan

### Automated Tests
```bash
# Después de cada fase, correr el backtester completo:
$env:PYTHONIOENCODING='utf-8'; .venv\Scripts\python.exe backtester.py
```

### Métricas de Éxito
1. **TSPulse (S14/S15/S16):** Después de corregir el data leakage, si los retornos caen significativamente (ej. de 3679% a ~1500%), confirma que los resultados anteriores estaban inflados. Si se mantienen similares, confirma que el modelo genuinamente aprendió patrones válidos.
2. **MiniRocket (S17/S18):** Con XGBoost, esperamos un aumento del AvgReturn del 10-30% manteniendo o reduciendo el MaxDrawdown.
3. **TimesFM (S11/S12/S13):** Después de corregir los bugs de desnormalización, las señales deberían ser más precisas, impactando positivamente los retornos.

### Manual Verification
- Comparar el ranking antes/después de cada fase.
- Verificar que ningún ticker genere señales para fechas que caen dentro del período de entrenamiento (anti-leakage check).
