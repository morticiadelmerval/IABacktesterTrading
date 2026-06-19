"""
MiniRocket: Entrenamiento + Pre-cálculo de Señales para S17
============================================================
Usa convoluciones aleatorias fijas (MiniRocket) + RidgeClassifierCV
para clasificar si el precio subirá (1) o no (0) en los próximos 5 días.

Genera minirocket_signals.json con formato idéntico a tspulse_signals.json.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backtester import fetch_data, TICKERS

# Hiperparámetros
CONTEXT_LEN = 256        # Ventana de contexto (días de historia)
FORECAST_HORIZON = 5     # Días a futuro para el target
BUY_THRESHOLD = 0.015    # Umbral para etiquetar como "compra" (1.5%)
OUTPUT_FILE = "data/minirocket_signals.json"


def build_windows_and_labels(returns, context_len, forecast_horizon, buy_threshold):
    """Construye ventanas deslizantes y etiquetas binarias."""
    windows = []
    labels = []
    for i in range(context_len, len(returns) - forecast_horizon):
        window = returns[i - context_len : i]
        # Retorno acumulado en los próximos N días
        future_return = np.sum(returns[i : i + forecast_horizon])
        label = 1 if future_return > buy_threshold else 0
        windows.append(window)
        labels.append(label)
    return np.array(windows), np.array(labels)


def run_minirocket_training():
    print("=" * 60)
    print("  MiniRocket: Entrenamiento + Pre-cálculo de Señales (S17)")
    print("=" * 60)
    
    # Importar MiniRocket
    print("\nImportando MiniRocketFeatures desde tsai (GPU)...")
    import torch
    from tsai.models.MINIROCKET_Pytorch import MiniRocketFeatures
    from sklearn.linear_model import RidgeClassifierCV
    from sklearn.preprocessing import StandardScaler
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"MiniRocketFeatures cargado correctamente. Usando: {device}")
    
    # Cargar datos
    print("\nCargando datos históricos de 18 activos...")
    data = fetch_data()
    try:
        with open(OUTPUT_FILE, "r") as f:
            predictions_cache = json.load(f)
            print(f"Caché cargado con {len(predictions_cache)} tickers.")
    except FileNotFoundError:
        predictions_cache = {tk: {} for tk in TICKERS}
    
    for tk in TICKERS:
        if tk not in data:
            print(f"  {tk}: Sin datos, saltando.")
            continue
        
        df = data[tk]
        if len(df) <= CONTEXT_LEN + FORECAST_HORIZON:
            print(f"  {tk}: Historia demasiado corta ({len(df)} barras), saltando.")
            continue
        
        closes = df['Close'].values
        returns = np.zeros_like(closes)
        returns[1:] = (closes[1:] - closes[:-1]) / closes[:-1]
        
        # Construir ventanas y etiquetas
        windows, labels = build_windows_and_labels(
            returns, CONTEXT_LEN, FORECAST_HORIZON, BUY_THRESHOLD
        )
        
        if len(windows) < 100:
            print(f"  {tk}: Muy pocas muestras ({len(windows)}), saltando.")
            continue
        
        # sktime espera shape (n_samples, n_timepoints) como DataFrame panel
        # Convertimos a formato 3D numpy: (n_samples, 1, n_timepoints) para univariado
        X_3d = windows.reshape(len(windows), 1, CONTEXT_LEN)
        
        print(f"\n  {tk}: {len(windows)} muestras | "
              f"Balance: {labels.sum()}/{len(labels)} compras "
              f"({100*labels.mean():.1f}%)")
        
        # --- Walk-Forward Expanding Window (sin look-ahead bias) ---
        # Entrenamos con los primeros N datos, predecimos el siguiente bloque,
        # expandimos la ventana de entrenamiento y repetimos.
        min_train_size = max(500, int(len(windows) * 0.30))  # Mínimo 30% para entrenar
        step_size = max(50, int(len(windows) * 0.05))  # Avanzar 5% a la vez
        
        print(f"    Walk-Forward (min_train={min_train_size}, step={step_size})...", end=" ", flush=True)
        
        all_oos_preds = predictions_cache.get(tk, {})
        
        for train_end in range(min_train_size, len(windows), step_size):
            test_end = min(train_end + step_size, len(windows))
            
            test_dates = []
            for j in range(test_end - train_end):
                date_idx = CONTEXT_LEN + train_end + j
                if date_idx < len(df):
                    test_dates.append(df.index[date_idx].strftime("%Y-%m-%d"))
                    
            if len(test_dates) > 0 and all(d in all_oos_preds for d in test_dates):
                continue
                
            X_train = X_3d[:train_end]
            y_train = labels[:train_end]
            X_test = X_3d[train_end:test_end]
            
            if len(X_test) == 0:
                break
            
            minirocket = MiniRocketFeatures(c_in=1, seq_len=CONTEXT_LEN).to(device)
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            
            minirocket.fit(X_train_tensor)
            
            with torch.no_grad():
                X_train_tf = minirocket(X_train_tensor).cpu().numpy()
                X_test_tf = minirocket(X_test_tensor).cpu().numpy()
            
            scaler = StandardScaler(with_mean=False)
            X_train_scaled = scaler.fit_transform(X_train_tf)
            X_test_scaled = scaler.transform(X_test_tf)
            
            from sklearn.linear_model import RidgeClassifier
            clf = RidgeClassifier()
            clf.fit(X_train_scaled, y_train)
            
            test_preds = clf.predict(X_test_scaled)
            
            for j in range(len(test_preds)):
                idx = train_end + j
                date_idx = CONTEXT_LEN + idx
                if date_idx < len(df):
                    date_str = df.index[date_idx].strftime("%Y-%m-%d")
                    all_oos_preds[date_str] = int(test_preds[j])
        
        # Calcular accuracy OOS usando índice pre-construido
        date_to_label = {}
        for wi in range(len(windows)):
            di = CONTEXT_LEN + wi
            if di < len(df):
                date_to_label[df.index[di].strftime("%Y-%m-%d")] = labels[wi]
        
        oos_correct = sum(1 for d, p in all_oos_preds.items() if date_to_label.get(d) == p)
        oos_total = len(all_oos_preds)
        
        acc = (oos_correct / oos_total * 100) if oos_total > 0 else 0.0
        print(f"OOS Accuracy: {acc:.1f}% ({oos_total} signals)")
        
        predictions_cache[tk] = all_oos_preds
    
    # Guardar señales
    print(f"\nGuardando señales en '{OUTPUT_FILE}'...")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(predictions_cache, f)
    
    total_signals = sum(len(v) for v in predictions_cache.values())
    print(f"✓ {total_signals} señales guardadas para {len(TICKERS)} activos.")
    print("\n¡Pre-cálculo MiniRocket finalizado! Ya puedes correr el backtester.")


if __name__ == "__main__":
    run_minirocket_training()
