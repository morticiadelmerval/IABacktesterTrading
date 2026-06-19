"""
MiniRocket GPU: Entrenamiento + Pre-cálculo de Señales (S18)
============================================================
Utiliza la librería tsai (fastai/PyTorch) para ejecutar 
MiniRocketPlus nativamente en CUDA (GPU). 

Beneficios:
1. Extrae features en GPU (ultra rápido).
2. Entrena una cabeza neuronal (MiniRocketPlus) con SGD/Adam.
3. Extrae PROBABILIDADES reales (0.0 a 1.0) en vez de etiquetas binarias.
"""
import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime

import matplotlib
matplotlib.use('Agg')

# Desactivar warnings pesados de fastai/tsai
import warnings
warnings.filterwarnings("ignore")

# Importar tsai
try:
    from tsai.all import *
except ImportError:
    print("Error: tsai no está instalado. Ejecuta 'pip install tsai'")
    sys.exit(1)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backtester import fetch_data, TICKERS

# Hiperparámetros
CONTEXT_LEN = 256
FORECAST_HORIZON = 5
BUY_THRESHOLD = 0.015
OUTPUT_FILE = "data/minirocket_gpu_signals.json"
EPOCHS = 5 # Solo 5 epochs por paso walk-forward gracias a MiniRocketFeatures

# Verificar CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dispositivo PyTorch: {device}")


def build_windows_and_labels(returns, context_len, forecast_horizon, buy_threshold):
    windows = []
    labels = []
    for i in range(context_len, len(returns) - forecast_horizon):
        window = returns[i - context_len : i]
        future_return = np.sum(returns[i : i + forecast_horizon])
        label = 1 if future_return > buy_threshold else 0
        windows.append(window)
        labels.append(label)
    return np.array(windows), np.array(labels)


def run_gpu_training():
    print("=" * 60)
    print("  MiniRocket GPU (tsai): Entrenamiento S18")
    print("=" * 60)
    
    data = fetch_data()
    try:
        with open(OUTPUT_FILE, "r") as f:
            predictions_cache = json.load(f)
            print(f"Caché cargado con {len(predictions_cache)} tickers.")
    except FileNotFoundError:
        predictions_cache = {tk: {} for tk in TICKERS}
    
    for tk in TICKERS:
        if tk not in data: continue
        df = data[tk]
        if len(df) <= CONTEXT_LEN + FORECAST_HORIZON: continue
        
        closes = df['Close'].values
        returns = np.zeros_like(closes)
        returns[1:] = (closes[1:] - closes[:-1]) / closes[:-1]
        
        windows, labels = build_windows_and_labels(
            returns, CONTEXT_LEN, FORECAST_HORIZON, BUY_THRESHOLD
        )
        if len(windows) < 100: continue
        
        # Formato tsai: (samples, variables, timesteps)
        X_3d = windows.reshape(len(windows), 1, CONTEXT_LEN).astype(np.float32)
        y = labels.astype(np.int64)
        
        print(f"\n  {tk}: {len(windows)} muestras | "
              f"Balance: {y.sum()}/{len(y)} compras "
              f"({100*y.mean():.1f}%)")
              
        min_train_size = max(500, int(len(windows) * 0.30))
        step_size = max(50, int(len(windows) * 0.05))
        
        print(f"    Walk-Forward (GPU)...", end=" ", flush=True)
        
        all_oos_probs = predictions_cache.get(tk, {}) # date -> probability of class 1
        
        for train_end in range(min_train_size, len(windows), step_size):
            test_end = min(train_end + step_size, len(windows))
            if test_end <= train_end: break
            
            test_dates = []
            for j in range(test_end - train_end):
                date_idx = CONTEXT_LEN + train_end + j
                if date_idx < len(df):
                    test_dates.append(df.index[date_idx].strftime("%Y-%m-%d"))
                    
            if len(test_dates) > 0 and all(d in all_oos_probs for d in test_dates):
                continue
                
            # Crear splits predefinidos para fastai
            train_idx = list(range(train_end))
            valid_idx = list(range(train_end, test_end))
            splits = (train_idx, valid_idx)
            
            # Datasets & DataLoaders de tsai
            tfms  = [None, [Categorize()]]
            dsets = TSDatasets(X_3d, y, tfms=tfms, splits=splits)
            dls   = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=256, num_workers=0)
            
            # Modelo Deep Learning (MiniRocketPlus)
            model = build_ts_model(MiniRocketPlus, dls=dls)
            
            # Learner
            learn = ts_learner(dls, model, metrics=accuracy)
            # Desactivar output
            learn.logger = lambda *args, **kwargs: None
            
            learn.fit_one_cycle(EPOCHS, 1e-3)
            
            # Obtener predicciones en validación (out of sample)
            probs, targets, preds = learn.get_preds(dl=dls.valid, with_decoded=True)
            # probs tiene shape (test_samples, 2). Queremos la prob de la clase 1.
            probs_1 = probs[:, 1].numpy()
            
            # Guardar OOS
            for j in range(len(valid_idx)):
                idx = valid_idx[j]
                date_idx = CONTEXT_LEN + idx
                if date_idx < len(df):
                    date_str = df.index[date_idx].strftime("%Y-%m-%d")
                    all_oos_probs[date_str] = float(probs_1[j])
                    
        # Accuracy binario usando prob > 0.5
        date_to_label = {}
        for wi in range(len(windows)):
            di = CONTEXT_LEN + wi
            if di < len(df):
                date_to_label[df.index[di].strftime("%Y-%m-%d")] = labels[wi]
        
        oos_correct = sum(1 for d, p in all_oos_probs.items() if date_to_label.get(d) == (1 if p > 0.5 else 0))
        oos_total = len(all_oos_probs)
        acc = (oos_correct / oos_total * 100) if oos_total > 0 else 0.0
        
        print(f"OOS Accuracy (>0.5): {acc:.1f}% ({oos_total} signals)")
        
        predictions_cache[tk] = all_oos_probs
        
    print(f"\nGuardando probabilidades puras en '{OUTPUT_FILE}'...")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(predictions_cache, f, indent=4)
    print(f"✓ Guardado exitoso.")

if __name__ == "__main__":
    run_gpu_training()
