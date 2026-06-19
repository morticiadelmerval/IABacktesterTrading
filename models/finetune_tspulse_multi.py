import json
import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PatchTSMixerForPrediction

import warnings
warnings.filterwarnings("ignore")

MODEL_NAME = "ibm-granite/granite-timeseries-patchtsmixer"
OUTPUT_FILE = "data/tspulse_multi_signals.json"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backtester import fetch_data, TICKERS

class TSTrainDatasetMulti(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return (torch.tensor(self.X[idx], dtype=torch.float32),
                torch.tensor(self.y[idx], dtype=torch.float32))

class TSTestDatasetMulti(Dataset):
    def __init__(self, X):
        self.X = X
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32)

def run_walk_forward_multi():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("ERROR: No se detectó CUDA. Abortando entrenamiento para forzar GPU.")
        sys.exit(1)
        
    print(f"Iniciando Walk-Forward MULTI en: {device}")
    data = fetch_data()
    try:
        with open(OUTPUT_FILE, "r") as f:
            predictions_cache = json.load(f)
            print(f"Caché cargado con {len(predictions_cache)} tickers.")
    except FileNotFoundError:
        predictions_cache = {tk: {} for tk in TICKERS}
    
    context_len = 512
    forecast_len = 96
    
    for tk in TICKERS:
        if tk not in data: continue
        df = data[tk]
        if len(df) <= context_len + forecast_len: continue
        
        closes = df['Close'].values
        returns = np.zeros_like(closes)
        returns[1:] = (closes[1:] - closes[:-1]) / closes[:-1]
        
        rsi = df['RSI_14'].values
        rel_vol = df['RelVol'].values
        
        sma_50 = df['SMA_50'].values
        sma_dist = np.zeros_like(closes)
        valid_sma = sma_50 != 0
        sma_dist[valid_sma] = (closes[valid_sma] - sma_50[valid_sma]) / sma_50[valid_sma]
        
        features = np.stack([returns, rsi, rel_vol, sma_dist], axis=1) # (N, 4)
        
        samples_past = []
        samples_future = []
        for i in range(len(features) - context_len - forecast_len + 1):
            window = features[i : i + context_len].copy()
            # Normalización Rolling Z-Score por ventana para canales 1, 2, 3
            for c in [1, 2, 3]:
                mean = window[:, c].mean()
                std = window[:, c].std() + 1e-8
                window[:, c] = (window[:, c] - mean) / std
                
            samples_past.append(window)
            samples_future.append(features[i + context_len : i + context_len + forecast_len])
            
        samples_past = np.array(samples_past)
        samples_future = np.array(samples_future)
        
        if len(samples_past) < 100: continue
        print(f"\n[{tk}] Procesando {len(samples_past)} ventanas para Walk-Forward Multi...")
        
        min_train_size = max(500, int(len(samples_past) * 0.30))
        step_size = max(50, int(len(samples_past) * 0.05))
        
        all_oos_preds = predictions_cache.get(tk, {})
        
        for train_end in range(min_train_size, len(samples_past), step_size):
            test_end = min(train_end + step_size, len(samples_past))
            if test_end <= train_end: break
            
            test_dates = []
            for j in range(test_end - train_end):
                date_idx = train_end + j + context_len
                if date_idx < len(df):
                    test_dates.append(df.index[date_idx].strftime("%Y-%m-%d"))
                    
            if len(test_dates) > 0 and all(d in all_oos_preds for d in test_dates):
                continue
            
            print(f"  Entrenando [{0}:{train_end}]...", end=" ", flush=True)
            
            train_dataset = TSTrainDatasetMulti(samples_past[:train_end], samples_future[:train_end])
            train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
            
            model = PatchTSMixerForPrediction.from_pretrained(
                MODEL_NAME,
                num_input_channels=4,
                ignore_mismatched_sizes=True
            )
            model.to(device)
            model.train()
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            for epoch in range(5):
                for p, f in train_loader:
                    p, f = p.to(device), f.to(device)
                    optimizer.zero_grad()
                    outputs = model(past_values=p, future_values=f)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
            
            print(f"OK. Prediciendo [{train_end}:{test_end}]...", end=" ", flush=True)
            
            model.eval()
            test_dataset = TSTestDatasetMulti(samples_past[train_end:test_end])
            test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
            
            preds_list = []
            with torch.no_grad():
                for p in test_loader:
                    p = p.to(device)
                    outputs = model(past_values=p)
                    preds_list.append(outputs.prediction_outputs.cpu().numpy())
            
            preds_arr = np.concatenate(preds_list, axis=0) # (batch, forecast, channels)
            signal_vals = np.sum(preds_arr[:, :5, 0], axis=1) # Suma de 5 días canal 0
            
            for j in range(len(signal_vals)):
                idx = train_end + j
                date_idx = idx + context_len
                if date_idx < len(df):
                    date_str = df.index[date_idx].strftime("%Y-%m-%d")
                    all_oos_preds[date_str] = float(signal_vals[j])
                    
            print(f"OK. ({len(signal_vals)} preds)")
            
        predictions_cache[tk] = all_oos_preds
        
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(predictions_cache, f, indent=4)
    print("Inferencia OOS Multi guardada exitosamente.")

if __name__ == "__main__":
    run_walk_forward_multi()
