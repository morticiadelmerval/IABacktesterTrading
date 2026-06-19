import json
import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backtester import fetch_data, TICKERS

import xgboost as xgb

def build_features_and_labels(df, minirocket_probs, context_len, forecast_horizon, buy_threshold):
    # Asegurarnos de usar los datos donde MiniRocket tiene probabilidad.
    # MiniRocket probs son un dict date -> prob
    
    dates = []
    features = []
    labels = []
    
    closes = df['Close'].values
    returns = np.zeros_like(closes)
    returns[1:] = (closes[1:] - closes[:-1]) / closes[:-1]
    
    # Pre-calcular el threshold real como lo hacíamos (esto es label, compramos o no en el futuro)
    future_returns = np.zeros_like(closes)
    for i in range(len(closes) - forecast_horizon):
        future_returns[i] = np.sum(returns[i : i+forecast_horizon])
        
    for i in range(context_len, len(closes) - forecast_horizon):
        date_str = df.index[i].strftime("%Y-%m-%d")
        if date_str not in minirocket_probs: continue
        
        # Target
        label = 1 if future_returns[i] > buy_threshold else 0
        
        # Features
        mr_prob = minirocket_probs[date_str]
        rsi = df['RSI_14'].iloc[i]
        mfi = df['MFI_14'].iloc[i]
        rel_vol = df['RelVol'].iloc[i]
        
        # Distancia a SMAs (Tendencia)
        c = closes[i]
        sma50 = df['SMA_50'].iloc[i]
        sma200 = df['SMA_200'].iloc[i]
        dist50 = (c - sma50) / sma50 if sma50 != 0 else 0
        dist200 = (c - sma200) / sma200 if sma200 != 0 else 0
        
        # Volatilidad
        roc = df['ROC_5'].iloc[i]
        
        feat_vector = [mr_prob, rsi, mfi, rel_vol, dist50, dist200, roc]
        
        dates.append(date_str)
        features.append(feat_vector)
        labels.append(label)
        
    return np.array(features), np.array(labels), dates

def run_xgboost_stacking():
    print("=" * 60)
    print("  XGBoost Meta-Labeling Stacking (S20)")
    print("=" * 60)
    
    data = fetch_data()
    
    with open("data/minirocket_gpu_signals.json", "r") as f:
        mr_cache = json.load(f)
        
    predictions_cache = {tk: {} for tk in TICKERS}
    
    CONTEXT_LEN = 256
    FORECAST_HORIZON = 5
    BUY_THRESHOLD = 0.015
    
    for tk in TICKERS:
        if tk not in data: continue
        if tk not in mr_cache: continue
        
        df = data[tk]
        mr_probs = mr_cache[tk]
        
        X, y, dates = build_features_and_labels(df, mr_probs, CONTEXT_LEN, FORECAST_HORIZON, BUY_THRESHOLD)
        if len(X) < 100: continue
        
        print(f"\n  {tk}: {len(X)} muestras de features construidas.")
        
        min_train_size = max(500, int(len(X) * 0.30))
        step_size = max(50, int(len(X) * 0.05))
        
        all_oos_probs = {}
        
        for train_end in range(min_train_size, len(X), step_size):
            test_end = min(train_end + step_size, len(X))
            if test_end <= train_end: break
            
            X_train, y_train = X[:train_end], y[:train_end]
            X_test, y_test = X[train_end:test_end], y[train_end:test_end]
            test_dates = dates[train_end:test_end]
            
            # Usar XGBoost (Device CUDA)
            clf = xgb.XGBClassifier(
                n_estimators=150, 
                max_depth=4, 
                learning_rate=0.05,
                objective='binary:logistic',
                tree_method='hist',
                device='cuda', # Usa la GPU!
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0
            )
            
            clf.fit(X_train, y_train)
            
            probs_1 = clf.predict_proba(X_test)[:, 1]
            
            for j in range(len(test_dates)):
                all_oos_probs[test_dates[j]] = float(probs_1[j])
                
        # Calcular Accuracy del stack:
        oos_correct = sum(1 for d, p in all_oos_probs.items() if (1 if p > 0.5 else 0) == y[dates.index(d)])
        oos_total = len(all_oos_probs)
        acc = (oos_correct / oos_total * 100) if oos_total > 0 else 0.0
        print(f"OOS Stack Accuracy: {acc:.1f}% ({oos_total} signals)")
        
        predictions_cache[tk] = all_oos_probs
        
    os.makedirs("data", exist_ok=True)
    with open("data/xgboost_stack_signals.json", "w") as f:
        json.dump(predictions_cache, f, indent=4)
    print("\n✓ Stacking de XGBoost completado.")

if __name__ == "__main__":
    run_xgboost_stacking()
