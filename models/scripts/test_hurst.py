import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from backtester import fetch_data

def hurst_ernie_chan(p, max_lag=20):
    """
    Hurst exponent calc from EP Chan.
    """
    if len(p) < max_lag:
        return np.nan
    p_log = np.log10(p.values)
    variancetau = []
    tau = []
    lags = range(2, max_lag)
    for lag in lags:
        tau.append(lag)
        pp = np.subtract(p_log[lag:], p_log[:-lag])
        variancetau.append(np.var(pp))
    
    # Avoid log(0) issues if variance is 0
    if 0 in variancetau:
        return np.nan
        
    m = np.polyfit(np.log10(tau), np.log10(variancetau), 1)
    hurst = m[0] / 2
    return hurst

def test_hurst():
    print("Cargando datos...")
    data_dict = fetch_data()
    df = data_dict["SPY"].copy()
    
    print("Calculando Hurst Rolling (Ventana: 100 dias, Max Lag: 20 dias)...")
    window = 100
    df['Hurst'] = df['Close'].rolling(window=window).apply(lambda x: hurst_ernie_chan(x, max_lag=20), raw=False)
    
    # Retornos a futuro
    df['Fwd_Ret_20d'] = df['Close'].shift(-20) / df['Close'] - 1
    
    # Test basico RSI Reversion
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['Fwd_Ret_5d'] = df['Close'].shift(-5) / df['Close'] - 1
    
    df = df.dropna()
    
    print("\n=== ANALISIS DEL EXPONENTE DE HURST (SPY) ===")
    
    print(f"Hurst Promedio: {df['Hurst'].mean():.4f}")
    print(f"Hurst Max: {df['Hurst'].max():.4f}")
    print(f"Hurst Min: {df['Hurst'].min():.4f}")
    
    # 2. Predictive Power for Trend Following
    trend_days = df[df['Hurst'] > 0.55]
    non_trend_days = df[df['Hurst'] <= 0.55]
    
    print(f"\n[TREND FOLLOWING] Retornos Futuros a 20 días")
    print(f"Dias en Fuerte Tendencia (Hurst > 0.55): {len(trend_days)}")
    print(f"  -> Retorno Medio 20d: {trend_days['Fwd_Ret_20d'].mean()*100:.2f}%")
    print(f"  -> Win Rate 20d: {len(trend_days[trend_days['Fwd_Ret_20d']>0])/len(trend_days)*100:.1f}%")
    
    print(f"Dias Sin Tendencia (Hurst <= 0.55): {len(non_trend_days)}")
    print(f"  -> Retorno Medio 20d: {non_trend_days['Fwd_Ret_20d'].mean()*100:.2f}%")
    print(f"  -> Win Rate 20d: {len(non_trend_days[non_trend_days['Fwd_Ret_20d']>0])/len(non_trend_days)*100:.1f}%")
    
    # 3. Predictive Power for Mean Reversion (RSI < 30)
    mr_days = df[df['Hurst'] < 0.45]
    
    rsi_buy_all = df[df['RSI'] < 30]
    rsi_buy_mr = mr_days[mr_days['RSI'] < 30]
    
    print(f"\n[MEAN REVERSION] Comprar Sobrevendido (RSI < 30) a 5 días")
    print(f"Comprar siempre que RSI < 30 (Sin Filtro Hurst): {len(rsi_buy_all)} trades")
    print(f"  -> Retorno Medio 5d: {rsi_buy_all['Fwd_Ret_5d'].mean()*100:.2f}%")
    print(f"  -> Win Rate 5d: {len(rsi_buy_all[rsi_buy_all['Fwd_Ret_5d']>0])/len(rsi_buy_all)*100:.1f}%")
    
    if len(rsi_buy_mr) > 0:
        print(f"Comprar RSI < 30 SOLO si Hurst < 0.45 (Regimen Reversión): {len(rsi_buy_mr)} trades")
        print(f"  -> Retorno Medio 5d: {rsi_buy_mr['Fwd_Ret_5d'].mean()*100:.2f}%")
        print(f"  -> Win Rate 5d: {len(rsi_buy_mr[rsi_buy_mr['Fwd_Ret_5d']>0])/len(rsi_buy_mr)*100:.1f}%")
    else:
        print("No hubieron señales de RSI < 30 en régimen de Hurst < 0.45.")

    print("\n[CONCLUSION HURST]")
    print("Si el Win Rate y Retorno de RSI mejora cuando Hurst < 0.45, es un excelente filtro de reversión.")
    print("Si el Win Rate a 20d mejora cuando Hurst > 0.55, es un excelente filtro de tendencias.")

if __name__ == '__main__':
    test_hurst()
