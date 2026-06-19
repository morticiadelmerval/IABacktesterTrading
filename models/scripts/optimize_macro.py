import os
import sys
import json
import numpy as np
import pandas as pd

# Add root folder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backtester import fetch_data, generate_signals, run_simulation, compute_metrics, STRATEGY_INFO, TICKERS

def optimize_macro():
    print("=== OPTIMIZACIÓN DE MACRO BASE PURA AISLADA ===")
    print("Cargando datos (esto se hace solo una vez)...")
    data_dict = fetch_data()
    
    spy_df = data_dict["SPY"]
    spy_ret = spy_df['Close'].pct_change()
    spy_idx = spy_df.index
    commission = 0.004
    
    # 3.5% a 10% de caida en pasos de 0.1%
    thresholds = np.arange(-0.035, -0.101, -0.001)
    days_out_list = range(3, 11)
    
    s_info = STRATEGY_INFO["S21"].copy()
    
    best_avg_return = -9999
    best_params = None
    
    print("\nIniciando iteraciones...")
    total_iters = len(thresholds) * len(days_out_list)
    current_iter = 0
    
    for th in thresholds:
        for days in days_out_list:
            current_iter += 1
            s_info["macro_threshold"] = th
            s_info["macro_days_out"] = days
            
            ticker_rets = []
            for ticker in TICKERS:
                df = data_dict[ticker]
                opens = df['Open'].values
                closes= df['Close'].values
                dates = df.index
                
                ls, es = generate_signals(df, ticker, spy_idx, spy_ret, s_info, commission)
                eq, tr = run_simulation(ls, es, opens, closes, dates, commission=commission)
                mets, _ = compute_metrics(eq, tr, dates)
                
                ticker_rets.append(mets["total_return"])
                
            avg_ret = np.nanmean(ticker_rets)
            
            # Mostrar progreso ocasionalmente
            if current_iter % 10 == 0 or current_iter == total_iters:
                print(f"Progreso: {current_iter}/{total_iters} | th: {th*100:.1f}%, days: {days} | Retorno: {avg_ret:.2f}%")
                
            if avg_ret > best_avg_return:
                best_avg_return = avg_ret
                best_params = (th, days)
                
    print("\n=== MEJOR CONFIGURACIÓN ENCONTRADA ===")
    print(f"Mejor macro_threshold: {best_params[0]*100:.2f}%")
    print(f"Mejor macro_days_out:  {best_params[1]} días")
    print(f"Retorno Promedio:      {best_avg_return:.2f}%")
    
    print("\nPara comparar contra el resto, revisa 'data/results.json' (el valor anterior de S21 era aprox 598%).")

if __name__ == '__main__':
    optimize_macro()
