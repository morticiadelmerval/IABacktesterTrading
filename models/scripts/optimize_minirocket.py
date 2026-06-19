import os
import sys
import numpy as np

# Add root folder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backtester import fetch_data, generate_signals, run_simulation, compute_metrics, STRATEGY_INFO, TICKERS

def optimize_minirocket():
    print("=== OPTIMIZACIÓN AISLADA DE MINIROCKET (S18) Y XGBOOST (S20) ===")
    print("Cargando datos históricos y predicciones IA...")
    data_dict = fetch_data()
    
    spy_df = data_dict["SPY"]
    spy_ret = spy_df['Close'].pct_change()
    spy_idx = spy_df.index
    commission = 0.004
    
    # Valores de entrada: 55 a 85 (0.55 a 0.85 en pasos de 0.05)
    # np.arange para llegar a 0.85 inclusivo necesita 0.86
    in_thresholds = np.arange(0.55, 0.86, 0.05)
    
    # Valores de salida: 25 a 45 (0.25 a 0.45 en pasos de 0.05)
    out_thresholds = np.arange(0.25, 0.46, 0.05)
    
    strategies_to_test = ["S18", "S20"]
    
    for s_id in strategies_to_test:
        print(f"\n--- Evaluando {s_id} ---")
        s_info = STRATEGY_INFO[s_id].copy()
        
        results = []
        total_iters = len(in_thresholds) * len(out_thresholds)
        current_iter = 0
        
        for in_th in in_thresholds:
            for out_th in out_thresholds:
                current_iter += 1
                s_info["ai_in_th"] = float(in_th)
                s_info["ai_out_th"] = float(out_th)
                
                ticker_rets = []
                total_trades = 0
                
                for ticker in TICKERS:
                    df = data_dict[ticker]
                    opens = df['Open'].values
                    closes= df['Close'].values
                    dates = df.index
                    
                    ls, es = generate_signals(df, ticker, spy_idx, spy_ret, s_info, commission)
                    eq, tr = run_simulation(ls, es, opens, closes, dates, commission=commission)
                    mets, _ = compute_metrics(eq, tr, dates)
                    
                    ticker_rets.append(mets["total_return"])
                    total_trades += mets["num_trades"]
                    
                avg_ret = np.nanmean(ticker_rets)
                results.append({
                    "in_th": in_th,
                    "out_th": out_th,
                    "avg_ret": avg_ret,
                    "trades": total_trades
                })
                
                if current_iter % 10 == 0 or current_iter == total_iters:
                    print(f"  Progreso {s_id}: {current_iter}/{total_iters} ...")
        
        # Rankear por retorno promedio
        results.sort(key=lambda x: x["avg_ret"], reverse=True)
        
        print(f"\n--- TOP 5 CONFIGURACIONES PARA {s_id}: ---")
        for i in range(5):
            res = results[i]
            print(f"  Rank {i+1}: Entrada > {res['in_th']*100:.1f}% | Salida < {res['out_th']*100:.1f}% --> Retorno: {res['avg_ret']:.2f}% (Trades: {res['trades']})")

if __name__ == '__main__':
    optimize_minirocket()
