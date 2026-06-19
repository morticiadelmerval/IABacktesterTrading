import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from backtester import fetch_data, STRATEGY_INFO, generate_signals, run_simulation, compute_metrics

def main():
    print("Fetching data...")
    data_cache = fetch_data()
    df = data_cache["SPY"]
    df = df[~df.index.duplicated(keep='first')]
    spy_idx = df['Close']
    spy_ret = spy_idx.pct_change().fillna(0)

    stop_losses = [None, -5, -10, -15, -20, -25, -30, -35]
    
    print("\n--- RESULTS FOR SPY ---")
    print(f"| {'Strat':<5} | {'SL':<4} | {'Total Ret':>9} | {'Max DD':>9} | {'Win Rate':>8} | {'Trades':>6} {'(Diff)':<5} | {'Ret Diff':<10} |")
    print("|" + "-" * 75 + "|")

    for strat_id, info in STRATEGY_INFO.items():
        # Generate signals once per strategy
        ls, es = generate_signals(df, "SPY", spy_idx, spy_ret, info, commission=0.004)
        opens, closes, dates = df['Open'].values, df['Close'].values, df.index
        
        baseline_ret = 0
        for sl in stop_losses:
            eq, tr = run_simulation(ls, es, opens, closes, dates, commission=0.004, stop_loss_pct=sl)
            metrics, ec = compute_metrics(eq, tr, dates)
            
            sl_str = f"{sl}%" if sl is not None else "None"
            ret = metrics['total_return']
            trades_count = metrics['num_trades']
            
            if sl is None:
                baseline_ret = ret
                baseline_trades = trades_count
            
            diff_str = ""
            diff_trades_str = ""
            if sl is not None:
                diff = ret - baseline_ret
                diff_str = f"({'+' if diff>=0 else ''}{diff:.1f}%)"
                diff_trades = trades_count - baseline_trades
                diff_trades_str = f"({'+' if diff_trades>=0 else ''}{diff_trades})"
                
            print(f"| {strat_id:<5} | {sl_str:<4} | {ret:>8.1f}% | {metrics['max_drawdown']:>8.1f}% | {metrics['win_rate']:>6.1f}% | {trades_count:>6} {diff_trades_str:<5} | {diff_str} |")
        print("-" * 50)

if __name__ == "__main__":
    main()
