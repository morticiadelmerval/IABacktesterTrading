import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backtester import fetch_data, STRATEGY_INFO, generate_signals, compute_metrics, TICKERS

def run_simulation_strict(signals_long, signals_exit, signals_strict, opens, closes, dates, initial_capital=10000.0, commission=0.004, stop_loss_pct=None):
    n = len(closes)
    equity = np.full(n, float(initial_capital))
    cash   = float(initial_capital)
    pos    = 0.0
    in_pos = False
    entry_price = 0.0
    entry_idx   = 0
    trades = []
    
    in_sl_recovery = False

    for i in range(1, n):
        hit_stop_loss = False
        if in_pos and stop_loss_pct is not None:
            current_loss = (closes[i-1] / entry_price - 1.0) * 100.0
            if current_loss <= stop_loss_pct:
                hit_stop_loss = True

        can_enter = False
        if not in_pos:
            if in_sl_recovery:
                if signals_strict[i-1] and signals_long[i-1]:
                    can_enter = True
                    in_sl_recovery = False
            else:
                if signals_long[i-1]:
                    can_enter = True

        if can_enter:
            pos = (cash * (1.0 - commission)) / opens[i]
            cash = 0.0
            in_pos = True
            entry_price = opens[i]
            entry_idx   = i
        elif in_pos and (signals_exit[i-1] or hit_stop_loss):
            revenue = pos * opens[i] * (1.0 - commission)
            pnl     = revenue - (pos * entry_price / (1.0 - commission))
            pct_ret = (revenue / (pos * entry_price / (1.0 - commission)) - 1.0) * 100.0
            trades.append({
                "pct_return": float(pct_ret),
                "pnl": float(pnl),
                "duration_days": int((dates[i] - dates[entry_idx]).days)
            })
            cash   = revenue
            pos    = 0.0
            in_pos = False
            
            if hit_stop_loss:
                in_sl_recovery = True

        equity[i] = cash + pos * closes[i]

    if in_pos:
        revenue = pos * closes[-1] * (1.0 - commission)
        pnl     = revenue - (pos * entry_price / (1.0 - commission))
        pct_ret = (revenue / (pos * entry_price / (1.0 - commission)) - 1.0) * 100.0
        trades.append({
            "pct_return": float(pct_ret),
            "pnl": float(pnl),
            "duration_days": int((dates[-1] - dates[entry_idx]).days)
        })
        equity[-1] = cash + revenue

    return equity, trades

def main():
    print("Fetching data...")
    data_dict = fetch_data()
    spy_df = data_dict["SPY"]
    spy_ret = spy_df['Close'].pct_change()
    spy_idx = spy_df.index
    
    strategies_to_test = [f"SS{i:02d}" for i in range(1, 11)]
    
    options = ["baseline", "strict_sma20", "strict_rsi40", "strict_roc3", "strict_sma50"]
    
    overall_results = {opt: {"ret": [], "mdd": []} for opt in options}
    
    print(f"Testing strategies: {strategies_to_test}")
    
    for s_id in strategies_to_test:
        s_info = STRATEGY_INFO[s_id]
        
        # We only care if the strategy actually has a stop loss
        if s_info.get("stop_loss_pct") is None:
            continue
            
        sl_pct = s_info["stop_loss_pct"]
        results = {opt: [] for opt in options}
        
        for ticker in TICKERS:
            df = data_dict[ticker]
            opens = df['Open'].values
            closes = df['Close'].values
            dates = df.index
            
            ls, es = generate_signals(df, ticker, spy_idx, spy_ret, s_info, commission=0.004)
            
            # Define strict signals
            sig_baseline = np.ones(len(df), dtype=bool) # always true, identical to old logic
            sig_sma20 = (df['Close'] > df['SMA_20']).values
            sig_rsi40 = (df['RSI_14'] > 40).values
            sig_roc3 = (df['ROC_3'] > 0).values
            sig_sma50 = (df['Close'] > df['SMA_50']).values
            
            strict_map = {
                "baseline": sig_baseline,
                "strict_sma20": sig_sma20,
                "strict_rsi40": sig_rsi40,
                "strict_roc3": sig_roc3,
                "strict_sma50": sig_sma50
            }
            
            for opt in options:
                sig_strict = strict_map[opt]
                eq_mod, tr_mod = run_simulation_strict(ls, es, sig_strict, opens, closes, dates, stop_loss_pct=sl_pct)
                met_mod, _ = compute_metrics(eq_mod, tr_mod, dates)
                results[opt].append(met_mod)

        print(f"\n--- Strategy {s_id} ---")
        for opt in options:
            avg_ret = np.mean([r['total_return'] for r in results[opt]])
            avg_mdd = np.mean([r['max_drawdown'] for r in results[opt]])
            print(f"  {opt}: Avg Ret = {avg_ret:7.2f}% | Avg MaxDD = {avg_mdd:6.2f}%")
            
            overall_results[opt]["ret"].append(avg_ret)
            overall_results[opt]["mdd"].append(avg_mdd)

    print("\n=== FINAL OVERALL AVERAGES ACROSS ALL TESTED STRATEGIES ===")
    for opt in options:
        if len(overall_results[opt]["ret"]) > 0:
            avg_ret = np.mean(overall_results[opt]["ret"])
            avg_mdd = np.mean(overall_results[opt]["mdd"])
            print(f"{opt}: OVERALL Avg Ret = {avg_ret:7.2f}% | OVERALL Avg MaxDD = {avg_mdd:6.2f}%")

if __name__ == '__main__':
    main()
