import pandas as pd
import numpy as np
import sys
sys.path.append("e:/ProyectosIA/TradingViewPersonalsIndicators")
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import backtester

def eval_logic(ticker, commission, name, logic_exit):
    df = backtester.CACHED_DATA_DICT[ticker]
    
    spy_df = backtester.CACHED_DATA_DICT["SPY"]
    spy_ret = spy_df['Close'].pct_change()
    
    spy_exit_mask = np.zeros(len(spy_ret), dtype=bool)
    days_out = 0
    for i in range(1, len(spy_ret)):
        if spy_ret.iloc[i] < -0.078:
            days_out = 8
        if days_out > 0:
            spy_exit_mask[i] = True
            days_out -= 1
            
    macro_aligned = pd.Series(spy_exit_mask, index=spy_df.index).reindex(df.index, fill_value=False).values
    
    import json
    with open("e:/ProyectosIA/TradingViewPersonalsIndicators/timesfm_signals.json", "r") as f:
        tsfm_cache = json.load(f)
        
    preds_dict = tsfm_cache.get(ticker, {})
    preds_series = pd.Series(preds_dict)
    if not preds_series.empty:
        preds_series.index = pd.to_datetime(preds_series.index)
    preds_aligned = preds_series.reindex(df.index, fill_value=0.0)
    
    active_exit = logic_exit(df, preds_aligned, commission)
    
    # NEW LOGIC: Long by default!
    es = macro_aligned | active_exit.values
    ls = ~es
    
    eq, tr = backtester.run_simulation(ls, es, df['Open'].values, df['Close'].values, df.index, commission=commission)
    mets, _ = backtester.compute_metrics(eq, tr, df.index)
    
    return mets['total_return'], mets['num_trades']

if __name__ == "__main__":
    backtester.CACHED_DATA_DICT = backtester.fetch_data()
    df = backtester.CACHED_DATA_DICT["GOOG"]
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['SMA_100'] = df['Close'].rolling(100).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()
    
    df_v = backtester.CACHED_DATA_DICT["V"]
    df_v['SMA_50'] = df_v['Close'].rolling(50).mean()
    df_v['SMA_100'] = df_v['Close'].rolling(100).mean()
    df_v['SMA_200'] = df_v['Close'].rolling(200).mean()
    
    logics = [
        ("Exit < -1% + Broken SMA100", lambda d, p, c: (p < -0.01) & (~(d['Close'] > d['SMA_100']))),
        ("Exit < -1% + Broken SMA200", lambda d, p, c: (p < -0.01) & (~(d['Close'] > d['SMA_200']))),
        ("Exit < -1.5% + Broken SMA100", lambda d, p, c: (p < -0.015) & (~(d['Close'] > d['SMA_100']))),
        ("Exit < -1.5% + Broken SMA200", lambda d, p, c: (p < -0.015) & (~(d['Close'] > d['SMA_200']))),
    ]
    
    bh_goog = backtester.simulate_buy_and_hold(backtester.CACHED_DATA_DICT["GOOG"], commission=0.004)['total_return']
    bh_v = backtester.simulate_buy_and_hold(backtester.CACHED_DATA_DICT["V"], commission=0.004)['total_return']
    
    print(f"B&H GOOG: {bh_goog:.2f}% | B&H V: {bh_v:.2f}%")
    
    for name, le in logics:
        ret_g, tr_g = eval_logic("GOOG", 0.004, name, le)
        ret_v, tr_v = eval_logic("V", 0.004, name, le)
        
        g_beat = "YES" if ret_g > bh_goog else "NO"
        v_beat = "YES" if ret_v > bh_v else "NO"
        
        print(f"{name:30s} | GOOG: {ret_g:8.2f}% ({tr_g} tr) [{g_beat}] | V: {ret_v:8.2f}% ({tr_v} tr) [{v_beat}]")

