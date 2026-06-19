import pandas as pd
import numpy as np
import sys
import json
sys.path.append("e:/ProyectosIA/TradingViewPersonalsIndicators")
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import backtester

def get_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

if __name__ == "__main__":
    backtester.CACHED_DATA_DICT = backtester.fetch_data()
    with open("e:/ProyectosIA/TradingViewPersonalsIndicators/timesfm_signals.json", "r") as f:
        tsfm_cache = json.load(f)

    beats = 0
    for tk in backtester.TICKERS:
        df = backtester.CACHED_DATA_DICT[tk]
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
        
        preds_dict = tsfm_cache.get(tk, {})
        preds_series = pd.Series(preds_dict)
        if not preds_series.empty:
            preds_series.index = pd.to_datetime(preds_series.index)
        preds_aligned = preds_series.reindex(df.index, fill_value=0.0)
        
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['RSI_14'] = get_rsi(df, 14)
        
        comm = 0.004
        cost_barrier = comm * 2.0
        ai_long = preds_aligned > (0.015 + cost_barrier)
        
        bb_lower = df['SMA_20'] - 2 * df['STD_20']
        mr_long = (df['Close'] < bb_lower) & (df['MFI_14'] < 30)
        active_long = ai_long | mr_long
        
        if "SMA_200" not in df.columns:
            df["SMA_200"] = df['Close'].rolling(200).mean()
            
        bb_width = 4.0 * df['STD_20'] / df['SMA_20']
        
        # Volatility filter: if volatile, use long-term trend (hold). If stable, use short-term (trade).
        # We will loop over threshold but let's test a simple dynamic condition:
        # Actually I can't loop easily without changing the file structure, so I'll just write one:
        high_vol = bb_width > 0.05
        
        # Where high_vol is true, use SMA_200, else use SMA_20
        trend_line = np.where(high_vol, df['SMA_200'], df['SMA_20'])
        strong_uptrend = df['Close'] > trend_line
        
        ai_wants_out = preds_aligned < -0.015
        
        active_exit = ai_wants_out & (~strong_uptrend)
        
        es = macro_aligned | active_exit.values
        ls = active_long.values & (~macro_aligned)
        
        eq, tr = backtester.run_simulation(ls, es, df['Open'].values, df['Close'].values, df.index, commission=comm)
        mets, _ = backtester.compute_metrics(eq, tr, df.index)
        bh = backtester.simulate_buy_and_hold(df, commission=comm)['total_return']
        
        if mets['total_return'] > bh:
            beats += 1
            print(f"{tk:4s}: {mets['total_return']:8.2f}% vs BH {bh:8.2f}% [BEAT]")
        else:
            print(f"{tk:4s}: {mets['total_return']:8.2f}% vs BH {bh:8.2f}% [NO]")
            
    print(f"\nFINAL SCORE: {beats}/11 BEATS")

