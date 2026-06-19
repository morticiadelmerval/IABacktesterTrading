import pandas as pd
import numpy as np
import sys
import json
sys.path.append("e:/ProyectosIA/TradingViewPersonalsIndicators")
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import backtester

def test_strategy(active_long_func, active_exit_func, commission=0.004):
    backtester.CACHED_DATA_DICT = backtester.fetch_data()
    
    with open("e:/ProyectosIA/TradingViewPersonalsIndicators/timesfm_signals.json", "r") as f:
        tsfm_cache = json.load(f)
        
    beats = 0
    total_tickers = len(backtester.TICKERS)
    
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
        
        # Calculate extra indicators if needed
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['SMA_100'] = df['Close'].rolling(100).mean()
        df['SMA_200'] = df['Close'].rolling(200).mean()
        
        # Execute custom logic
        active_long = active_long_func(df, preds_aligned, commission)
        active_exit = active_exit_func(df, preds_aligned, commission)
        
        es = macro_aligned | active_exit.values
        ls = active_long.values & (~macro_aligned)
        
        eq, tr = backtester.run_simulation(ls, es, df['Open'].values, df['Close'].values, df.index, commission=commission)
        mets, _ = backtester.compute_metrics(eq, tr, df.index)
        
        bh = backtester.simulate_buy_and_hold(df, commission=commission)['total_return']
        
        if mets['total_return'] > bh:
            beats += 1
            print(f"{tk:4s}: {mets['total_return']:8.2f}% vs BH {bh:8.2f}% [BEAT]")
        else:
            print(f"{tk:4s}: {mets['total_return']:8.2f}% vs BH {bh:8.2f}% [NO]")
            
    print(f"\nFINAL SCORE: {beats}/{total_tickers} BEATS")

if __name__ == "__main__":
    # --- EDIT THESE TWO FUNCTIONS TO TEST ---
    
    def my_active_long(df, preds, comm):
        bb_width = 4.0 * df['STD_20'] / df['SMA_20']
        is_volatile_asset = bb_width.mean() > 0.21
        
        if is_volatile_asset:
            cost_barrier = comm * 2.0
            ai_long_strict = preds > (0.015 + cost_barrier)
            bb_lower = df['SMA_20'] - 2 * df['STD_20']
            mr_long = (df['Close'] < bb_lower) & (df['MFI_14'] < 30)
            return ai_long_strict | mr_long
        else:
            return preds > -0.01

    def my_active_exit(df, preds, comm):
        bb_width = 4.0 * df['STD_20'] / df['SMA_20']
        is_volatile_asset = bb_width.mean() > 0.21
        
        if is_volatile_asset:
            strong_uptrend = df['Close'] > df['SMA_200']
            ai_wants_out = preds < -0.015
            return ai_wants_out & (~strong_uptrend)
        else:
            strong_uptrend = df['Close'] > df['SMA_50']
            ai_wants_out = preds < -0.02
            return ai_wants_out & (~strong_uptrend)

    # ----------------------------------------
    
    test_strategy(my_active_long, my_active_exit, commission=0.004)

