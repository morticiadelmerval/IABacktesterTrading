import pandas as pd
import numpy as np
import sys
import json
sys.path.append("e:/ProyectosIA/TradingViewPersonalsIndicators")
import backtester

def get_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def test_combination(rsi_buy_th, mfi_buy_th, mfi_sell_th, bb_std, bb_sell_std):
    commission = 0.004
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
        
        rsi = get_rsi(df, 14)
        mfi = df['MFI_14']
        
        bb_lower = df['SMA_20'] - bb_std * df['STD_20']
        bb_upper = df['SMA_20'] + bb_sell_std * df['STD_20']
        
        cost_barrier = commission * 2.0
        base_long = preds_aligned > (0.015 + cost_barrier)
        
        bb_buy = df['Close'] <= bb_lower
        active_long = base_long | (bb_buy & ((mfi < mfi_buy_th) | (rsi < rsi_buy_th)))
        
        strong_uptrend = df['Close'] > df['SMA_20']
        ai_wants_out = preds_aligned < -commission
        base_exit = ai_wants_out & (~strong_uptrend | (preds_aligned < -0.02))
        
        bb_sell = df['Close'] >= bb_upper
        active_exit = base_exit | (bb_sell & (mfi > mfi_sell_th))
        
        es = macro_aligned | active_exit.values
        ls = active_long.values & (~macro_aligned)
        
        eq, tr = backtester.run_simulation(ls, es, df['Open'].values, df['Close'].values, df.index, commission=commission)
        mets, _ = backtester.compute_metrics(eq, tr, df.index)
        
        bh = backtester.simulate_buy_and_hold(df, commission=commission)['total_return']
        
        if mets['total_return'] > bh:
            beats += 1
            
    return beats

if __name__ == "__main__":
    backtester.CACHED_DATA_DICT = backtester.fetch_data()
    with open("e:/ProyectosIA/TradingViewPersonalsIndicators/timesfm_signals.json", "r") as f:
        tsfm_cache = json.load(f)
        
    for rsi_b in [20, 30, 40]:
        for mfi_b in [20, 30, 40]:
            for mfi_s in [80, 90, 100]:
                for bb_s in [2.0, 2.5]:
                    for bb_s_s in [2.0, 2.5]:
                        score = test_combination(rsi_b, mfi_b, mfi_s, bb_s, bb_s_s)
                        if score >= 10:
                            print(f"RSI_B:{rsi_b} MFI_B:{mfi_b} MFI_S:{mfi_s} BB_S:{bb_s} BB_SS:{bb_s_s} -> {score}/11")
                        if score == 11:
                            print("FOUND 11!")
                            sys.exit(0)
