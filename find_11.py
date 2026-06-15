import pandas as pd
import numpy as np
import sys
import json
import itertools
sys.path.append("e:/ProyectosIA/TradingViewPersonalsIndicators")
import backtester

def calculate_indicators(df):
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['SMA_100'] = df['Close'].rolling(100).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

def run_search():
    commission = 0.004
    backtester.CACHED_DATA_DICT = backtester.fetch_data()
    
    with open("e:/ProyectosIA/TradingViewPersonalsIndicators/timesfm_signals.json", "r") as f:
        tsfm_cache = json.load(f)
        
    # Pre-calculate data per ticker
    ticker_data = {}
    for tk in backtester.TICKERS:
        df = backtester.CACHED_DATA_DICT[tk].copy()
        
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
        
        calculate_indicators(df)
        
        bh = backtester.simulate_buy_and_hold(df, commission=commission)['total_return']
        
        ticker_data[tk] = {
            'df': df,
            'preds': preds_aligned,
            'macro': macro_aligned,
            'bh': bh
        }
    
    # We define search space
    # The user asked to try "Smart Hold" (estar comprado por defecto) -> pd.Series(True, index=df.index)
    

    trend_options = [
        "df['Close'] > df['SMA_20']",
        "df['Close'] > df['SMA_50']",
        "df['Close'] > df['SMA_100']",
        "df['Close'] > df['SMA_200']",
        "df['SMA_20'] > df['SMA_50']",
        "df['SMA_50'] > df['SMA_200']",
        "df['MACD'] > df['MACD_Signal']",
        "df['MACD'] > 0"
    ]
    
    best_score = -1
    best_combo = None
    
    for long_logic in [
        "pd.Series(True, index=df.index)"
    ]:
        for trend in trend_options:
            for force_exit_th in [-0.01, -0.02, -0.03, -0.05, -1]:
                for ai_out_th in ["-comm", "0", "0.01", "-0.02", "-0.05"]:
                    for and_or in ["&", "|"]:
                        
                        def eval_combo():
                            beats = 0
                            for tk in backtester.TICKERS:
                                dat = ticker_data[tk]
                                df = dat['df']
                                preds = dat['preds']
                                macro_aligned = dat['macro']
                                comm = commission
                                
                                active_long = eval(long_logic)
                                strong_uptrend = eval(trend)
                                ai_wants_out = preds < eval(ai_out_th)
                                
                                if force_exit_th == -1:
                                    active_exit = ai_wants_out & (~strong_uptrend)
                                else:
                                    if and_or == "&":
                                        active_exit = ai_wants_out & (~strong_uptrend | (preds < force_exit_th))
                                    else:
                                        active_exit = ai_wants_out | (~strong_uptrend & (preds < force_exit_th))
                                    
                                es = macro_aligned | active_exit.values
                                ls = active_long.values & (~macro_aligned)
                                
                                eq, tr = backtester.run_simulation(ls, es, df['Open'].values, df['Close'].values, df.index, commission=commission)
                                mets, _ = backtester.compute_metrics(eq, tr, df.index)
                                
                                if mets['total_return'] > dat['bh']:
                                    beats += 1
                                else:
                                    pass # do not short circuit
                            return beats
                            
                        b = eval_combo()
                        if b > best_score:
                            best_score = b
                            best_combo = {
                                'long': long_logic,
                                'trend': trend,
                                'ai_out_th': ai_out_th,
                                'force_exit_th': force_exit_th,
                                'and_or': and_or
                            }
                            print(f"New Best: {b}/11 - {best_combo}")
                            if b >= 11:
                                return best_combo

if __name__ == "__main__":
    combo = run_search()
    if combo:
        print("\n\nFOUND 11/11")
        print(combo)
    else:
        print("Not found")
