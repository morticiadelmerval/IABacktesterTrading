import os
import json
import pandas as pd
import numpy as np
import itertools
from backtester_trend import fetch_data, simulate_buy_and_hold, run_simulation, compute_metrics, TICKERS

print("Fetching data...")
data_dict = fetch_data()

print("Computing benchmarks...")
benchmarks = {}
for ticker, df in data_dict.items():
    benchmarks[ticker] = simulate_buy_and_hold(df, commission=0.004)

spy_df = data_dict["SPY"]
spy_ret = spy_df['Close'].pct_change()
spy_idx = spy_df.index
macro_mask = np.zeros(len(spy_ret), dtype=bool)
days_out = 0
for i in range(1, len(spy_ret)):
    if spy_ret.iloc[i] < -0.078:
        days_out = 8
    if days_out > 0:
        macro_mask[i] = True
        days_out -= 1
spy_exit_series = pd.Series(macro_mask, index=spy_idx)

try:
    with open("timesfm_signals.json", "r") as f:
        tsfm_cache = json.load(f)
except:
    tsfm_cache = {}

# We will try mixing timesfm and trend filters
def test_params(sma_l, sma_s, use_macd, require_tsfm_long, tsfm_exit_override):
    beats = 0
    results = {}
    for ticker in TICKERS:
        df = data_dict[ticker]
        macro_aligned = spy_exit_series.reindex(df.index, fill_value=False).values
        
        c = df['Close']
        sma_long = c.rolling(sma_l).mean() if sma_l else None
        sma_short = c.rolling(sma_s).mean() if sma_s else None
        
        ema_12 = c.ewm(span=12, adjust=False).mean()
        ema_26 = c.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()
        
        preds_dict = tsfm_cache.get(ticker, {})
        preds_series = pd.Series(preds_dict)
        if not preds_series.empty:
            preds_series.index = pd.to_datetime(preds_series.index)
        preds_aligned = preds_series.reindex(df.index, fill_value=0.0)
        
print("Starting grid search...")
print("Starting grid search...")
for combo in ["S06_SMA200", "S06_GOLDEN", "S10_SMA200", "TIMESFM_SMA200", "JUST_SMA200_MACRO", "S04_SMA200"]:
    beats = 0
    results = {}
    for ticker in TICKERS:
        df = data_dict[ticker]
        c = df['Close']
        sma_50 = c.rolling(50).mean()
        sma_200 = c.rolling(200).mean()
        
        preds_dict = tsfm_cache.get(ticker, {})
        preds_series = pd.Series(preds_dict)
        if not preds_series.empty:
            preds_series.index = pd.to_datetime(preds_series.index)
        preds_aligned = preds_series.reindex(df.index, fill_value=0.0)
        
        macro_aligned = spy_exit_series.reindex(df.index, fill_value=False).values
        
        # S06 Logic
        roc_3 = c.pct_change(3) * 100
        s06_exit = roc_3 > 10.0
        
        # S10 Logic
        sma_20 = c.rolling(20).mean()
        std_20 = c.rolling(20).std()
        bb_up = sma_20 + 3.1 * std_20
        s10_exit = c > bb_up
        
        # S04 Logic
        vol_sma20 = df['Volume'].rolling(20).mean()
        rel_vol = df['Volume'] / vol_sma20
        s04_exit = (roc_3 > 10.0) | ((rel_vol > 3.0) & (c > df['Open']))
        
        if combo == "S06_SMA200":
            active_long = c > sma_200
            active_exit = s06_exit | (c < sma_200)
        elif combo == "S06_GOLDEN":
            active_long = sma_50 > sma_200
            active_exit = s06_exit | (sma_50 < sma_200)
        elif combo == "S10_SMA200":
            active_long = c > sma_200
            active_exit = s10_exit | (c < sma_200)
        elif combo == "S04_SMA200":
            active_long = c > sma_200
            active_exit = s04_exit | (c < sma_200)
        elif combo == "TIMESFM_SMA200":
            # Just buy if TIMESFM says so, but only if above SMA200
            active_long = (c > sma_200) & (preds_aligned > (0.015 + 0.008))
            ai_wants_out = preds_aligned < -0.004
            active_exit = (c < sma_200) | (ai_wants_out & (preds_aligned < -0.02))
        elif combo == "JUST_SMA200_MACRO":
            # Just hold above SMA_200, no other active exit
            active_long = c > sma_200
            active_exit = c < sma_200
            
        es = macro_aligned | active_exit.values
        # But wait, if active_long is false, we don't enter. If active_exit triggers, we exit.
        # When does it re-enter? Next day if active_long is true again!
        ls = active_long.values & (~macro_aligned)
        
        eq, tr = run_simulation(ls, es, df['Open'].values, df['Close'].values, df.index, commission=0.004)
        mets, ec = compute_metrics(eq, tr, df.index)
        
        bh_ret = benchmarks[ticker]["total_return"]
        strat_ret = mets["total_return"]
        if strat_ret > bh_ret:
            beats += 1
            
    print(f"Combo: {combo} -> BEATS: {beats}")

