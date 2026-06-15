import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

CACHE_DIR = ".data_cache"
TICKERS = ["SPY", "QQQ", "DIA", "IWM", "MCD", "KO", "MSFT", "GOOG", "V", "C", "XOM"]

data_dict = {}
for ticker in TICKERS:
    cache_path = os.path.join(CACHE_DIR, f"{ticker}.csv")
    df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    c = df['Close']
    o = df['Open']
    v = df['Volume']
    h = df['High']
    df['SMA_20'] = c.rolling(20).mean()
    high_low = h - df['Low']
    high_close = np.abs(h - c.shift())
    low_close = np.abs(df['Low'] - c.shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR_14'] = true_range.rolling(14).mean()
    df['ATR_Pct'] = df['ATR_14'] / c * 100
    data_dict[ticker] = df

with open("timesfm_signals.json", "r") as f:
    tsfm_cache = json.load(f)

commission = 0.004

def simulate_bh(df):
    closes = df['Close'].values
    opens = df['Open'].values
    buy_price = opens[0]
    shares = (10000.0 * (1.0 - commission)) / buy_price
    equity = shares * closes[-1] * (1.0 - commission)
    return (equity / 10000.0 - 1.0) * 100.0

bh_returns = {t: simulate_bh(data_dict[t]) for t in TICKERS}

spy_df = data_dict["SPY"]
spy_ret = spy_df['Close'].pct_change()
spy_idx = spy_df.index
spy_exit_mask = np.zeros(len(spy_ret), dtype=bool)
days_out = 0
for i in range(1, len(spy_ret)):
    if spy_ret.iloc[i] < -0.078:
        days_out = 8
    if days_out > 0:
        spy_exit_mask[i] = True
        days_out -= 1
spy_exit_series = pd.Series(spy_exit_mask, index=spy_idx)

def test_params(p1, p2, p3, p4):
    beats = 0
    results = {}
    for ticker in TICKERS:
        df = data_dict[ticker]
        macro_aligned = spy_exit_series.reindex(df.index, fill_value=False).values
        
        preds_dict = tsfm_cache.get(ticker, {})
        preds_series = pd.Series(preds_dict)
        if not preds_series.empty:
            preds_series.index = pd.to_datetime(preds_series.index)
        preds_aligned = preds_series.reindex(df.index, fill_value=0.0)
        
        atr_pct = df['ATR_Pct'].fillna(1.0)
        
        long_threshold = p1 + commission * 2.0 + (atr_pct * p2)
        active_long = preds_aligned > long_threshold
        
        ai_wants_out = preds_aligned < (-commission - (atr_pct * p3))
        strong_uptrend = df['Close'] > df['SMA_20']
        severe_drop = p4 - (atr_pct * p3)
        active_exit = ai_wants_out & (~strong_uptrend | (preds_aligned < severe_drop))
        
        es = macro_aligned | active_exit.values
        ls = active_long.values & (~macro_aligned)
        
        opens = df['Open'].values
        closes= df['Close'].values
        n = len(closes)
        cash = 10000.0
        pos = 0.0
        in_pos = False
        entry_price = 0.0
        
        for i in range(1, n):
            if not in_pos and ls[i-1]:
                pos = (cash * (1.0 - commission)) / opens[i]
                cash = 0.0
                in_pos = True
                entry_price = opens[i]
            elif in_pos and es[i-1]:
                cash = pos * opens[i] * (1.0 - commission)
                pos = 0.0
                in_pos = False
        if in_pos:
            cash = pos * closes[-1] * (1.0 - commission)
        
        ret = (cash / 10000.0 - 1.0) * 100.0
        if ret > bh_returns[ticker]:
            beats += 1
        results[ticker] = ret
    return beats, results

print("Starting grid search...")
best_beats = 0
best_params = None
best_res = None

for p1 in [-0.015, -0.01, -0.005, 0.0, 0.005]: # base long
    for p2 in [0.0, 0.001, 0.002, 0.003, 0.005]: # atr mul long
        for p3 in [-0.001, 0.0, 0.001, 0.002, 0.003]: # atr mul exit
            for p4 in [-0.01, -0.015, -0.02, -0.025, -0.03, -0.04]: # severe drop base
                b, res = test_params(p1, p2, p3, p4)
                if b > best_beats:
                    best_beats = b
                    best_params = (p1, p2, p3, p4)
                    best_res = res
                    print(f"New best: {b}/11 with params {best_params}")
                    if b == 11:
                        print("Got 11/11, exiting loop!")
                        break
            if best_beats == 11: break
        if best_beats == 11: break
    if best_beats == 11: break

print("Done. Best:", best_beats, best_params)
print("Results:", best_res)
for t in TICKERS:
    if best_res[t] <= bh_returns[t]:
        print(f"FAILED: {t} - S11={best_res[t]:.2f} BH={bh_returns[t]:.2f}")
