import os
import json
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from numba import njit

CACHE_DIR = ".data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)
TICKERS = ["SPY", "QQQ", "DIA", "IWM", "MCD", "KO", "MSFT", "GOOG", "V", "C", "XOM", "NU", "GLD", "IVE", "EWZ", "PBR", "BRK-B", "O"]
START_DATE = "1996-01-01"

def sigmoid_norm(x, scale=1.0):
    return 100.0 / (1.0 + np.exp(-x / scale))

def prepare_data():
    data = {}
    for ticker in TICKERS:
        cache_path = os.path.join(CACHE_DIR, f"{ticker}.csv")
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        else:
            df = yf.download(ticker, start=START_DATE)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
            df.to_csv(cache_path)
            
        c, h, l, v = df['Close'], df['High'], df['Low'], df['Volume']
        
        df['SMA_50'] = c.rolling(50).mean()
        df['SMA_200'] = c.rolling(200).mean()
        df['Macro_Trend'] = c > df['SMA_200']
        df['Macro_Crash'] = ~df['Macro_Trend'] & (c < c.rolling(50).min().shift(1) * 1.05)
        
        low14 = l.rolling(14).min()
        high14 = h.rolling(14).max()
        df['STOCH_14'] = 100 * (c - low14) / (high14 - low14 + 1e-8)
        
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_sig = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - macd_sig
        scale_macd = macd_hist.rolling(100).std().bfill() + 1e-8
        df['MACD_NORM'] = sigmoid_norm(macd_hist, scale_macd)
        
        roc5 = c.pct_change(5) * 100
        df['ROC_NORM'] = sigmoid_norm(roc5, 5.0)
        
        vol_sma20 = v.rolling(20).mean()
        relvol = v / (vol_sma20 + 1e-8)
        df['VOL_NORM'] = sigmoid_norm(relvol - 1.0, 1.0)
        
        delta = c.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        df['RSI_NORM'] = rsi
        
        tp = (h + l + c) / 3
        rmf = tp * v
        pos_mf = rmf.where(delta > 0, 0).rolling(14).sum()
        neg_mf = rmf.where(delta < 0, 0).rolling(14).sum()
        mfi = 100 - (100 / (1 + pos_mf / (neg_mf + 1e-8)))
        df['MFI_NORM'] = mfi
        
        sma20 = c.rolling(20).mean()
        std20 = c.rolling(20).std()
        bb_upper = sma20 + 2*std20
        bb_lower = sma20 - 2*std20
        df['BB_POS'] = (c - bb_lower) / (bb_upper - bb_lower + 1e-8) * 100
        df['BB_POS'] = df['BB_POS'].clip(0, 100)
        
        tr1 = h - l
        tr2 = (h - c.shift()).abs()
        tr3 = (l - c.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean()
        df['ATR_NORM'] = sigmoid_norm(atr14 / c * 100, 2.0)
        
        ret = c.pct_change().fillna(0)
        vol_up = v > v.shift(1)
        pvi_change = np.where(vol_up, ret, 0.0)
        pvi = 1000.0 * np.cumprod(1 + pvi_change)
        ema_pvi = pd.Series(pvi).ewm(span=15, adjust=False).mean()
        scale_pvi = pd.Series(pvi).rolling(100).std().bfill() + 1e-8
        df['KONCORDE_MD'] = sigmoid_norm(pvi - ema_pvi, scale_pvi)
        
        trend = (c - df['SMA_50']) / df['SMA_50'] * 100
        df['TREND_SMA'] = sigmoid_norm(trend, 5.0)
        
        df.fillna(50.0, inplace=True)
        data[ticker] = df
        
    return data

@njit
def simulate_ticker(opens, closes, lows, macro, score, entry_th, exit_th, commission):
    n = len(opens)
    in_pos = False
    
    cash = 10000.0
    pos = 0.0
    entry_price = 0.0
    trades = 0
    
    for i in range(1, n):
        is_macro_crash = macro[i-1]
        
        hit_stop_loss = False
        if in_pos and (lows[i-1] <= entry_price * 0.85):
            hit_stop_loss = True
            
        can_enter = False
        if not in_pos:
            if score[i-1] > entry_th and not is_macro_crash:
                can_enter = True
                    
        if can_enter:
            pos = (cash * (1.0 - commission)) / opens[i]
            cash = 0.0
            in_pos = True
            entry_price = opens[i]
            trades += 1
            
        elif in_pos and (score[i-1] < exit_th or hit_stop_loss or is_macro_crash):
            cash = pos * opens[i] * (1.0 - commission)
            pos = 0.0
            in_pos = False
            
    if in_pos:
        cash = pos * closes[-1] * (1.0 - commission)
        
    ret = (cash / 10000.0) - 1.0
    return ret * 100.0, trades

def optimize_commission(data, commission, num_iterations=100000):
    indicators = ['STOCH_14', 'MACD_NORM', 'ROC_NORM', 'VOL_NORM', 'RSI_NORM', 'MFI_NORM', 'BB_POS', 'ATR_NORM', 'KONCORDE_MD', 'TREND_SMA']
    
    spy_macro_series = data["SPY"]['Macro_Crash']
    
    arrays = {}
    for t in TICKERS:
        ticker_macro = spy_macro_series.reindex(data[t].index).ffill().fillna(False).values
        arrays[t] = {
            'opens': data[t]['Open'].values,
            'closes': data[t]['Close'].values,
            'lows': data[t]['Low'].values,
            'macro': ticker_macro,
            'inds': {i: data[t][i].values for i in indicators}
        }
    
    best_params = None
    best_fitness = -999999
    
    for i in range(num_iterations):
        sel_inds = list(np.random.choice(indicators, 4, replace=False))
        weights = np.random.randint(1, 80, size=4)
        weights = np.round((weights / weights.sum()) * 100).astype(int)
        
        entry_th = np.random.randint(50, 95)
        exit_th = np.random.randint(5, 50)
        
        if entry_th <= exit_th:
            continue
            
        total_ret = 0.0
        beats = 0
        total_trades = 0
        
        for t in TICKERS:
            arrs = arrays[t]
            score = np.zeros(len(arrs['opens']))
            for idx, ind in enumerate(sel_inds):
                score += arrs['inds'][ind] * weights[idx]
            score /= 100.0
            
            bh_ret = (arrs['closes'][-1] / arrs['opens'][0] - 1) * 100
            strat_ret, trades = simulate_ticker(
                arrs['opens'], arrs['closes'], arrs['lows'], 
                arrs['macro'], score, entry_th, exit_th, commission
            )
            
            total_ret += strat_ret
            total_trades += trades
            if strat_ret > bh_ret:
                beats += 1
                
        avg_ret = total_ret / len(TICKERS)
        
        if total_trades >= 360:
            fitness = avg_ret
        else:
            diff = 360 - total_trades
            fitness = avg_ret * np.exp(-diff / 100.0)
            
        if fitness > best_fitness:
            best_fitness = fitness
            best_params = {
                "indicators": sel_inds,
                "weights": weights.tolist(),
                "entry_th": int(entry_th),
                "exit_th": int(exit_th),
                "beats_bh": beats,
                "avg_return": avg_ret,
                "total_trades": total_trades,
                "fitness": fitness
            }
            print(f"[{commission:.3f}] New Best! Fit: {fitness:.0f} | Ret: {avg_ret:.0f}% | Beats: {beats} | Trades: {total_trades}")
            
    return best_params

if __name__ == "__main__":
    print("Loading data...")
    data = prepare_data()
    
    print("\n--- Optimizing for ZERO Commission ---")
    best_zero = optimize_commission(data, 0.000, 100000)
    with open("data/ss12_zero_params.json", "w") as f:
        json.dump(best_zero, f, indent=4)
        
    print("\n--- Optimizing for 0.4% Commission ---")
    best_com = optimize_commission(data, 0.004, 100000)
    with open("data/ss12_com_params.json", "w") as f:
        json.dump(best_com, f, indent=4)
        
    print("Done!")
