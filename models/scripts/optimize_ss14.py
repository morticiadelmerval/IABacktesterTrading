import os
import json
import time
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from numba import njit
import itertools

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
        
        df['SMA_20'] = c.rolling(20).mean()
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
        
        # New ROC_3 for SS01 style
        roc3 = c.pct_change(3) * 100
        df['ROC_3_NORM'] = sigmoid_norm(roc3, 5.0)
        
        vol_sma20 = v.rolling(20).mean()
        relvol = v / (vol_sma20 + 1e-8)
        df['VOL_NORM'] = sigmoid_norm(relvol - 1.0, 1.0)
        
        # New Extreme Vol for SS01 style
        df['VOL_EXT_NORM'] = sigmoid_norm(relvol - 2.0, 1.0)
        
        delta = c.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        df['RSI_NORM'] = rsi
        
        tp = (h + l + c) / 3
        rmf = tp * v
        tp_delta = tp.diff()
        pos_mf = rmf.where(tp_delta > 0, 0).rolling(14).sum()
        neg_mf = rmf.where(tp_delta < 0, 0).rolling(14).sum()
        mfi = 100 - (100 / (1 + pos_mf / (neg_mf + 1e-8)))
        df['MFI_NORM'] = mfi
        
        sma20 = c.rolling(20).mean()
        std20 = c.rolling(20).std()
        bb_upper = sma20 + 2*std20
        bb_lower = sma20 - 2*std20
        df['BB_POS'] = (c - bb_lower) / (bb_upper - bb_lower + 1e-8) * 100
        df['BB_POS'] = df['BB_POS'].clip(0, 100)
        
        # New BB 3.1 for SS10 style
        bb_upper_3 = sma20 + 3.1*std20
        bb_lower_3 = sma20 - 3.1*std20
        df['BB_3_POS'] = (c - bb_lower_3) / (bb_upper_3 - bb_lower_3 + 1e-8) * 100
        df['BB_3_POS'] = df['BB_3_POS'].clip(0, 100)
        
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
def simulate_ticker(opens, closes, macro, sma20_mask, score, entry_th, exit_th, commission):
    n = len(opens)
    in_pos = False
    in_sl_recovery = False
    
    cash = 10000.0
    pos = 0.0
    entry_price = 0.0
    trades = 0
    
    for i in range(1, n):
        is_macro_crash = macro[i-1]
        
        hit_stop_loss = False
        if in_pos:
            current_loss = (closes[i-1] / entry_price - 1.0) * 100.0
            if current_loss <= -15.0:
                hit_stop_loss = True
            
        can_enter = False
        if not in_pos:
            if in_sl_recovery:
                if sma20_mask[i-1] and score[i-1] > entry_th and not is_macro_crash:
                    can_enter = True
                    in_sl_recovery = False
            else:
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
            if hit_stop_loss:
                in_sl_recovery = True
            
    if in_pos:
        cash = pos * closes[-1] * (1.0 - commission)
        
    ret = (cash / 10000.0) - 1.0
    return ret * 100.0, trades

def optimize_strategies():
    indicators = ['STOCH_14', 'MACD_NORM', 'ROC_NORM', 'ROC_3_NORM', 'VOL_NORM', 'VOL_EXT_NORM', 'RSI_NORM', 'MFI_NORM', 'BB_POS', 'BB_3_POS', 'ATR_NORM', 'KONCORDE_MD', 'TREND_SMA']
    
    print("Loading data...")
    data = prepare_data()
    
    spy_df = data["SPY"]
    spy_ret = spy_df['Close'].pct_change()
    spy_exit_mask = np.zeros(len(spy_ret), dtype=bool)
    
    days_out = 0
    for i in range(1, len(spy_ret)):
        if spy_ret.iloc[i] < -0.042:
            days_out = 8
        if days_out > 0:
            spy_exit_mask[i] = True
            days_out -= 1
            
    spy_macro_series = pd.Series(spy_exit_mask, index=spy_df.index)
    
    arrays = {}
    for t in TICKERS:
        ticker_macro = spy_macro_series.reindex(data[t].index).ffill().fillna(False).values
        sma20_mask = (data[t]['Close'] > data[t]['SMA_20']).values
        arrays[t] = {
            'opens': data[t]['Open'].values,
            'closes': data[t]['Close'].values,
            'macro': ticker_macro,
            'sma20_mask': sma20_mask,
            'inds': {i: data[t][i].values for i in indicators}
        }
        
    # We maintain the best fitness. If we hit 18/18, we can prioritize that if we want,
    # but fitness directly reflects profitability. A hybrid check ensures we get high returns.
    best_zero = {"fitness": -999999, "beats_bh": 0}
    best_com = {"fitness": -999999, "beats_bh": 0}
    
    start_time = time.time()
    max_duration = 1800 # 30 mins
    
    def evaluate(sel_inds, weights, entry_th, exit_th, commission):
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
                arrs['opens'], arrs['closes'], 
                arrs['macro'], arrs['sma20_mask'], score, entry_th, exit_th, commission
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
            
        return fitness, avg_ret, beats, total_trades

    # Generate random weights summing exactly to 100, max 80, multiples of 5
    valid_random_weights = []
    for w1 in range(0, 85, 5):
        for w2 in range(0, 85, 5):
            for w3 in range(0, 85, 5):
                w4 = 100 - (w1 + w2 + w3)
                if 0 <= w4 <= 80:
                    valid_random_weights.append((w1, w2, w3, w4))

    fixed_thresholds = [(70, 30), (75, 25), (80, 20)]
    
    iteration = 0
    while time.time() - start_time < max_duration:
        # Check if both achieved 18/18
        if best_zero["beats_bh"] >= 18 and best_com["beats_bh"] >= 18:
            print("Both zero and 0.4% commission reached 18/18! Stopping early.")
            break
            
        sel_inds = list(np.random.choice(indicators, 4, replace=False))
        configs_to_test = []
        
        # 1. Combinations of 2 (50, 50)
        for i, j in itertools.combinations(range(4), 2):
            w = [0, 0, 0, 0]
            w[i], w[j] = 50, 50
            for en, ex in fixed_thresholds:
                configs_to_test.append((w, en, ex))
                
        # 2. Combinations of 2 (75, 25)
        for i, j in itertools.permutations(range(4), 2):
            w = [0, 0, 0, 0]
            w[i], w[j] = 75, 25
            for en, ex in fixed_thresholds:
                configs_to_test.append((w, en, ex))
                
        # 3. Combinations of 3 (33.33 -> 33, 33, 34)
        for i, j, k in itertools.combinations(range(4), 3):
            w = [0, 0, 0, 0]
            w[i], w[j], w[k] = 33, 33, 34
            for en, ex in fixed_thresholds:
                configs_to_test.append((w, en, ex))
                
        # 4. Combinations of 3 (70, 20, 10)
        for c in itertools.combinations(range(4), 3):
            for p in itertools.permutations(c):
                w = [0, 0, 0, 0]
                w[p[0]], w[p[1]], w[p[2]] = 70, 20, 10
                for en, ex in fixed_thresholds:
                    configs_to_test.append((w, en, ex))
                    
        # 5. 2000 Random configurations
        for _ in range(2000):
            w = valid_random_weights[np.random.randint(len(valid_random_weights))]
            en = np.random.randint(11, 20) * 5  # 55 to 95
            ex = np.random.randint(1, min(10, en//5)) * 5  # 5 to 45
            if en > ex:
                configs_to_test.append((w, en, ex))
                
        # Test all configurations for this group of 4 indicators
        for w, en, ex in configs_to_test:
            if time.time() - start_time >= max_duration:
                break
                
            # Evaluamos para ambas comisiones
            for c_name, comm, best_dict in [("SS14", 0.0, best_zero), ("SS15", 0.004, best_com)]:
                if best_dict["beats_bh"] < 18 or True:  # always try to improve fitness
                    fit, ret, beats, tr = evaluate(sel_inds, w, en, ex, comm)
                    
                    # Criterio de mejora: más beats, o igual beats pero mejor fitness
                    is_better = False
                    if beats > best_dict["beats_bh"]:
                        is_better = True
                    elif beats == best_dict["beats_bh"] and fit > best_dict["fitness"]:
                        is_better = True
                        
                    if is_better:
                        best_dict.update({
                            "indicators": sel_inds, "weights": w, "entry_th": en, "exit_th": ex,
                            "beats_bh": beats, "avg_return": ret, "total_trades": tr, "fitness": fit
                        })
                        print(f"[{c_name} - {comm*100:.1f}%] New Best! Fit: {fit:.0f} | Ret: {ret:.0f}% | Beats: {beats}/18 | Trades: {tr} | {sel_inds} w={w} en={en} ex={ex}")

        print(f"[{iteration}] Evaluated group {sel_inds} ({len(configs_to_test)} configs)...")
        iteration += 1
        
    print(f"\nOptimization Finished! Iterations evaluated: {iteration} groups of 4")
    
    with open("data/ss14_zero_params.json", "w") as f:
        json.dump(best_zero, f, indent=4)
        
    with open("data/ss15_com_params.json", "w") as f:
        json.dump(best_com, f, indent=4)
        
    print("Saved ss14_zero_params.json and ss15_com_params.json")

if __name__ == "__main__":
    optimize_strategies()
