import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

CACHE_DIR = ".data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

TICKERS = ["SPY", "QQQ", "DIA", "IWM", "MCD", "KO", "MSFT", "GOOG", "V", "C", "XOM"]
START_DATE = "1996-01-01"
END_DATE = datetime.today().strftime('%Y-%m-%d')

def fetch_data():
    data = {}
    for ticker in TICKERS:
        cache_path = os.path.join(CACHE_DIR, f"{ticker}.csv")
        cache_valid = False
        if os.path.exists(cache_path):
            mod_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
            if mod_time.date() == datetime.today().date():
                cache_valid = True

        if cache_valid:
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            df.index.name = "Date"
        else:
            df = yf.download(ticker, start=START_DATE, end=END_DATE)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
            df.to_csv(cache_path)
        
        c = df['Close']
        o = df['Open']
        v = df['Volume']
        h = df['High']
        
        # Bollinger
        df['SMA_20'] = c.rolling(20).mean()
        df['STD_20'] = c.rolling(20).std()
        df['BB_Upper'] = df['SMA_20'] + 2 * df['STD_20']
        df['BB_Upper_1'] = df['SMA_20'] + 1 * df['STD_20']
        df['BB_Lower'] = df['SMA_20'] - 2 * df['STD_20']
        
        # RSI 14
        delta = c.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=13, adjust=False).mean()
        ema_down = down.ewm(com=13, adjust=False).mean()
        rs = ema_up / ema_down
        df['RSI_14'] = 100 - (100 / (1 + rs))

        df['SMA_50'] = c.rolling(50).mean()
        df['SMA_200'] = c.rolling(200).mean()

        data[ticker] = df
    return data

def run_simulation(signals_long, signals_exit, opens, closes, dates, initial_capital=10000.0, commission=0.004):
    n = len(closes)
    equity = np.full(n, float(initial_capital))
    cash   = float(initial_capital)
    pos    = 0.0
    in_pos = False
    entry_price = 0.0

    for i in range(1, n):
        if not in_pos and signals_long[i-1]:
            pos = (cash * (1.0 - commission)) / opens[i]
            cash = 0.0
            in_pos = True
            entry_price = opens[i]
        elif in_pos and signals_exit[i-1]:
            revenue = pos * opens[i] * (1.0 - commission)
            cash   = revenue
            pos    = 0.0
            in_pos = False

        equity[i] = cash + pos * closes[i]

    if in_pos:
        revenue = pos * closes[-1] * (1.0 - commission)
        equity[-1] = cash + revenue

    return equity

def simulate_buy_and_hold(df, initial_capital=10000.0, commission=0.004):
    dates = df.index
    closes = df['Close'].values
    opens = df['Open'].values
    buy_price = opens[0]
    shares = (initial_capital * (1.0 - commission)) / buy_price
    equity = shares * closes * (1.0 - commission)
    return equity[-1]

def search_best_strategy():
    data_dict = fetch_data()
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

    bhs = {}
    for t in TICKERS:
        bhs[t] = simulate_buy_and_hold(data_dict[t])

    print("Testing combinations...")
    best_score = -1
    best_params = None

    for rsi_exit in [60, 65, 70, 75, 80, 85]:
        for sma_trend in [50, 200]:
            for strategy_type in [1, 3, 4, 5]:
                beat_count = 0
                total_excess = 0
                
                for ticker in TICKERS:
                    df = data_dict[ticker]
                    macro_aligned = spy_exit_series.reindex(df.index, fill_value=False).values
                    
                    if strategy_type == 1:
                        # Exit during downtrend if RSI is overbought (failed bounce)
                        downtrend = df['Close'] < df[f'SMA_{sma_trend}']
                        active_exit = downtrend & (df['RSI_14'] > rsi_exit)
                    elif strategy_type == 3:
                        # Exit when touches upper BB AND RSI is overbought
                        active_exit = (df['Close'] > df['BB_Upper']) & (df['RSI_14'] > rsi_exit)
                    elif strategy_type == 4:
                        # Exit when touches BB_Upper unconditionally
                        active_exit = (df['Close'] > df['BB_Upper'])
                    elif strategy_type == 5:
                        # Exit when RSI > rsi_exit AND Close < SMA
                        active_exit = (df['RSI_14'] > rsi_exit) & (df['Close'] < df[f'SMA_{sma_trend}'])

                    es = macro_aligned | active_exit.values
                    ls = ~es

                    eq = run_simulation(ls, es, df['Open'].values, df['Close'].values, df.index)
                    
                    ret = eq[-1]
                    bh = bhs[ticker]
                    
                    if ret > bh:
                        beat_count += 1
                    total_excess += (ret / bh)

                if beat_count > best_score:
                    best_score = beat_count
                    best_params = (strategy_type, rsi_exit, sma_trend)
                    print(f"New Best: {best_score}/11 | Type {strategy_type}, RSI {rsi_exit}, SMA {sma_trend}")
                if beat_count == 11:
                    print(f"FOUND 11/11! Type: {strategy_type}, RSI Exit: {rsi_exit}, SMA: {sma_trend}")
                    return

    print(f"Final Best: {best_score}/11 with Type {best_params[0]}, Exit {best_params[1]}, SMA {best_params[2]}")

if __name__ == "__main__":
    search_best_strategy()
