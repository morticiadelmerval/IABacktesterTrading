import os
import sys
import yfinance as yf
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from backtester import generate_signals, STRATEGY_INFO, CACHE_DIR

def add_indicators(df):
    c = df['Close']
    h = df['High']
    v = df['Volume']
    
    df['SMA_20'] = c.rolling(20).mean()
    df['STD_20'] = c.rolling(20).std()
    df['SMA_50'] = c.rolling(50).mean()
    df['SMA_200'] = c.rolling(200).mean()
    
    typ_price = (h + df['Low'] + c) / 3
    raw_mf = typ_price * v
    delta_tp = typ_price.diff()
    pos_mf = np.where(delta_tp > 0, raw_mf, 0.0)
    neg_mf = np.where(delta_tp < 0, raw_mf, 0.0)
    pos_mf_sum = pd.Series(pos_mf, index=df.index).rolling(14).sum()
    neg_mf_sum = pd.Series(neg_mf, index=df.index).rolling(14).sum()
    mfr = pos_mf_sum / np.where(neg_mf_sum == 0, 1, neg_mf_sum)
    df['MFI_14'] = 100 - (100 / (1 + mfr))
    
    df['Donchian_33_High'] = h.rolling(33).max()
    
    delta = c.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / np.where(loss == 0, 1e-10, loss)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    df['ROC_3'] = c.pct_change(3) * 100
    df['ROC_5'] = c.pct_change(5) * 100
    
    df['Vol_SMA20'] = v.rolling(20).mean()
    df['RelVol'] = v / df['Vol_SMA20']
    
    return df

# 100 de los activos más populares/operados de Wall Street (Aproximación S&P 100 / Nasdaq 100)
TICKERS_100 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA", "BRK-B", "LLY",
    "AVGO", "V", "JPM", "UNH", "MA", "PG", "JNJ", "HD", "MRK", "COST",
    "ABBV", "CVX", "CRM", "AMD", "KO", "PEP", "BAC", "WMT", "NFLX", "LIN",
    "TMO", "MCD", "WFC", "CSCO", "INTU", "DIS", "DHR", "ORCL", "ABT", "QCOM",
    "GE", "IBM", "CAT", "VZ", "TXN", "NKE", "AXP", "AMGN", "PM", "NOW",
    "INTC", "COP", "SPGI", "BA", "HON", "UNP", "PLD", "LOW", "SYK", "GS",
    "BKNG", "ELV", "LMT", "TJX", "BLK", "MDT", "SBUX", "CI", "CB", "MMC",
    "ADI", "AMT", "ISRG", "GILD", "ADP", "C", "REGN", "VRTX", "PGR", "T",
    "MDLZ", "SLB", "BSX", "ZTS", "MO", "EOG", "CME", "SO", "DUK", "KLAC",
    "ICE", "ITW", "CSX", "NOC", "SHW", "HUM", "TGT", "AON", "WM", "MCO"
]

def load_data(ticker):
    file_path = os.path.join(CACHE_DIR, f"{ticker}_daily.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    else:
        print(f"Descargando {ticker}...")
        df = yf.download(ticker, start="1994-01-01", end="2024-12-31", progress=False)
        if len(df) == 0:
            return None
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.dropna(inplace=True)
        # Fix duplicates
        df = df[~df.index.duplicated(keep='first')]
        df.to_csv(file_path)
    return df

def run_test():
    # Only simple strategies
    simple_strategies = {k: v for k, v in STRATEGY_INFO.items() if k.startswith("SS")}
    
    # Load SPY for Macro Filter
    spy_df = load_data("SPY")
    spy_idx = spy_df.index
    spy_ret = spy_df['Close'].pct_change().fillna(0)
    
    results_by_strategy = {s: {'beats': 0, 'trades': [], 'returns': [], 'maxdd': [], 'cagr': []} for s in simple_strategies}
    
    valid_tickers = 0
    for ticker in TICKERS_100:
        df = load_data(ticker)
        if df is None or len(df) < 500:
            print(f"Saltando {ticker} por falta de datos.")
            continue
            
        valid_tickers += 1
        
        # B&H
        ret = df['Close'].pct_change().fillna(0)
        bh_cum = (1 + ret).cumprod()
        bh_ret = bh_cum.iloc[-1] - 1
        
        years = (df.index[-1] - df.index[0]).days / 365.25
        bh_cagr = (bh_cum.iloc[-1] ** (1 / years)) - 1 if years > 0 and bh_cum.iloc[-1] > 0 else 0
        
        df = add_indicators(df)
        
        for s_code, s_info in simple_strategies.items():
            ls, es = generate_signals(df, ticker, spy_idx, spy_ret, s_info, commission=0.004)
            # Apply stop loss logic exactly as in backtester.py
            sl_pct = s_info.get("stop_loss_pct", None)
            
            # Simple simulation loop
            position = 0
            entry_price = 0
            trades = 0
            
            daily_returns = np.zeros(len(df))
            closes = df['Close'].values
            
            for i in range(len(df)):
                # Stop loss
                if position > 0 and sl_pct is not None:
                    sl_price = entry_price * (1 + (sl_pct / 100.0))
                    if closes[i] <= sl_price:
                        position = 0
                        trades += 1
                        daily_returns[i] = (sl_price / closes[i-1]) - 1.0 - 0.004
                        continue
                
                if position > 0:
                    daily_returns[i] = (closes[i] / closes[i-1]) - 1.0
                    
                if ls[i] and position == 0:
                    position = 1
                    entry_price = closes[i]
                    daily_returns[i] -= 0.004
                elif es[i] and position > 0:
                    position = 0
                    trades += 1
                    daily_returns[i] -= 0.004
            
            daily_series = pd.Series(daily_returns, index=df.index)
            strat_cum = (1 + daily_series).cumprod()
            strat_ret = strat_cum.iloc[-1] - 1
            strat_cagr = (strat_cum.iloc[-1] ** (1 / years)) - 1 if years > 0 and strat_cum.iloc[-1] > 0 else 0
            
            roll_max = strat_cum.cummax()
            dd = (strat_cum / roll_max) - 1
            max_dd = dd.min()
            
            results_by_strategy[s_code]['trades'].append(trades)
            results_by_strategy[s_code]['returns'].append(strat_ret)
            results_by_strategy[s_code]['maxdd'].append(max_dd)
            results_by_strategy[s_code]['cagr'].append(strat_cagr)
            if strat_ret > bh_ret:
                results_by_strategy[s_code]['beats'] += 1
                
    # Compile report
    print(f"\n--- REPORTE FINAL ({valid_tickers} Activos) ---")
    
    report_lines = []
    report_lines.append(f"# Test de 100 Activos (Wall Street Top 100)\n")
    report_lines.append(f"Activos válidos analizados: {valid_tickers}\n")
    report_lines.append("| Estrategia | WinRate vs B&H | Retorno Promedio | CAGR Promedio | Max DD Promedio | Trades Promedio |")
    report_lines.append("|---|---|---|---|---|---|")
    
    for s_code, data in results_by_strategy.items():
        winrate = (data['beats'] / valid_tickers) * 100
        avg_ret = np.mean(data['returns']) * 100
        avg_cagr = np.mean(data['cagr']) * 100
        avg_dd = np.mean(data['maxdd']) * 100
        avg_trades = np.mean(data['trades'])
        
        report_lines.append(f"| {s_code} | {winrate:.1f}% ({data['beats']}/{valid_tickers}) | {avg_ret:.1f}% | {avg_cagr:.2f}% | {avg_dd:.1f}% | {avg_trades:.0f} |")
        
        print(f"{s_code}: {data['beats']}/{valid_tickers} Beats | Ret: {avg_ret:.1f}% | CAGR: {avg_cagr:.2f}% | MaxDD: {avg_dd:.1f}%")
        
    with open('artifacts_report_100.md', 'w') as f:
        f.write("\n".join(report_lines))
        
if __name__ == "__main__":
    run_test()
