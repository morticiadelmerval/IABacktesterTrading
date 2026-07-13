import os
import sys
import json
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from backtester import (
    generate_signals,
    run_simulation,
    compute_metrics,
    simulate_buy_and_hold,
    sigmoid_norm,
    STRATEGY_INFO,
    CACHE_DIR
)

# 100 de los activos más operados y representativos de la década del 90,
# incluyendo una fuerte proporción de activos que sufrieron mercados bajistas seculares,
# burbujas punto com, caídas severas post-2008 o lateralización de varias décadas.
TICKERS_100_BEARISH_90S = [
    # Tech / Telecom / Hardware de los 90s con severos crash o lateralidad secular
    "CSCO", "INTC", "NOK", "ERIC", "GLW", "JNPR", "CIEN", "LUMN", "T", "VZ",
    "IBM", "ORCL", "QCOM", "HPQ", "XRX", "BB", "EBAY", "AKAM", "VRSN", "NTAP",
    "FFIV", "CHKP", "FLEX", "JBL", "SANM", "ARW", "AVT", "WDC", "STX", "MU",
    
    # Financieros con colapsos o décadas bajo el agua desde 1998-2007
    "C", "AIG", "BAC", "WFC", "KEY", "RF", "HBAN", "ZION", "CMA", "MTB",
    "PRU", "MET", "TRV", "BK", "STT", "FITB",
    
    # Retail / Medios / Automotriz / Consumo con fuertes caídas o pérdida de valor en décadas
    "WBA", "PARA", "F", "GM", "GPS", "M", "KSS", "JWN", "FL", "BBY",
    "DIS", "GT", "BWA", "WHR", "SWK", "NWL", "CLX", "CPB", "CAG", "K",
    "GIS", "TSN", "HRL", "TAP", "MO",
    
    # Industriales / Minería / Materias Primas / Energía con caídas o ciclos laterales violentos
    "AA", "X", "NEM", "FCX", "HL", "CDE", "APA", "HAL", "SLB", "DVN",
    "OXY", "MMM", "IP", "DD", "EMN", "PPG",
    
    # Farmacéuticas y BioTech clásicas de los 90 con décadas de estancamiento
    "PFE", "BMY", "GILD", "BIIB", "AMGN",
    
    # Utilities y defensivas con retornos nominales aplastados
    "ED", "SO", "DUK", "D", "EXC", "EIX", "PEG"
]

def load_ai_signals():
    print("Loading AI JSON signals for 100 90s/Bearish tickers test...")
    try:
        with open("data/minirocket_gpu_signals.json", "r") as f:
            mr_gpu = json.load(f)
    except Exception: mr_gpu = {}
    try:
        with open("data/minirocket_signals.json", "r") as f:
            mr_bin = json.load(f)
    except Exception: mr_bin = {}
    try:
        with open("data/timesfm_signals.json", "r") as f:
            tsfm = json.load(f)
    except Exception: tsfm = {}
    try:
        with open("data/tspulse_signals.json", "r") as f:
            tsp = json.load(f)
    except Exception: tsp = {}
    try:
        with open("data/xgboost_stack_signals.json", "r") as f:
            xgb = json.load(f)
    except Exception: xgb = {}
    return mr_gpu, mr_bin, tsfm, tsp, xgb

def load_data(ticker):
    file_path = os.path.join(CACHE_DIR, f"{ticker}_daily.csv")
    if not os.path.exists(file_path):
        alt_path = os.path.join(CACHE_DIR, f"{ticker}.csv")
        if os.path.exists(alt_path):
            file_path = alt_path

    if os.path.exists(file_path) and os.path.getsize(file_path) > 100:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    else:
        print(f"Descargando {ticker} (desde 1990 o inicio disponible)...")
        df = yf.download(ticker, start="1990-01-01", progress=False)
        if len(df) == 0:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        df.dropna(inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        df.to_csv(file_path)
    return df

def prepare_df_for_ticker(df, ticker, mr_gpu, mr_bin, tsfm, tsp, xgb):
    c = df['Close']
    o = df['Open']
    v = df['Volume']
    h = df['High']
    l = df['Low']
    
    df['SMA_20'] = c.rolling(20).mean()
    df['STD_20'] = c.rolling(20).std()
    df['SMA_50'] = c.rolling(50).mean()
    df['SMA_200'] = c.rolling(200).mean()
    
    df['Macro_Trend'] = c > df['SMA_200']
    df['Macro_Crash'] = ~df['Macro_Trend'] & (c < c.rolling(50).min().shift(1) * 1.05)
    
    # MFI
    typ_price = (h + l + c) / 3
    raw_mf = typ_price * v
    delta_tp = typ_price.diff()
    pos_mf = np.where(delta_tp > 0, raw_mf, 0.0)
    neg_mf = np.where(delta_tp < 0, raw_mf, 0.0)
    pos_mf_sum = pd.Series(pos_mf, index=df.index).rolling(14).sum()
    neg_mf_sum = pd.Series(neg_mf, index=df.index).rolling(14).sum()
    mfr = pos_mf_sum / np.where(neg_mf_sum == 0, 1, neg_mf_sum)
    df['MFI_14'] = 100 - (100 / (1 + mfr))
    
    # Donchian
    df['Donchian_33_High'] = h.rolling(33).max()
    
    # RSI 14
    delta = c.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / np.where(loss == 0, 1e-10, loss)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # ROC
    df['ROC_3'] = c.pct_change(3) * 100
    df['ROC_5'] = c.pct_change(5) * 100
    
    # RelVol
    df['Vol_SMA20'] = v.rolling(20).mean()
    df['RelVol'] = v / df['Vol_SMA20']
    
    # V10/v0.2.5 Normalized Indicators
    df['VOL_NORM'] = sigmoid_norm((df['RelVol'] - 1.0).astype(float), 1.0)
    roc5 = (c.pct_change(5) * 100).astype(float)
    df['ROC_NORM'] = sigmoid_norm(roc5, 5.0)
    gain_norm = (delta.where(delta > 0, 0)).rolling(14).mean().astype(float)
    loss_norm = (-delta.where(delta < 0, 0)).rolling(14).mean().astype(float)
    rs_norm = gain_norm / (loss_norm + 1e-8)
    df['RSI_NORM'] = 100 - (100 / (1 + rs_norm))
    
    tr1 = h - l
    tr2 = (h - c.shift()).abs()
    tr3 = (l - c.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).astype(float)
    atr14 = tr.rolling(14).mean().astype(float)
    df['ATR_NORM'] = sigmoid_norm(atr14 / c * 100, 2.0)
    
    ret = c.pct_change().fillna(0).astype(float)
    vol_up = v > v.shift(1)
    pvi_change = np.where(vol_up, ret, 0.0)
    pvi = 1000.0 * np.cumprod(1 + pvi_change)
    ema_pvi = pd.Series(pvi, index=df.index).ewm(span=15, adjust=False).mean().astype(float)
    scale_pvi = pd.Series(pvi, index=df.index).rolling(100).std().bfill().astype(float) + 1e-8
    df['KONCORDE_MD'] = sigmoid_norm(pvi - ema_pvi, scale_pvi)
    
    trend = ((c - df['SMA_50']) / df['SMA_50'] * 100).astype(float)
    df['TREND_SMA'] = sigmoid_norm(trend, 5.0)
    
    df['VOL_EXT_NORM'] = sigmoid_norm((df['RelVol'] - 2.0).astype(float), 1.0)
    ema12 = c.ewm(span=12, adjust=False).mean().astype(float)
    ema26 = c.ewm(span=26, adjust=False).mean().astype(float)
    macd = ema12 - ema26
    macd_sig = macd.ewm(span=9, adjust=False).mean().astype(float)
    macd_hist = macd - macd_sig
    scale_macd = macd_hist.rolling(100).std().bfill().astype(float) + 1e-8
    df['MACD_NORM'] = sigmoid_norm(macd_hist, scale_macd)
    
    roc3 = (c.pct_change(3) * 100).astype(float)
    df['ROC_3_NORM'] = sigmoid_norm(roc3, 5.0)
    
    bb_upper_3 = df['SMA_20'] + 3.1*df['STD_20']
    bb_lower_3 = df['SMA_20'] - 3.1*df['STD_20']
    df['BB_3_POS'] = (c - bb_lower_3) / (bb_upper_3 - bb_lower_3 + 1e-8) * 100
    df['BB_3_POS'] = df['BB_3_POS'].clip(0, 100)
    df['MFI_NORM'] = df['MFI_14']
    
    low14 = l.rolling(14).min().astype(float)
    high14 = h.rolling(14).max().astype(float)
    df['STOCH_14'] = 100 * (c - low14) / (high14 - low14 + 1e-8)
    
    df.fillna(50.0, inplace=True)
    
    # AI Columns
    s_gpu = pd.Series(mr_gpu.get(ticker, {}), dtype=float)
    if not s_gpu.empty: s_gpu.index = pd.to_datetime(s_gpu.index)
    df['AI_MINIROCKET_GPU'] = s_gpu.reindex(df.index).astype(float) * 100.0

    s_bin = pd.Series(mr_bin.get(ticker, {}), dtype=float)
    if not s_bin.empty: s_bin.index = pd.to_datetime(s_bin.index)
    df['AI_MINIROCKET_BIN'] = s_bin.reindex(df.index).astype(float) * 100.0

    s_tfm = pd.Series(tsfm.get(ticker, {}), dtype=float)
    if not s_tfm.empty: s_tfm.index = pd.to_datetime(s_tfm.index)
    s_tfm_aligned = s_tfm.reindex(df.index).astype(float)
    df['AI_TIMESFM'] = sigmoid_norm(s_tfm_aligned * 100.0, 2.0)

    s_tsp = pd.Series(tsp.get(ticker, {}), dtype=float)
    if not s_tsp.empty: s_tsp.index = pd.to_datetime(s_tsp.index)
    s_tsp_aligned = s_tsp.reindex(df.index).astype(float)
    df['AI_TSPULSE'] = sigmoid_norm(s_tsp_aligned * 100.0, 2.0)

    s_xgb = pd.Series(xgb.get(ticker, {}), dtype=float)
    if not s_xgb.empty: s_xgb.index = pd.to_datetime(s_xgb.index)
    df['AI_XGBOOST'] = s_xgb.reindex(df.index).astype(float) * 100.0

    return df

def run_test():
    all_strategies = STRATEGY_INFO.copy()
    print(f"Total estrategias a evaluar en activos 90s/Bearish: {len(all_strategies)}")
    
    mr_gpu, mr_bin, tsfm, tsp, xgb = load_ai_signals()
    
    spy_df = load_data("SPY")
    if spy_df is None or len(spy_df) < 500:
        raise ValueError("No se pudo cargar datos suficientes para SPY.")
        
    spy_idx = spy_df.index
    spy_ret = spy_df['Close'].pct_change().fillna(0)
    
    results_by_strategy = {s: {'beats': 0, 'trades': [], 'returns': [], 'maxdd': [], 'cagr': []} for s in all_strategies}
    
    valid_tickers = 0
    for ticker in TICKERS_100_BEARISH_90S:
        df = load_data(ticker)
        if df is None or len(df) < 500:
            print(f"Saltando {ticker} por falta de datos suficientes.")
            continue
            
        valid_tickers += 1
        
        # B&H Benchmark exacto
        bh_metrics = simulate_buy_and_hold(df, initial_capital=10000.0, commission=0.004)
        bh_ret = bh_metrics['total_return']
        
        df = prepare_df_for_ticker(df, ticker, mr_gpu, mr_bin, tsfm, tsp, xgb)
        opens = df['Open'].values
        closes = df['Close'].values
        dates = df.index
        
        for s_code, s_info in all_strategies.items():
            ls, es = generate_signals(df, ticker, spy_idx, spy_ret, s_info, commission=0.004)
            
            signals_strict = None
            if (s_code.startswith("SS") and s_code != "SS11") or s_code in ["AIS10", "AIS11"]:
                signals_strict = (df['Close'] > df['SMA_20']).values
                
            eq, tr, _ = run_simulation(
                ls, es, opens, closes, dates,
                initial_capital=10000.0,
                commission=0.004,
                stop_loss_pct=s_info.get("stop_loss_pct"),
                signals_strict=signals_strict
            )
            
            mets, _ = compute_metrics(eq, tr, dates, initial_capital=10000.0)
            
            strat_ret = mets['total_return']
            strat_cagr = mets['cagr']
            max_dd = mets['max_drawdown']
            trades = mets['num_trades']
            
            results_by_strategy[s_code]['trades'].append(trades)
            results_by_strategy[s_code]['returns'].append(strat_ret)
            results_by_strategy[s_code]['maxdd'].append(max_dd)
            results_by_strategy[s_code]['cagr'].append(strat_cagr)
            if strat_ret > bh_ret:
                results_by_strategy[s_code]['beats'] += 1
                
        if valid_tickers % 10 == 0:
            print(f"Progreso: {valid_tickers} tickers de los 90s procesados...")

    print(f"\n==========================================================================")
    print(f"   REPORTE FINAL GENERAL ({valid_tickers} Activos 90s/Bearish - v0.2.5)   ")
    print(f"==========================================================================")
    
    report_lines = []
    report_lines.append(f"# Test de 100 Activos de los Años 90 con Sesgo Bajista/Lateral (v0.2.5)\n")
    report_lines.append(f"Activos válidos analizados: {valid_tickers} | Comisiones Reales: 0.4% por trade\n")
    report_lines.append("| Ranking | Estrategia | WinRate vs B&H | Retorno Promedio | CAGR Promedio | Max DD Promedio | Trades Promedio |")
    report_lines.append("|---|---|---|---|---|---|---|")
    
    sorted_strats = sorted(
        results_by_strategy.items(),
        key=lambda x: (x[1]['beats'], np.mean(x[1]['returns'])),
        reverse=True
    )
    
    for rank, (s_code, data) in enumerate(sorted_strats, 1):
        winrate = (data['beats'] / valid_tickers) * 100
        avg_ret = np.mean(data['returns'])
        avg_cagr = np.mean(data['cagr'])
        avg_dd = np.mean(data['maxdd'])
        avg_trades = np.mean(data['trades'])
        
        report_lines.append(f"| #{rank} | **{s_code}** | **{winrate:.1f}% ({data['beats']}/{valid_tickers})** | {avg_ret:,.1f}% | {avg_cagr:.2f}% | {avg_dd:.1f}% | {avg_trades:.0f} |")
        print(f"Rank #{rank:2d} | {s_code:15s} | Beats: {data['beats']:2d}/{valid_tickers} ({winrate:5.1f}%) | Ret: {avg_ret:10,.1f}% | CAGR: {avg_cagr:6.2f}% | MaxDD: {avg_dd:5.1f}% | Trades: {avg_trades:4.0f}")
        
    with open('artifacts_report_100_bearish.md', 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
        
    try:
        os.makedirs('data', exist_ok=True)
        with open('data/report_100_bearish_v0.2.5.md', 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))
    except Exception as e:
        print(f"No se pudo guardar copia en data/: {e}")
        
    print("\nReporte guardado exitosamente en 'artifacts_report_100_bearish.md' y 'data/report_100_bearish_v0.2.5.md'.")

if __name__ == "__main__":
    run_test()
