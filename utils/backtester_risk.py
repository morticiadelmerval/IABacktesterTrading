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

# -------------------------------------------------------------------------
# 1. Data Fetching
# -------------------------------------------------------------------------
def fetch_data():
    data = {}
    for ticker in TICKERS:
        cache_path = os.path.join(CACHE_DIR, f"{ticker}.csv")
        
        # Invalidate cache if it's older than today
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
        
        # Pre-compute indicators for V10 strategies
        c = df['Close']
        o = df['Open']
        v = df['Volume']
        h = df['High']
        
        # Bollinger
        df['SMA_20'] = c.rolling(20).mean()
        df['STD_20'] = c.rolling(20).std()
        
        # MFI (Money Flow Index)
        typ_price = (h + df['Low'] + c) / 3
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
        
        # ROC
        df['ROC_3'] = c.pct_change(3) * 100
        df['ROC_5'] = c.pct_change(5) * 100
        
        # RelVol
        df['Vol_SMA20'] = v.rolling(20).mean()
        df['RelVol'] = v / df['Vol_SMA20']
        
        # ATR 14
        high_low = h - df['Low']
        high_close = np.abs(h - c.shift())
        low_close = np.abs(df['Low'] - c.shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR_14'] = true_range.rolling(14).mean()
        df['ATR_Pct'] = df['ATR_14'] / c * 100
        
        data[ticker] = df
    return data

# -------------------------------------------------------------------------
# 2. Simulation Engine
# -------------------------------------------------------------------------
def run_simulation(signals_long, signals_exit, opens, closes, dates, initial_capital=10000.0, commission=0.004):
    n = len(closes)
    equity = np.full(n, float(initial_capital))
    cash   = float(initial_capital)
    pos    = 0.0
    in_pos = False
    entry_price = 0.0
    entry_idx   = 0
    trades = []

    for i in range(1, n):
        if not in_pos and signals_long[i-1]:
            pos = (cash * (1.0 - commission)) / opens[i]
            cash = 0.0
            in_pos = True
            entry_price = opens[i]
            entry_idx   = i
        elif in_pos and signals_exit[i-1]:
            revenue = pos * opens[i] * (1.0 - commission)
            pnl     = revenue - (pos * entry_price / (1.0 - commission)) # Approx cost
            pct_ret = (revenue / (pos * entry_price / (1.0 - commission)) - 1.0) * 100.0
            trades.append({
                "entry_date":   dates[entry_idx].strftime("%Y-%m-%d"),
                "entry_price":  float(entry_price),
                "exit_date":    dates[i].strftime("%Y-%m-%d"),
                "exit_price":   float(opens[i]),
                "pct_return":   float(pct_ret),
                "pnl":          float(pnl),
                "reason":       "Active/Macro Signal",
                "duration_days": int((dates[i] - dates[entry_idx]).days)
            })
            cash   = revenue
            pos    = 0.0
            in_pos = False

        equity[i] = cash + pos * closes[i]

    if in_pos:
        revenue = pos * closes[-1] * (1.0 - commission)
        pnl     = revenue - (pos * entry_price / (1.0 - commission))
        pct_ret = (revenue / (pos * entry_price / (1.0 - commission)) - 1.0) * 100.0
        trades.append({
            "entry_date":   dates[entry_idx].strftime("%Y-%m-%d"),
            "entry_price":  float(entry_price),
            "exit_date":    dates[-1].strftime("%Y-%m-%d"),
            "exit_price":   float(closes[-1]),
            "pct_return":   float(pct_ret),
            "pnl":          float(pnl),
            "reason":       "End of History",
            "duration_days": int((dates[-1] - dates[entry_idx]).days)
        })
        equity[-1] = cash + revenue

    return equity, trades

def compute_metrics(equity, trades, dates, initial_capital=10000.0):
    equity_series = pd.Series(equity, index=dates)
    daily_ret = equity_series.pct_change().dropna()
    total_return = (equity[-1] / initial_capital - 1.0) * 100.0
    years = (dates[-1] - dates[0]).days / 365.25
    cagr  = ((equity[-1] / initial_capital) ** (1.0 / years) - 1.0) * 100.0 if years > 0 else 0.0
    std   = daily_ret.std()
    sharpe= (daily_ret.mean() / std * np.sqrt(252)) if std > 0 else 0.0
    ds    = daily_ret[daily_ret < 0].std()
    sortino = (daily_ret.mean() / ds * np.sqrt(252)) if ds > 0 else 0.0
    roll_max  = equity_series.cummax()
    drawdowns = (equity_series - roll_max) / roll_max
    max_dd    = drawdowns.min() * 100.0
    n = len(trades)
    win_rate = (sum(1 for t in trades if t['pct_return'] > 0) / n * 100.0) if n > 0 else 0.0
    gains = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    losses = sum(abs(t['pnl']) for t in trades if t['pnl'] < 0)
    pf    = (gains / losses) if losses > 0 else (999.0 if gains > 0 else 1.0)
    avg_dur = np.mean([t['duration_days'] for t in trades]) if n > 0 else 0.0

    step = max(1, len(equity_series) // 300)
    ec = [{"date": d.strftime("%Y-%m-%d"), "value": float(v)}
          for d, v in zip(equity_series.index[::step], equity_series.values[::step])]
    if ec[-1]['date'] != dates[-1].strftime("%Y-%m-%d"):
        ec.append({"date": dates[-1].strftime("%Y-%m-%d"), "value": float(equity[-1])})

    return {
        "total_return":  float(total_return),
        "cagr":          float(cagr),
        "sharpe":        float(sharpe),
        "sortino":       float(sortino),
        "max_drawdown":  float(max_dd),
        "num_trades":    int(n),
        "win_rate":      float(win_rate),
        "profit_factor": float(pf),
        "avg_duration":  float(avg_dur),
        "ending_val":    float(equity[-1])
    }, ec

def simulate_buy_and_hold(df, initial_capital=10000.0, commission=0.004):
    dates = df.index
    closes = df['Close'].values
    opens = df['Open'].values
    buy_price = opens[0]
    shares = (initial_capital * (1.0 - commission)) / buy_price
    # Assume commission on final liquidation for equity curve
    equity = shares * closes * (1.0 - commission)
    es = pd.Series(equity, index=dates)
    dr = es.pct_change().dropna()
    total_ret = (equity[-1] / initial_capital - 1.0) * 100.0
    years = (dates[-1] - dates[0]).days / 365.25
    cagr  = ((equity[-1] / initial_capital) ** (1.0 / years) - 1.0) * 100.0 if years > 0 else 0.0
    std   = dr.std()
    sharpe= (dr.mean() / std * np.sqrt(252)) if std > 0 else 0.0
    rm = es.cummax()
    max_dd = ((es - rm) / rm).min() * 100.0
    step = max(1, len(es) // 300)
    ec = [{"date": d.strftime("%Y-%m-%d"), "value": float(v)}
          for d, v in zip(es.index[::step], es.values[::step])]
    if ec[-1]['date'] != dates[-1].strftime("%Y-%m-%d"):
        ec.append({"date": dates[-1].strftime("%Y-%m-%d"), "value": float(equity[-1])})
    return {"total_return": float(total_ret), "cagr": float(cagr),
            "sharpe": float(sharpe), "max_drawdown": float(max_dd),
            "ending_val": float(equity[-1]), "equity_curve": ec}

# -------------------------------------------------------------------------
# 3. Strategy Definitions (V10 - Supreme Master Set)
# -------------------------------------------------------------------------
STRATEGY_INFO = {
    "S01": {"type": "ROC_VOL", "roc": 3, "roc_val": 10.0, "rel_vol": 2.0, "name": "Macro + Extremo Volumen & Momentum (V10)"},
    "S02": {"type": "ROC_VOL", "roc": 5, "roc_val": 15.0, "rel_vol": 2.0, "name": "Macro + Volatilidad de Precio 5D (V10)"},
    "S03": {"type": "ROC_VOL", "roc": 5, "roc_val": 10.0, "rel_vol": 2.0, "name": "Macro + Despegue Agresivo (V10)"},
    "S04": {"type": "ROC_VOL", "roc": 3, "roc_val": 10.0, "rel_vol": 3.0, "name": "Macro + Spike Volumen x3 (V10)"},
    "S05": {"type": "ROC_VOL", "roc": 3, "roc_val": 15.0, "rel_vol": 2.0, "name": "Macro + ROC Extremo 15% (V10)"},
    "S06": {"type": "ROC", "roc": 3, "roc_val": 10.0, "name": "Macro + Momentum Puro a 3 Días (V10)"},
    "S07": {"type": "RELVOL", "rel_vol": 2.0, "name": "Macro + Anomalía de Volumen Alcista (V10)"},
    "S08": {"type": "DONCHIAN", "period": 33, "name": "Macro + Donchian Channel Breakout (V10)"},
    "S09": {"type": "MFI", "thresh": 85.0, "name": "Macro + Exhaustion Flow Index (V10)"},
    "S10": {"type": "BB", "mult": 3.1, "name": "Macro + Bollinger Ultra Estirado (V9 Clásico)"},
    "S11": {"type": "TIMESFM", "name": "Macro + TimesFM 200M AI Oracle (GPU Edition)"},
}

for s_id, params in STRATEGY_INFO.items():
    stype = params["type"]
    if stype == "ROC_VOL":
        roc, rval, rv = params["roc"], params["roc_val"], params["rel_vol"]
        desc = f"La 'Joya Oculta' encontrada por subagentes en V10. Además del Filtro Macro, hace toma de ganancias cuando el precio salta un >{rval}% en {roc} días, O si hay un pico de volumen >{rv}x del promedio en un día verde. Aprovecha euforias súbitas."
        inds = ["SPY Macro Crash Guard", f"ROC {roc} > {rval}%", f"Relative Volume > {rv}x"]
        pine_active = f"roc = ta.roc(close, {roc})\nvolSma = ta.sma(volume, 20)\nrelVol = volume / volSma\nactiveExit = (roc > {rval}) or (relVol > {rv} and close > open)"
    elif stype == "ROC":
        roc, rval = params["roc"], params["roc_val"]
        desc = f"V10 Momentum Puro: Vende al hacer un salto de >{rval}% en solo {roc} días para bloquear ganancias rápidas."
        inds = ["SPY Macro Crash Guard", f"ROC {roc} > {rval}%"]
        pine_active = f"roc = ta.roc(close, {roc})\nactiveExit = (roc > {rval})"
    elif stype == "RELVOL":
        rv = params["rel_vol"]
        desc = f"V10 Anomalía de Volumen: Vende solo en días donde el precio sube pero inyectando un volumen brutal >{rv}x del promedio, indicando el climax de un rally."
        inds = ["SPY Macro Crash Guard", f"Relative Volume > {rv}x"]
        pine_active = f"volSma = ta.sma(volume, 20)\nrelVol = volume / volSma\nactiveExit = (relVol > {rv} and close > open)"
    elif stype == "DONCHIAN":
        p = params["period"]
        desc = f"V10 Donchian Channels: Vende cuando el precio perfora el máximo absoluto de los últimos {p} días. Permite aguantar mucho, y salir justo en el quiebre de la cima."
        inds = ["SPY Macro Crash Guard", f"Donchian Channel ({p})"]
        pine_active = f"donchianHi = ta.highest(high, {p})\nactiveExit = (close > donchianHi[1])"
    elif stype == "MFI":
        th = params["thresh"]
        desc = f"V10 Money Flow Index: MFI es RSI pesado por volumen. Si llega a >{th}, el activo está hiper-sobrecomprado de forma peligrosa. Toma de ganancias inminente."
        inds = ["SPY Macro Crash Guard", f"MFI 14 > {th}"]
        pine_active = f"mfi = ta.mfi(close, 14)\nactiveExit = (mfi > {th})"
    elif stype == "BB":
        mult = params["mult"]
        desc = f"El clásico de V9: Toma de ganancias en la banda de Bollinger >{mult}x desviaciones estándar."
        inds = ["SPY Macro Crash Guard", f"Bollinger Bands (20, {mult:.1f})"]
        pine_active = f"bbUp = ta.sma(close, 20) + {mult:.1f} * ta.stdev(close, 20)\nactiveExit = close > bbUp"
    elif stype == "TIMESFM":
        desc = "V11: Inteligencia Artificial Pura. Usa Google TimesFM ejecutado en tu GPU GPU local para predecir numéricamente los próximos 5 días de trading. Compra si la predicción espera > 1.5% de ganancia."
        inds = ["Google TimesFM 200M/500M", "Batch GPU Inference", "SPY Macro Crash Guard"]
        pine_active = "//@version=5\n// [ADVERTENCIA] ESTA ESTRATEGIA NO SE PUEDE EJECUTAR EN TRADINGVIEW.\n// TradingView y Pine Script v5 no soportan importar modelos de Redes Neuronales locales (Hugging Face / PyTorch).\n// Las señales de S11 solo pueden procesarse en tu GPU local local."

    params["desc"] = desc
    params["indicators"] = inds
    params["pinescript"] = f"""//@version=5
strategy("{params['name']}", overlay=true, initial_capital=10000, default_qty_type=strategy.percent_of_equity, default_qty_value=100)

// 1. Filtro Macro Global
spyClose = request.security("SPY", "D", close)
spyRet   = (spyClose - spyClose[1]) / spyClose[1]

var int daysOut = 0
if spyRet < -0.078
    daysOut := 8

// 2. Filtro Activo (V10 Subagent AI)
{pine_active}

// Lógica de Trading Combinada
if daysOut > 0 or activeExit
    strategy.close("Long")
    if daysOut > 0
        daysOut := daysOut - 1
else
    strategy.entry("Long", strategy.long)

if barstate.isfirst
    strategy.entry("Long", strategy.long)
"""

def generate_signals(df, ticker, spy_idx, spy_ret, params, commission=0.0):
    # Macro Exit
    spy_exit_mask = np.zeros(len(spy_ret), dtype=bool)
    days_out = 0
    for i in range(1, len(spy_ret)):
        if spy_ret.iloc[i] < -0.078:
            days_out = 8
        if days_out > 0:
            spy_exit_mask[i] = True
            days_out -= 1
            
    spy_exit_series = pd.Series(spy_exit_mask, index=spy_idx)
    macro_aligned = spy_exit_series.reindex(df.index, fill_value=False).values
    
    # Active Exit
    stype = params["type"]
    if stype == "ROC_VOL":
        cond1 = df[f'ROC_{params["roc"]}'] > params["roc_val"]
        cond2 = (df['RelVol'] > params["rel_vol"]) & (df['Close'] > df['Open'])
        active_exit = cond1 | cond2
    elif stype == "ROC":
        active_exit = df[f'ROC_{params["roc"]}'] > params["roc_val"]
    elif stype == "RELVOL":
        active_exit = (df['RelVol'] > params["rel_vol"]) & (df['Close'] > df['Open'])
    elif stype == "DONCHIAN":
        # Shift the donchian high so we are comparing close with PREVIOUS high
        dh = df[f'Donchian_{params["period"]}_High'].shift(1)
        active_exit = df['Close'] > dh
    elif stype == "MFI":
        active_exit = df['MFI_14'] > params["thresh"]
    elif stype == "BB":
        bb_up = df['SMA_20'] + params["mult"] * df['STD_20']
        active_exit = df['Close'] > bb_up
    elif stype == "TIMESFM":
        if not hasattr(generate_signals, 'tsfm_cache'):
            try:
                import json
                with open("timesfm_signals.json", "r") as f:
                    generate_signals.tsfm_cache = json.load(f)
            except:
                generate_signals.tsfm_cache = {}
        
        preds_dict = generate_signals.tsfm_cache.get(ticker, {})
        preds_series = pd.Series(preds_dict)
        if not preds_series.empty:
            preds_series.index = pd.to_datetime(preds_series.index)
        preds_aligned = preds_series.reindex(df.index, fill_value=0.0)
        
        cost_barrier = commission * 2.0
        
        # Volatility-adjusted thresholds
        atr_pct = df['ATR_Pct'].fillna(1.0)
        
        # Dynamic threshold based on ATR
        long_threshold = 0.012 + cost_barrier + (atr_pct * 0.002)
        active_long = preds_aligned > long_threshold
        
        # AI predicting a drop worse than commission cost
        ai_wants_out = preds_aligned < (-commission - (atr_pct * 0.001))
        
        # Guardrail: Is the stock in a strong short-term uptrend?
        strong_uptrend = df['Close'] > df['SMA_20']
        
        # Only exit if AI wants out AND (uptrend is broken OR AI predicts a severe drop > dynamic)
        severe_drop = -0.015 - (atr_pct * 0.002)
        active_exit = ai_wants_out & (~strong_uptrend | (preds_aligned < severe_drop))
        
        es = macro_aligned | active_exit.values
        ls = active_long.values & (~macro_aligned)
        return ls, es
        
    es = macro_aligned | active_exit.values
    ls = ~es
    return ls, es

# -------------------------------------------------------------------------
# 4. Main Pipeline / API Entry
# -------------------------------------------------------------------------
CACHED_DATA_DICT = None

def run_all(commission=0.0):
    global CACHED_DATA_DICT
    print(f"=== RECALCULANDO PIPELINE V10 (Comision: {commission*100:.2f}%) ===")
    
    if CACHED_DATA_DICT is None:
        CACHED_DATA_DICT = fetch_data()
    data_dict = CACHED_DATA_DICT
    
    spy_df = data_dict["SPY"]
    spy_ret = spy_df['Close'].pct_change()
    spy_idx = spy_df.index
    
    benchmarks = {}
    for ticker, df in data_dict.items():
        benchmarks[ticker] = simulate_buy_and_hold(df, commission=commission)
        print(f"B&H {ticker:5s}: {benchmarks[ticker]['total_return']:8.2f}% | MaxDD={benchmarks[ticker]['max_drawdown']:.2f}%")
        
    print("\n--- Running 10 V10 Strategies (30-Year) ---")
    all_results = {}
    strategy_ids = ["S11"] # Test only S11 for now to make it faster
    # strategy_ids = list(STRATEGY_INFO.keys())

    for s_id in strategy_ids:
        s_info = STRATEGY_INFO[s_id]
        all_results[s_id] = {}
        print(f"\n{s_id} ({s_info['name']}):")
        
        for ticker in TICKERS:
            df = data_dict[ticker]
            opens = df['Open'].values
            closes= df['Close'].values
            dates = df.index
            
            ls, es = generate_signals(df, ticker, spy_idx, spy_ret, s_info, commission)
            
            eq, tr = run_simulation(ls, es, opens, closes, dates, commission=commission)
            mets, ec = compute_metrics(eq, tr, dates)
            
            bh = benchmarks[ticker]
            beat = mets["total_return"] > bh["total_return"]
            
            all_results[s_id][ticker] = {
                "metrics": mets,
                "outperformed": beat,
                "trades": tr,
                "equity_curve": ec
            }
            flag = "BEAT" if beat else "----"
            print(f"  {ticker:5s}: {mets['total_return']:8.2f}% vs BH {bh['total_return']:8.2f}% [{flag}]  MaxDD={mets['max_drawdown']:.1f}%  Trades={mets['num_trades']}")

    print("\n--- FINAL RANKING ---")
    ranking = []
    spy_raw = []

    for s_id in strategy_ids:
        s_info = STRATEGY_INFO[s_id]
        td     = all_results[s_id]
        avg_ret  = np.mean([d["metrics"]["total_return"] for d in td.values()])
        avg_cagr = np.mean([d["metrics"]["cagr"]         for d in td.values()])
        avg_sh   = np.mean([d["metrics"]["sharpe"]        for d in td.values()])
        avg_dd   = np.mean([d["metrics"]["max_drawdown"]  for d in td.values()])
        avg_wr   = np.mean([d["metrics"]["win_rate"]      for d in td.values()])
        pf_vals  = [d["metrics"]["profit_factor"] for d in td.values() if d["metrics"]["profit_factor"] != 999.0]
        avg_pf   = float(np.mean(pf_vals)) if pf_vals else 1.0
        tot_trd  = sum(d["metrics"]["num_trades"] for d in td.values())
        out_cnt  = sum(1 for d in td.values() if d["outperformed"])

        score = (avg_sh * 50) + avg_cagr - (avg_dd * 0.5) + (out_cnt * 5)

        ranking.append({
            "strategy_id": s_id,
            "name":        s_info["name"],
            "description": s_info["desc"],
            "indicators":  s_info["indicators"],
            "pinescript":  s_info["pinescript"],
            "score":       float(score),
            "aggregate_metrics": {
                "avg_return":       float(avg_ret),
                "avg_cagr":         float(avg_cagr),
                "avg_sharpe":       float(avg_sh),
                "avg_max_dd":       float(avg_dd),
                "avg_win_rate":     float(avg_wr),
                "avg_profit_factor":float(avg_pf if not np.isnan(avg_pf) else 1.0),
                "total_trades":     int(tot_trd),
                "outperform_count": int(out_cnt)
            },
            "ticker_results": td
        })

        spy_raw.append({
            "strategy_id": s_id,
            "metrics":     td["SPY"]["metrics"],
            "outperformed":td["SPY"]["outperformed"],
            "trades":      td["SPY"]["trades"],
            "equity_curve":td["SPY"]["equity_curve"]
        })

    ranking = sorted(ranking, key=lambda x: x["aggregate_metrics"]["avg_return"], reverse=True)
    spy_raw = sorted(spy_raw, key=lambda x: x["metrics"]["total_return"], reverse=True)

    for i, r in enumerate(ranking):
        print(f"Rank {i+1}: {r['strategy_id']} ({r['name']}) | Trades={r['aggregate_metrics']['total_trades']} | "
              f"AvgReturn={r['aggregate_metrics']['avg_return']:.2f}% | "
              f"Beat B&H: {r['aggregate_metrics']['outperform_count']}/11")

    simplified_benchmarks = {t: {k: v for k, v in d.items()} for t, d in benchmarks.items()}

    output = {
        "metadata": {
            "start_date":            START_DATE,
            "end_date":              END_DATE,
            "tickers":               TICKERS,
            "num_strategies_tested": len(STRATEGY_INFO),
            "version":               "V10 - Subagent Supreme AI",
            "generated_at":          datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "benchmarks":      simplified_benchmarks,
        "ranking":         ranking,
        "spy_raw_ranking": spy_raw
    }

    return output

if __name__ == "__main__":
    output = run_all(commission=0.004)
    with open("results_risk.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print("\nResults saved to results_risk.json")

