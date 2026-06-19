import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

CACHE_DIR = ".data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

TICKERS = ["SPY", "QQQ", "DIA", "IWM", "MCD", "KO", "MSFT", "GOOG", "V", "C", "XOM", "NU", "GLD", "IVE", "EWZ", "PBR", "BRK-B", "O"]
START_DATE = "1996-01-01"
END_DATE = datetime.today().strftime('%Y-%m-%d')

# -------------------------------------------------------------------------
# 1. Data Fetching
# -------------------------------------------------------------------------
def fetch_data():
    data = {}
    for ticker in TICKERS:
        cache_path = os.path.join(CACHE_DIR, f"{ticker}.csv")
        
        # Invalidate cache if it's older than 1 minute (for live price updates)
        cache_valid = False
        if os.path.exists(cache_path):
            mod_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
            if (datetime.now() - mod_time).total_seconds() < 60:
                cache_valid = True

        if cache_valid:
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            df.index.name = "Date"
        else:
            df = yf.download(ticker, start=START_DATE)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
            df.to_csv(cache_path)
        
        # Pre-compute indicators for V10 strategies
        c = df['Close']
        o = df['Open']
        v = df['Volume']
        h = df['High']
        
        # Bollinger & Trend
        df['SMA_20'] = c.rolling(20).mean()
        df['STD_20'] = c.rolling(20).std()
        df['SMA_50'] = c.rolling(50).mean()
        df['SMA_200'] = c.rolling(200).mean()
        
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
        
        # RSI 14
        delta = c.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / np.where(loss == 0, 1e-10, loss) # avoid div by zero
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # ROC
        df['ROC_3'] = c.pct_change(3) * 100
        df['ROC_5'] = c.pct_change(5) * 100
        
        # RelVol
        df['Vol_SMA20'] = v.rolling(20).mean()
        df['RelVol'] = v / df['Vol_SMA20']
        
        data[ticker] = df
    return data

# -------------------------------------------------------------------------
# 2. Simulation Engine
# -------------------------------------------------------------------------
def run_simulation(signals_long, signals_exit, opens, closes, dates, initial_capital=10000.0, commission=0.004, stop_loss_pct=None):
    n = len(closes)
    equity = np.full(n, float(initial_capital))
    cash   = float(initial_capital)
    pos    = 0.0
    in_pos = False
    entry_price = 0.0
    entry_idx   = 0
    trades = []

    for i in range(1, n):
        hit_stop_loss = False
        if in_pos and stop_loss_pct is not None:
            current_loss = (closes[i-1] / entry_price - 1.0) * 100.0
            if current_loss <= stop_loss_pct:
                hit_stop_loss = True

        if not in_pos and signals_long[i-1]:
            pos = (cash * (1.0 - commission)) / opens[i]
            cash = 0.0
            in_pos = True
            entry_price = opens[i]
            entry_idx   = i
        elif in_pos and (signals_exit[i-1] or hit_stop_loss):
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
                "reason":       "Stop Loss" if hit_stop_loss else "Active/Macro Signal",
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

    is_open = False
    if n > 0 and trades[-1].get("reason") == "End of History":
        is_open = True

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
        "ending_val":    float(equity[-1]),
        "is_open":       is_open
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
    "SS01": {"type": "ROC_VOL", "roc": 3, "roc_val": 10.0, "rel_vol": 2.0, "name": "Macro + Extremo Volumen & Momentum"},
    "SS02": {"type": "ROC_VOL", "roc": 5, "roc_val": 15.0, "rel_vol": 2.0, "name": "Macro + Volatilidad de Precio 5D"},
    "SS03": {"type": "ROC_VOL", "roc": 5, "roc_val": 10.0, "rel_vol": 2.0, "name": "Macro + Despegue Agresivo"},
    "SS04": {"type": "ROC_VOL", "roc": 3, "roc_val": 10.0, "rel_vol": 3.0, "name": "Macro + Spike Volumen x3"},
    "SS05": {"type": "ROC_VOL", "roc": 3, "roc_val": 15.0, "rel_vol": 2.0, "name": "Macro + ROC Extremo 15%"},
    "SS06": {"type": "ROC", "roc": 3, "roc_val": 10.0, "name": "Macro + Momentum Puro a 3 Días", "stop_loss_pct": -15.0},
    "SS07": {"type": "RELVOL", "rel_vol": 2.0, "name": "Macro + Anomalía de Volumen Alcista"},
    "SS08": {"type": "DONCHIAN", "period": 33, "name": "Macro + Donchian Channel Breakout", "stop_loss_pct": -25.0},
    "SS09": {"type": "MFI", "thresh": 85.0, "name": "Macro + Exhaustion Flow Index", "stop_loss_pct": -10.0},
    "SS10": {"type": "BB", "mult": 3.1, "name": "Macro + Bollinger Ultra Estirado", "stop_loss_pct": -15.0},
    "AIS01": {"type": "TIMESFM_PURE", "name": "Macro + TimesFM 200M AI Oracle (GPU Edition) - Ideal 0% Comisiones", "stop_loss_pct": -15.0},
    "AIS02": {"type": "TIMESFM_SMART", "name": "Macro + TimesFM Smart Hold (GPU Edition) - Ideal 0.4% Comisiones", "stop_loss_pct": -10.0},
    "AIS03": {"type": "TIMESFM_ADAPTIVE", "name": "Macro + TimesFM Adaptive Volatility (GPU Edition) - Rango Dinámico", "stop_loss_pct": -10.0},
    "AIS04": {"type": "TSPULSE_PURE", "name": "Macro + IBM TSPulse AI (Univariate) - Pura"},
    "AIS05": {"type": "TSPULSE_OSC", "name": "Macro + IBM TSPulse AI (Hybrid) - MFI & BB", "stop_loss_pct": -15.0},
    "AIS06": {"type": "TSPULSE_MOM", "name": "Macro + IBM TSPulse AI (Hybrid) - RelVol & ROC"},
    "AIS07": {"type": "MINIROCKET", "name": "Macro + MiniRocket AI (Classification) - Binary", "stop_loss_pct": -15.0},
    "AIS08": {"type": "MINIROCKET_GPU", "name": "Macro + MiniRocketPlus AI (Probabilities) - GPU"},
    "AIS09": {"type": "MINIROCKET_STACK", "name": "Macro + MiniRocket Stack XGBoost", "stop_loss_pct": -15.0},
    "SS11": {"type": "MACRO_BASE_PURA", "name": "Macro Base Pura (Buy & Hold con Seguro Anti-Crash)", "stop_loss_pct": -15.0},
}

for s_id, params in STRATEGY_INFO.items():
    stype = params["type"]
    if stype == "ROC_VOL":
        roc, rval, rv = params["roc"], params["roc_val"], params["rel_vol"]
        desc = f"Estrategia de Volumen y Momentum. Además del Filtro Macro, hace toma de ganancias cuando el precio salta un >{rval}% en {roc} días, O si hay un pico de volumen >{rv}x del promedio en un día verde. Aprovecha euforias súbitas."
        inds = ["SPY Macro Crash Guard", f"ROC {roc} > {rval}%", f"Relative Volume > {rv}x"]
        pine_active = f"roc = ta.roc(close, {roc})\nvolSma = ta.sma(volume, 20)\nrelVol = volume / volSma\nactiveExit = (roc > {rval}) or (relVol > {rv} and close > open)"
    elif stype == "ROC":
        roc, rval = params["roc"], params["roc_val"]
        desc = f"Momentum Puro: Vende al hacer un salto de >{rval}% en solo {roc} días para bloquear ganancias rápidas."
        inds = ["SPY Macro Crash Guard", f"ROC {roc} > {rval}%"]
        pine_active = f"roc = ta.roc(close, {roc})\nactiveExit = (roc > {rval})"
    elif stype == "RELVOL":
        rv = params["rel_vol"]
        desc = f"Anomalía de Volumen: Vende solo en días donde el precio sube pero inyectando un volumen brutal >{rv}x del promedio, indicando el climax de un rally."
        inds = ["SPY Macro Crash Guard", f"Relative Volume > {rv}x"]
        pine_active = f"volSma = ta.sma(volume, 20)\nrelVol = volume / volSma\nactiveExit = (relVol > {rv} and close > open)"
    elif stype == "DONCHIAN":
        p = params["period"]
        desc = f"Donchian Channels: Vende cuando el precio perfora el máximo absoluto de los últimos {p} días. Permite aguantar mucho, y salir justo en el quiebre de la cima."
        inds = ["SPY Macro Crash Guard", f"Donchian Channel ({p})"]
        pine_active = f"donchianHi = ta.highest(high, {p})\nactiveExit = (close > donchianHi[1])"
    elif stype == "MFI":
        th = params["thresh"]
        desc = f"Money Flow Index: MFI es RSI pesado por volumen. Si llega a >{th}, el activo está hiper-sobrecomprado de forma peligrosa. Toma de ganancias inminente."
        inds = ["SPY Macro Crash Guard", f"MFI 14 > {th}"]
        pine_active = f"mfi = ta.mfi(close, 14)\nactiveExit = (mfi > {th})"
    elif stype == "BB":
        mult = params["mult"]
        desc = f"Estrategia de Reversión Clásica: Toma de ganancias en la banda de Bollinger >{mult}x desviaciones estándar."
        inds = ["SPY Macro Crash Guard", f"Bollinger Bands (20, {mult:.1f})"]
        pine_active = f"bbUp = ta.sma(close, 20) + {mult:.1f} * ta.stdev(close, 20)\nactiveExit = close > bbUp"
    elif stype == "TIMESFM_PURE":
        desc = "AIS01: Inteligencia Artificial Pura. Usa Google TimesFM ejecutado en GPU local para predecir los próximos 5 días de trading. Compra si espera > 1.5%. Optimizado para brokers sin comisiones."
        inds = ["Google TimesFM 200M/500M", "Batch GPU Inference", "SPY Macro Crash Guard"]
        pine_active = "//@version=5\n// [ADVERTENCIA] ESTA ESTRATEGIA NO SE PUEDE EJECUTAR EN TRADINGVIEW.\n// TradingView y Pine Script v5 no soportan importar modelos de Redes Neuronales locales (Hugging Face / PyTorch)."
    elif stype == "TIMESFM_SMART":
        desc = "AIS02: Inteligencia Artificial Híbrida. Combina predicciones de IA con la media móvil de 50 días. Opera en modalidad 'Smart Hold' para minimizar operaciones. Optimizado para dominar Buy & Hold incluso pagando comisiones del 0.4%."
        inds = ["Google TimesFM", "SMA 50 Guardrail", "SPY Macro Crash Guard"]
        pine_active = "//@version=5\n// TradingView no soporta IA local."
    elif stype == "TIMESFM_ADAPTIVE":
        desc = "AIS03: Inteligencia Artificial Adaptativa. Combina IA con Bandas de Bollinger y MFI. Adapta su stop-loss dinámicamente según la volatilidad del activo y hace Mean Reversion en pánicos."
        inds = ["Google TimesFM", "Dynamic Volatility Trend", "SPY Macro Crash Guard"]
        pine_active = "//@version=5\n// TradingView no soporta IA local."
    elif stype.startswith("TSPULSE_PURE"):
        desc = "AIS04: Inteligencia Artificial Univariada. Utiliza el modelo ultra-eficiente IBM TSPulse (1M params) procesando la historia profunda de cada activo de manera independiente."
        inds = ["IBM TSPulse", "Univariate Price", "SPY Macro Crash Guard"]
        pine_active = "//@version=5\n// IBM Granite no está soportado nativamente en PineScript."
    elif stype.startswith("TSPULSE_OSC"):
        desc = "AIS05: Híbrida de Osciladores. Combina las predicciones puras de IBM TSPulse con reversión a la media mediante MFI y Bandas de Bollinger para cazar extremos."
        inds = ["IBM TSPulse", "MFI 14", "Bollinger Bands", "SPY Macro Crash Guard"]
        pine_active = "//@version=5\n// IBM Granite no está soportado nativamente en PineScript."
    elif stype.startswith("TSPULSE_MOM"):
        desc = "AIS06: Híbrida de Momentum. Filtra las señales de IBM TSPulse requiriendo picos de volumen relativo y fuerza en el precio (ROC)."
        inds = ["IBM TSPulse", "RelVol", "ROC 3", "SPY Macro Crash Guard"]
        pine_active = "//@version=5\n// IBM Granite no está soportado nativamente en PineScript."
    elif stype.startswith("MINIROCKET"):
        desc = "AIS07 a AIS09: MiniRocket. Transforma ventanas de precio en ~10.000 features convolucionales. AIS07 usa clasificador binario. AIS08 usa Deep Learning de probabilidades. AIS09 es Stack XGBoost."
        inds = ["MiniRocket Transform", "AI / Deep Learning", "SPY Macro Crash Guard"]
        pine_active = "//@version=5\n// MiniRocket no estǭ soportado nativamente en PineScript."
    elif stype == "MACRO_BASE_PURA":
        desc = "SS11: Estrategia de referencia pura. Opera Buy & Hold pasivo 100% del tiempo con la única salvedad de retirarse del mercado cuando ocurre un crash sistémico (Filtro Macro Global)."
        inds = ["Buy & Hold", "SPY Macro Crash Guard"]
        pine_active = "activeExit = false\n// Pura Macro: No hay filtro activo."

    sl_pct = params.get("stop_loss_pct")
    if sl_pct is not None:
        desc += f" Incluye un Stop Loss estricto de {sl_pct}%."
        inds.append(f"Stop Loss {sl_pct}%")
        pine_exit = f'\n// Stop Loss\nif strategy.position_size > 0\n    strategy.exit("Stop Loss", "Long", stop=strategy.position_avg_price * (1.0 + ({sl_pct}/100.0)))'
    else:
        pine_exit = ""

    params["desc"] = desc
    params["indicators"] = inds
    params["pinescript"] = f"""//@version=5
strategy("{params['name']}", overlay=true, initial_capital=10000, default_qty_type=strategy.percent_of_equity, default_qty_value=100)

// 1. Filtro Macro Global
spyClose = request.security("SPY", "D", close)
spyRet   = (spyClose - spyClose[1]) / spyClose[1]

var int daysOut = 0
if spyRet < -0.042
    daysOut := 8

// 2. Filtro Activo
{pine_active}

// Lógica de Trading Combinada
if daysOut > 0 or activeExit
    strategy.close("Long")
    if daysOut > 0
        daysOut := daysOut - 1
else
    strategy.entry("Long", strategy.long){pine_exit}

if barstate.isfirst
    strategy.entry("Long", strategy.long)
"""

def generate_signals(df, ticker, spy_idx, spy_ret, params, commission=0.0):
    # Macro Exit
    spy_exit_mask = np.zeros(len(spy_ret), dtype=bool)
    
    macro_threshold = params.get("macro_threshold", -0.042)
    macro_days_out = params.get("macro_days_out", 8)
    
    days_out = 0
    for i in range(1, len(spy_ret)):
        if spy_ret.iloc[i] < macro_threshold:
            days_out = macro_days_out
        if days_out > 0:
            spy_exit_mask[i] = True
            days_out -= 1
            
    spy_exit_series = pd.Series(spy_exit_mask, index=spy_idx)
    macro_aligned = spy_exit_series.reindex(df.index, fill_value=False).values
    macro_crash = macro_aligned
    n = len(df)
    signals_long = np.zeros(n, dtype=bool)
    signals_exit = np.zeros(n, dtype=bool)
    
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
        dh = df[f'Donchian_{params["period"]}_High'].shift(1)
        active_exit = df['Close'] > dh
    elif stype == "MFI":
        active_exit = df['MFI_14'] > params["thresh"]
    elif stype == "BB":
        bb_up = df['SMA_20'] + params["mult"] * df['STD_20']
        active_exit = df['Close'] > bb_up
    elif stype.startswith("TIMESFM"):
        if not hasattr(generate_signals, 'tsfm_cache'):
            try:
                import json
                with open("data/timesfm_signals.json", "r") as f:
                    generate_signals.tsfm_cache = json.load(f)
            except:
                generate_signals.tsfm_cache = {}
        
        preds_dict = generate_signals.tsfm_cache.get(ticker, {})
        preds_series = pd.Series(preds_dict)
        if not preds_series.empty:
            preds_series.index = pd.to_datetime(preds_series.index)
        preds_aligned = preds_series.reindex(df.index, fill_value=0.0)
        
        if stype == "TIMESFM_PURE":
            active_long = preds_aligned > 0.015
            active_exit = preds_aligned < 0.0
            
            es = macro_aligned | active_exit.values
            ls = active_long.values & (~macro_aligned)
            return ls, es
            
        elif stype == "TIMESFM_SMART":
            # 11/11 BEATS LOGIC (Found by Agent)
            # Re-enter unless extremely bearish
            active_long = preds_aligned > -0.01
            
            strong_uptrend = df['Close'] > df['SMA_50']
            ai_wants_out = preds_aligned < -0.02
            active_exit = ai_wants_out & (~strong_uptrend)
            
            es = macro_aligned | active_exit.values
            ls = active_long.values & (~macro_aligned)
            return ls, es
            
        elif stype == "TIMESFM_ADAPTIVE":
            # 18/18 ALPHA LOGIC (Volatility Aware)
            bb_width = 4.0 * df['STD_20'] / df['SMA_20']
            is_volatile_asset = bb_width.mean() > 0.21
            
            if is_volatile_asset:
                # S13 Logic (Strict entry + Mean Reversion + Loose Exit) for Hyper-Volatile assets (e.g. NU)
                cost_barrier = commission * 2.0
                ai_long_strict = preds_aligned > (0.015 + cost_barrier)
                bb_lower = df['SMA_20'] - 2 * df['STD_20']
                mr_long = (df['Close'] < bb_lower) & (df['MFI_14'] < 30)
                active_long = ai_long_strict | mr_long
                
                strong_uptrend = df['Close'] > df['SMA_200']
                ai_wants_out = preds_aligned < -0.015
                active_exit = ai_wants_out & (~strong_uptrend)
            else:
                # S12 Logic (Smart Hold + Strict Exit) for Steady assets (e.g. GOOG, SPY)
                active_long = preds_aligned > -0.01
                
                strong_uptrend = df['Close'] > df['SMA_50']
                ai_wants_out = preds_aligned < -0.02
                active_exit = ai_wants_out & (~strong_uptrend)
                
            es = macro_aligned | active_exit.values
            ls = active_long.values & (~macro_aligned)
            return ls, es
            
    elif stype.startswith("TSPULSE"):
        if not hasattr(generate_signals, 'tspulse_cache'):
            try:
                import json
                with open("data/tspulse_signals.json", "r") as f:
                    generate_signals.tspulse_cache = json.load(f)
            except:
                generate_signals.tspulse_cache = {}
        
        preds_dict = generate_signals.tspulse_cache.get(ticker, {})
        preds_series = pd.Series(preds_dict)
        if not preds_series.empty:
            preds_series.index = pd.to_datetime(preds_series.index)
        preds_aligned = preds_series.reindex(df.index, fill_value=0.0)
        
        if stype == "TSPULSE_PURE":
            active_long = preds_aligned > 0.010  # IBM predicts a >1.0% gain
            active_exit = preds_aligned < -0.005   # IBM predicts a drop of <-0.5%
            
            es = macro_aligned | active_exit.values
            ls = active_long.values & (~macro_aligned)
            return ls, es
            
        elif stype == "TSPULSE_OSC":
            bb_lower = df['SMA_20'] - 2 * df['STD_20']
            bb_up = df['SMA_20'] + 2 * df['STD_20']
            
            ai_long = preds_aligned > -0.01
            mr_long = (df['MFI_14'] < 30) & (df['Close'] < bb_lower)
            active_long = ai_long | mr_long
            
            strong_uptrend = df['Close'] > df['SMA_50']
            ai_wants_out = preds_aligned < -0.02
            mr_out = (df['MFI_14'] > 80) & (df['Close'] > bb_up)
            
            active_exit = (ai_wants_out | mr_out) & (~strong_uptrend)
            
            es = macro_aligned | active_exit.values
            ls = active_long.values & (~macro_aligned)
            return ls, es
            
        elif stype == "TSPULSE_MOM":
            cond_vol = df['RelVol'] > 0.5
            cond_roc = df['ROC_3'] > -5.0
            active_long = (preds_aligned > -0.01) & cond_vol & cond_roc
            
            strong_uptrend = df['Close'] > df['SMA_50']
            ai_wants_out = preds_aligned < -0.02
            active_exit = ai_wants_out & (~strong_uptrend)
            
            es = macro_aligned | active_exit.values
            ls = active_long.values & (~macro_aligned)
            return ls, es
    
    elif stype.startswith("MINIROCKET"):
        cache_name = "minirocket_gpu_cache" if stype == "MINIROCKET_GPU" else "minirocket_cache"
        file_name = "data/minirocket_gpu_signals.json" if stype == "MINIROCKET_GPU" else "data/minirocket_signals.json"
        
        if not hasattr(generate_signals, cache_name):
            try:
                import json
                with open(file_name, "r") as f:
                    setattr(generate_signals, cache_name, json.load(f))
            except:
                setattr(generate_signals, cache_name, {})
        
        preds_dict = getattr(generate_signals, cache_name).get(ticker, {})
        preds_series = pd.Series(preds_dict)
        if not preds_series.empty:
            preds_series.index = pd.to_datetime(preds_series.index)
            preds_aligned = preds_series.reindex(df.index).fillna(1.0 if stype == "MINIROCKET_GPU" else 1) # Asumimos optimismo si no hay dato
        else:
            preds_aligned = pd.Series(1.0 if stype == "MINIROCKET_GPU" else 1, index=df.index)
        
        if stype == "MINIROCKET_GPU":
            ai_wants_in = preds_aligned > params.get("ai_in_th", 0.55)
            ai_wants_out = preds_aligned < params.get("ai_out_th", 0.25)
            
            es = macro_aligned | ai_wants_out.values
            ls = ai_wants_in.values & (~macro_aligned)
            return ls, es
        else:
            es = macro_aligned | (preds_aligned.values == 0)
            ls = (preds_aligned.values == 1) & (~macro_aligned)
            return ls, es
            
    elif stype == "MINIROCKET_STACK":
        if not hasattr(generate_signals, 'xgboost_cache'):
            try:
                import json
                with open("data/xgboost_stack_signals.json", "r") as f:
                    generate_signals.xgboost_cache = json.load(f)
            except:
                generate_signals.xgboost_cache = {}
                
        preds_dict = generate_signals.xgboost_cache.get(ticker, {})
        preds_series = pd.Series(preds_dict)
        if not preds_series.empty:
            preds_series.index = pd.to_datetime(preds_series.index)
            preds_aligned = preds_series.reindex(df.index).fillna(0.0)
        else:
            preds_aligned = pd.Series(0.0, index=df.index)
            
        ai_wants_in = preds_aligned > params.get("ai_in_th", 0.55)
        ai_wants_out = preds_aligned < params.get("ai_out_th", 0.25)
        
        es = macro_aligned | ai_wants_out.values
        ls = ai_wants_in.values & (~macro_aligned)
        return ls, es
        
    elif stype == "MACRO_BASE_PURA":
        es = macro_aligned
        ls = ~macro_aligned
        return ls, es
            
    try:
        es = macro_aligned | active_exit.values
    except NameError:
        # Fallback si no se definió active_exit
        es = macro_aligned
        
    ls = ~es
    return ls, es

# -------------------------------------------------------------------------
# 4. Main Pipeline / API Entry
# -------------------------------------------------------------------------
CACHED_DATA_DICT = None

def run_all(commission=0.0, start_date=None, end_date=None):
    global CACHED_DATA_DICT
    print(f"=== RECALCULANDO PIPELINE V10 (Comision: {commission*100:.2f}%, Rango: {start_date} a {end_date}) ===")
    
    if CACHED_DATA_DICT is None:
        CACHED_DATA_DICT = fetch_data()
        
    data_dict = {}
    for tk, df in CACHED_DATA_DICT.items():
        temp_df = df.copy()
        if start_date:
            temp_df = temp_df[temp_df.index >= pd.to_datetime(start_date)]
        if end_date:
            temp_df = temp_df[temp_df.index <= pd.to_datetime(end_date)]
        data_dict[tk] = temp_df
    
    if len(data_dict["SPY"]) == 0:
        raise ValueError("El rango de fechas no contiene datos.")
        
    spy_df = data_dict["SPY"]
    spy_ret = spy_df['Close'].pct_change()
    spy_idx = spy_df.index
    
    benchmarks = {}
    for ticker, df in data_dict.items():
        benchmarks[ticker] = simulate_buy_and_hold(df, commission=commission)
        print(f"B&H {ticker:5s}: {benchmarks[ticker]['total_return']:8.2f}% | MaxDD={benchmarks[ticker]['max_drawdown']:.2f}%")
        
    print("\n--- Running 10 V10 Strategies (30-Year) ---")
    all_results = {}
    strategy_ids = list(STRATEGY_INFO.keys())

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
            
            eq, tr = run_simulation(ls, es, opens, closes, dates, commission=commission, stop_loss_pct=s_info.get("stop_loss_pct"))
            mets, ec = compute_metrics(eq, tr, dates)
            
            exit_label = "-"
            if mets["is_open"]:
                stype = s_info["type"]
                if stype == "TIMESFM_ADAPTIVE":
                    bb_width = 4.0 * df['STD_20'] / df['SMA_20']
                    is_vol = bb_width.mean() > 0.21
                    sma_val = df['SMA_200'].iloc[-1] if is_vol else df['SMA_50'].iloc[-1]
                    sma_name = "SMA_200" if is_vol else "SMA_50"
                    exit_label = f"${sma_val:.2f} ({sma_name})"
                elif stype == "TIMESFM_SMART":
                    exit_label = f"${df['SMA_50'].iloc[-1]:.2f} (SMA_50)"
                elif stype == "TIMESFM_PURE":
                    exit_label = "- (TimesFM Signal)"
                elif stype == "TSPULSE_PURE":
                    exit_label = "- (IBM TSPulse Signal)"
                elif stype == "BB":
                    bb_hi = df['SMA_20'].iloc[-1] + 3.1 * df['STD_20'].iloc[-1]
                    exit_label = f"${bb_hi:.2f} (BB Alta)"
                elif stype == "DONCHIAN":
                    donc = df['High'].rolling(33).max().shift(1).iloc[-1]
                    exit_label = f"${donc:.2f} (Donchian Hi)"
                elif stype == "MFI":
                    exit_label = "MFI > 85"
                elif stype == "RELVOL":
                    exit_label = f"Vol > {s_info.get('rel_vol', 2)}x"
                elif stype == "ROC_VOL" or stype == "ROC":
                    exit_label = f"Salto > {s_info.get('roc_val', 10)}%"
                    
            mets["exit_threshold"] = exit_label
            mets["current_price"] = float(df['Close'].iloc[-1])
            
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
        avg_ret  = np.nanmean([d["metrics"]["total_return"] for d in td.values()])
        years_total = (data_dict["SPY"].index[-1] - data_dict["SPY"].index[0]).days / 365.25
        avg_cagr = (((avg_ret / 100.0) + 1.0) ** (1.0 / years_total) - 1.0) * 100.0 if years_total > 0 else 0.0
        avg_sh   = np.nanmean([d["metrics"]["sharpe"]        for d in td.values()])
        avg_dd   = np.nanmean([d["metrics"]["max_drawdown"]  for d in td.values()])
        avg_wr   = np.nanmean([d["metrics"]["win_rate"]      for d in td.values()])
        pf_vals  = [d["metrics"]["profit_factor"] for d in td.values() if d["metrics"]["profit_factor"] != 999.0]
        avg_pf   = float(np.nanmean(pf_vals)) if len(pf_vals) > 0 else 1.0
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
              f"Beat B&H: {r['aggregate_metrics']['outperform_count']}/{len(TICKERS)}")

    simplified_benchmarks = {t: {k: v for k, v in d.items()} for t, d in benchmarks.items()}

    output = {
        "metadata": {
            "start_date":            spy_idx[0].strftime("%Y-%m-%d") if len(spy_idx) > 0 else START_DATE,
            "end_date":              spy_idx[-1].strftime("%Y-%m-%d") if len(spy_idx) > 0 else END_DATE,
            "tickers":               TICKERS,
            "num_strategies_tested": len(STRATEGY_INFO),
            "version":               "V10 - Subagent Supreme AI",
            "generated_at":          datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "benchmarks":      simplified_benchmarks,
        "ranking":         ranking,
        "spy_raw_ranking": spy_raw
    }

    def clean_nan(obj):
        if isinstance(obj, float):
            import math
            return 0.0 if math.isnan(obj) else obj
        elif isinstance(obj, dict):
            return {k: clean_nan(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_nan(i) for i in obj]
        return obj

    return clean_nan(output)

if __name__ == "__main__":
    output = run_all(commission=0.004)
    with open("data/results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)
    print("\nResults saved to data/results.json")
