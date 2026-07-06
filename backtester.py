import os
import json
import time
# pyrefly: ignore [missing-import]
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
def sigmoid_norm(x, scale=1.0):
    return 100.0 / (1.0 + np.exp(-x / scale))

def fetch_data():
    print("Loading AI JSON signals...")
    try:
        with open("data/minirocket_gpu_signals.json", "r") as f:
            minirocket_gpu_data = json.load(f)
    except Exception: minirocket_gpu_data = {}
    try:
        with open("data/minirocket_signals.json", "r") as f:
            minirocket_bin_data = json.load(f)
    except Exception: minirocket_bin_data = {}
    try:
        with open("data/timesfm_signals.json", "r") as f:
            timesfm_data = json.load(f)
    except Exception: timesfm_data = {}
    try:
        with open("data/tspulse_signals.json", "r") as f:
            tspulse_data = json.load(f)
    except Exception: tspulse_data = {}
    try:
        with open("data/xgboost_stack_signals.json", "r") as f:
            xgboost_data = json.load(f)
    except Exception: xgboost_data = {}

    # Invalidate cache if it's older than specified seconds (default 3600 for 1-hour updates)
    cache_expire = float(os.environ.get("YF_CACHE_SECONDS", 3600))
    data = {}
    for ticker in TICKERS:
        cache_path = os.path.join(CACHE_DIR, f"{ticker}.csv")
        
        cache_valid = False
        if os.path.exists(cache_path) and os.path.getsize(cache_path) > 100:
            mod_time = os.path.getmtime(cache_path)
            if 0 <= (time.time() - mod_time) < cache_expire:
                cache_valid = True

        if cache_valid:
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            df.index.name = "Date"
        else:
            df = pd.DataFrame()
            for attempt in range(3):
                try:
                    df = yf.download(ticker, start=START_DATE, progress=False)
                    if not df.empty:
                        break
                    time.sleep(2)
                except Exception:
                    time.sleep(2)
                    
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
                
            if df.empty and os.path.exists(cache_path) and os.path.getsize(cache_path) > 100:
                df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                df.index.name = "Date"
            elif not df.empty:
                df.to_csv(cache_path)
            elif ticker == "C" and os.path.exists(os.path.join(CACHE_DIR, "C_daily.csv")):
                df = pd.read_csv(os.path.join(CACHE_DIR, "C_daily.csv"), index_col=0, parse_dates=True)
                df.index.name = "Date"
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
        
        # SS12 Winning Indicators (100k iterations - Dual Mode)
        df['VOL_NORM'] = sigmoid_norm(df['RelVol'] - 1.0, 1.0)
        roc5 = df['Close'].pct_change(5) * 100
        df['ROC_NORM'] = sigmoid_norm(roc5, 5.0)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        df['RSI_NORM'] = 100 - (100 / (1 + rs))
        
        tr1 = df['High'] - df['Low']
        tr2 = (df['High'] - df['Close'].shift()).abs()
        tr3 = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean()
        df['ATR_NORM'] = sigmoid_norm(atr14 / df['Close'] * 100, 2.0)
        
        ret = df['Close'].pct_change().fillna(0)
        vol_up = df['Volume'] > df['Volume'].shift(1)
        pvi_change = np.where(vol_up, ret, 0.0)
        pvi = 1000.0 * np.cumprod(1 + pvi_change)
        ema_pvi = pd.Series(pvi).ewm(span=15, adjust=False).mean()
        scale_pvi = pd.Series(pvi).rolling(100).std().bfill() + 1e-8
        df['KONCORDE_MD'] = sigmoid_norm(pvi - ema_pvi, scale_pvi)
        
        trend = (df['Close'] - df['SMA_50']) / df['SMA_50'] * 100
        df['TREND_SMA'] = sigmoid_norm(trend, 5.0)
        
        # New for SS14/SS15
        df['VOL_EXT_NORM'] = sigmoid_norm(df['RelVol'] - 2.0, 1.0)
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_sig = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - macd_sig
        scale_macd = macd_hist.rolling(100).std().bfill() + 1e-8
        df['MACD_NORM'] = sigmoid_norm(macd_hist, scale_macd)
        
        roc3 = df['Close'].pct_change(3) * 100
        df['ROC_3_NORM'] = sigmoid_norm(roc3, 5.0)
        
        bb_upper_3 = df['SMA_20'] + 3.1*df['STD_20']
        bb_lower_3 = df['SMA_20'] - 3.1*df['STD_20']
        df['BB_3_POS'] = (df['Close'] - bb_lower_3) / (bb_upper_3 - bb_lower_3 + 1e-8) * 100
        df['BB_3_POS'] = df['BB_3_POS'].clip(0, 100)
        df['MFI_NORM'] = df['MFI_14']
        
        low14 = df['Low'].rolling(14).min()
        high14 = df['High'].rolling(14).max()
        df['STOCH_14'] = 100 * (df['Close'] - low14) / (high14 - low14 + 1e-8)
        
        df.fillna(50.0, inplace=True)
        
        # AI 1: AI_MINIROCKET_GPU
        s_gpu = pd.Series(minirocket_gpu_data.get(ticker, {}))
        if not s_gpu.empty: s_gpu.index = pd.to_datetime(s_gpu.index)
        df['AI_MINIROCKET_GPU'] = s_gpu.reindex(df.index) * 100.0

        # AI 2: AI_MINIROCKET_BIN
        s_bin = pd.Series(minirocket_bin_data.get(ticker, {}))
        if not s_bin.empty: s_bin.index = pd.to_datetime(s_bin.index)
        df['AI_MINIROCKET_BIN'] = s_bin.reindex(df.index) * 100.0

        # AI 3: AI_TIMESFM
        s_tfm = pd.Series(timesfm_data.get(ticker, {}))
        if not s_tfm.empty: s_tfm.index = pd.to_datetime(s_tfm.index)
        s_tfm_aligned = s_tfm.reindex(df.index)
        df['AI_TIMESFM'] = sigmoid_norm(s_tfm_aligned * 100.0, 2.0)

        # AI 4: AI_TSPULSE
        s_tsp = pd.Series(tspulse_data.get(ticker, {}))
        if not s_tsp.empty: s_tsp.index = pd.to_datetime(s_tsp.index)
        s_tsp_aligned = s_tsp.reindex(df.index)
        df['AI_TSPULSE'] = sigmoid_norm(s_tsp_aligned * 100.0, 2.0)

        # AI 5: AI_XGBOOST
        s_xgb = pd.Series(xgboost_data.get(ticker, {}))
        if not s_xgb.empty: s_xgb.index = pd.to_datetime(s_xgb.index)
        df['AI_XGBOOST'] = s_xgb.reindex(df.index) * 100.0

        data[ticker] = df
    return data

# -------------------------------------------------------------------------
# 2. Simulation Engine
# -------------------------------------------------------------------------
def run_simulation(signals_long, signals_exit, opens, closes, dates, initial_capital=10000.0, commission=0.004, stop_loss_pct=None, signals_strict=None):
    n = len(closes)
    equity = np.full(n, float(initial_capital))
    cash   = float(initial_capital)
    pos    = 0.0
    in_pos = False
    entry_price = 0.0
    entry_idx   = 0
    trades = []
    
    in_sl_recovery = False

    for i in range(1, n):
        hit_stop_loss = False
        if in_pos and stop_loss_pct is not None:
            current_loss = (closes[i-1] / entry_price - 1.0) * 100.0
            if current_loss <= stop_loss_pct:
                hit_stop_loss = True

        can_enter = False
        if not in_pos:
            if in_sl_recovery and signals_strict is not None:
                if signals_strict[i-1] and signals_long[i-1]:
                    can_enter = True
                    in_sl_recovery = False
            else:
                if signals_long[i-1]:
                    can_enter = True

        if can_enter:
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
            
            if hit_stop_loss:
                in_sl_recovery = True

        equity[i] = cash + pos * closes[i]

    # --- LIVE EDGE EVALUATION ---
    if in_pos:
        tomorrow_hit_stop_loss = False
        if stop_loss_pct is not None:
            current_loss = (closes[-1] / entry_price - 1.0) * 100.0
            if current_loss <= stop_loss_pct:
                tomorrow_hit_stop_loss = True
        
        if signals_exit[-1] or tomorrow_hit_stop_loss:
            current_signal = "SELL"
        else:
            current_signal = "HOLD"
    else:
        can_enter = False
        if in_sl_recovery and signals_strict is not None:
            if signals_strict[-1] and signals_long[-1]:
                can_enter = True
        else:
            if signals_long[-1]:
                can_enter = True

        if can_enter:
            current_signal = "BUY"
        else:
            current_signal = "WAIT"

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

    return equity, trades, current_signal

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
    "SS12": {"type": "SS12", "name": "Macro + SS12 AI (Optimizado p/ 0.4% Comisiones)", "stop_loss_pct": -15.0},
    "SS13": {"type": "SS13", "name": "Macro + SS13 AI (Optimizado p/ 0% Comisiones)", "stop_loss_pct": -15.0},
    "SS14": {"type": "SS14", "name": "Macro + SS14 AI (Optimizado p/ 0% Comisiones)", "stop_loss_pct": -15.0},
    "SS15": {"type": "SS15", "name": "Macro + SS15 AI (Optimizado p/ 0.4% Comisiones)", "stop_loss_pct": -15.0},
    "AIS10": {"type": "AIS10", "name": "Macro + AIS10 AI Multi-Model (Optimizado p/ 0% Comisiones)", "stop_loss_pct": -15.0},
    "AIS11": {"type": "AIS11", "name": "Macro + AIS11 AI Multi-Model (Optimizado p/ 0.4% Comisiones)", "stop_loss_pct": -15.0},
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
    elif stype == "SS13":
        desc = "SS13: Optimizado para comisiones al 0%. Score: ATR (34%), Koncorde (32%), Tendencia SMA (28%), RSI (6%)."
        inds = ["SS13 Zero Score", "SPY Macro Crash Guard"]
        pine_active = """// Indicadores SS13
sma50 = ta.sma(close, 50)
trendSma = (close - sma50) / sma50 * 100.0
trendNorm = 100.0 / (1.0 + math.exp(-trendSma / 5.0))

atr14 = ta.sma(ta.tr(true), 14)
atrNorm = 100.0 / (1.0 + math.exp(-((atr14 / close) * 100.0) / 2.0))

rsiNorm = ta.rsi(close, 14)

ret1 = (close - close[1]) / close[1]
pviChange = volume > volume[1] ? ret1 : 0.0
var float pvi = 1000.0
pvi := pvi * (1.0 + nz(pviChange))
emaPvi = ta.ema(pvi, 15)
scalePvi = ta.stdev(pvi, 100)
scalePviSafe = scalePvi == 0 or na(scalePvi) ? 1.0 : scalePvi
koncordeMd = 100.0 / (1.0 + math.exp(-(pvi - emaPvi) / scalePviSafe))

ss13Score = (trendNorm * 28.0 + atrNorm * 34.0 + rsiNorm * 6.0 + koncordeMd * 32.0) / 100.0
activeEntry = ss13Score > 52.0
activeExit  = ss13Score < 35.0"""
    elif stype == "SS12":
        desc = "SS12: Optimizado para comisiones realistas del 0.4%. Score: Volatilidad (46%), Koncorde (32%), Tendencia SMA (14%), ROC (9%)."
        inds = ["SS12 Com Score", "SPY Macro Crash Guard"]
        pine_active = """// Indicadores SS12
volSma20 = ta.sma(volume, 20)
relVol = volume / volSma20
volNorm = 100.0 / (1.0 + math.exp(-(relVol - 1.0) / 1.0))

roc5 = ta.roc(close, 5)
rocNorm = 100.0 / (1.0 + math.exp(-roc5 / 5.0))

sma50 = ta.sma(close, 50)
trendSma = (close - sma50) / sma50 * 100.0
trendNorm = 100.0 / (1.0 + math.exp(-trendSma / 5.0))

ret1 = (close - close[1]) / close[1]
pviChange = volume > volume[1] ? ret1 : 0.0
var float pvi = 1000.0
pvi := pvi * (1.0 + nz(pviChange))
emaPvi = ta.ema(pvi, 15)
scalePvi = ta.stdev(pvi, 100)
scalePviSafe = scalePvi == 0 or na(scalePvi) ? 1.0 : scalePvi
koncordeMd = 100.0 / (1.0 + math.exp(-(pvi - emaPvi) / scalePviSafe))

ss12Score = (rocNorm * 9.0 + trendNorm * 14.0 + koncordeMd * 32.0 + volNorm * 46.0) / 100.0
activeEntry = ss12Score > 57.0
activeExit  = ss12Score < 39.0"""
    elif stype == "SS14":
        try:
            with open("data/ss14_zero_params.json") as f: p = json.load(f)
        except Exception: p = {"indicators": ["ROC_3_NORM", "ROC_NORM", "VOL_EXT_NORM", "ATR_NORM"], "weights": [40, 10, 5, 45], "entry_th": 55, "exit_th": 45}
        ind_w = sorted(zip(p["indicators"], p["weights"]), key=lambda x: x[1], reverse=True)
        score_str = ", ".join([f"{ind} ({w}%)" for ind, w in ind_w if w > 0])
        desc = f"SS14: Optimizado para comisiones al 0% (v0.2.5). Score: {score_str}."
        inds = ["SS14 Zero Score", "SPY Macro Crash Guard"]
        pine_active = f"""// Indicadores SS14
atr14 = ta.sma(ta.tr(true), 14)
atrNorm = 100.0 / (1.0 + math.exp(-((atr14 / close) * 100.0) / 2.0))

roc3 = ta.roc(close, 3)
roc3Norm = 100.0 / (1.0 + math.exp(-roc3 / 5.0))

roc5 = ta.roc(close, 5)
rocNorm = 100.0 / (1.0 + math.exp(-roc5 / 5.0))

volSma20 = ta.sma(volume, 20)
relVol = volume / volSma20
volExtNorm = 100.0 / (1.0 + math.exp(-(relVol - 2.0) / 1.0))

ss14Score = (atrNorm * 45.0 + roc3Norm * 40.0 + rocNorm * 10.0 + volExtNorm * 5.0) / 100.0
activeEntry = ss14Score > {p['entry_th']}.0
activeExit  = ss14Score < {p['exit_th']}.0"""
    elif stype == "SS15":
        try:
            with open("data/ss15_com_params.json") as f: p = json.load(f)
        except Exception: p = {"indicators": ["ATR_NORM", "VOL_EXT_NORM"], "weights": [50, 50], "entry_th": 55, "exit_th": 10}
        ind_w = sorted(zip(p["indicators"], p["weights"]), key=lambda x: x[1], reverse=True)
        score_str = ", ".join([f"{ind} ({w}%)" for ind, w in ind_w if w > 0])
        desc = f"SS15: Optimizado para comisiones al 0.4% (v0.2.5). Score: {score_str}."
        inds = ["SS15 Com Score", "SPY Macro Crash Guard"]
        pine_active = f"""// Indicadores SS15
atr14 = ta.sma(ta.tr(true), 14)
atrNorm = 100.0 / (1.0 + math.exp(-((atr14 / close) * 100.0) / 2.0))

volSma20 = ta.sma(volume, 20)
relVol = volume / volSma20
volExtNorm = 100.0 / (1.0 + math.exp(-(relVol - 2.0) / 1.0))

ss15Score = (atrNorm * 50.0 + volExtNorm * 50.0) / 100.0
activeEntry = ss15Score > {p['entry_th']}.0
activeExit  = ss15Score < {p['exit_th']}.0"""
    elif stype == "AIS10":
        try:
            with open("data/ais10_zero_params.json") as f: p = json.load(f)
        except Exception: p = {"indicators": ["AI_MINIROCKET_BIN", "AI_TIMESFM", "KONCORDE_MD", "ROC_NORM"], "weights": [10, 55, 15, 20], "entry_th": 60, "exit_th": 15}
        ind_w = sorted(zip(p["indicators"], p["weights"]), key=lambda x: x[1], reverse=True)
        score_str = ", ".join([f"{ind} ({w}%)" for ind, w in ind_w if w > 0])
        desc = f"AIS10: Multi-IA Optimizada para comisiones 0% (v0.2.5). Score: {score_str}."
        inds = ["AIS10 Zero Score", "SPY Macro Crash Guard"]
        pine_active = "// [ADVERTENCIA] Los modelos locales de IA (TimesFM/TSPulse) no se pueden evaluar en TradingView.\n// Se aplica únicamente el Filtro Macro Global:\nactiveExit = false"
    elif stype == "AIS11":
        try:
            with open("data/ais11_com_params.json") as f: p = json.load(f)
        except Exception: p = {"indicators": ["AI_MINIROCKET_GPU", "AI_TIMESFM", "AI_TSPULSE", "ROC_3_NORM"], "weights": [5, 20, 65, 10], "entry_th": 55, "exit_th": 5}
        ind_w = sorted(zip(p["indicators"], p["weights"]), key=lambda x: x[1], reverse=True)
        score_str = ", ".join([f"{ind} ({w}%)" for ind, w in ind_w if w > 0])
        desc = f"AIS11: Multi-IA Optimizada para comisiones 0.4% (v0.2.5). Score: {score_str}."
        inds = ["AIS11 Com Score", "SPY Macro Crash Guard"]
        pine_active = "// [ADVERTENCIA] Los modelos locales de IA (TimesFM/TSPulse) no se pueden evaluar en TradingView.\n// Se aplica únicamente el Filtro Macro Global:\nactiveExit = false"


    sl_pct = params.get("stop_loss_pct")
    
    # Check if this strategy gets the strict SMA20 recovery logic
    s_id = next((k for k, v in STRATEGY_INFO.items() if v == params), "")
    is_strict_reentry = ((s_id.startswith("SS") and s_id != "SS11") or s_id in ["AIS10", "AIS11"]) and sl_pct is not None
    
    if sl_pct is not None:
        desc += f" Incluye un Stop Loss estricto de {sl_pct}%."
        if is_strict_reentry:
            desc += " Tras tocar el stop-loss, exige que el precio supere la SMA 20 para volver a comprar."
            inds.append("SMA 20 Re-entry Filter")
        inds.append(f"Stop Loss {sl_pct}%")
        pine_exit = f'\n// Stop Loss\nif strategy.position_size > 0\n    strategy.exit("Stop Loss", "Long", stop=strategy.position_avg_price * (1.0 + ({sl_pct}/100.0)))'
        
        if is_strict_reentry:
            pine_exit += f'''
// Lógica de recuperación post-StopLoss
var bool in_sl_recovery = false
if strategy.position_size > 0 and low <= strategy.position_avg_price * (1.0 + ({sl_pct}/100.0))
    in_sl_recovery := true

sma20 = ta.sma(close, 20)
if in_sl_recovery and close > sma20
    in_sl_recovery := false
'''
    else:
        pine_exit = ""

    params["desc"] = desc
    params["indicators"] = inds
    
    if stype in ["SS12", "SS13", "SS14", "SS15"]:
        entry_cmd = 'if activeEntry and not in_sl_recovery\n        strategy.entry("Long", strategy.long)' if is_strict_reentry else 'if activeEntry\n        strategy.entry("Long", strategy.long)'
    else:
        entry_cmd = 'if not in_sl_recovery\n        strategy.entry("Long", strategy.long)' if is_strict_reentry else 'strategy.entry("Long", strategy.long)'

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
    {entry_cmd}{pine_exit}

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
    macro_aligned = spy_exit_series.reindex(df.index).ffill().fillna(False).values
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
    elif stype == "SS13":
        score = (df['TREND_SMA'] * 28 + df['ATR_NORM'] * 34 + df['RSI_NORM'] * 6 + df['KONCORDE_MD'] * 32) / 100.0
        active_long = score > 52
        active_exit = score < 35
        
        es = macro_aligned | active_exit.values
        ls = active_long.values & (~macro_aligned)
        return ls, es
    elif stype == "SS12":
        score = (df['ROC_NORM'] * 9 + df['TREND_SMA'] * 14 + df['KONCORDE_MD'] * 32 + df['VOL_NORM'] * 46) / 100.0
        active_long = score > 57
        active_exit = score < 39
        
        es = macro_aligned | active_exit.values
        ls = active_long.values & (~macro_aligned)
        return ls, es
    elif stype in ["SS14", "SS15", "AIS10", "AIS11"]:
        if stype == "SS14":
            try:
                with open("data/ss14_zero_params.json") as f: p = json.load(f)
            except Exception: p = {"indicators": ["ROC_3_NORM", "ROC_NORM", "VOL_EXT_NORM", "ATR_NORM"], "weights": [40, 10, 5, 45], "entry_th": 55, "exit_th": 45}
        elif stype == "SS15":
            try:
                with open("data/ss15_com_params.json") as f: p = json.load(f)
            except Exception: p = {"indicators": ["ATR_NORM", "VOL_EXT_NORM"], "weights": [50, 50], "entry_th": 55, "exit_th": 10}
        elif stype == "AIS10":
            try:
                with open("data/ais10_zero_params.json") as f: p = json.load(f)
            except Exception: p = {"indicators": ["AI_MINIROCKET_BIN", "AI_TIMESFM", "KONCORDE_MD", "ROC_NORM"], "weights": [10, 55, 15, 20], "entry_th": 60, "exit_th": 15}
        else: # AIS11
            try:
                with open("data/ais11_com_params.json") as f: p = json.load(f)
            except Exception: p = {"indicators": ["AI_MINIROCKET_GPU", "AI_TIMESFM", "AI_TSPULSE", "ROC_3_NORM"], "weights": [5, 20, 65, 10], "entry_th": 55, "exit_th": 5}
            
        sel_inds = p["indicators"]
        weights = p["weights"]
        entry_th = p["entry_th"]
        exit_th = p["exit_th"]
        
        score = np.zeros(len(df))
        total_w = np.zeros(len(df))
        for idx, ind in enumerate(sel_inds):
            vals = df[ind].values
            valid = (~np.isnan(vals)).astype(float)
            safe_vals = np.where(np.isnan(vals), 0.0, vals)
            score += safe_vals * weights[idx]
            total_w += valid * weights[idx]
            
        valid_rows = total_w > 0
        score = np.where(valid_rows, score / total_w, 50.0)
        active_long = score > entry_th
        active_exit = score < exit_th

        es = macro_aligned | active_exit
        ls = active_long & (~macro_aligned)
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
            
            signals_strict = None
            if (s_id.startswith("SS") and s_id != "SS11") or s_id in ["AIS10", "AIS11"]:
                signals_strict = (df['Close'] > df['SMA_20']).values
                
            eq, tr, current_signal = run_simulation(ls, es, opens, closes, dates, commission=commission, stop_loss_pct=s_info.get("stop_loss_pct"), signals_strict=signals_strict)
            mets, ec = compute_metrics(eq, tr, dates)
            mets["current_signal"] = current_signal
            
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
                elif stype == "SS13":
                    score_val = (df['TREND_SMA'].iloc[-1] * 28 + df['ATR_NORM'].iloc[-1] * 34 + df['RSI_NORM'].iloc[-1] * 6 + df['KONCORDE_MD'].iloc[-1] * 32) / 100.0
                    exit_label = f"Score < 35 (Current: {score_val:.1f})"
                elif stype == "SS12":
                    score_val = (df['ROC_NORM'].iloc[-1] * 9 + df['TREND_SMA'].iloc[-1] * 14 + df['KONCORDE_MD'].iloc[-1] * 32 + df['VOL_NORM'].iloc[-1] * 46) / 100.0
                    exit_label = f"Score < 39 (Current: {score_val:.1f})"
                elif stype in ["SS14", "SS15", "AIS10", "AIS11"]:
                    exit_label = f"Score Dinámico ({stype})"
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
