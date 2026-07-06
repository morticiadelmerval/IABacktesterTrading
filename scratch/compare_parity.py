import sys
sys.path.append(".")
import json
import numpy as np
import pandas as pd
from backtester import fetch_data, generate_signals, run_simulation, compute_metrics, TICKERS, STRATEGY_INFO
from models.scripts.optimize_ais10 import prepare_data, simulate_ticker

print("=== PARITY COMPARISON SCRIPT (4 STRATEGIES) ===")
print("Loading data from both engines...")
data_backtester = fetch_data()
data_opt = prepare_data()

spy_df = data_backtester["SPY"]
spy_ret = spy_df['Close'].pct_change()

strategies_to_test = [
    ("AIS10", "data/ais10_zero_params.json", 0.0),
    ("AIS11", "data/ais11_com_params.json", 0.004),
    ("SS14", "data/ss14_zero_params.json", 0.0),
    ("SS15", "data/ss15_com_params.json", 0.004)
]

all_ok = True

for stype, json_path, comm in strategies_to_test:
    print(f"\n--- Verifying Parity for {stype} (com={comm*100:.1f}%) ---")
    with open(json_path) as f:
        p = json.load(f)
    
    sel_inds = p["indicators"]
    weights = p["weights"]
    entry_th = p["entry_th"]
    exit_th = p["exit_th"]
    s_info = STRATEGY_INFO[stype]
    
    for t in TICKERS:
        df_bt = data_backtester[t]
        df_opt = data_opt[t]
        
        # Backtester signals and simulation
        ls, es = generate_signals(df_bt, t, spy_df.index, spy_ret, s_info, comm)
        signals_strict = (df_bt['Close'] > df_bt['SMA_20']).values
        eq_bt, tr_bt, _ = run_simulation(ls, es, df_bt['Open'].values, df_bt['Close'].values, df_bt.index, commission=comm, stop_loss_pct=s_info["stop_loss_pct"], signals_strict=signals_strict)
        ret_bt = (eq_bt[-1] / 10000.0 - 1.0) * 100.0
        
        # Optimizer macro and signals
        spy_ret_opt = data_opt["SPY"]['Close'].pct_change()
        spy_exit_mask = np.zeros(len(spy_ret_opt), dtype=bool)
        days_out = 0
        for i in range(1, len(spy_ret_opt)):
            if spy_ret_opt.iloc[i] < -0.042: days_out = 8
            if days_out > 0:
                spy_exit_mask[i] = True
                days_out -= 1
        spy_macro_series = pd.Series(spy_exit_mask, index=data_opt["SPY"].index)
        macro_opt = spy_macro_series.reindex(df_opt.index).ffill().fillna(False).values
        
        score_opt = np.zeros(len(df_opt))
        total_w_opt = np.zeros(len(df_opt))
        for idx, ind in enumerate(sel_inds):
            vals = df_opt[ind].values
            valid = (~np.isnan(vals)).astype(float)
            safe_vals = np.where(np.isnan(vals), 0.0, vals)
            score_opt += safe_vals * weights[idx]
            total_w_opt += valid * weights[idx]
        valid_rows = total_w_opt > 0
        score_opt = np.where(valid_rows, score_opt / total_w_opt, 50.0)
        
        ls_opt = (score_opt > entry_th) & (~macro_opt)
        es_opt = macro_opt | (score_opt < exit_th)
        sma20_mask_opt = (df_opt['Close'] > df_opt['SMA_20']).values
        
        ret_opt_t, tr_opt_t = simulate_ticker(
            df_opt['Open'].values, df_opt['Close'].values, macro_opt, sma20_mask_opt, score_opt, entry_th, exit_th, comm
        )
        
        if abs(ret_bt - ret_opt_t) > 0.01 or len(tr_bt) != tr_opt_t:
            print(f"DIFF [{stype} - {t}]: BT Ret={ret_bt:.2f}% (Tr={len(tr_bt)}) vs OPT Ret={ret_opt_t:.2f}% (Tr={tr_opt_t})")
            all_ok = False
        else:
            print(f"OK   [{stype} - {t}]: Ret={ret_bt:.2f}% | Trades={len(tr_bt)}")

if all_ok:
    print("\n✅ 100% PARITY VERIFIED ACROSS ALL 4 STRATEGIES AND 18 TICKERS!")
    sys.exit(0)
else:
    print("\n❌ PARITY FAILED FOR SOME TICKERS!")
    sys.exit(1)
