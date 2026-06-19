import os
import sys
import itertools
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from backtester import run_all

in_thresholds = [0.33, 0.34, 0.35, 0.36, 0.37, 0.38]
out_thresholds = [0.28, 0.29, 0.30, 0.31, 0.32, 0.33]

best_avg_return = -9999
best_params = None
best_result = None

for in_th, out_th in itertools.product(in_thresholds, out_thresholds):
    os.environ["MINIROCKET_GPU_IN_TH"] = str(in_th)
    os.environ["MINIROCKET_GPU_OUT_TH"] = str(out_th)
    
    # print(f"Testing in_th={in_th}, out_th={out_th}...")
    output = run_all(commission=0.004)
    
    # find S18 inside ranking
    ranking = output.get("ranking", [])
    s18_data = None
    s18_rank = -1
    for idx, r in enumerate(ranking):
        if r["strategy_id"] == "S18":
            s18_data = r
            s18_rank = idx + 1
            break
            
    if s18_data:
        avg_return = s18_data["aggregate_metrics"]["avg_return"]
        total_trades = s18_data["aggregate_metrics"]["total_trades"]
        # print(f"Result for S18: AvgReturn={avg_return:.2f}%, Trades={total_trades}, Rank={s18_rank}")
        if avg_return > best_avg_return:
            best_avg_return = avg_return
            best_params = (in_th, out_th)
            best_result = {
                "avg_return": avg_return,
                "total_trades": total_trades,
                "rank": s18_rank
            }

print("\n--- BEST RESULT ---")
print(f"IN: {best_params[0]}, OUT: {best_params[1]}")
print(f"AvgReturn: {best_result['avg_return']:.2f}%")
print(f"Trades: {best_result['total_trades']}")
print(f"Rank: {best_result['rank']}")
