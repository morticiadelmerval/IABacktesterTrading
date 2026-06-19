import json

with open("results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for strat in data["spy_raw_ranking"]:
    if strat["strategy_id"] in ["S06", "S10"]:
        print(f"\n=== {strat['strategy_id']} ===")
        trades = strat["trades"]
        print(f"Total trades: {len(trades)}")
        if len(trades) > 0:
            print("Últimos 3 trades:")
            for t in trades[-3:]:
                print(f"  Entrada: {t['entry_date']} -> Salida: {t['exit_date']} | Razón: {t['reason']}")

