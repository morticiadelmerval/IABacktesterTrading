import json

with open("results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for r in data["ranking"]:
    print(f"\n--- {r['strategy_id']} ---")
    for tk, res in r["ticker_results"].items():
        trades = res["trades"]
        real_exits = [t for t in trades if t["reason"] != "End of History"]
        if real_exits:
            last = real_exits[-1]
            print(f"  {tk:5s}: Último cierre real -> {last['exit_date']}")
        else:
            print(f"  {tk:5s}: Sin cierres reales")
