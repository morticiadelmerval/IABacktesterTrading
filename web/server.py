import http.server
import socketserver
import urllib.parse
import json
import traceback
import sys
import os
import shutil
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import backtester
import threading

PORT = 8000
IS_LOADING = True

AI_STATUS = {
    "is_running": False,
    "current_step": 0,
    "total_steps": 6,
    "step_name": "",
    "error": None
}

def run_ai_update_background():
    global AI_STATUS
    AI_STATUS["is_running"] = True
    AI_STATUS["error"] = None
    AI_STATUS["current_step"] = 0

    steps = [
        ("models/precalculate_timesfm.py", "[1/6] Google TimesFM (Oráculo Principal)"),
        ("models/finetune_tspulse.py", "[2/6] IBM TSPulse Univariate"),
        ("models/finetune_tspulse_multi.py", "[3/6] IBM TSPulse Multivariate"),
        ("models/train_minirocket.py", "[4/6] MiniRocket Clasificación"),
        ("models/train_minirocket_gpu.py", "[5/6] MiniRocketPlus GPU Probabilidades"),
        ("models/train_xgboost_stack.py", "[6/6] XGBoost Stack Ensamble")
    ]

    py_cmd = shutil.which("uv")
    if py_cmd:
        base_cmd = [py_cmd, "run", "python"]
    else:
        base_cmd = [sys.executable]

    env = os.environ.copy()
    env["YF_CACHE_SECONDS"] = "3600"

    try:
        import subprocess
        for idx, (script, desc) in enumerate(steps, 1):
            AI_STATUS["current_step"] = idx
            AI_STATUS["step_name"] = desc
            print(f"\n[AI Background Update] {desc}...")
            res = subprocess.run(base_cmd + [script], env=env)
            if res.returncode != 0:
                print(f"[AI Background Update] Advertencia en {script} (Código {res.returncode})")
        
        AI_STATUS["step_name"] = "Regenerando rankings y backtests finales..."
        import importlib
        importlib.reload(backtester)
        result_dict = backtester.run_all(commission=0.004)
        with open("data/results.json", "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        AI_STATUS["error"] = str(e)
        traceback.print_exc()
    finally:
        AI_STATUS["is_running"] = False
        AI_STATUS["step_name"] = "Completado"

class APIHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()

    def do_POST(self):
        parsed_path = urllib.parse.urlparse(self.path)
        if parsed_path.path == '/api/update-ai':
            if not AI_STATUS["is_running"]:
                threading.Thread(target=run_ai_update_background, daemon=True).start()
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "started", "is_running": True}).encode('utf-8'))
            return
        self.send_error(404)

    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        
        if parsed_path.path == '/api/ai-status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            last_ai_date = None
            if os.path.exists("data/xgboost_stack_signals.json"):
                try:
                    with open("data/xgboost_stack_signals.json", "r") as f:
                        d = json.load(f)
                        spy_dates = list(d.get("SPY", {}).keys())
                        if spy_dates:
                            last_ai_date = sorted(spy_dates)[-1][:10]
                except Exception:
                    pass
            
            last_market_date = None
            if os.path.exists(".data_cache/SPY.csv"):
                try:
                    import pandas as pd
                    df_spy = pd.read_csv(".data_cache/SPY.csv")
                    dates = df_spy.iloc[:, 0].dropna()
                    if not dates.empty:
                        last_market_date = str(dates.iloc[-1])[:10]
                except Exception:
                    pass
                    
            status_code = "up_to_date"
            if AI_STATUS["is_running"]:
                status_code = "running"
            elif last_ai_date and last_market_date and last_ai_date < last_market_date:
                status_code = "outdated"
                
            res = {
                "status": status_code,
                "is_running": AI_STATUS["is_running"],
                "step": AI_STATUS["current_step"],
                "total": AI_STATUS["total_steps"],
                "step_name": AI_STATUS["step_name"],
                "last_ai_date": last_ai_date or "Desconocido",
                "latest_market_date": last_market_date or "Desconocido",
                "error": AI_STATUS["error"]
            }
            self.wfile.write(json.dumps(res).encode('utf-8'))
            return

        # API Endpoint
        if parsed_path.path == '/api/status':
            global IS_LOADING
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"loading": IS_LOADING}).encode('utf-8'))
            return

        if parsed_path.path == '/api/recalculate':
            qs = urllib.parse.parse_qs(parsed_path.query)
            comm_str = qs.get('commission', ['0.0'])[0]
            try:
                # User enters percentage e.g. 0.4, we need decimal 0.004
                comm_pct = float(comm_str)
                comm_decimal = comm_pct / 100.0
            except ValueError:
                comm_decimal = 0.004
                
            start_date = qs.get('start_date', [None])[0]
            end_date = qs.get('end_date', [None])[0]
            
            try:
                print(f"Server received request to recalculate with commission: {comm_pct}%, start: {start_date}, end: {end_date}")
                import importlib
                importlib.reload(backtester)
                result_dict = backtester.run_all(commission=comm_decimal, start_date=start_date, end_date=end_date)
                
                # Also save to results.json to keep it updated for future static loads
                with open("data/results.json", "w", encoding="utf-8") as f:
                    json.dump(result_dict, f, indent=2, ensure_ascii=False)
                    
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(result_dict).encode('utf-8'))
            except Exception as e:
                print("Error during recalculation:")
                traceback.print_exc()
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
            return
            
        if parsed_path.path == '/api/live-prices':
            try:
                import yfinance as yf
                import pandas as pd
                from backtester import TICKERS
                
                # Fetch only 1 day, 1 minute interval to be lightning fast
                df = yf.download(TICKERS, period="1d", interval="1m", progress=False)
                prices = {}
                
                if isinstance(df.columns, pd.MultiIndex):
                    # MultiIndex: (PriceType, Ticker)
                    # Get the last row of 'Close'
                    last_row = df['Close'].iloc[-1]
                    for tk in TICKERS:
                        prices[tk] = float(last_row[tk])
                else:
                    # Single ticker or flat index (shouldn't happen with 18 tickers but fallback)
                    for tk in TICKERS:
                        prices[tk] = float(df['Close'].iloc[-1])
                        
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(prices).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
            return

        # Serve static files normally
        super().do_GET()

if __name__ == "__main__":
    def load_data_thread():
        global IS_LOADING
        print("Pre-loading Yahoo Finance data into memory cache...")
        try:
            backtester.run_all(commission=0.004)
            print("\nData loaded.")
        except Exception as e:
            print(f"\nError loading data: {e}")
            traceback.print_exc()
        IS_LOADING = False

    # Start data loading in background
    threading.Thread(target=load_data_thread, daemon=True).start()
    
    with socketserver.TCPServer(("", PORT), APIHandler) as httpd:
        print(f"Serving at port {PORT}. Web Dashboard available at http://localhost:{PORT}")
        httpd.serve_forever()
