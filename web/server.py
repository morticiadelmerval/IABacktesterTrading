import http.server
import socketserver
import urllib.parse
import json
import traceback
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import backtester
import threading

PORT = 8000
IS_LOADING = True

class APIHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()

    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        
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
