import http.server
import socketserver
import urllib.parse
import json
import traceback
import backtester

PORT = 8000

class APIHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        
        # API Endpoint
        if parsed_path.path == '/api/recalculate':
            qs = urllib.parse.parse_qs(parsed_path.query)
            comm_str = qs.get('commission', ['0.0'])[0]
            try:
                # User enters percentage e.g. 0.4, we need decimal 0.004
                comm_pct = float(comm_str)
                comm_decimal = comm_pct / 100.0
            except ValueError:
                comm_decimal = 0.004
                
            try:
                print(f"Server received request to recalculate with commission: {comm_pct}%")
                import importlib
                importlib.reload(backtester)
                result_dict = backtester.run_all(commission=comm_decimal)
                
                # Also save to results.json to keep it updated for future static loads
                with open("results.json", "w", encoding="utf-8") as f:
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
    # Pre-load cache so first request is fast
    print("Pre-loading Yahoo Finance data into memory cache...")
    backtester.run_all(commission=0.004)
    print("\nData loaded. Starting server...")
    
    with socketserver.TCPServer(("", PORT), APIHandler) as httpd:
        print(f"Serving at port {PORT}. Web Dashboard available at http://localhost:{PORT}")
        httpd.serve_forever()
