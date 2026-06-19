import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from backtester import fetch_data

@njit
def kalman_filter_1d(prices, Q=1e-4, R=0.01):
    """
    1D Kalman Filter (Local Level Model).
    Q: Process Noise (cuánto cambia el "valor real" internamente). Mayor Q -> más rápido se adapta.
    R: Measurement Noise (cuánto ruido de mercado hay). Mayor R -> más suavizado y lento.
    La relación Q/R define el "lag" y la velocidad de respuesta.
    """
    n = len(prices)
    kf_mean = np.zeros(n)
    P = 1.0 # Error covariance
    
    kf_mean[0] = prices[0]
    
    for t in range(1, n):
        # Predict (asumimos que el próximo valor es igual al actual + ruido)
        x_pred = kf_mean[t-1]
        P_pred = P + Q
        
        # Update (corrección basada en el precio observado)
        K = P_pred / (P_pred + R) # Kalman Gain
        kf_mean[t] = x_pred + K * (prices[t] - x_pred)
        P = (1 - K) * P_pred
        
    return kf_mean

def backtest_crossover(df, indicator_col, name):
    # Buy when Close > Indicator, Sell when Close < Indicator
    signals = (df['Close'] > df[indicator_col]).astype(int)
    # Return del siguiente dia
    daily_returns = df['Close'].pct_change().shift(-1)
    
    # Estrategia
    strategy_returns = signals * daily_returns
    
    # Performance
    cum_ret = (1 + strategy_returns).cumprod()
    total_return = (cum_ret.iloc[-2] - 1) * 100 # -2 porque la ultima es NaN
    
    # Win rate de los trades individuales
    trades = []
    in_trade = False
    entry_price = 0
    for i in range(len(signals)-1):
        if signals.iloc[i] == 1 and not in_trade:
            in_trade = True
            entry_price = df['Close'].iloc[i]
        elif signals.iloc[i] == 0 and in_trade:
            in_trade = False
            exit_price = df['Close'].iloc[i]
            trades.append(exit_price / entry_price - 1)
            
    # Si termino la serie y quedo comprado
    if in_trade:
        trades.append(df['Close'].iloc[-1] / entry_price - 1)
        
    trades = np.array(trades)
    win_rate = len(trades[trades > 0]) / len(trades) * 100 if len(trades) > 0 else 0
    
    # Max DD
    roll_max = cum_ret.cummax()
    dd = cum_ret / roll_max - 1
    max_dd = dd.min() * 100
    
    print(f"\n=== ESTRATEGIA: {name} ===")
    print(f"Retorno Total Acumulado: {total_return:.2f}%")
    print(f"Cantidad de Trades: {len(trades)}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Max Drawdown: {max_dd:.2f}%")
    
    return cum_ret

def test_kalman():
    print("Cargando datos...")
    data_dict = fetch_data()
    df = data_dict["SPY"].copy()
    
    print("Calculando SMA 50...")
    df['SMA_50'] = df['Close'].rolling(50).mean()
    
    print("Calculando Filtro de Kalman...")
    # Parametros de prueba: Q=1e-4, R=0.01 (Equilibrio entre reaccion rapida y suavizado)
    df['Kalman'] = kalman_filter_1d(df['Close'].values, Q=1e-4, R=0.01)
    
    df = df.dropna()
    
    # Run simple backtests
    cum_ret_sma = backtest_crossover(df, 'SMA_50', 'Cruce SMA 50')
    cum_ret_kalman = backtest_crossover(df, 'Kalman', 'Cruce Kalman Filter')
    
    # Buy and hold base
    bh = (1 + df['Close'].pct_change().shift(-1)).cumprod()
    print(f"\n=== BUY & HOLD ===")
    print(f"Retorno Total: {(bh.iloc[-2] - 1)*100:.2f}%")
    
    # Visualizacion de las lineas en el grafico
    plt.figure(figsize=(14,7))
    plt.plot(df.index[-500:], df['Close'].iloc[-500:], label='SPY Close', alpha=0.5, color='gray')
    plt.plot(df.index[-500:], df['SMA_50'].iloc[-500:], label='SMA 50', color='blue', linestyle='--')
    plt.plot(df.index[-500:], df['Kalman'].iloc[-500:], label='Kalman Filter', color='red', linewidth=2)
    plt.title('Comparación: SMA 50 vs Kalman Filter (Últimos 500 días)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('kalman_analysis.png')
    print("\nGrafico comparativo guardado en kalman_analysis.png")

if __name__ == '__main__':
    test_kalman()
