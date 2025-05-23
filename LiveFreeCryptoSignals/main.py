"""
Main entry point for the Live Free Crypto Signals application.

This script initializes and runs the Flask web server. It also contains the core
logic for orchestrating data fetching from Binance, calculating various technical
indicators, and generating trading signals based on these indicators. The generated
signals are then displayed via a web interface.
"""

from webapp import app # Flask app instance
import pandas as pd

# Project-specific imports
# Ensure these modules are in PYTHONPATH; typically, running from the project root handles this.
from binance_client import get_klines
from indicators import (
    calculate_sma,
    calculate_ema,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands
)
from signals import (
    generate_rsi_signal,
    generate_macd_signal,
    generate_ma_crossover_signal,
    generate_bb_signal
)

# --- Configuration ---
# Define the cryptocurrency symbols to process
SYMBOLS = ['BTCUSDT', 'ETHUSDT'] 
# Define the timeframes for kline data
# Example: ['1m', '5m', '15m', '1h', '4h', '1d']
# Using a reduced set for faster processing during development/testing
TIMEFRAMES = ['1h', '4h'] 

# Standard column names for Binance kline data.
# Reference: https://github.com/sammchardy/python-binance/blob/master/binance/client.py#L1030
# These are: Open time, Open, High, Low, Close, Volume, Close time, Quote asset volume, Number of trades,
# Taker buy base asset volume, Taker buy quote asset volume, Ignore.
KLINE_COLUMN_NAMES = [
    'Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 
    'Close time', 'Quote asset volume', 'Number of trades', 
    'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
]

def process_all_signals():
    """
    Fetches kline data, calculates various technical indicators, and generates trading
    signals for all configured symbols and timeframes.

    The process for each symbol/timeframe pair is:
    1. Fetch kline data from Binance.
    2. Convert data to a Pandas DataFrame.
    3. Calculate indicators: SMA (10, 30), EMA (12, 26, 50), RSI (14),
       MACD (12, 26, 9), Bollinger Bands (20, 2).
    4. Generate signals based on these indicators (RSI, MACD crossover,
       MA crossover, Bollinger Bands breakout).
    5. Store the signal information including the indicator values that led to it.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary represents a signal
                    and contains keys like 'coin', 'timeframe', 'indicator',
                    'signal', and 'value' (current indicator readings).
    """
    all_signals_data = [] # List to store all generated signal data

    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            print(f"Processing {symbol} - {timeframe}...")
            # Fetch kline data from Binance API; limit to 200 periods for sufficient data for indicators
            klines_data = get_klines(symbol, timeframe, limit=200) 

            if not klines_data:
                print(f"  No kline data for {symbol} - {timeframe}. Skipping.")
                continue

            # Convert raw kline list to a Pandas DataFrame with named columns
            df = pd.DataFrame(klines_data, columns=KLINE_COLUMN_NAMES)
            
            # Ensure essential columns ('Open', 'High', 'Low', 'Close', 'Volume') are numeric
            # Errors during conversion will be coerced to NaN
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Drop rows where 'Close' price is NaN, as it's crucial for most calculations
            df.dropna(subset=['Close'], inplace=True) 

            if df.empty:
                print(f"  DataFrame empty after processing for {symbol} - {timeframe} (e.g., all Close prices were NaN). Skipping.")
                continue

            close_prices = df['Close'] # Series of close prices for indicator calculations

            # --- Calculate Indicators ---
            # Simple Moving Averages (e.g., 10-period and 30-period for crossover strategy)
            sma10 = calculate_sma(close_prices, 10)
            sma30 = calculate_sma(close_prices, 30)
            
            # Exponential Moving Averages (e.g., 12, 26 for MACD; 50 as a standalone trend indicator)
            ema12 = calculate_ema(close_prices, 12) # Used in MACD
            ema26 = calculate_ema(close_prices, 26) # Used in MACD
            ema50 = calculate_ema(close_prices, 50) # Example of a standalone EMA

            # Relative Strength Index (typically 14-period)
            rsi = calculate_rsi(close_prices, 14)
            
            # Moving Average Convergence Divergence (standard periods: 12-fast, 26-slow, 9-signal)
            macd_line, signal_line, _ = calculate_macd(close_prices, fast_period=12, slow_period=26, signal_period=9)
            
            # Bollinger Bands (typically 20-period SMA with 2 standard deviations)
            upper_band, middle_band, lower_band = calculate_bollinger_bands(close_prices, window=20, num_std_dev=2)

            # --- Generate Signals & Store Data ---
            # For each signal type, get the latest indicator value(s) and the generated signal.
            # Store this information in the all_signals_data list.
            # Using .dropna().iloc[-1] safely gets the last valid (non-NaN) value.
            
            # RSI Signal
            latest_rsi_val = rsi.dropna().iloc[-1] if not rsi.dropna().empty else 'N/A'
            rsi_signal = generate_rsi_signal(rsi, buy_threshold=30, sell_threshold=70)
            all_signals_data.append({
                'coin': symbol, 'timeframe': timeframe, 'indicator': 'RSI(14)', 
                'signal': rsi_signal, 
                'value': f"{latest_rsi_val:.2f}" if isinstance(latest_rsi_val, float) else latest_rsi_val
            })

            # MACD Signal
            latest_macd_val = macd_line.dropna().iloc[-1] if not macd_line.dropna().empty else 'N/A'
            latest_signal_val = signal_line.dropna().iloc[-1] if not signal_line.dropna().empty else 'N/A'
            macd_crossover_signal = generate_macd_signal(macd_line, signal_line) # Renamed for clarity
            macd_display_value = 'N/A'
            if isinstance(latest_macd_val, float) and isinstance(latest_signal_val, float):
                macd_display_value = f"MACD:{latest_macd_val:.2f}, Signal:{latest_signal_val:.2f}"
            all_signals_data.append({
                'coin': symbol, 'timeframe': timeframe, 'indicator': 'MACD(12,26,9)', 
                'signal': macd_crossover_signal, 
                'value': macd_display_value
            })
            
            # MA Crossover Signal (SMA10 vs SMA30)
            latest_sma10 = sma10.dropna().iloc[-1] if not sma10.dropna().empty else 'N/A'
            latest_sma30 = sma30.dropna().iloc[-1] if not sma30.dropna().empty else 'N/A'
            ma_cross_signal = generate_ma_crossover_signal(sma10, sma30)
            ma_cross_display_value = 'N/A'
            if isinstance(latest_sma10, float) and isinstance(latest_sma30, float):
                ma_cross_display_value = f"SMA10:{latest_sma10:.2f}, SMA30:{latest_sma30:.2f}"
            all_signals_data.append({
                'coin': symbol, 'timeframe': timeframe, 'indicator': 'MA Cross (SMA10/SMA30)', 
                'signal': ma_cross_signal, 
                'value': ma_cross_display_value
            })

            # Bollinger Bands Signal
            latest_close_val = close_prices.dropna().iloc[-1] if not close_prices.dropna().empty else 'N/A'
            latest_lower_band = lower_band.dropna().iloc[-1] if not lower_band.dropna().empty else 'N/A'
            latest_upper_band = upper_band.dropna().iloc[-1] if not upper_band.dropna().empty else 'N/A'
            bb_breakout_signal = generate_bb_signal(close_prices, lower_band, upper_band) # Renamed for clarity
            bb_display_value = 'N/A'
            if isinstance(latest_close_val, float) and isinstance(latest_lower_band, float) and isinstance(latest_upper_band, float):
                 bb_display_value = f"P:{latest_close_val:.2f}, L:{latest_lower_band:.2f}, U:{latest_upper_band:.2f}"
            all_signals_data.append({
                'coin': symbol, 'timeframe': timeframe, 'indicator': 'Bollinger Bands(20,2)', 
                'signal': bb_breakout_signal, 
                'value': bb_display_value
            })
            
            # Example: Standalone EMA50 value (no direct signal, just for informational display)
            latest_ema50 = ema50.dropna().iloc[-1] if not ema50.dropna().empty else 'N/A'
            all_signals_data.append({
                'coin': symbol, 'timeframe': timeframe, 'indicator': 'EMA(50)', 
                'signal': 'INFO', # 'INFO' indicates this is informational, not a direct buy/sell signal
                'value': f"{latest_ema50:.2f}" if isinstance(latest_ema50, float) else latest_ema50
            })
            print(f"  Finished processing {symbol} - {timeframe}.")

    return all_signals_data

if __name__ == '__main__':
    # This block runs when the script is executed directly (e.g., `python main.py`)
    
    # --- Optional: Manual test of signal processing ---
    # You can uncomment these lines to test signal generation directly without running the web app.
    # This is useful for debugging the core logic.
    # print("Performing manual signal processing test...")
    # test_signals = process_all_signals()
    # for sig_data in test_signals: # Renamed variable to avoid conflict
    #     print(sig_data)
    # print("Manual signal processing test complete.")
    
    # --- Start the Flask Web Application ---
    print("Starting Flask app...")
    # `debug=True` enables Flask's debugger and auto-reloader. Good for development.
    # `host='0.0.0.0'` makes the server accessible from any network interface,
    # not just localhost. Useful for testing from other devices on the same network.
    # `port=8080` sets the listening port.
    app.run(debug=True, host='0.0.0.0', port=8080)
