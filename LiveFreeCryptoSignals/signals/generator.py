"""
Generates trading signals based on various technical indicator conditions.

Each function in this module takes one or more Pandas Series (representing
indicator values or prices) and returns a string signal: 'BUY', 'SELL', or 'HOLD'.
The logic aims to identify common trading patterns like threshold crossings for RSI,
crossovers for MACD and Moving Averages, and price breakouts for Bollinger Bands.

The functions are designed to be robust to insufficient data by defaulting to 'HOLD'.
"""
import pandas as pd
import numpy as np

def generate_rsi_signal(rsi_series: pd.Series, buy_threshold: int = 30, sell_threshold: int = 70) -> str:
    """
    Generates a buy, sell, or hold signal based on the latest RSI value.

    - 'BUY' if RSI < buy_threshold (oversold).
    - 'SELL' if RSI > sell_threshold (overbought).
    - 'HOLD' otherwise, or if data is insufficient.

    Args:
        rsi_series (pd.Series): Pandas Series of RSI values.
        buy_threshold (int): The RSI level below which a 'BUY' signal is generated.
        sell_threshold (int): The RSI level above which a 'SELL' signal is generated.

    Returns:
        str: 'BUY', 'SELL', or 'HOLD'.
    """
    if rsi_series is None or rsi_series.empty:
        return 'HOLD' # Not enough data
    
    # Get the last valid (non-NaN) RSI value
    latest_rsi = rsi_series.dropna().iloc[-1] if not rsi_series.dropna().empty else np.nan

    if pd.isna(latest_rsi):
        return 'HOLD' # Not enough valid data points

    if latest_rsi < buy_threshold:
        return 'BUY'
    elif latest_rsi > sell_threshold:
        return 'SELL'
    else:
        return 'HOLD'

def generate_macd_signal(macd_line: pd.Series, signal_line: pd.Series) -> str:
    """
    Generates a buy, sell, or hold signal based on MACD line and Signal line crossover.

    - 'BUY' if the MACD line crosses above the Signal line.
    - 'SELL' if the MACD line crosses below the Signal line.
    - 'HOLD' otherwise, or if data is insufficient for a crossover.

    Args:
        macd_line (pd.Series): Pandas Series of MACD line values.
        signal_line (pd.Series): Pandas Series of Signal line values.

    Returns:
        str: 'BUY', 'SELL', or 'HOLD'.
    """
    if macd_line is None or signal_line is None or macd_line.empty or signal_line.empty:
        return 'HOLD'

    # Drop NaNs from both series individually first
    macd_nona = macd_line.dropna()
    signal_nona = signal_line.dropna()

    # Need at least two points from each series to check for a crossover
    if len(macd_nona) < 2 or len(signal_nona) < 2:
        return 'HOLD' 

    # Align series by index for correct comparison, then drop rows where either value is NaN
    # This ensures we are comparing corresponding points in time.
    combined = pd.DataFrame({'macd': macd_nona, 'signal': signal_nona}).dropna()
    
    # Need at least two aligned data points to check for a crossover
    if len(combined) < 2:
        return 'HOLD' 

    # Get the latest two aligned (non-NaN) values for comparison
    last_macd = combined['macd'].iloc[-1]
    prev_macd = combined['macd'].iloc[-2]
    last_signal = combined['signal'].iloc[-1]
    prev_signal = combined['signal'].iloc[-2]

    # Bullish Crossover: MACD was below Signal, now it's above.
    if last_macd > last_signal and prev_macd < prev_signal:
        return 'BUY'
    # Bearish Crossover: MACD was above Signal, now it's below.
    elif last_macd < last_signal and prev_macd > prev_signal:
        return 'SELL'
    else:
        return 'HOLD' # No crossover in the last period

def generate_ma_crossover_signal(fast_ma_series: pd.Series, slow_ma_series: pd.Series) -> str:
    """
    Generates a buy, sell, or hold signal based on a Moving Average (MA) crossover.

    - 'BUY' if the Fast MA crosses above the Slow MA.
    - 'SELL' if the Fast MA crosses below the Slow MA.
    - 'HOLD' otherwise, or if data is insufficient.

    Args:
        fast_ma_series (pd.Series): Pandas Series of the Fast Moving Average values.
        slow_ma_series (pd.Series): Pandas Series of the Slow Moving Average values.

    Returns:
        str: 'BUY', 'SELL', or 'HOLD'.
    """
    if fast_ma_series is None or slow_ma_series is None or fast_ma_series.empty or slow_ma_series.empty:
        return 'HOLD'

    fast_nona = fast_ma_series.dropna()
    slow_nona = slow_ma_series.dropna()

    if len(fast_nona) < 2 or len(slow_nona) < 2:
        return 'HOLD' 
    
    combined = pd.DataFrame({'fast': fast_nona, 'slow': slow_nona}).dropna()
    if len(combined) < 2:
        return 'HOLD'

    last_fast = combined['fast'].iloc[-1]
    prev_fast = combined['fast'].iloc[-2]
    last_slow = combined['slow'].iloc[-1]
    prev_slow = combined['slow'].iloc[-2]

    # Bullish Crossover (Golden Cross): Fast MA was below Slow MA, now it's above.
    if last_fast > last_slow and prev_fast < prev_slow:
        return 'BUY'
    # Bearish Crossover (Death Cross): Fast MA was above Slow MA, now it's below.
    elif last_fast < last_slow and prev_fast > prev_slow:
        return 'SELL'
    else:
        return 'HOLD'

def generate_bb_signal(close_prices: pd.Series, lower_band: pd.Series, upper_band: pd.Series) -> str:
    """
    Generates a buy, sell, or hold signal based on Bollinger Bands breakouts.

    - 'BUY' if the price crosses below the lower band (from above to below).
    - 'SELL' if the price crosses above the upper band (from below to above).
    - 'HOLD' otherwise, or if data is insufficient.

    Note: This is a simplified breakout strategy. More complex strategies might
    look for bounces off bands or mean reversion.

    Args:
        close_prices (pd.Series): Pandas Series of closing prices.
        lower_band (pd.Series): Pandas Series of Bollinger Lower Band values.
        upper_band (pd.Series): Pandas Series of Bollinger Upper Band values.

    Returns:
        str: 'BUY', 'SELL', or 'HOLD'.
    """
    if close_prices is None or lower_band is None or upper_band is None or \
       close_prices.empty or lower_band.empty or upper_band.empty:
        return 'HOLD'

    close_nona = close_prices.dropna()
    lower_nona = lower_band.dropna()
    upper_nona = upper_band.dropna()

    # Need at least two data points for close prices and bands to determine a crossover
    if len(close_nona) < 2 or len(lower_nona) < 2 or len(upper_nona) < 2:
        return 'HOLD' 

    # Align all series by index and drop rows with any NaNs to ensure valid comparisons
    combined = pd.DataFrame({
        'close': close_nona, 
        'lower': lower_nona, 
        'upper': upper_nona
    }).dropna()
    
    if len(combined) < 2: # Still need at least two aligned points after combining
        return 'HOLD'

    # Get the latest two aligned values
    latest_close = combined['close'].iloc[-1]
    prev_close = combined['close'].iloc[-2]
    latest_lower = combined['lower'].iloc[-1]
    prev_lower = combined['lower'].iloc[-2] # Previous lower band value
    latest_upper = combined['upper'].iloc[-1]
    prev_upper = combined['upper'].iloc[-2] # Previous upper band value

    # Buy signal: Price crossed below the lower band
    # (Previous close was above or on previous lower band, current close is below current lower band)
    if latest_close < latest_lower and prev_close > prev_lower:
        return 'BUY'
    # Sell signal: Price crossed above the upper band
    # (Previous close was below or on previous upper band, current close is above current upper band)
    elif latest_close > latest_upper and prev_close < prev_upper:
        return 'SELL'
    else:
        return 'HOLD'

if __name__ == '__main__':
    # This block provides example usage and basic tests when the script is run directly.
    print("--- Signal Generation Examples ---")

    # --- RSI Example ---
    print("--- RSI Signals ---")
    rsi_data_buy = pd.Series([50, 40, 35, 25]) # Ends below 30
    rsi_data_sell = pd.Series([50, 60, 65, 75]) # Ends above 70
    rsi_data_hold = pd.Series([40, 50, 60])    # Ends between 30-70
    rsi_data_empty = pd.Series([], dtype=float)
    rsi_data_nan = pd.Series([np.nan, np.nan])
    rsi_data_short = pd.Series([50])

    print(f"RSI {rsi_data_buy.tolist()}: {generate_rsi_signal(rsi_data_buy)}")
    print(f"RSI {rsi_data_sell.tolist()}: {generate_rsi_signal(rsi_data_sell)}")
    print(f"RSI {rsi_data_hold.tolist()}: {generate_rsi_signal(rsi_data_hold)}")
    print(f"RSI {rsi_data_empty.tolist()}: {generate_rsi_signal(rsi_data_empty)}")
    print(f"RSI {rsi_data_nan.tolist()}: {generate_rsi_signal(rsi_data_nan)}")
    print(f"RSI {rsi_data_short.tolist()}: {generate_rsi_signal(rsi_data_short)}")
    print(f"RSI with NaN at end {pd.Series([30,20,np.nan]).tolist()}: {generate_rsi_signal(pd.Series([30,20,np.nan]))}")


    # --- MACD Example ---
    print("\n--- MACD Signals ---")
    # Buy crossover: MACD was below signal, now above
    macd_buy_m = pd.Series([0.1, 0.05, 0.15]) # 0.05 (prev) < 0.1 (prev_signal), 0.15 (curr) > 0.12 (curr_signal)
    macd_buy_s = pd.Series([0.12, 0.1, 0.12])
    print(f"MACD BUY: M={macd_buy_m.tolist()}, S={macd_buy_s.tolist()} -> {generate_macd_signal(macd_buy_m, macd_buy_s)}")

    # Sell crossover: MACD was above signal, now below
    macd_sell_m = pd.Series([0.1, 0.15, 0.05]) # 0.15 (prev) > 0.12 (prev_signal), 0.05 (curr) < 0.1 (curr_signal)
    macd_sell_s = pd.Series([0.08, 0.12, 0.1])
    print(f"MACD SELL: M={macd_sell_m.tolist()}, S={macd_sell_s.tolist()} -> {generate_macd_signal(macd_sell_m, macd_sell_s)}")

    # Hold (no crossover)
    macd_hold_m = pd.Series([0.1, 0.15, 0.2]) # MACD rising, but always above signal
    macd_hold_s = pd.Series([0.05, 0.1, 0.15])
    print(f"MACD HOLD: M={macd_hold_m.tolist()}, S={macd_hold_s.tolist()} -> {generate_macd_signal(macd_hold_m, macd_hold_s)}")
    
    macd_hold_m2 = pd.Series([0.1, np.nan, 0.15, 0.2]) 
    macd_hold_s2 = pd.Series([0.05, 0.08, 0.1, 0.15])
    print(f"MACD HOLD (with NaN): M={macd_hold_m2.tolist()}, S={macd_hold_s2.tolist()} -> {generate_macd_signal(macd_hold_m2, macd_hold_s2)}")

    print(f"MACD Insufficient: M={[0.1]}, S={[0.05]} -> {generate_macd_signal(pd.Series([0.1]), pd.Series([0.05]))}")

    # --- MA Crossover Example ---
    print("\n--- MA Crossover Signals ---")
    # Buy crossover: Fast MA was below Slow MA, now above
    ma_buy_fast = pd.Series([10, 10.5, 11.5]) # 10.5 (prev_fast) < 11 (prev_slow), 11.5 (curr_fast) > 11.2 (curr_slow)
    ma_buy_slow = pd.Series([10.8, 11, 11.2])
    print(f"MA BUY: Fast={ma_buy_fast.tolist()}, Slow={ma_buy_slow.tolist()} -> {generate_ma_crossover_signal(ma_buy_fast, ma_buy_slow)}")

    # Sell crossover: Fast MA was above Slow MA, now below
    ma_sell_fast = pd.Series([10, 11, 10.5]) # 11 (prev_fast) > 10.8 (prev_slow), 10.5 (curr_fast) < 10.9 (curr_slow)
    ma_sell_slow = pd.Series([9.8, 10.8, 10.9])
    print(f"MA SELL: Fast={ma_sell_fast.tolist()}, Slow={ma_sell_slow.tolist()} -> {generate_ma_crossover_signal(ma_sell_fast, ma_sell_slow)}")
    
    ma_hold_fast = pd.Series([10, 10.5, np.nan, 11])
    ma_hold_slow = pd.Series([10.8, 11, 11.2, 11.5])
    print(f"MA HOLD (with NaN): Fast={ma_hold_fast.tolist()}, Slow={ma_hold_slow.tolist()} -> {generate_ma_crossover_signal(ma_hold_fast, ma_hold_slow)}")
    print(f"MA Insufficient: Fast={[10]}, Slow={[11]} -> {generate_ma_crossover_signal(pd.Series([10]), pd.Series([11]))}")


    # --- Bollinger Bands Example ---
    print("\n--- Bollinger Bands Signals ---")
    # Buy signal: Price crossed below lower band
    bb_prices_buy = pd.Series([25, 22, 21])    # prev_close=22 > prev_lower=21.5, latest_close=21 < latest_lower=21.5
    bb_lower_buy  = pd.Series([24, 21.5, 21.5])
    bb_upper_buy  = pd.Series([26, 25.5, 25.5])
    print(f"BB BUY: P={bb_prices_buy.tolist()}, L={bb_lower_buy.tolist()}, U={bb_upper_buy.tolist()} -> {generate_bb_signal(bb_prices_buy, bb_lower_buy, bb_upper_buy)}")

    # Sell signal: Price crossed above upper band
    bb_prices_sell = pd.Series([25, 28, 29])   # prev_close=28 < prev_upper=28.5, latest_close=29 > latest_upper=28.5
    bb_lower_sell  = pd.Series([24, 24.5, 24.5])
    bb_upper_sell  = pd.Series([26, 28.5, 28.5])
    print(f"BB SELL: P={bb_prices_sell.tolist()}, L={bb_lower_sell.tolist()}, U={bb_upper_sell.tolist()} -> {generate_bb_signal(bb_prices_sell, bb_lower_sell, bb_upper_sell)}")

    # Hold signal: Price within bands
    bb_prices_hold = pd.Series([25, 24, 23])
    bb_lower_hold  = pd.Series([22, 22, 22])
    bb_upper_hold  = pd.Series([26, 26, 26])
    print(f"BB HOLD: P={bb_prices_hold.tolist()}, L={bb_lower_hold.tolist()}, U={bb_upper_hold.tolist()} -> {generate_bb_signal(bb_prices_hold, bb_lower_hold, bb_upper_hold)}")
    
    # Hold signal: Price touched lower band but did not cross
    bb_prices_touch_lower = pd.Series([25, 22, 22])
    bb_lower_touch_lower  = pd.Series([24, 22, 22]) # prev_close=22 == prev_lower=22, so not a cross *from above*
    bb_upper_touch_lower  = pd.Series([26, 26, 26])
    print(f"BB HOLD (touch lower): P={bb_prices_touch_lower.tolist()}, L={bb_lower_touch_lower.tolist()}, U={bb_upper_touch_lower.tolist()} -> {generate_bb_signal(bb_prices_touch_lower, bb_lower_touch_lower, bb_upper_touch_lower)}")

    bb_prices_short = pd.Series([25])
    bb_lower_short  = pd.Series([24])
    bb_upper_short  = pd.Series([26])
    print(f"BB Insufficient: P={bb_prices_short.tolist()} -> {generate_bb_signal(bb_prices_short, bb_lower_short, bb_upper_short)}")
    
    bb_prices_nan = pd.Series([25, np.nan, 23])
    bb_lower_nan  = pd.Series([22, 22, 22])
    bb_upper_nan  = pd.Series([26, 26, 26])
    print(f"BB HOLD (with NaN): P={bb_prices_nan.tolist()} -> {generate_bb_signal(bb_prices_nan, bb_lower_nan, bb_upper_nan)}")
