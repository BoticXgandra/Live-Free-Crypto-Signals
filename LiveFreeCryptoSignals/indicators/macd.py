"""
Calculates the Moving Average Convergence Divergence (MACD) indicator.

MACD is a trend-following momentum indicator that shows the relationship between
two exponential moving averages (EMAs) of a security's price. The MACD line is
calculated by subtracting the longer-period EMA (typically 26-period) from the
shorter-period EMA (typically 12-period). A signal line, which is an EMA of the
MACD line (typically 9-period), is then plotted with the MACD line. The histogram
represents the difference between the MACD line and the signal line.
"""
import pandas as pd
import numpy as np
from .moving_average import calculate_ema # Import EMA from the same package

def calculate_macd(data: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
    """
    Calculates the Moving Average Convergence Divergence (MACD) line, Signal line, and Histogram.

    Args:
        data (pd.Series): A Pandas Series of prices (e.g., close prices).
        fast_period (int): The window period for the fast EMA (default is 12).
        slow_period (int): The window period for the slow EMA (default is 26).
                           Must be greater than `fast_period`.
        signal_period (int): The window period for the signal line EMA (default is 9).

    Returns:
        tuple[pd.Series, pd.Series, pd.Series]: 
            A tuple containing three Pandas Series:
            1. MACD Line (fast_ema - slow_ema)
            2. Signal Line (EMA of the MACD line)
            3. Histogram (MACD Line - Signal Line)
            All series are padded with NaN where values are not yet defined due to
            EMA calculation warm-up periods.
    """
    if not isinstance(data, pd.Series):
        data = pd.Series(data, dtype=float)

    if not (fast_period > 0 and slow_period > 0 and signal_period > 0):
        raise ValueError("All window periods (fast, slow, signal) must be positive integers.")
    if fast_period >= slow_period:
        raise ValueError("The fast_period must be less than the slow_period for standard MACD.")

    if data.empty:
        empty_series = pd.Series([], dtype=float, index=data.index)
        return empty_series, empty_series, empty_series
        
    # Calculate Fast EMA using the provided window period
    ema_fast = calculate_ema(data, window=fast_period)
    
    # Calculate Slow EMA using the provided window period
    ema_slow = calculate_ema(data, window=slow_period)
    
    # Calculate MACD Line: Fast EMA - Slow EMA
    # The MACD line will have NaNs until ema_slow has enough data (i.e., up to slow_period - 1)
    macd_line = ema_fast - ema_slow
    
    # Calculate Signal Line: EMA of the MACD Line
    # The signal line calculation should only start after enough MACD values are available.
    # calculate_ema handles min_periods internally. We apply it on the macd_line.
    # Since macd_line can have NaNs at the beginning, calculate_ema will correctly handle them.
    # The first valid signal value will appear after (slow_period - 1) + (signal_period - 1) data points.
    signal_line = calculate_ema(macd_line, window=signal_period)
    # No need to dropna before passing to calculate_ema, as calculate_ema itself handles initial NaNs.
    # Reindexing is also handled if calculate_ema returns series with same index as input.
    
    # Calculate Histogram: MACD Line - Signal Line
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

if __name__ == '__main__':
    # This block provides example usage when the script is run directly.
    prices_list = [
        22.27, 22.19, 22.08, 22.17, 22.18, 22.13, 22.23, 22.43, 22.24, 22.29,
        22.15, 22.39, 22.38, 22.61, 23.36, 24.05, 23.75, 23.83, 23.95, 23.63,
        23.82, 23.87, 23.65, 23.19, 23.10, 23.33, 22.68, 23.10, 22.40, 22.17,
        21.94, 21.20, 21.22, 21.54, 21.29, 21.01, 21.34, 20.97, 20.99, 21.20
    ] # Longer series for better MACD illustration
    prices_series = pd.Series(prices_list)

    print("Prices:", prices_series.tolist())

    # MACD Example
    fast_p, slow_p, signal_p = 12, 26, 9
    macd, signal, hist = calculate_macd(prices_series, fast_p, slow_p, signal_p)

    print(f"\nMACD ({fast_p}, {slow_p}, {signal_p}):")
    print("MACD Line:")
    print(macd.tolist())
    print("\nSignal Line:")
    print(signal.tolist())
    print("\nHistogram:")
    print(hist.tolist())

    macd_list_input, signal_list_input, hist_list_input = calculate_macd(
        prices_list, fast_p, slow_p, signal_p
    )
    print(f"\nMACD ({fast_p}, {slow_p}, {signal_p}) (list input):")
    print("MACD Line:")
    print(macd_list_input.tolist())
    print("\nSignal Line:")
    print(signal_list_input.tolist())
    print("\nHistogram:")
    print(hist_list_input.tolist())

    # Test with data shorter than slow_period
    short_series = prices_series.iloc[:20] # slow_period is 26
    print(f"\nMACD with data shorter than slow_period ({len(short_series)} items):")
    macd_short, signal_short, hist_short = calculate_macd(short_series, fast_p, slow_p, signal_p)
    print("MACD Line (short):")
    print(macd_short.tolist()) # Expect many NaNs
    print("Signal Line (short):")
    print(signal_short.tolist()) # Expect many NaNs
    print("Histogram (short):")
    print(hist_short.tolist()) # Expect many NaNs

    # Test with empty data
    empty_series = pd.Series([], dtype=float)
    print("\nMACD with empty data:")
    macd_empty, signal_empty, hist_empty = calculate_macd(empty_series)
    print("MACD Line (empty):", macd_empty.tolist())
    print("Signal Line (empty):", signal_empty.tolist())
    print("Histogram (empty):", hist_empty.tolist())
