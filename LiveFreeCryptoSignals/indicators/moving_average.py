"""
Calculates moving average indicators like Simple Moving Average (SMA)
and Exponential Moving Average (EMA).

These functions take a Pandas Series of price data and a window period
as input and return a Pandas Series of the calculated moving average.
"""
import pandas as pd
import numpy as np

def calculate_sma(data: pd.Series, window: int) -> pd.Series:
    """
    Calculates the Simple Moving Average (SMA).

    The SMA is the unweighted mean of the previous 'window' data points.
    It's a lagging indicator used to identify trends.

    Args:
        data (pd.Series): A Pandas Series of prices (e.g., close prices).
        window (int): The lookback period for the SMA calculation.

    Returns:
        pd.Series: A Pandas Series containing the SMA values. Early values
                   (before enough data is available for the window) will be NaN.
    """
    if not isinstance(data, pd.Series):
        # Attempt to convert input to a Pandas Series if it's not already one (e.g., a list)
        data = pd.Series(data, dtype=float) 

    if window <= 0:
        raise ValueError("Window period must be a positive integer.")
    
    if data.empty:
        return pd.Series([], dtype=float, index=data.index) # Return empty series if input is empty

    if window > len(data):
        # If window is larger than data length, all SMA values will be NaN
        return pd.Series([np.nan] * len(data), index=data.index)
        
    # Calculate SMA using pandas rolling mean
    # min_periods=window ensures that the SMA is calculated only when there's enough data for the full window
    sma = data.rolling(window=window, min_periods=window).mean()
    return sma

def calculate_ema(data: pd.Series, window: int) -> pd.Series:
    """
    Calculates the Exponential Moving Average (EMA).

    The EMA gives more weight to recent prices, making it more responsive to
    new information compared to an SMA.

    Args:
        data (pd.Series): A Pandas Series of prices (e.g., close prices).
        window (int): The lookback period (span) for the EMA calculation.

    Returns:
        pd.Series: A Pandas Series containing the EMA values. Early values
                   (before enough data is available for the window) will be NaN.
    """
    if not isinstance(data, pd.Series):
        data = pd.Series(data, dtype=float)

    if window <= 0:
        raise ValueError("Window period must be a positive integer.")

    if data.empty:
        return pd.Series([], dtype=float, index=data.index)

    if window > len(data):
        # If window is larger than data length, all EMA values will be NaN
        return pd.Series([np.nan] * len(data), index=data.index)

    # Calculate EMA using pandas ewm (exponentially weighted moving)
    # span=window: Corresponds to the 'N-day EMA' commonly referred to in trading. Alpha = 2 / (span + 1).
    # adjust=False: Uses the recursive formula without adjustment for early data points,
    #               which is common in many technical analysis libraries.
    # min_periods=window: Ensures that EMA is calculated only when there's enough data for the window.
    ema = data.ewm(span=window, adjust=False, min_periods=window).mean()
    return ema

if __name__ == '__main__':
    # This block provides example usage when the script is run directly.
    prices_list = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] # Sample price data
    prices_series = pd.Series(prices_list)

    print("Prices:", prices_series.tolist())

    # SMA Example
    sma_window = 5
    sma_values = calculate_sma(prices_series, sma_window)
    print(f"\nSMA with window {sma_window}:")
    print(sma_values.tolist()) # Expected: [NaN, NaN, NaN, NaN, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]

    sma_values_list_input = calculate_sma(prices_list, sma_window)
    print(f"\nSMA with window {sma_window} (list input):")
    print(sma_values_list_input.tolist())

    # EMA Example
    ema_window = 5
    ema_values = calculate_ema(prices_series, ema_window)
    print(f"\nEMA with window {ema_window}:")
    # Pandas EMA calculation with adjust=False and min_periods matches common TA library behavior
    # The first EMA is the SMA of the first 'window' periods, then recursive.
    # For adjust=False, ewm() uses N for span, so alpha = 2 / (N + 1)
    # If min_periods is set, the first few values will be NaN until enough data points are available.
    print(ema_values.tolist())

    ema_values_list_input = calculate_ema(prices_list, ema_window)
    print(f"\nEMA with window {ema_window} (list input):")
    print(ema_values_list_input.tolist())
    
    # Test with window larger than data
    print(f"\nSMA with window 20 on {len(prices_list)} items:")
    print(calculate_sma(prices_series, 20).tolist())
    print(f"\nEMA with window 20 on {len(prices_list)} items:")
    print(calculate_ema(prices_series, 20).tolist())

    # Test with empty data
    empty_series = pd.Series([], dtype=float)
    print("\nSMA with empty data:")
    print(calculate_sma(empty_series, 5).tolist())
    print("\nEMA with empty data:")
    print(calculate_ema(empty_series, 5).tolist())

    # Test with data shorter than window
    short_series = pd.Series([1, 2, 3])
    print("\nSMA with data shorter than window:")
    print(calculate_sma(short_series, 5).tolist()) # Expected: [NaN, NaN, NaN]
    print("\nEMA with data shorter than window:")
    print(calculate_ema(short_series, 5).tolist()) # Expected: [NaN, NaN, NaN]
