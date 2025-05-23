"""
Calculates Bollinger Bands.

Bollinger Bands are a type of statistical chart characterizing the prices and
volatility over time of a financial instrument or commodity. They consist of:
- A middle band being an N-period simple moving average (SMA).
- An upper band at K standard deviations above the middle band.
- A lower band at K standard deviations below the middle band.
"""
import pandas as pd
import numpy as np
from .moving_average import calculate_sma # Import SMA for the middle band

def calculate_bollinger_bands(data: pd.Series, window: int = 20, num_std_dev: float = 2):
    """
    Calculates the Upper Band, Middle Band (SMA), and Lower Band for Bollinger Bands.

    Args:
        data (pd.Series): A Pandas Series of prices (e.g., close prices).
        window (int): The window period for the Middle Band (SMA) and for calculating
                      the rolling standard deviation (default is 20).
        num_std_dev (float): The number of standard deviations to use for the
                             Upper and Lower Bands (default is 2).

    Returns:
        tuple[pd.Series, pd.Series, pd.Series]: 
            A tuple containing three Pandas Series:
            1. Upper Band
            2. Middle Band (SMA of the data)
            3. Lower Band
            All series are padded with NaN where values are not yet defined due to
            the SMA and rolling standard deviation warm-up periods.
    """
    if not isinstance(data, pd.Series):
        data = pd.Series(data, dtype=float)

    if window <= 0:
        raise ValueError("Window period must be a positive integer.")
    if num_std_dev <= 0:
        raise ValueError("Number of standard deviations must be positive.")
    
    if data.empty:
        empty_series = pd.Series([], dtype=float, index=data.index)
        return empty_series, empty_series, empty_series

    if window > len(data):
        # If window is larger than data length, all bands will be NaN
        nan_series = pd.Series([np.nan] * len(data), index=data.index)
        return nan_series, nan_series, nan_series

    # Calculate the Middle Band (Simple Moving Average)
    middle_band = calculate_sma(data, window)
    
    # Calculate the rolling standard deviation over the same window
    # `min_periods=window` ensures alignment with SMA's NaN values at the beginning.
    # Pandas' .std() calculates the sample standard deviation (ddof=1) by default.
    rolling_std = data.rolling(window=window, min_periods=window).std() 
    
    # Calculate Upper and Lower Bands
    upper_band = middle_band + (rolling_std * num_std_dev)
    lower_band = middle_band - (rolling_std * num_std_dev)
    
    return upper_band, middle_band, lower_band

if __name__ == '__main__':
    # This block provides example usage when the script is run directly.
    prices_list = [
        22.27, 22.19, 22.08, 22.17, 22.18, 22.13, 22.23, 22.43, 22.24, 22.29,
        22.15, 22.39, 22.38, 22.61, 23.36, 24.05, 23.75, 23.83, 23.95, 23.63, # Index 19 (20th item)
        23.82, 23.87, 23.65, 23.19, 23.10, 23.33, 22.68, 23.10, 22.40, 22.17
    ]
    prices_series = pd.Series(prices_list)

    print("Prices:", prices_series.tolist())

    # Bollinger Bands Example
    bb_window, bb_std_dev = 20, 2
    upper, middle, lower = calculate_bollinger_bands(prices_series, bb_window, bb_std_dev)

    print(f"\nBollinger Bands (window={bb_window}, std_dev={bb_std_dev}):")
    print("Upper Band:")
    print(upper.tolist())
    print("\nMiddle Band (SMA):")
    print(middle.tolist())
    print("\nLower Band:")
    print(lower.tolist())

    # Verify that the first (window-1) values are NaN
    print(f"\nFirst {bb_window-1} values of Middle Band should be NaN: {middle.iloc[:bb_window-1].isnull().all()}")
    
    upper_list, middle_list, lower_list = calculate_bollinger_bands(
        prices_list, bb_window, bb_std_dev
    )
    print(f"\nBollinger Bands (window={bb_window}, std_dev={bb_std_dev}) (list input):")
    print("Upper Band:")
    print(upper_list.tolist())
    print("\nMiddle Band (SMA):")
    print(middle_list.tolist())
    print("\nLower Band:")
    print(lower_list.tolist())

    # Test with data shorter than window
    short_series = prices_series.iloc[:15] # window is 20
    print(f"\nBollinger Bands with data shorter than window ({len(short_series)} items):")
    upper_short, middle_short, lower_short = calculate_bollinger_bands(short_series, bb_window, bb_std_dev)
    print("Upper Band (short):")
    print(upper_short.tolist()) # Expect all NaNs
    print("Middle Band (short):")
    print(middle_short.tolist()) # Expect all NaNs
    print("Lower Band (short):")
    print(lower_short.tolist()) # Expect all NaNs

    # Test with empty data
    empty_series = pd.Series([], dtype=float)
    print("\nBollinger Bands with empty data:")
    upper_empty, middle_empty, lower_empty = calculate_bollinger_bands(empty_series)
    print("Upper Band (empty):", upper_empty.tolist())
    print("Middle Band (empty):", middle_empty.tolist())
    print("Lower Band (empty):", lower_empty.tolist())
