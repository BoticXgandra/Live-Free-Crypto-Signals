"""
Calculates the Relative Strength Index (RSI).

RSI is a momentum oscillator that measures the speed and change of price movements.
It oscillates between zero and 100. Traditionally, RSI is considered overbought
when above 70 and oversold when below 30.
"""
import pandas as pd
import numpy as np

def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculates the Relative Strength Index (RSI).

    This implementation uses a simple moving average (SMA) for the initial
    calculation of average gains and losses, followed by Wilder's smoothing method
    for subsequent values.

    Args:
        data (pd.Series): A Pandas Series of prices (e.g., close prices).
        window (int): The lookback period for RSI calculation (default is 14).

    Returns:
        pd.Series: A Pandas Series containing the RSI values (0-100).
                   Early values (before enough data is available) will be NaN.
                   The first valid RSI value is at index `window`.
    """
    if not isinstance(data, pd.Series):
        data = pd.Series(data, dtype=float)

    if window <= 0:
        raise ValueError("Window period must be a positive integer.")

    if data.empty:
        return pd.Series([], dtype=float, index=data.index)

    # RSI requires at least 'window' + 1 data points to calculate the first RSI value,
    # because it needs 'window' periods of price changes.
    if len(data) < window + 1: 
        return pd.Series([np.nan] * len(data), index=data.index)

    # Calculate price changes
    delta = data.diff()

    # Separate gains and losses
    gain = delta.copy()
    loss = delta.copy()

    gain[gain < 0] = 0  # Zero out losses for gains
    loss[loss > 0] = 0  # Zero out gains for losses
    loss = abs(loss)    # Make losses positive

    # Calculate initial average gain and loss using SMA for the first 'window' periods of change.
    # This means we look at delta from index 1 to 'window' (inclusive).
    # The first RSI value will correspond to the price at index 'window'.
    initial_avg_gain = gain.iloc[1:window+1].mean() 
    initial_avg_loss = loss.iloc[1:window+1].mean()
    
    # Initialize series for average gains and losses
    avg_gain = pd.Series([np.nan] * len(data), index=data.index)
    avg_loss = pd.Series([np.nan] * len(data), index=data.index)

    # Set the first calculated average gain/loss at index 'window'.
    # This corresponds to the end of the first calculation period.
    if len(avg_gain) > window : # Check to prevent index out of bounds if data is exactly window+1 long
        avg_gain.iloc[window] = initial_avg_gain
        avg_loss.iloc[window] = initial_avg_loss
    else: # Should not happen due to len(data) < window + 1 check, but as a safeguard
        return pd.Series([np.nan] * len(data), index=data.index)


    # Apply Wilder's Smoothing for subsequent average gains and losses
    # Formula: subsequent_avg = (previous_avg * (window - 1) + current_value) / window
    for i in range(window + 1, len(data)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (window - 1) + gain.iloc[i]) / window
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (window - 1) + loss.iloc[i]) / window

    # Calculate Relative Strength (RS)
    # RS = Average Gain / Average Loss
    # Handle division by zero by replacing resulting infinities with NaN if avg_loss is 0.
    rs = avg_gain / avg_loss
    rs.replace([np.inf, -np.inf], np.nan, inplace=True) # if avg_loss is 0, rs is inf. RSI should be 100.

    # Calculate RSI
    # RSI = 100 - (100 / (1 + RS))
    rsi = 100 - (100 / (1 + rs))
    
    # If avg_loss was zero, RS is NaN (after replacement of inf) or was inf.
    # If avg_loss is 0, then RS is effectively infinite, meaning RSI should be 100.
    # This is correctly handled by the formula if RS becomes very large (e.g. avg_loss is tiny but not zero).
    # If avg_loss is strictly zero and avg_gain is positive, RS was inf, now NaN.
    # We need to ensure RSI is 100 in such cases.
    # Where avg_loss is 0 and avg_gain > 0, RSI should be 100.
    # avg_loss.iloc[i] == 0 and avg_gain.iloc[i] > 0 implies rsi.iloc[i] should be 100.
    # The formula 100 - (100 / (1 + large_rs)) approaches 100.
    # If rs is NaN due to avg_loss being 0, then rsi will be NaN.
    # Correcting positions where avg_loss is 0 and avg_gain > 0 to RSI = 100.
    rsi.loc[(avg_loss == 0) & (avg_gain > 0) & (avg_gain.notna())] = 100
    # If both avg_gain and avg_loss are 0, RS is NaN, RSI is NaN. This is usually treated as RSI 50 by some,
    # but NaN is also acceptable indicating no trend. Current formula results in NaN.

    # Ensure the first 'window' RSI values are NaN, as RSI calculation starts from index 'window'.
    rsi.iloc[:window] = np.nan 
    
    return rsi

if __name__ == '__main__':
    # This block provides example usage when the script is run directly.
    prices_list = [
        44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08,
        45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64,
        46.21, 46.25, 45.71, 46.39, 45.78, 45.35, 44.03, 44.18
    ]
    prices_series = pd.Series(prices_list)

    print("Prices:", prices_series.tolist())

    # RSI Example
    rsi_window = 14
    rsi_values = calculate_rsi(prices_series, rsi_window)
    print(f"\nRSI with window {rsi_window}:")
    # Expected: First 14 values are NaN
    print(rsi_values.tolist())
    
    rsi_values_list_input = calculate_rsi(prices_list, rsi_window)
    print(f"\nRSI with window {rsi_window} (list input):")
    print(rsi_values_list_input.tolist())

    # Test with a common example from online sources
    # For data [10,11,12,13,14,15,14,13,12,11,10,9], window 5
    # Expected RSI for the 6th value (index 5) should be around 83.33 if using simple MA for first avg gain/loss
    # Wilder's RSI is more complex.
    test_prices = pd.Series([10,11,12,13,14,15,14,13,12,11,10,9,8,7,6,7,8,9,10,11,12])
    rsi_5 = calculate_rsi(test_prices, window=5)
    print("\nTest RSI (window 5):")
    print(test_prices.tolist())
    print(rsi_5.tolist())

    # Test with window larger than data
    print(f"\nRSI with window 30 on {len(prices_list)} items:")
    print(calculate_rsi(prices_series, 30).tolist())
    
    # Test with empty data
    empty_series = pd.Series([], dtype=float)
    print("\nRSI with empty data:")
    print(calculate_rsi(empty_series, 14).tolist())

    # Test with data shorter than window + 1
    short_series = pd.Series([1,2,3,4,5])
    print("\nRSI with data shorter than window+1 (window=5):")
    print(calculate_rsi(short_series, 5).tolist()) # Expect all NaNs
    
    short_series_14 = pd.Series(prices_list[:10])
    print("\nRSI with data shorter than window+1 (window=14, 10 items):")
    print(calculate_rsi(short_series_14, 14).tolist()) # Expect all NaNs
