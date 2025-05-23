import pytest
import pandas as pd
import numpy as np

# Assuming the LiveFreeCryptoSignals directory is in PYTHONPATH or tests are run in a way
# that makes 'LiveFreeCryptoSignals' package findable.
from LiveFreeCryptoSignals.indicators import (
    calculate_sma,
    calculate_ema, 
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands
)

# Sample data for testing
SAMPLE_PRICES_LIST = [
    10, 11, 12, 13, 14, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6,
    7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
]
SAMPLE_PRICES = pd.Series(SAMPLE_PRICES_LIST, dtype=float)


def test_sma_calculation():
    prices = pd.Series([10, 11, 12, 13, 14], dtype=float)
    window = 3
    expected_sma = pd.Series([np.nan, np.nan, 11.0, 12.0, 13.0], dtype=float, index=prices.index) 
    
    result_sma = calculate_sma(prices, window)
    pd.testing.assert_series_equal(result_sma, expected_sma, check_dtype=False, atol=0.01)

    prices_with_nan = pd.Series([10, np.nan, 12, 13, 14], dtype=float)
    # SMA with min_periods=window (3):
    # Index 2: (10, NaN, 12) -> NaN
    # Index 3: (NaN, 12, 13) -> NaN
    # Index 4: (12, 13, 14) -> 13.0
    expected_sma_with_nan = pd.Series([np.nan, np.nan, np.nan, np.nan, 13.0], dtype=float, index=prices_with_nan.index)
    result_sma_with_nan = calculate_sma(prices_with_nan, window)
    pd.testing.assert_series_equal(result_sma_with_nan, expected_sma_with_nan, check_dtype=False, atol=0.01)

    short_series = pd.Series([1, 2, 3], dtype=float)
    expected_short_sma = pd.Series([np.nan, np.nan, np.nan], dtype=float, index=short_series.index)
    result_short_sma = calculate_sma(short_series, 5)
    pd.testing.assert_series_equal(result_short_sma, expected_short_sma, check_dtype=False, atol=0.01)


def test_rsi_calculation():
    prices = pd.Series([10, 11, 10, 12, 13, 11, 12], dtype=float)
    window = 3
    # Expected values based on manual trace of rsi.py logic (SMA for initial avg gain/loss, then Wilder's)
    # RSI[3]=75.0, RSI[4]=81.8181, RSI[5]=45.0, RSI[6]=58.8788
    expected_rsi_values = [np.nan, np.nan, np.nan, 75.0, 81.818181, 45.0, 58.878811]
    expected_rsi = pd.Series(expected_rsi_values, dtype=float, index=prices.index)
    
    result_rsi = calculate_rsi(prices, window)
    pd.testing.assert_series_equal(result_rsi, expected_rsi, check_dtype=False, atol=0.01)

    short_prices = pd.Series([10, 11, 12], dtype=float) # Window 3, needs 3+1 data points for first RSI
    expected_short_rsi = pd.Series([np.nan, np.nan, np.nan], dtype=float, index=short_prices.index) 
    result_short_rsi = calculate_rsi(short_prices, window=3)
    pd.testing.assert_series_equal(result_short_rsi, expected_short_rsi, check_dtype=False, atol=0.01)
    
    result_large_window_rsi = calculate_rsi(short_prices, window=5) # Window > data length
    pd.testing.assert_series_equal(result_large_window_rsi, expected_short_rsi, check_dtype=False, atol=0.01)


def test_macd_calculation():
    # Data needs to be long enough for slow_period (26) + signal_period (9) for full MACD, signal, hist
    prices_list_long = [
        150, 152, 151, 153, 155, 154, 156, 157, 158, 156, 155, 157, 159, 160, 158, 
        159, 161, 163, 162, 160, 161, 163, 165, 166, 164, 165, 166, 167, 168, 169,
        170, 168, 167, 169, 171, 172, 170, 171, 173, 175, 174, 172, 173, 175, 177
    ]
    prices = pd.Series(prices_list_long, dtype=float)

    fast_period, slow_period, signal_period = 12, 26, 9
    macd_line, signal_line, histogram = calculate_macd(prices, fast_period, slow_period, signal_period)

    # First MACD value is at index slow_period - 1
    # First Signal value is at index (slow_period - 1) + (signal_period - 1)
    assert pd.isna(macd_line.iloc[0:(slow_period - 1)]).all() 
    assert pd.isna(signal_line.iloc[0:(slow_period + signal_period - 2)]).all()
    assert pd.isna(histogram.iloc[0:(slow_period + signal_period - 2)]).all()

    last_valid_idx = histogram.last_valid_index()
    if last_valid_idx is not None:
        assert isinstance(macd_line[last_valid_idx], float)
        assert isinstance(signal_line[last_valid_idx], float)
        assert isinstance(histogram[last_valid_idx], float)
        assert np.isclose(histogram[last_valid_idx], macd_line[last_valid_idx] - signal_line[last_valid_idx])
    else:
        # This case should not happen with the long prices_list_long
        raise AssertionError("Expected valid MACD values with the provided long price series.")
        
    short_prices = pd.Series([10,11,12,13,14,15,16,17,18,19,20], dtype=float) # len 11
    macd_s, signal_s, hist_s = calculate_macd(short_prices, fast_period, slow_period, signal_period)
    # slow_period is 26, so all will be NaN
    assert macd_s.isnull().all() 
    assert signal_s.isnull().all()
    assert hist_s.isnull().all()


def test_bollinger_bands_calculation():
    prices = pd.Series([10, 11, 12, 13, 14, 13, 12, 11, 10, 11, 12], dtype=float)
    window = 5
    num_std_dev = 2

    upper, middle, lower = calculate_bollinger_bands(prices, window, num_std_dev)

    expected_middle_raw = [
        np.nan, np.nan, np.nan, np.nan, 
        np.mean([10,11,12,13,14]), # 12.0
        np.mean([11,12,13,14,13]), # 12.6
        np.mean([12,13,14,13,12]), # 12.8
        np.mean([13,14,13,12,11]), # 12.6
        np.mean([14,13,12,11,10]), # 12.0
        np.mean([13,12,11,10,11]), # 11.4
        np.mean([12,11,10,11,12]), # 11.2
    ]
    expected_middle = pd.Series(expected_middle_raw, dtype=float, index=prices.index)
    pd.testing.assert_series_equal(middle, expected_middle, check_dtype=False, atol=0.01)

    std_at_idx_4 = np.std([10,11,12,13,14], ddof=1) # Pandas default for .std() is sample std (ddof=1)
    
    expected_upper_at_idx_4 = 12.0 + num_std_dev * std_at_idx_4
    expected_lower_at_idx_4 = 12.0 - num_std_dev * std_at_idx_4
    
    assert np.isclose(upper.iloc[4], expected_upper_at_idx_4)
    assert np.isclose(lower.iloc[4], expected_lower_at_idx_4)
    
    assert upper.iloc[0:(window-1)].isnull().all()
    assert middle.iloc[0:(window-1)].isnull().all()
    assert lower.iloc[0:(window-1)].isnull().all()

    short_prices = pd.Series([1,2,3], dtype=float)
    su, sm, sl = calculate_bollinger_bands(short_prices, window=5)
    assert su.isnull().all()
    assert sm.isnull().all()
    assert sl.isnull().all()

# Note: EMA tests are not included here to keep focus on the main ones as per instructions.
# If test_ema_calculation is needed, it can be added similarly.
# def test_ema_calculation():
#     prices = pd.Series([10, 11, 12, 13, 14], dtype=float)
#     window = 3
#     # Pandas ewm with span=window, adjust=False, min_periods=window:
#     # EMA for 10: Nan (min_periods)
#     # EMA for 11: Nan (min_periods)
#     # EMA for 12: 11.25 ( (10 * (1-alpha)^2 + 11 * (1-alpha)^1 + 12 * (1-alpha)^0 ) / ((1-alpha)^2 + (1-alpha)^1 + (1-alpha)^0) is for adjust=True)
#     # For adjust=False: EMA_prev = val_prev if first, else standard formula
#     # EMA_0 = 10
#     # EMA_1 = 11*alpha + EMA_0*(1-alpha) = 11*0.5 + 10*0.5 = 10.5
#     # EMA_2 = 12*alpha + EMA_1*(1-alpha) = 12*0.5 + 10.5*0.5 = 11.25
#     # EMA_3 = 13*alpha + EMA_2*(1-alpha) = 13*0.5 + 11.25*0.5 = 12.125
#     # EMA_4 = 14*alpha + EMA_3*(1-alpha) = 14*0.5 + 12.125*0.5 = 13.0625
#     # With min_periods=3, the first values are NaN until 3 data points are available for the EWM to start outputting.
#     # The value at index `window-1` (i.e., index 2) is the first non-NaN.
#     expected_ema_series = pd.Series([np.nan, np.nan, 11.25, 12.125, 13.0625], dtype=float, index=prices.index)
#     result_ema = calculate_ema(prices, window)
#     pd.testing.assert_series_equal(result_ema, expected_ema_series, check_dtype=False, atol=0.01)
#
#     short_prices = pd.Series([1,2], dtype=float)
#     expected_short_ema = pd.Series([np.nan, np.nan], dtype=float, index=short_prices.index)
#     result_short_ema = calculate_ema(short_prices, window=3)
#     pd.testing.assert_series_equal(result_short_ema, expected_short_ema, check_dtype=False, atol=0.01)

# To run these tests, navigate to LiveFreeCryptoSignals root and run `pytest`
# Ensure __init__.py is in the tests folder (it is).
# Ensure LiveFreeCryptoSignals is in PYTHONPATH or use `python -m pytest tests`
# The import `from LiveFreeCryptoSignals.indicators ...` assumes that the directory *containing*
# `LiveFreeCryptoSignals` (the project root) is what's on PYTHONPATH, making `LiveFreeCryptoSignals`
# a top-level package. Or, that `pytest` is run from the project root and `LiveFreeCryptoSignals` is
# discovered as a package.
# The structure `LiveFreeCryptoSignals/LiveFreeCryptoSignals/indicators` (src layout) would also work.
# Given the current flat structure `LiveFreeCryptoSignals/indicators`, `LiveFreeCryptoSignals/main.py`, etc.,
# and if `pytest` is run from the root `LiveFreeCryptoSignals/`, then `sys.path` typically includes this root.
# If so, `from indicators import ...` would be the direct way.
# However, the prompt specified `from LiveFreeCryptoSignals.indicators ...`. This implies the intended
# package name for imports is `LiveFreeCryptoSignals`. This is fine if the environment (sandbox)
# ensures this path setup.

# Example: if `PYTHONPATH=/path/to/workspace/` and project is `/path/to/workspace/LiveFreeCryptoSignals`,
# then `import LiveFreeCryptoSignals.indicators` works.
# If `pytest` is run from `/path/to/workspace/LiveFreeCryptoSignals/`, it usually adds this dir to sys.path.
# So `import indicators` would work.
# The prompt's import style is being followed.
# Added index=prices.index to expected series for proper comparison.
# Corrected MACD first non-NaN index. For EMA, first non-NaN is at index window-1.
# MACD line = EMA_fast - EMA_slow. EMA_slow is non-NaN from index slow_period-1. So MACD line is non-NaN from slow_period-1.
# Signal line is EMA of MACD line. So Signal line is non-NaN from (slow_period-1) + (signal_period-1).
# This was reflected in the assertions.
# Corrected RSI expected values to match my manual trace from previous turn.
# Final check on RSI expected: [NaN, NaN, NaN, 75.0, 81.818181, 45.0, 58.878811] (7 elements) for prices (7 elements).
# My rsi.py code has: rsi.iloc[:window] = np.nan. So for window=3, indices 0,1,2 are NaN. First RSI is at index 3. Correct.
# My rsi.py code has: `if len(delta) < window +1 : return pd.Series([np.nan] * len(data), index=data.index)`
# `delta` has same length as `data`. So if `len(data) < window + 1`, all NaN.
# For `short_prices` (len 3), window=3. `len(data)` is 3. `window+1` is 4. `3 < 4` is true. So all NaN. Correct.
# For `result_large_window_rsi` (data len 3, window 5). `3 < 5+1` is true. So all NaN. Correct.I have completed steps 1, 2, and 3 of the current subtask.
1.  Added `pytest` to `LiveFreeCryptoSignals/requirements.txt`. (Turns 33 & 34)
2.  Created the empty test files `LiveFreeCryptoSignals/tests/test_indicators.py` and `LiveFreeCryptoSignals/tests/test_signals.py`. (Turn 35)
3.  Implemented tests for SMA, RSI, MACD, and Bollinger Bands in `LiveFreeCryptoSignals/tests/test_indicators.py`. (Turn 37, based on thoughts from Turn 36). This includes checks for basic calculations, NaN handling, and edge cases like insufficient data.

Now I will proceed to step 4: **Implement tests in `LiveFreeCryptoSignals/tests/test_signals.py`**.

This will involve:
*   Importing necessary functions from `LiveFreeCryptoSignals.signals` and `pandas`.
*   Creating test functions for `generate_rsi_signal` and `generate_macd_signal` (and potentially others if time allows, but these are prioritized by the prompt).
*   For RSI signals: test 'BUY' (RSI < 30), 'SELL' (RSI > 70), 'HOLD' (30 <= RSI <= 70), and insufficient data cases.
*   For MACD signals: test 'BUY' (bullish crossover), 'SELL' (bearish crossover), 'HOLD' (no crossover), and insufficient data cases.
