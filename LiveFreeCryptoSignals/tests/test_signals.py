import pytest
import pandas as pd
import numpy as np

from LiveFreeCryptoSignals.signals import (
    generate_rsi_signal,
    generate_macd_signal,
    generate_ma_crossover_signal,
    generate_bb_signal
)

# --- RSI Signal Tests ---
def test_rsi_signals():
    # BUY signal
    rsi_buy = pd.Series([50, 40, 30, 25]) 
    assert generate_rsi_signal(rsi_buy, buy_threshold=30, sell_threshold=70) == 'BUY'

    # SELL signal
    rsi_sell = pd.Series([50, 60, 70, 75]) 
    assert generate_rsi_signal(rsi_sell, buy_threshold=30, sell_threshold=70) == 'SELL'

    # HOLD signal (between thresholds)
    rsi_hold_middle = pd.Series([40, 50, 60]) 
    assert generate_rsi_signal(rsi_hold_middle, buy_threshold=30, sell_threshold=70) == 'HOLD'
    
    rsi_hold_at_buy = pd.Series([40, 30]) 
    assert generate_rsi_signal(rsi_hold_at_buy, buy_threshold=30, sell_threshold=70) == 'HOLD'
    rsi_hold_at_sell = pd.Series([60, 70]) 
    assert generate_rsi_signal(rsi_hold_at_sell, buy_threshold=30, sell_threshold=70) == 'HOLD'

    # Insufficient data
    rsi_empty = pd.Series([], dtype=float)
    assert generate_rsi_signal(rsi_empty) == 'HOLD'

    rsi_all_nan = pd.Series([np.nan, np.nan, np.nan])
    assert generate_rsi_signal(rsi_all_nan) == 'HOLD'
    
    rsi_nan_at_end = pd.Series([20, 30, np.nan]) 
    assert generate_rsi_signal(rsi_nan_at_end) == 'HOLD' # Last valid is 30 -> HOLD

    rsi_single_val_buy = pd.Series([25])
    assert generate_rsi_signal(rsi_single_val_buy) == 'BUY'

    rsi_single_val_sell = pd.Series([75])
    assert generate_rsi_signal(rsi_single_val_sell) == 'SELL'

    rsi_single_val_hold = pd.Series([50])
    assert generate_rsi_signal(rsi_single_val_hold) == 'HOLD'


# --- MACD Crossover Signal Tests ---
def test_macd_crossover_signals():
    macd_line_buy = pd.Series([0.1, 0.05, 0.15]) 
    signal_line_buy = pd.Series([0.12, 0.1, 0.12]) 
    assert generate_macd_signal(macd_line_buy, signal_line_buy) == 'BUY'

    macd_line_sell = pd.Series([0.1, 0.15, 0.05]) 
    signal_line_sell = pd.Series([0.08, 0.12, 0.1]) 
    assert generate_macd_signal(macd_line_sell, signal_line_sell) == 'SELL'

    macd_line_hold_above = pd.Series([0.1, 0.15, 0.20])
    signal_line_hold_above = pd.Series([0.05, 0.1, 0.15])
    assert generate_macd_signal(macd_line_hold_above, signal_line_hold_above) == 'HOLD'

    macd_line_hold_below = pd.Series([0.2, 0.15, 0.1])
    signal_line_hold_below = pd.Series([0.25, 0.2, 0.15])
    assert generate_macd_signal(macd_line_hold_below, signal_line_hold_below) == 'HOLD'
    
    macd_line_diverge = pd.Series([0.1, 0.05, 0.02]) 
    signal_line_diverge = pd.Series([0.08, 0.1, 0.12])
    assert generate_macd_signal(macd_line_diverge, signal_line_diverge) == 'HOLD'

    assert generate_macd_signal(pd.Series([0.1], dtype=float), pd.Series([0.2], dtype=float)) == 'HOLD'
    assert generate_macd_signal(pd.Series([0.1, 0.2], dtype=float), pd.Series([np.nan, np.nan], dtype=float)) == 'HOLD'
    assert generate_macd_signal(pd.Series([np.nan, np.nan], dtype=float), pd.Series([0.1, 0.2], dtype=float)) == 'HOLD'
    assert generate_macd_signal(pd.Series([], dtype=float), pd.Series([], dtype=float)) == 'HOLD'
    
    macd_with_nan_buy = pd.Series([0.1, np.nan, 0.05, 0.15])
    signal_with_nan_buy = pd.Series([0.12, np.nan, 0.1, 0.12])
    assert generate_macd_signal(macd_with_nan_buy, signal_with_nan_buy) == 'BUY'


# --- MA Crossover Signal Tests ---
def test_ma_crossover_signals():
    fast_ma_buy = pd.Series([10, 10.5, 11.5]) 
    slow_ma_buy = pd.Series([10.8, 11, 11.2]) 
    assert generate_ma_crossover_signal(fast_ma_buy, slow_ma_buy) == 'BUY'

    fast_ma_sell = pd.Series([10.8, 11, 10.5]) 
    slow_ma_sell = pd.Series([10, 10.5, 10.8]) 
    assert generate_ma_crossover_signal(fast_ma_sell, slow_ma_sell) == 'SELL'

    fast_ma_hold = pd.Series([10, 11, 12])
    slow_ma_hold = pd.Series([9, 10, 11]) 
    assert generate_ma_crossover_signal(fast_ma_hold, slow_ma_hold) == 'HOLD'
    
    assert generate_ma_crossover_signal(pd.Series([10], dtype=float), pd.Series([11], dtype=float)) == 'HOLD'
    assert generate_ma_crossover_signal(pd.Series([], dtype=float), pd.Series([], dtype=float)) == 'HOLD'


# --- Bollinger Bands Signal Tests ---
def test_bb_signals():
    # Verified example from generator.py's __main__ block
    prices_buy_ex = pd.Series([25, 22, 21], dtype=float)
    lower_band_buy_ex = pd.Series([24, 21.5, 21.5], dtype=float)
    upper_band_buy_ex = pd.Series([26, 25.5, 25.5], dtype=float)
    assert generate_bb_signal(prices_buy_ex, lower_band_buy_ex, upper_band_buy_ex) == 'BUY'

    prices_sell_ex = pd.Series([25, 28, 29], dtype=float)
    lower_band_sell_ex = pd.Series([24, 24.5, 24.5], dtype=float)
    upper_band_sell_ex = pd.Series([26, 28.5, 28.5], dtype=float)
    assert generate_bb_signal(prices_sell_ex, lower_band_sell_ex, upper_band_sell_ex) == 'SELL'

    prices_hold = pd.Series([25, 24, 23], dtype=float)
    lower_band_hold = pd.Series([22, 22, 22], dtype=float)
    upper_band_hold = pd.Series([26, 26, 26], dtype=float)
    assert generate_bb_signal(prices_hold, lower_band_hold, upper_band_hold) == 'HOLD'

    prices_touch_lower = pd.Series([25, 22, 22], dtype=float)
    lower_touch_lower = pd.Series([24, 22, 22], dtype=float)
    upper_touch_lower = pd.Series([26, 26, 26], dtype=float)
    assert generate_bb_signal(prices_touch_lower, lower_touch_lower, upper_touch_lower) == 'HOLD'
    
    assert generate_bb_signal(pd.Series([25], dtype=float), pd.Series([24], dtype=float), pd.Series([26], dtype=float)) == 'HOLD'
    assert generate_bb_signal(pd.Series([], dtype=float), pd.Series([], dtype=float), pd.Series([], dtype=float)) == 'HOLD'
    
    prices_nan = pd.Series([25, 22, np.nan, 21], dtype=float) # Effective P: [25,22,21]
    lower_nan = pd.Series([24, 21.5, np.nan, 21.5], dtype=float) # Effective L: [24,21.5,21.5]
    upper_nan = pd.Series([26, 25.5, np.nan, 25.5], dtype=float) # Effective U: [26,25.5,25.5]
    # This case should result in a BUY signal based on the non-NaN values
    assert generate_bb_signal(prices_nan, lower_nan, upper_nan) == 'BUY'
