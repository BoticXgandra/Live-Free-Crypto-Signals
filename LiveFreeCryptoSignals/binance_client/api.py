"""
Handles interactions with the Binance API.

This module provides functions to fetch cryptocurrency data, such as kline
(candlestick) data, from the Binance exchange. It uses the python-binance library.
"""
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException

def get_klines(symbol: str, interval: str, limit: int = 500):
    """
    Fetches kline/candlestick data for a specific symbol and interval from Binance.

    Args:
        symbol (str): The trading symbol (e.g., 'BTCUSDT').
        interval (str): The interval for klines (e.g., Client.KLINE_INTERVAL_1MINUTE, 
                        Client.KLINE_INTERVAL_1HOUR, Client.KLINE_INTERVAL_1DAY).
        limit (int): The number of klines to retrieve (default is 500, max is 1000
                     as per Binance API documentation for this specific endpoint).

    Returns:
        list: A list of kline data points (list of lists), or None if an error occurs.
              Each kline is a list representing:
              [
                  Kline open time (ms),
                  Open price (str),
                  High price (str),
                  Low price (str),
                  Close price (str),
                  Volume (str),
                  Kline close time (ms),
                  Quote asset volume (str),
                  Number of trades (int),
                  Taker buy base asset volume (str),
                  Taker buy quote asset volume (str),
                  Ignore (str)
              ]
              Note: Prices and volumes are returned as strings by the API.
    """
    # Initialize an unauthenticated client.
    # For authenticated endpoints, you would pass api_key and api_secret.
    client = Client()

    try:
        # Fetch klines using the python-binance client
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        return klines
    except BinanceAPIException as e:
        # Handle API errors (e.g., invalid symbol, rate limits)
        print(f"Binance API Exception while fetching klines for {symbol} ({interval}): {e}")
        return None
    except BinanceRequestException as e:
        # Handle request errors (e.g., network issues before API response)
        print(f"Binance Request Exception while fetching klines for {symbol} ({interval}): {e}")
        return None
    except Exception as e:
        # Handle any other unexpected errors
        print(f"An unexpected error occurred while fetching klines for {symbol} ({interval}): {e}")
        return None

if __name__ == '__main__':
    # This block provides example usage when the script is run directly.
    # It's useful for testing the get_klines function independently.
    
    print("--- Example: Fetching 1-minute klines for BTCUSDT ---")
    # Using Client constants for intervals is recommended for clarity and to avoid typos.
    btc_klines_1m = get_klines('BTCUSDT', Client.KLINE_INTERVAL_1MINUTE, limit=10)
    if btc_klines_1m:
        print(f"Successfully fetched {len(btc_klines_1m)} BTCUSDT 1-minute klines.")
        print("First kline data point:")
        print(btc_klines_1m[0]) # Print the first kline
    else:
        print("Failed to retrieve BTCUSDT 1-minute klines.")

    print("\n" + "-" * 50 + "\n")

    print("--- Example: Fetching 1-day klines for ETHUSDT ---")
    eth_klines_1d = get_klines('ETHUSDT', Client.KLINE_INTERVAL_1DAY, limit=5)
    if eth_klines_1d:
        print(f"Successfully fetched {len(eth_klines_1d)} ETHUSDT 1-day klines.")
        print("First kline data point:")
        print(eth_klines_1d[0])
    else:
        print("Failed to retrieve ETHUSDT 1-day klines.")

    print("\n" + "-" * 50 + "\n")
    
    print("--- Example: Testing error handling with an invalid symbol ---")
    invalid_klines = get_klines('INVALIDPAIRXYZ', Client.KLINE_INTERVAL_1MINUTE, limit=10)
    if invalid_klines is None:
        print("Correctly handled invalid symbol and returned None.")
    else:
        print("Error handling test failed for invalid symbol.")
        
    print("\n" + "-" * 50 + "\n")

    print("--- Example: Fetching XRPUSDT 1-hour klines ---")
    # Note: '1h' is a valid string alias for Client.KLINE_INTERVAL_1HOUR
    xrp_klines = get_klines('XRPUSDT', '1h', limit=2)
    if xrp_klines:
        print(f"Successfully fetched {len(xrp_klines)} XRPUSDT 1-hour klines.")
        print("First kline data point:")
        print(xrp_klines[0])
    else:
        print("Failed to retrieve XRPUSDT 1-hour klines.")
