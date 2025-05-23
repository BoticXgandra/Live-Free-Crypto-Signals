# Live Free Crypto Signals

## Description
Live Free Crypto Signals is a Python-based application that automates the process of generating cryptocurrency trading signals. It fetches historical kline data from Binance, calculates various technical indicators, generates buy/sell/hold signals based on predefined strategies, and displays these signals through a simple web interface.

## Features
*   **Binance Data Fetching**: Retrieves kline (candlestick) data for multiple cryptocurrency symbols (e.g., BTCUSDT, ETHUSDT) and timeframes (e.g., 1h, 4h) from the Binance API.
*   **Technical Indicator Calculation**:
    *   Simple Moving Average (SMA)
    *   Exponential Moving Average (EMA)
    *   Relative Strength Index (RSI)
    *   Moving Average Convergence Divergence (MACD)
    *   Bollinger Bands
*   **Signal Generation**: Generates trading signals (BUY, SELL, HOLD) based on:
    *   RSI thresholds
    *   MACD crossovers
    *   Moving Average (SMA) crossovers
    *   Bollinger Bands price breakouts
*   **Web Interface**: A Flask-based web application displays the generated signals in a clear tabular format, showing the coin, timeframe, indicator, indicator value(s), and the resulting signal.
*   **Configurability**: Symbols and timeframes for signal generation are currently hardcoded in `main.py` but can be easily modified.
*   **Unit Tests**: Basic unit tests for indicators and signal generation logic are included.

## Setup Instructions

1.  **Clone the Repository**:
    ```bash
    git clone <repository_url>
    cd LiveFreeCryptoSignals
    ```
    (Replace `<repository_url>` with the actual URL of the repository.)

2.  **Create a Virtual Environment** (Recommended):
    ```bash
    python -m venv venv
    ```
    Activate the virtual environment:
    *   On Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    *   On macOS and Linux:
        ```bash
        source venv/bin/activate
        ```

3.  **Install Dependencies**:
    Ensure your virtual environment is activated, then run:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

1.  **Start the Web Application**:
    From the project root directory (`LiveFreeCryptoSignals/`), run:
    ```bash
    python main.py
    ```
    You should see output indicating the Flask development server is running.

2.  **Access the Web Interface**:
    Open your web browser and navigate to:
    [http://0.0.0.0:8080/](http://0.0.0.0:8080/) or [http://localhost:8080/](http://localhost:8080/)

    The application will fetch data and generate signals upon loading the page, which might take a few moments.

## Running Tests

1.  **Execute Unit Tests**:
    Ensure you are in the project root directory (`LiveFreeCryptoSignals/`) and have installed dependencies (including `pytest`). Then run:
    ```bash
    pytest
    ```
    Or, to be more explicit if you have multiple Python versions or projects:
    ```bash
    python -m pytest
    ```

## Modules

*   **`main.py`**: The main entry point for the application. It initializes and runs the Flask web server and contains the core logic for orchestrating data fetching, indicator calculation, and signal generation (`process_all_signals` function).
*   **`binance_client/`**: Module responsible for interacting with the Binance API.
    *   `api.py`: Contains functions to fetch kline data.
*   **`indicators/`**: Module for calculating various technical indicators.
    *   `moving_average.py`: Calculates SMA and EMA.
    *   `rsi.py`: Calculates RSI.
    *   `macd.py`: Calculates MACD line, signal line, and histogram.
    *   `bollinger_bands.py`: Calculates Bollinger Bands.
*   **`signals/`**: Module for generating trading signals based on indicator values.
    *   `generator.py`: Contains functions for different signal strategies (RSI, MACD crossover, MA crossover, Bollinger Bands).
*   **`webapp/`**: The Flask web application package.
    *   `__init__.py`: Initializes the Flask app.
    *   `routes.py`: Defines the web routes (e.g., the main page to display signals).
    *   `templates/`: Contains HTML templates (e.g., `index.html`).
    *   `static/`: (Currently empty) Intended for static files like CSS or JavaScript if needed.
*   **`tests/`**: Contains unit tests for the application.
    *   `test_indicators.py`: Tests for the indicator calculation functions.
    *   `test_signals.py`: Tests for the signal generation functions.
*   **`requirements.txt`**: Lists the Python dependencies for the project.
*   **`.gitignore`**: Specifies intentionally untracked files that Git should ignore.
