"""
Defines the routes for the Flask web application.

This module handles the web page generation, primarily the main page
that displays the cryptocurrency trading signals. It uses Flask's routing
decorators to map URLs to Python functions.
"""
from flask import render_template
from . import app  # Import the app instance from webapp/__init__.py

# To import `process_all_signals` from `main.py` located in the project root:
import sys
from pathlib import Path

# Add the project root directory (LiveFreeCryptoSignals) to sys.path
# This allows us to import modules from the root, like `main.py`.
# Path(__file__) is the path to this current file (routes.py)
# .resolve() makes it an absolute path
# .parent is webapp/
# .parent.parent is LiveFreeCryptoSignals/ (the project root)
project_root = Path(__file__).resolve().parent.parent 
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now we can import process_all_signals from main.py
from main import process_all_signals


@app.route('/')
@app.route('/index') # Define alternative route '/index' for the same page
def index():
    """
    Handles requests to the main page ('/' or '/index').

    This function calls `process_all_signals()` from `main.py` to fetch the latest
    kline data, calculate indicators, and generate trading signals.
    It then renders the `index.html` template, passing the signal data
    and a title to be displayed on the web page.

    Returns:
        str: The rendered HTML content of the index page.
    """
    # Fetch all signal data by calling the core processing function.
    # This operation can be time-consuming as it involves API calls and calculations.
    print("Web request received, processing signals...") # Log to console
    signals_data = process_all_signals()
    print(f"Finished processing signals. Found {len(signals_data)} signals to display.")
    
    # Render the main HTML page, passing the title and the list of signals.
    # Flask will look for 'index.html' in the 'templates' directory
    # (LiveFreeCryptoSignals/webapp/templates/index.html).
    return render_template('index.html', title='Live Crypto Signals', signals=signals_data)
