from flask import Flask

# Create a Flask app instance
app = Flask(__name__)

# Import routes
from . import routes
