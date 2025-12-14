"""
fpl_predictor package
=====================

Tools for predicting Fantasy Premier League points and transfers.
"""

# Import key functions so they’re available at the top level
from .prediction import run_predictions
from .fixture_transfer import get_fixture_transfers
from .import_data_func import import_data
from .review_points import review_past_weeks

# Define what gets exported when someone does `from fpl_predictor import *`
__all__ = [
    "run_predictions",
    "get_fixture_transfers",
    "import_data",
    "review_past_weeks",
]

