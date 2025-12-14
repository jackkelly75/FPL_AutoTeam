from .fixture_transfer import get_fixture_transfers
from .import_data_func import import_data
from .review_points import review_past_weeks

def run_predictions(gameweek: int):
    data = import_data()
    transfers = get_fixture_transfers(data, gameweek)
    review = review_past_weeks(data, gameweek)
    return {"transfers": transfers, "review": review}
