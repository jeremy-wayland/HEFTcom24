import argparse
import os
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv

import utils
from loaders import get_hornsea_data, get_next_day_market_times, get_solar_data
from rebase_api import RebaseAPI

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--submit",
        type=bool,
        default=False,
        help="If True, submit a forecast for competition.",
    )
    args = parser.parse_args()
    this = sys.modules[__name__]
    # INIT API KEY
    load_dotenv()
    api_key = os.getenv("rebase_api_key")
    rebase_client = RebaseAPI(api_key)

    # LOAD DATA
    solar = get_solar_data(rebase_client)
    hornsea = get_hornsea_data(rebase_client)

    current_forecasts = utils.format_forecast_table(hornsea, solar)

    # FORECASTING
    for quantile in range(10, 100, 10):
        # stub
        current_forecasts[f"q{quantile}"] = np.inf
        # TODO: Load trained models and predict

    # PREPARE SUBMISSION
    submission_data = pd.DataFrame({"datetime": get_next_day_market_times()})
    submission_data = submission_data.merge(
        current_forecasts, how="left", left_on="datetime", right_on="valid_datetime"
    )
    submission_data["market_bid"] = submission_data["q50"]

    submission_data = utils.prep_submission_in_json_format(submission_data)
    print(submission_data)

    # EXCECUTE SUBMIT
    if args.submit:
        rebase_client.submit(submission_data)
