"Train, score, and save models"
import os

from dotenv import load_dotenv

import utils
from config import QuantRegConfig
from loaders import get_hornsea_data, get_solar_data, load_module
from rebase_api import RebaseAPI

# TODO: Write trainer class for specific quantile and input data


class Trainer:
    def __init__(self) -> None:
        pass

    def train(self):
        pass

    def score(self):
        pass

    def save(self):
        pass


if __name__ == "__main__":
    load_dotenv()
    root = os.getenv("root")
    data_dir = os.path.join(root, "data/")

    # INIT API KEY
    api_key = os.getenv("rebase_api_key")
    rebase_client = RebaseAPI(api_key)

    solar = get_solar_data(rebase_client)
    hornsea = get_hornsea_data(rebase_client)

    inputs = utils.format_forecast_table(hornsea, solar)
