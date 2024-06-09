"Testing Model Configurations based on `Getting Started` notebook"
import os

import statsmodels.formula.api as smf
from dotenv import load_dotenv
from omegaconf import OmegaConf
import numpy as np
import pandas as pd

from config import QuantRegConfig
from loaders import get_local_data, load_module
from utils import separate_target_data, pinball_score

if __name__ == "__main__":
    load_dotenv()
    root = os.getenv("root")
    data_dir = os.path.join(root, "data/")

    # Pick your target day and training window. 
    target_day = pd.to_datetime(np.datetime64("2023-09-17 00:00:00"), utc=True)
    training_window = -1 # use all

    inputs, target_data = separate_target_data(get_local_data(data_dir), target_day, training_window)
    config = QuantRegConfig()
    model = load_module("model", config, inputs)
    forecast_models = dict()

    predicted_quantiles = pd.DataFrame(index=target_data.index)
    predicted_quantiles["valid_datetime"] = target_data["valid_datetime"]

    for q in range(10, 100, 10):
        print(f"Starting Predictions for q{q}")

        forecast_models[f"q{q}"] = model.fit(quantile=q / 100)
        predicted_quantiles[f"q{q}"] = forecast_models[f"q{q}"].predict(target_data)
        predicted_quantiles.loc[predicted_quantiles[f"q{q}"] < 0, f"q{q}"] = 0

    score = pinball_score(predicted_quantiles, target_data)
    print("Pinball score:", score)
    print(predicted_quantiles.head())
