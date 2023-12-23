import datetime
import warnings

import pandas as pd


def format_forecast_table(hornsea_features, solar_features):
    table = hornsea_features.merge(
        solar_features, how="outer", on=["ref_datetime", "valid_datetime"]
    )
    table = (
        table.set_index("valid_datetime")
        .resample("30T")
        .interpolate("linear", limit=5)
        .reset_index()
    )
    table.rename(columns={"WindSpeed:100": "WindSpeed"}, inplace=True)
    return table


# Convert nwp data frame to xarray
def weather_df_to_xr(weather_data):
    weather_data["ref_datetime"] = pd.to_datetime(
        weather_data["ref_datetime"], utc=True
    )
    weather_data["valid_datetime"] = pd.to_datetime(
        weather_data["valid_datetime"], utc=True
    )

    if "point" in weather_data.columns:
        weather_data = weather_data.set_index(
            ["ref_datetime", "valid_datetime", "point"]
        )
    else:
        weather_data = pd.melt(weather_data, id_vars=["ref_datetime", "valid_datetime"])

        weather_data = pd.concat(
            [weather_data, weather_data["variable"].str.split("_", expand=True)], axis=1
        ).drop(["variable", 1, 3], axis=1)

        weather_data.rename(
            columns={0: "variable", 2: "latitude", 4: "longitude"}, inplace=True
        )

        weather_data = weather_data.set_index(
            ["ref_datetime", "valid_datetime", "longitude", "latitude"]
        )
        weather_data = weather_data.pivot(columns="variable", values="value")

    weather_data = weather_data.to_xarray()

    weather_data["ref_datetime"] = pd.DatetimeIndex(
        weather_data["ref_datetime"].values, tz="UTC"
    )
    weather_data["valid_datetime"] = pd.DatetimeIndex(
        weather_data["valid_datetime"].values, tz="UTC"
    )

    return weather_data


def prep_submission_in_json_format(
    submission_data, market_day=pd.to_datetime("today") + pd.Timedelta(1, unit="day")
):
    submission = []

    if any(submission_data["market_bid"] < 0):
        submission_data.loc[submission_data["market_bid"] < 0, "market_bid"] = 0
        warnings.warn(
            "Warning...Some market bids were less than 0 and have been set to 0"
        )

    if any(submission_data["market_bid"] > 1800):
        submission_data.loc[submission_data["market_bid"] > 1800, "market_bid"] = 1800
        warnings.warn(
            "Warning...Some market bids were greater than 1800 and have been set to 1800"
        )

    for i in range(len(submission_data.index)):
        submission.append(
            {
                "timestamp": submission_data["datetime"][i].isoformat(),
                "market_bid": submission_data["market_bid"][i],
                "probabilistic_forecast": {
                    10: submission_data["q10"][i],
                    20: submission_data["q20"][i],
                    30: submission_data["q30"][i],
                    40: submission_data["q40"][i],
                    50: submission_data["q50"][i],
                    60: submission_data["q60"][i],
                    70: submission_data["q70"][i],
                    80: submission_data["q80"][i],
                    90: submission_data["q90"][i],
                },
            }
        )

    data = {"market_day": market_day.strftime("%Y-%m-%d"), "submission": submission}

    return data

def separate_target_data(df, target_day, training_window=-1):
    """ Separates the data in the dataframe based on training data and the data from the target day.

    Parameters
    ----------
    df : Dataframe
    target_day: pandas.Timestamp
    training_window: int
        # of days to use for training. If -1, use all.

    Returns
    -------
    training_data : Dataframe
        The dataframe that should be trained on.
    target_data : Dataframe
        The dataframe that should be used to calculate the loss.
    """
    # If this is uncommented, values are only on a 6 hour interval. But not sure 
    # df = df[df['ref_datetime'] == df['valid_datetime']]
    df = df.sort_values(by=['ref_datetime', 'valid_datetime'])
    target = df.loc[(df['valid_datetime'] >= target_day) & (df['valid_datetime'] < target_day + pd.Timedelta(1, unit="day"))]
    training = df[df['valid_datetime'] < target_day]
    if training_window != -1:
        training = training[training['valid_datetime'] >= target_day - pd.Timedelta(training_window, unit="day")]
    return training.reset_index(), target

def pinball_score(predicted_quantiles, target):
    """ Calculates the pinball loss score for the predicted quantiles. Lower is better.

    Parameters
    ----------
    predicted_quantiles : Dataframe
    target: Dataframe
    
    Returns
    -------
    score : int
    """
    def pinball(y,q,alpha):
        return (y-q)*alpha*(y>=q) + (q-y)*(1-alpha)*(y<q)
    score = list()
    for qu in range(10,100,10):
        score.append(pinball(y=target["total_generation_MWh"],
            q=predicted_quantiles[f"q{qu}"],
            alpha=qu/100).mean())
    return sum(score)/len(score)