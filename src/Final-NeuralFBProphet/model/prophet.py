from typing import Dict, Union

import numpy as np
import pandas as pd
from fbprophet import Prophet
from neuralprophet import NeuralProphet


def neural_prophet(
    params: Dict[str, Union[bool, float, int]], df: pd.DataFrame, freq: str
) -> pd.DataFrame:
    model = NeuralProphet(**params)
    model.fit(df, freq=freq)
    future = model.make_future_dataframe(df, periods=144)
    forecast = model.predict(future)
    return forecast


def prophet(
    params: Dict[str, Union[bool, float, int]], df: pd.DataFrame
) -> pd.DataFrame:
    # fit_model
    model = Prophet(
        changepoint_range=params["changepoint_prior_scale"],
        n_changepoints=params["n_changepoints"],
        seasonality_prior_scale=params["seasonality_prior_scale"],
        changepoint_prior_scale=params["changepoint_prior_scale"],
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        growth="logistic",
        seasonality_mode="additive",
    )
    model.add_seasonality(
        name="yearly",
        period=365.25,
        fourier_order=params["yearly_fourier"],
        prior_scale=params["yearly_prior"],
    )
    model.add_seasonality(
        name="monthly",
        period=30.5,
        fourier_order=params["monthly_fourier"],
        prior_scale=params["monthly_prior"],
    )
    model.add_seasonality(
        name="weekly",
        period=7,
        fourier_order=params["weekly_fourier"],
        prior_scale=params["weekly_prior"],
    )
    model.add_seasonality(
        name="quaterly",
        period=365.25 / 4,
        fourier_order=params["quaterly_fourier"],
        prior_scale=params["quaterly_prior"],
    )
    cap = np.max(df.y)
    floor = np.min(df.y)
    df["cap"] = cap
    df["floor"] = floor
    model.fit(df)
    future = model.make_future_dataframe(periods=144, freq="d")
    future["cap"] = cap
    future["floor"] = floor

    forecast = model.predict(future)

    return forecast
