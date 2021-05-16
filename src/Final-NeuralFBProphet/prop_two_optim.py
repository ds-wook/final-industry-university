import argparse

import joblib
import numpy as np
import optuna
from fbprophet import Prophet
from optuna import Trial
from optuna.samplers import TPESampler
from sklearn.metrics import mean_squared_error

from data.dataset import two_seconds_dataset

parse = argparse.ArgumentParser("Optimize")
parse.add_argument("--path", type=str, default="../../input/")
parse.add_argument("--trials", type=int, default=20)
parse.add_argument("--params", type=str, default="two_second_params.pkl")
args = parse.parse_args()

df, train, valid = two_seconds_dataset(args.path)
train = df[df["ds"] < "2021-2-10"]
valid = df[df["ds"] >= "2021-2-10"]
valid["days"] = valid["ds"].apply(lambda x: x.day)
valid["hour"] = valid["ds"].apply(lambda x: x.hour)
valid["days_hour"] = valid["days"].astype(str) + "_" + valid["hour"].astype(str)
valid = valid.groupby("days_hour")["y"].agg("mean").reset_index()
cap = np.max(train.y)
floor = np.min(train.y)


def objective(trial: Trial) -> float:
    params = {
        "changepoint_range": trial.suggest_discrete_uniform(
            "changepoint_range", 0.8, 0.95, 0.001
        ),
        "n_changepoints": trial.suggest_int("n_changepoints", 20, 35),
        "changepoint_prior_scale": trial.suggest_discrete_uniform(
            "changepoint_prior_scale", 0.001, 0.5, 0.001
        ),
        "seasonality_prior_scale": trial.suggest_discrete_uniform(
            "seasonality_prior_scale", 1, 25, 0.5
        ),
        "growth": "logistic",
        "seasonality_mode": "additive",
        "yearly_seasonality": False,
        "weekly_seasonality": True,
        "daily_seasonality": True,
    }
    # fit_model
    m = Prophet(**params)
    train["cap"] = cap
    train["floor"] = floor
    m.fit(train)
    future = m.make_future_dataframe(periods=163, freq="H")

    future["cap"] = cap
    future["floor"] = floor

    forecast = m.predict(future)
    valid_forecast = forecast.tail(163)
    val_rmse = mean_squared_error(valid.y, valid_forecast.yhat, squared=False)

    return val_rmse


if __name__ == "__main__":
    study = optuna.create_study(
        study_name="ontune hyperparameter",
        direction="minimize",
        sampler=TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=args.trials)
    prophet_params = study.best_params
    prophet_params["growth"] = "logistic"
    prophet_params["seasonality_mode"] = "additive"
    prophet_params["weekly_seasonality"] = True
    prophet_params["daily_seasonality"] = True
    prophet_params["yearly_seasonality"] = False
    joblib.dump(prophet_params, "../../parameters/" + args.params)
