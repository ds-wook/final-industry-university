import argparse

import joblib
import numpy as np
import optuna
from fbprophet import Prophet
from optuna import Trial
from optuna.samplers import TPESampler
from sklearn.metrics import mean_squared_error

from data.dataset import vcenter_dataset

parse = argparse.ArgumentParser("Optimize")
parse.add_argument("--path", type=str, default="../../input/")
parse.add_argument("--trials", type=int, default=20)
parse.add_argument("--params", type=str, default="two_second_params.pkl")
args = parse.parse_args()

df, train, valid = vcenter_dataset(args.path)
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
        "yearly_fourier": trial.suggest_int("yearly_fourier", 5, 15),
        "monthly_fourier": trial.suggest_int("monthly_fourier", 3, 12),
        "weekly_fourier": trial.suggest_int("weekly_fourier", 3, 7),
        "quaterly_fourier": trial.suggest_int("quaterly_fourier", 3, 10),
        "yearly_prior": trial.suggest_discrete_uniform("yearly_prior", 1, 25, 0.5),
        "monthly_prior": trial.suggest_discrete_uniform("monthly_prior", 1, 25, 0.5),
        "weekly_prior": trial.suggest_discrete_uniform("weekly_prior", 1, 25, 0.5),
        "quaterly_prior": trial.suggest_discrete_uniform("quaterly_prior", 1, 25, 0.5),
        "growth": "logistic",
        "seasonality_mode": "additive",
        "weekly_seasonality": True,
        "daily_seasonality": True,
    }
    # fit_model
    model = Prophet(
        changepoint_range=params["changepoint_prior_scale"],
        n_changepoints=params["n_changepoints"],
        changepoint_prior_scale=params["changepoint_prior_scale"],
        seasonality_prior_scale=params["seasonality_prior_scale"],
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=True,
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
    train["cap"] = cap
    train["floor"] = floor
    model.fit(train)
    future = model.make_future_dataframe(periods=144, freq="d")
    future["cap"] = cap
    future["floor"] = floor

    forecast = model.predict(future)
    valid_forecast = forecast.tail(7)

    rmse = mean_squared_error(valid.y, valid_forecast.yhat, squared=False)

    return rmse


if __name__ == "__main__":
    study = optuna.create_study(
        study_name="ontune hyperparameter",
        direction="minimize",
        sampler=TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=args.trials)
    prophet_params = study.best_params
    joblib.dump(prophet_params, "../../parameters/" + args.params)
