from typing import Callable

import joblib
import optuna
import pandas as pd
from fbprophet import Prophet
from optuna import Trial
from optuna.samplers import TPESampler
from sklearn.metrics import mean_squared_error


class BayesianOptimizer:
    def __init__(self, objective_function: object):
        self.objective_function = objective_function

    def build_study(self, trials: int, verbose: bool = False):
        sampler = TPESampler(seed=42)
        study = optuna.create_study(
            study_name="TPE hyperparameter",
            direction="minimize",
            sampler=sampler,
        )
        study.optimize(self.objective_function, n_trials=trials)
        if verbose:
            self.display_study_statistics(study)
        return study

    def display_study_statistics(study: optuna.create_study):
        print("Best Score:", study.best_value)
        print("Best trial:", study.best_trial.params)

    @staticmethod
    def save_params(study: optuna.create_study, params_name: str):
        params = study.best_trial.params
        joblib.dump(params, "../../parameters/" + params_name)

    @staticmethod
    def save_two_params(study: optuna.create_study, params_name: str):
        prophet_params = study.best_params
        prophet_params["growth"] = "logistic"
        prophet_params["seasonality_mode"] = "additive"
        prophet_params["weekly_seasonality"] = True
        prophet_params["daily_seasonality"] = True
        prophet_params["yearly_seasonality"] = False
        joblib.dump(prophet_params, "../../parameters/" + params_name)

    @staticmethod
    def plot_optimization_history(study: optuna.create_study) -> optuna.visualization:
        return optuna.visualization.plot_optimization_history(study)

    @staticmethod
    def plot_param_importances(study: optuna.create_study) -> optuna.visualization:
        return optuna.visualization.plot_param_importances(study)

    @staticmethod
    def plot_edf(study: optuna.create_study) -> optuna.visualization:
        return optuna.visualization.plot_edf(study)


def ontune_prophet_objective(
    train: pd.DataFrame, valid: pd.Series, cap: float, floor: float
) -> Callable[[Trial], float]:
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
            "monthly_prior": trial.suggest_discrete_uniform(
                "monthly_prior", 1, 25, 0.5
            ),
            "weekly_prior": trial.suggest_discrete_uniform("weekly_prior", 1, 25, 0.5),
            "quaterly_prior": trial.suggest_discrete_uniform(
                "quaterly_prior", 1, 25, 0.5
            ),
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

    return objective


def vcenter_prophet_objective(
    train: pd.DataFrame, valid: pd.Series, cap: float, floor: float
) -> Callable[[Trial], float]:
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
            "monthly_prior": trial.suggest_discrete_uniform(
                "monthly_prior", 1, 25, 0.5
            ),
            "weekly_prior": trial.suggest_discrete_uniform("weekly_prior", 1, 25, 0.5),
            "quaterly_prior": trial.suggest_discrete_uniform(
                "quaterly_prior", 1, 25, 0.5
            ),
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

    return objective


def two_second_prophet_objective(
    train: pd.DataFrame, valid: pd.Series, cap: float, floor: float
) -> Callable[[Trial], float]:
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

    return objective
