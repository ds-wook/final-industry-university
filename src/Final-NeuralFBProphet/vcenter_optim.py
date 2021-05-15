import argparse

import joblib
import optuna
from neuralprophet import NeuralProphet
from optuna import Trial
from optuna.samplers import TPESampler
from sklearn.metrics import mean_squared_error

from data.dataset import vcenter_dataset

parse = argparse.ArgumentParser("Optimize")
parse.add_argument("--path", type=str, default="../../input/")
parse.add_argument("--trials", type=int, default=360)
parse.add_argument("--params", type=str, default="two_second_params.pkl")
args = parse.parse_args()

df, train, valid = vcenter_dataset(args.path)


def objective(trial: Trial) -> float:
    params = {
        "epochs": trial.suggest_categorical("epochs", [50, 100, 200, 300, 400, 500]),
        "batch_size": 64,
        "num_hidden_layers": trial.suggest_int("num_hidden_layers", 0, 5),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1),
        "changepoints_range": trial.suggest_discrete_uniform(
            "changepoints_range", 0.8, 0.95, 0.001
        ),
        "n_changepoints": trial.suggest_int("n_changepoints", 20, 35),
        "seasonality_mode": "additive",
        "yearly_seasonality": False,
        "weekly_seasonality": True,
        "daily_seasonality": True,
        "loss_func": "MSE",
    }
    # fit_model
    m = NeuralProphet(**params)
    m.fit(train, freq="1D")
    future = m.make_future_dataframe(
        train, periods=len(valid), n_historic_predictions=True
    )

    forecast = m.predict(future)
    valid_forecast = forecast[forecast.y.isna()]
    val_rmse = mean_squared_error(valid_forecast.yhat1, valid, squared=False)

    return val_rmse


if __name__ == "__main__":
    study = optuna.create_study(
        study_name="ontune hyperparameter",
        direction="minimize",
        sampler=TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=args.trials)
    prophet_params = study.best_params
    prophet_params["batch_size"] = 64
    prophet_params["seasonality_mode"] = "additive"
    prophet_params["loss_func"] = "MSE"
    prophet_params["weekly_seasonality"] = True
    prophet_params["daily_seasonality"] = True
    prophet_params["yearly_seasonality"] = False
    joblib.dump(prophet_params, "../../parameters/" + args.params)
