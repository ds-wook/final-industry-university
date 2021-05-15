import argparse

import pandas as pd
from neuralprophet import NeuralProphet

from data.dataset import two_seconds_dataset

parse = argparse.ArgumentParser("Training!")
parse.add_argument("--path", type=str, default="../../input/")
parse.add_argument("--epoch", type=int, default=30)
parse.add_argument("--file", type=str, default="future.csv")
args = parse.parse_args()

df, train, valid = two_seconds_dataset(args.path)

if __name__ == "__main__":
    prophet_params = pd.read_pickle("../../parameters/best_two_params.pkl")
    prophet_params["epoch"] = args.epoch
    model = NeuralProphet(**prophet_params)
    metrics = model.fit(df, freq="1min")
    future = model.make_future_dataframe(df, periods=120)
    forecast = model.predict(future)
    forecast.to_csv("../../submission/" + args.file, index=False)
