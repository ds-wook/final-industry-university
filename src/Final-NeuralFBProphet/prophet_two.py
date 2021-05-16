import argparse

import pandas as pd

from data.dataset import two_seconds_dataset
from model.prophet import two_prophet

parse = argparse.ArgumentParser("Training!")
parse.add_argument("--path", type=str, default="../../input/")
parse.add_argument("--epoch", type=int, default=30)
parse.add_argument("--file", type=str, default="future.csv")
args = parse.parse_args()

df, train, valid = two_seconds_dataset(args.path)

if __name__ == "__main__":
    prophet_params = pd.read_pickle("../../parameters/best_two_params.pkl")
    forecast = two_prophet(prophet_params, df)
    forecast.to_csv("../../submission/" + args.file, index=False)
