import argparse

import pandas as pd

from data.dataset import vcenter_dataset
from model.prophet import prophet

parse = argparse.ArgumentParser("Training!")
parse.add_argument("--path", type=str, default="../../input/")
parse.add_argument("--epoch", type=int, default=30)
parse.add_argument("--file", type=str, default="future.csv")
args = parse.parse_args()

df, train, valid = vcenter_dataset(args.path)

if __name__ == "__main__":
    prophet_params = pd.read_pickle("../../parameters/best_two_params.pkl")
    forecast = prophet(prophet_params, df)
    forecast.to_csv("../../submission/" + args.file, index=False)
