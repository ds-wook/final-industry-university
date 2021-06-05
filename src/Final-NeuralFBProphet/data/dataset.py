from typing import Tuple

import pandas as pd


def two_seconds_dataset(path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(
        path + "2s/15day_0201_0215.csv", usecols=["ds", "y"], parse_dates=["ds"]
    )
    df["ds"] = pd.to_datetime(df["ds"])
    train = df[df["ds"] < "2021-2-10"]
    valid = df[df["ds"] >= "2021-2-10"].copy()
    valid["ds"] = valid["ds"].astype(str).map(lambda x: x[:16])
    valid["ds"] = pd.to_datetime(valid["ds"])
    valid = valid.groupby("ds")["y"].agg("mean").reset_index()
    return df, train, valid


def ontune_dataset(path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    df = pd.read_csv(
        path + "10min/ontune2016.csv", usecols=["date", "value"], parse_dates=["date"]
    )
    df.rename(columns={"date": "ds", "value": "y"}, inplace=True)
    train = df[df["ds"] < "2021-03-20"]
    valid = df[df["ds"] >= "2021-03-20"].copy()
    valid["days"] = valid["ds"].apply(lambda x: x.day)
    valid = valid.groupby("days")["y"].agg("mean").reset_index()
    return df, train, valid


def vcenter_dataset(path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    df = pd.read_csv(
        path + "10min/vcenter.csv", usecols=["date", "value"], parse_dates=["date"]
    )
    df.rename(columns={"date": "ds", "value": "y"}, inplace=True)
    train = df[df["ds"] < "2021-02-11"]
    valid = df[df["ds"] >= "2021-02-11"].copy()
    valid["days"] = valid["ds"].apply(lambda x: x.day)
    valid = valid.groupby("days")["y"].agg("mean").reset_index()
    return df, train, valid
