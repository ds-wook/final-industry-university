import argparse
import warnings

import numpy as np

from data.dataset import ontune_dataset
from optim.bayesian import BayesianOptimizer, ontune_prophet_objective

warnings.filterwarnings("ignore")


def define_argparser():
    parse = argparse.ArgumentParser("Optimize")
    parse.add_argument("--path", type=str, default="../../input/")
    parse.add_argument("--trials", type=int, default=20)
    parse.add_argument("--params", type=str, default="ontune_params.pkl")
    args = parse.parse_args()
    return args


def _main(args: argparse.Namespace):
    df, train, valid = ontune_dataset(args.path)
    cap = np.max(train.y)
    floor = np.min(train.y)
    objective = ontune_prophet_objective(train, valid, cap, floor)
    bayesian_optim = BayesianOptimizer(objective)
    study = bayesian_optim.build_study(trials=args.trials)
    bayesian_optim.save_params(study, args.params)


if __name__ == "__main__":
    args = define_argparser()
    _main(args)
