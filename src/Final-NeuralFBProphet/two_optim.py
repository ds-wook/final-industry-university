import warnings

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from neuralprophet import NeuralProphet
from optuna import Trial
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")
