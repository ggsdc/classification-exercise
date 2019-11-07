import pandas as pd
from .functions.KNN_classifier import knn_classifier

pd.set_option('display.max_columns', 210)

data = pd.read_csv('./data/train.csv')
# TODO: data preprocessing.
# TODO: cross-validation?
# TODO: model looping.
# TODO: model selection.